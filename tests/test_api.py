"""API tests. Embeddings and the LLM are faked, so no Ollama or model download is needed."""

import re
import shutil
import subprocess
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import config, main, sessions
from tests.conftest import FakeLLM


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM(reply="fake answer")


@pytest.fixture
def client(llm, monkeypatch):
    """A test client whose startup loads FAKE models instead of the real ones."""
    monkeypatch.setattr(main, "load_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(main, "load_llm", lambda: llm)
    with TestClient(main.app) as test_client:  # `with` runs the lifespan (startup)
        yield test_client


def new_session(client) -> str:
    return client.post("/session").json()["session_id"]


def upload_text(client, sid: str, name: str, text: str):
    return client.post(f"/upload/{sid}", files={"file": (name, text.encode(), "text/plain")})


def make_orders_db() -> None:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
        INSERT INTO customers VALUES (1, 'Alice');
        INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 20.0), (3, 1, 30.0);
        """
    )
    conn.commit()
    conn.close()


def order_count() -> int:
    conn = sqlite3.connect(config.SQLITE_PATH)
    try:
        return conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        conn.close()


# ----------------------------------------------------------------- basics ---
def test_health_when_ollama_is_up(client, monkeypatch):
    monkeypatch.setattr(main, "_ollama_reachable", lambda: True)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": True, "model": config.OLLAMA_MODEL}


def test_health_when_ollama_is_down(client, monkeypatch):
    monkeypatch.setattr(main, "_ollama_reachable", lambda: False)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"] is False


def test_ollama_reachable_returns_false_when_connection_fails(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", refuse)
    assert main._ollama_reachable() is False


def test_create_session(client):
    response = client.post("/session")
    assert response.status_code == 200
    sid = response.json()["session_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", sid)
    assert (config.DATA_DIR / "sessions" / sid / "uploads").is_dir()


def test_root_serves_chat_page_and_docs_exist(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "<html" in page.text.lower()
    assert client.get("/docs").status_code == 200


# ----------------------------------------------------------------- upload ---
def test_upload_txt_success(client):
    sid = new_session(client)
    response = upload_text(client, sid, "notes.txt", "Cats are wonderful pets. " * 50)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks"] >= 1
    assert "notes.txt" in body["message"]


def test_upload_sanitizes_filename_and_adds_uuid_prefix(client):
    sid = new_session(client)
    response = upload_text(client, sid, "../../evil.txt", "harmless text")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.txt"
    saved = list(sessions.uploads_dir(sid).iterdir())
    assert len(saved) == 1
    assert re.fullmatch(r"[0-9a-f]{8}_evil\.txt", saved[0].name)
    # Nothing was written outside the session's uploads folder.
    assert not (config.DATA_DIR / "evil.txt").exists()
    assert not (config.DATA_DIR / "sessions" / "evil.txt").exists()


def test_upload_unsupported_type_returns_415(client):
    sid = new_session(client)
    response = client.post(f"/upload/{sid}", files={"file": ("virus.exe", b"MZ...", "application/octet-stream")})

    assert response.status_code == 415
    assert list(sessions.uploads_dir(sid).iterdir()) == []


def test_upload_oversized_file_returns_413_and_keeps_nothing(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)

    # Slightly over the limit: caught while streaming to disk.
    response = upload_text(client, sid, "big.txt", "x" * 5000)

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()
    assert list(sessions.uploads_dir(sid).iterdir()) == []  # partial file removed


def test_upload_far_over_limit_is_rejected_early_by_content_length(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)

    response = client.post(f"/upload/{sid}", files={"file": ("huge.txt", b"x" * (2 * 1024 * 1024))})

    assert response.status_code == 413
    assert list(sessions.uploads_dir(sid).iterdir()) == []


def test_upload_exactly_at_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)
    assert upload_text(client, sid, "ok.txt", "a" * 1000).status_code == 200


def test_upload_to_unknown_session_returns_404(client):
    response = upload_text(client, "0" * 32, "a.txt", "hello")
    assert response.status_code == 404


def test_upload_to_malformed_session_returns_400(client):
    response = upload_text(client, "not-a-real-id", "a.txt", "hello")
    assert response.status_code == 400


def test_upload_empty_file_returns_400_and_keeps_nothing(client):
    sid = new_session(client)
    response = client.post(f"/upload/{sid}", files={"file": ("empty.txt", b"", "text/plain")})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert list(sessions.uploads_dir(sid).iterdir()) == []


def test_upload_corrupt_pdf_returns_400(client):
    sid = new_session(client)
    response = client.post(f"/upload/{sid}", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower()


def test_upload_without_a_file_returns_400(client):
    sid = new_session(client)
    assert client.post(f"/upload/{sid}").status_code == 400


# ------------------------------------------------------------------- chat ---
def test_chat_before_any_upload_returns_friendly_message(client, llm):
    sid = new_session(client)
    response = client.post("/chat", json={"session_id": sid, "question": "Hello?"})

    assert response.status_code == 200
    assert "upload" in response.json()["answer"].lower()
    assert response.json()["sources"] == []
    assert llm.prompts == []


def test_chat_after_upload_returns_answer_and_sources(client, llm):
    sid = new_session(client)
    upload_text(client, sid, "facts.txt", "The capital of Testland is Examplecity.")

    response = client.post("/chat", json={"session_id": sid, "question": "Capital of Testland?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["sources"] == [{"file": "facts.txt", "page": None}]
    assert body["answered"] is True
    assert body["passages"] == [
        {"file": "facts.txt", "page": None, "text": "The capital of Testland is Examplecity."}
    ]
    assert "Examplecity" in llm.prompts[0]


def test_chat_unknown_session_returns_404(client):
    response = client.post("/chat", json={"session_id": "f" * 32, "question": "hi"})
    assert response.status_code == 404


@pytest.mark.parametrize("bad_id", ["../x", "a/b", "/etc/passwd", "", "XYZ"])
def test_chat_invalid_session_id_returns_400(client, bad_id):
    response = client.post("/chat", json={"session_id": bad_id, "question": "hi"})
    assert response.status_code == 400


def test_session_a_cannot_see_session_b_documents(client, llm):
    a = new_session(client)
    b = new_session(client)
    upload_text(client, a, "secret.txt", "The vault password is swordfish.")

    # B has uploaded nothing, so B must get no answer, no sources, and no LLM call.
    from_b = client.post("/chat", json={"session_id": b, "question": "What is the vault password?"})
    assert from_b.status_code == 200
    assert from_b.json()["sources"] == []
    assert "swordfish" not in from_b.text
    assert llm.prompts == []

    # A can see its own document.
    from_a = client.post("/chat", json={"session_id": a, "question": "What is the vault password?"})
    assert from_a.json()["sources"] == [{"file": "secret.txt", "page": None}]
    assert "swordfish" in llm.prompts[0]


def test_session_b_with_its_own_document_never_gets_a_documents(client, llm):
    a = new_session(client)
    b = new_session(client)
    upload_text(client, a, "a.txt", "Alpha secret: AAAA-1111.")
    upload_text(client, b, "b.txt", "Bravo public: BBBB-2222.")

    response = client.post("/chat", json={"session_id": b, "question": "secret?"})

    assert response.json()["sources"] == [{"file": "b.txt", "page": None}]
    assert "AAAA-1111" not in llm.prompts[0]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_chat_rejects_bad_questions_with_400(client, question):
    sid = new_session(client)
    response = client.post("/chat", json={"session_id": sid, "question": question})
    assert response.status_code == 400
    assert "question" in response.json()["detail"]


def test_chat_accepts_a_question_of_exactly_1000_chars(client):
    sid = new_session(client)
    response = client.post("/chat", json={"session_id": sid, "question": "x" * 1000})
    assert response.status_code == 200


def test_chat_missing_fields_returns_400(client):
    assert client.post("/chat", json={}).status_code == 400


# -------------------------------------------------------------------- sql ---
def test_sql_endpoint_returns_sql_rows_and_answer(client, llm):
    make_orders_db()
    llm.replies = ["SELECT COUNT(*) AS n FROM orders", "There are 3 orders."]

    response = client.post("/sql", json={"question": "How many orders?"})

    assert response.status_code == 200
    assert response.json() == {
        "sql": "SELECT COUNT(*) AS n FROM orders",
        "rows": [{"n": 3}],
        "answer": "There are 3 orders.",
    }


def test_sql_delete_request_is_blocked_and_data_unchanged(client, llm):
    make_orders_db()
    llm.replies = ["DELETE FROM orders"]

    response = client.post("/sql", json={"question": "Delete all orders"})

    assert response.status_code == 200
    assert response.json() == {
        "sql": "DELETE FROM orders",
        "answer": "Blocked: only single SELECT queries are allowed.",
    }
    assert order_count() == 3


def test_sql_without_database_returns_400_with_hint(client):
    response = client.post("/sql", json={"question": "How many orders?"})
    assert response.status_code == 400
    assert "create_sample_db" in response.json()["detail"]


def test_sql_rejects_empty_question(client):
    assert client.post("/sql", json={"question": ""}).status_code == 400


# ------------------------------------------------------------ error paths ---
class BrokenLLM:
    """An LLM whose every call fails with the given exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, prompt, **kwargs):
        raise self.error


@pytest.mark.parametrize(
    "error",
    [ConnectionError("Failed to connect to Ollama"), httpx.ConnectError("refused")],
)
def test_chat_returns_503_with_helpful_message_when_ollama_is_down(client, error):
    sid = new_session(client)
    upload_text(client, sid, "a.txt", "Some content here.")
    client.app.state.llm = BrokenLLM(error)

    response = client.post("/chat", json={"session_id": sid, "question": "What?"})

    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_sql_returns_503_when_ollama_is_down(client):
    make_orders_db()
    client.app.state.llm = BrokenLLM(ConnectionError("down"))

    response = client.post("/sql", json={"question": "How many orders?"})

    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_llm_timeout_returns_503(client):
    make_orders_db()
    client.app.state.llm = BrokenLLM(httpx.ReadTimeout("too slow"))

    response = client.post("/sql", json={"question": "How many orders?"})

    assert response.status_code == 503
    assert "too long" in response.json()["detail"]


def test_unexpected_errors_do_not_leak_details(llm, monkeypatch):
    monkeypatch.setattr(main, "load_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(main, "load_llm", lambda: BrokenLLM(RuntimeError("secret internal detail /home/x")))
    make_orders_db()

    # raise_server_exceptions=False lets us see the response a real client would get.
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/sql", json={"question": "How many orders?"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "secret" not in response.text
    assert "Traceback" not in response.text


# ------------------------------------------------------------- chat page ---
PAGE = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")


def test_page_script_only_uses_element_ids_that_exist():
    html, script = PAGE.split("<script>", 1)
    used = set(re.findall(r'\$\("([\w-]+)"\)', script))
    defined = set(re.findall(r'id="([\w-]+)"', html))
    assert used <= defined, f"script uses ids missing from the page: {used - defined}"


def test_page_loads_nothing_from_the_internet():
    """The app is 'fully offline': no CDN scripts, fonts, images or stylesheets."""
    assert not re.search(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", PAGE)
    assert not re.search(r"url\(\s*[\"']?\s*(?:https?:)?//", PAGE)
    assert "@import" not in PAGE


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_has_valid_javascript_syntax(tmp_path):
    script_file = tmp_path / "page.js"
    script_file.write_text(PAGE.split("<script>", 1)[1].split("</script>", 1)[0])
    result = subprocess.run(["node", "--check", str(script_file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_page_only_offers_the_documents_flow():
    """The chat page is upload-and-ask only; SQL mode lives in the API, not the page."""
    assert "mode-sql" not in PAGE and "SQL database" not in PAGE
    assert "/sql" not in PAGE
    assert 'id="file"' in PAGE and 'id="new-session"' in PAGE and "/chat" in PAGE


# ------------------------------------------------------------ paste text ---
def paste(client, sid: str, text: str):
    return client.post(f"/paste/{sid}", json={"text": text})


def test_paste_adds_text_and_it_can_be_asked_about(client, llm):
    sid = new_session(client)

    response = paste(client, sid, "The vault code is 4321 and the office closes at five.")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "Pasted text 1"
    assert body["message"] == "Added Pasted text 1."
    assert body["chunks"] >= 1

    answer = client.post("/chat", json={"session_id": sid, "question": "What is the vault code?"}).json()
    assert answer["sources"] == [{"file": "Pasted text 1", "page": None}]
    assert "4321" in llm.prompts[0]


def test_each_paste_becomes_its_own_numbered_document(client):
    sid = new_session(client)
    assert paste(client, sid, "first note").json()["filename"] == "Pasted text 1"
    assert paste(client, sid, "second note").json()["filename"] == "Pasted text 2"
    # Pasted text and real uploads live side by side.
    assert upload_text(client, sid, "file.txt", "third note").status_code == 200
    assert paste(client, sid, "fourth note").json()["filename"] == "Pasted text 3"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n"])
def test_paste_of_blank_text_returns_400(client, text):
    sid = new_session(client)
    response = paste(client, sid, text)
    assert response.status_code == 400
    assert "no text" in response.json()["detail"].lower()
    assert list(sessions.uploads_dir(sid).iterdir()) == []


def test_paste_without_a_text_field_returns_400(client):
    sid = new_session(client)
    assert client.post(f"/paste/{sid}", json={}).status_code == 400


def test_paste_to_unknown_session_returns_404(client):
    assert paste(client, "0" * 32, "hello").status_code == 404


def test_paste_to_malformed_session_returns_400(client):
    assert paste(client, "not-a-real-id", "hello").status_code == 400


def test_paste_over_the_size_limit_returns_413_and_keeps_nothing(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)

    response = paste(client, sid, "x" * 5000)

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()
    assert list(sessions.uploads_dir(sid).iterdir()) == []


def test_paste_limit_counts_bytes_not_characters(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)
    # 600 characters, but each "é" is 2 bytes in UTF-8, so 1,200 bytes.
    assert paste(client, sid, "é" * 600).status_code == 413
    assert paste(client, sid, "é" * 400).status_code == 200  # 800 bytes


def test_paste_far_over_limit_is_rejected_early_by_content_length(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    sid = new_session(client)
    assert paste(client, sid, "x" * (2 * 1024 * 1024)).status_code == 413


def test_pasted_text_is_private_to_its_session(client, llm):
    a = new_session(client)
    b = new_session(client)
    paste(client, a, "The secret is swordfish.")

    from_b = client.post("/chat", json={"session_id": b, "question": "What is the secret?"}).json()

    assert from_b["sources"] == []
    assert llm.prompts == []


def test_paste_survives_odd_characters(client):
    sid = new_session(client)
    assert paste(client, sid, "Emoji 🚀 and accents café — “quotes”").status_code == 200


def test_page_has_a_working_paste_box():
    assert 'id="paste-btn"' in PAGE and 'id="paste-dialog"' in PAGE
    assert "/paste/" in PAGE


def test_page_treats_only_an_unknown_session_reply_as_an_expired_session():
    """A bare 404 (e.g. an older server with no /paste route) must not be reported as 'session expired'."""
    assert "status === 404)" not in PAGE  # no blanket 404 handling
    assert "function sessionGone" in PAGE and "unknown session" in PAGE
    assert PAGE.count("sessionGone(e)") == 3  # upload, paste and chat all use it
    assert "older version" in PAGE  # the helpful message for a plain "Not Found"


def test_page_shows_the_matching_text_under_answers():
    assert "function renderPassages" in PAGE
    assert "Show matching text" in PAGE and "Related text from your document" in PAGE
    assert "data.passages" in PAGE and "data.answered" in PAGE


def test_chat_endpoint_returns_related_text_instead_of_a_dead_end(client, llm):
    sid = new_session(client)
    upload_text(client, sid, "notes.txt", "Git is used to track changes. Apples are red.")
    llm.replies = ["I don't know based on the documents."]

    body = client.post("/chat", json={"session_id": sid, "question": "what is gti"}).json()

    assert body["answered"] is False
    assert "I don't know" not in body["answer"]
    assert body["passages"] and "Git is used" in body["passages"][0]["text"]
    assert body["sources"] == [{"file": "notes.txt", "page": None}]


# ------------------------------------------------------ debug endpoint ---
def test_debug_chat_does_not_exist_unless_debugging_is_switched_on(client):
    sid = new_session(client)
    response = client.post("/debug/chat", json={"session_id": sid, "question": "hello"})
    assert response.status_code == 404
    assert "/debug/chat" not in client.get("/openapi.json").text


def test_debug_chat_returns_the_answer_and_the_full_trace(client, llm, monkeypatch):
    monkeypatch.setattr(config, "RAG_DEBUG", True)
    sid = new_session(client)
    upload_text(client, sid, "facts.txt", "The capital of Testland is Examplecity.")

    response = client.post("/debug/chat", json={"session_id": sid, "question": "Capital of Testland?"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["answer"] == "fake answer"
    trace = body["trace"]
    assert trace["route"] == "search" and trace["prompt"] == llm.prompts[0]
    assert trace["candidates"][0]["file"] == "facts.txt"


def test_debug_chat_still_validates_the_session(client, monkeypatch):
    monkeypatch.setattr(config, "RAG_DEBUG", True)
    assert client.post("/debug/chat", json={"session_id": "0" * 32, "question": "hi"}).status_code == 404
    assert client.post("/debug/chat", json={"session_id": "../x", "question": "hi"}).status_code == 400
