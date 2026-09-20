"""Tests for ingest() and ask() using fake embeddings and a fake LLM."""

import threading
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS

from app import config, rag, sessions
from app.rag import (
    CorruptDocumentError,
    DocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from tests.conftest import FakeLLM


def put_file(sid: str, name: str, content: str | bytes) -> Path:
    """Write a file into the session's uploads folder and return its path."""
    path = sessions.uploads_dir(sid) / name
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


def make_text_pdf(text: str) -> bytes:
    """Build a tiny one-page PDF containing `text` (so we don't need a PDF library to write)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return out


def make_blank_pdf() -> bytes:
    """A valid PDF whose only page has no text (like a scanned image)."""
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def sid() -> str:
    return sessions.create_session()


def test_load_llm_caps_answer_length_and_sets_timeout(monkeypatch):
    """A runaway answer must be limited by num_predict (no Ollama needed to build the client)."""
    from app import config

    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 123)
    monkeypatch.setattr(config, "LLM_TIMEOUT", 45)

    llm = rag.load_llm()

    assert llm.num_predict == 123
    assert llm.temperature == config.LLM_TEMPERATURE
    assert llm.client_kwargs == {"timeout": 45}


# ---------------------------------------------------------------- ingest ---
def test_ingest_txt_returns_chunk_count_and_creates_index(sid, fake_embeddings):
    path = put_file(sid, "notes.txt", "Cats are great pets. " * 100)  # ~2100 chars

    chunks = rag.ingest(sid, path, fake_embeddings)

    assert chunks > 1  # 2100 chars / 500-char chunks
    assert (sessions.index_dir(sid) / "index.faiss").exists()


def test_ingest_pdf_records_one_based_page(sid, fake_embeddings, fake_llm):
    path = put_file(sid, "doc.pdf", make_text_pdf("The launch code is banana"))

    assert rag.ingest(sid, path, fake_embeddings) >= 1

    result = rag.ask(sid, "What is the launch code?", fake_embeddings, fake_llm)
    assert result["sources"] == [{"file": "doc.pdf", "page": 1}]
    assert "banana" in fake_llm.prompts[0]


def test_ingest_adds_to_existing_index(sid, fake_embeddings, fake_llm):
    first = rag.ingest(sid, put_file(sid, "a.txt", "Alpha text. " * 30), fake_embeddings)
    second = rag.ingest(sid, put_file(sid, "b.txt", "Beta text. " * 30), fake_embeddings)

    store = FAISS.load_local(
        str(sessions.index_dir(sid)), fake_embeddings, allow_dangerous_deserialization=True
    )
    assert store.index.ntotal == first + second


def test_ingest_uses_display_name_not_path(sid, fake_embeddings, fake_llm):
    path = put_file(sid, "1a2b3c4d_report.txt", "Quarterly revenue was 5 million.")
    rag.ingest(sid, path, fake_embeddings, display_name="report.txt")

    result = rag.ask(sid, "revenue?", fake_embeddings, fake_llm)
    assert result["sources"] == [{"file": "report.txt", "page": None}]


def test_ingest_rejects_unsupported_extension(sid, fake_embeddings):
    path = put_file(sid, "malware.exe", "hello")
    with pytest.raises(UnsupportedFileTypeError):
        rag.ingest(sid, path, fake_embeddings)


def test_ingest_rejects_empty_and_blank_files(sid, fake_embeddings):
    with pytest.raises(EmptyDocumentError):
        rag.ingest(sid, put_file(sid, "empty.txt", ""), fake_embeddings)
    with pytest.raises(EmptyDocumentError):
        rag.ingest(sid, put_file(sid, "blank.txt", "   \n\n  "), fake_embeddings)


def test_ingest_rejects_pdf_without_text_and_mentions_scan(sid, fake_embeddings):
    path = put_file(sid, "scan.pdf", make_blank_pdf())
    with pytest.raises(EmptyDocumentError, match="scanned image"):
        rag.ingest(sid, path, fake_embeddings)


def test_ingest_rejects_corrupt_pdf(sid, fake_embeddings):
    path = put_file(sid, "broken.pdf", b"this is definitely not a pdf")
    with pytest.raises(CorruptDocumentError):
        rag.ingest(sid, path, fake_embeddings)


def test_ingest_rejects_non_utf8_text(sid, fake_embeddings):
    path = put_file(sid, "latin.txt", b"caf\xe9 \xff\xfe bytes")
    with pytest.raises(CorruptDocumentError):
        rag.ingest(sid, path, fake_embeddings)


def test_ingest_rejects_file_outside_session_uploads(sid, fake_embeddings, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(DocumentError):
        rag.ingest(sid, outside, fake_embeddings)


def test_failed_ingest_does_not_create_an_index(sid, fake_embeddings):
    with pytest.raises(DocumentError):
        rag.ingest(sid, put_file(sid, "empty.txt", ""), fake_embeddings)
    assert not (sessions.index_dir(sid) / "index.faiss").exists()


def test_concurrent_ingests_to_one_session_keep_every_chunk(sid, fake_embeddings):
    """The per-session lock must stop parallel uploads from overwriting each other."""
    paths = [put_file(sid, f"f{i}.txt", f"File number {i}. " * 40) for i in range(6)]
    counts: list[int] = []
    errors: list[Exception] = []

    def worker(path: Path) -> None:
        try:
            counts.append(rag.ingest(sid, path, fake_embeddings))
        except Exception as exc:  # collected so the test can fail with details
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    store = FAISS.load_local(
        str(sessions.index_dir(sid)), fake_embeddings, allow_dangerous_deserialization=True
    )
    assert store.index.ntotal == sum(counts)


# ------------------------------------------------------------------- ask ---
def test_ask_without_documents_returns_friendly_message(sid, fake_embeddings, fake_llm):
    result = rag.ask(sid, "Anything?", fake_embeddings, fake_llm)

    assert "upload" in result["answer"].lower()
    assert result["sources"] == []
    assert fake_llm.prompts == []  # the LLM must not even be called


def test_ask_prompt_contains_rules_context_and_question(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "fruit.txt", "Apples are red."), fake_embeddings)

    result = rag.ask(sid, "What colour are apples?", fake_embeddings, fake_llm)

    prompt = fake_llm.prompts[0]
    assert "using ONLY the context" in prompt
    assert "I don't know based on the documents." in prompt
    assert "Apples are red." in prompt
    assert "What colour are apples?" in prompt
    assert result["answer"] == "fake answer"
    assert result["sources"] == [{"file": "fruit.txt", "page": None}]


def test_short_document_is_sent_to_the_model_whole(sid, fake_embeddings, fake_llm, monkeypatch):
    """The search can miss a chunk (e.g. a contact header), so small documents skip it."""
    monkeypatch.setattr(config, "TOP_K", 1)  # search alone would return just ONE chunk
    paragraphs = [f"Section {n}: " + f"filler text {n}. " * 25 for n in range(6)]  # ~2,500 chars
    paragraphs[0] = "Jane Doe. Email: jane@example.com. " + paragraphs[0]
    rag.ingest(sid, put_file(sid, "resume.txt", "\n\n".join(paragraphs)), fake_embeddings)

    result = rag.ask(sid, "Which section is the longest", fake_embeddings, fake_llm)

    prompt = fake_llm.prompts[0]
    assert "jane@example.com" in prompt
    assert all(f"Section {n}:" in prompt for n in range(6))  # every part is present
    assert result["sources"] == [{"file": "resume.txt", "page": None}]


def test_long_document_still_uses_similarity_search(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 100)  # pretend this doc is "long"
    monkeypatch.setattr(config, "TOP_K", 2)
    text = "\n\n".join(f"Topic {n}: " + f"detail {n}. " * 40 for n in range(6))
    rag.ingest(sid, put_file(sid, "long.txt", text), fake_embeddings)

    rag.ask(sid, "anything", fake_embeddings, fake_llm)

    context = fake_llm.prompts[0].split("---\n")[1]
    assert len(context.strip().split("\n\n")) <= 2


def test_ask_retrieves_at_most_top_k_chunks(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)  # force the search path
    rag.ingest(sid, put_file(sid, "long.txt", "Sentence about topic. " * 300), fake_embeddings)

    rag.ask(sid, "topic?", fake_embeddings, fake_llm)

    # Context chunks are separated by blank lines; TOP_K is 3.
    context = fake_llm.prompts[0].split("---\n")[1]
    assert len(context.strip().split("\n\n")) <= 3


def test_ask_dont_know_shows_the_closest_parts_instead_of_a_dead_end(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Some unrelated text."), fake_embeddings)
    llm = FakeLLM(reply="I don't know based on the documents.")

    result = rag.ask(sid, "Who is the CEO?", fake_embeddings, llm)

    assert result["answered"] is False
    assert result["answer"].startswith("I couldn't find anything about that in your document.")
    assert "may not be related" in result["answer"]  # honest: none of the question's words occur
    assert result["passages"] == [{"file": "a.txt", "page": None, "text": "Some unrelated text."}]
    assert result["sources"] == [{"file": "a.txt", "page": None}]


def test_wrapped_and_paraphrased_dont_know_answers_also_show_the_closest_parts(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Some unrelated text."), fake_embeddings)

    replies = [
        'Based on the context, the answer is "I don\'t know based on the documents."',  # wrapped sentence
        "The CEO's favorite color is not mentioned in the given text.",  # a paraphrase
        "The address is not available in this context.",  # another paraphrase
        "",  # nothing at all
    ]
    for reply in replies:
        result = rag.ask(sid, "Q?", fake_embeddings, FakeLLM(reply=reply))
        assert result["answered"] is False, reply
        assert result["passages"], reply
        assert "unrelated text" in result["passages"][0]["text"], reply


def test_ask_sources_are_deduplicated_filenames_only(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "one.txt", "Word. " * 400), fake_embeddings)

    result = rag.ask(sid, "Word?", fake_embeddings, fake_llm)

    assert result["sources"] == [{"file": "one.txt", "page": None}]  # 3 chunks, 1 source
    assert "/" not in result["sources"][0]["file"]


def test_ask_unknown_session_raises(fake_embeddings, fake_llm):
    with pytest.raises(sessions.SessionNotFoundError):
        rag.ask("0" * 32, "hi", fake_embeddings, fake_llm)


def test_session_a_cannot_see_session_b_documents(fake_embeddings):
    a = sessions.create_session()
    b = sessions.create_session()
    rag.ingest(a, put_file(a, "secret.txt", "The password is swordfish."), fake_embeddings)

    llm_b = FakeLLM()
    result = rag.ask(b, "What is the password?", fake_embeddings, llm_b)

    assert result["sources"] == []
    assert llm_b.prompts == []  # B has no index, so nothing from A can reach the prompt
    assert "swordfish" not in result["answer"]


# ------------------------------------------------------- short answers ---
def test_ask_tells_the_model_to_stop_at_a_blank_line(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)
    rag.ask(sid, "What colour are apples?", fake_embeddings, fake_llm)
    assert "\n\n" in fake_llm.stops[0]


def test_ask_retries_when_the_model_starts_with_a_blank_line(sid, fake_embeddings):
    """A leading blank line would trip the stop sequence and give an empty answer."""

    class ShyLLM:
        def invoke(self, prompt, stop=None, **kwargs):
            return "" if stop else "Real answer.\n\nMore rambling that should be dropped."

    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)
    result = rag.ask(sid, "What colour are apples?", fake_embeddings, ShyLLM())
    assert result["answer"] == "Real answer."


def test_trim_if_cut_off_only_touches_long_unfinished_answers(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 20)  # "long" now means 50+ characters
    cut = "First sentence is complete. Second one is also fine. Third one is cut of"
    assert rag._trim_if_cut_off(cut) == "First sentence is complete. Second one is also fine."
    # A short answer, a finished answer, and a long list with no sentence end are left alone.
    assert rag._trim_if_cut_off("Python, React") == "Python, React"
    finished = "This answer is long enough to count but it ends properly."
    assert rag._trim_if_cut_off(finished) == finished
    a_list = "Python, React, Angular, TypeScript, Node.js, AWS, Docker and more tools"
    assert rag._trim_if_cut_off(a_list) == a_list


def test_ask_trims_an_answer_that_hit_the_token_cap(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)
    llm = FakeLLM(reply="This sentence is complete. " * 30 + "and this part is cut o")

    answer = rag.ask(sid, "What colour are apples?", fake_embeddings, llm)["answer"]

    assert answer.endswith("complete.")
    assert "cut o" not in answer


# ------------------------------------------------------ overview questions ---
@pytest.mark.parametrize(
    "question",
    [
        "what is this document about",
        "Summarize this",
        "whose resume is this",
        "who is the candidate",
        "Tell me about the document",
        "who is the owner of the resume",
        "give me an overview",
        "What is the candidate's name and job title?",
        "what is this about",
        "What's this about?",
    ],
)
def test_is_overview_question_true(question):
    assert rag.is_overview_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "what is the email address",
        "How many days do customers have to return a robot?",
        "What is the CEO's favorite color?",
        "What projects mention machine learning in the document that talks about databases and cloud tools and more?",
    ],
)
def test_is_overview_question_false(question):
    assert not rag.is_overview_question(question)


def test_overview_describes_the_document_using_its_own_first_lines(sid, fake_embeddings):
    text = "Jane Doe\nData Analyst\njane@example.com | +1 555 0100\nSkills: SQL, Excel, Tableau"
    path = put_file(sid, "1a2b3c4d_Jane_Doe_Resume.txt", text)
    rag.ingest(sid, path, fake_embeddings, display_name="Jane_Doe_Resume.txt")
    llm = FakeLLM(reply="This is the resume of a data analyst. It lists skills. More text.")

    result = rag.ask(sid, "what is this document about", fake_embeddings, llm)

    assert result["answer"] == (
        "This is a resume (Jane_Doe_Resume.txt). "
        "It begins with “Jane Doe” and “Data Analyst”. "  # copied exactly; the email line is skipped
        "This is the resume of a data analyst."  # only the first sentence of the model's summary
    )
    assert result["sources"] == [{"file": "Jane_Doe_Resume.txt", "page": None}]
    assert "jane@example.com" not in result["answer"]
    assert len(llm.prompts) == 1  # the type is guessed from words; only the summary asks the model


def test_overview_uses_an_before_vowels_and_names_unknown_files_plainly(fake_embeddings):
    invoice_sid = sessions.create_session()
    rag.ingest(invoice_sid, put_file(invoice_sid, "inv.txt", "Invoice 42\nTotal due: 10"), fake_embeddings)
    invoice = rag.ask(invoice_sid, "summarize", fake_embeddings, FakeLLM(reply="Money owed."))
    assert invoice["answer"].startswith("This is an invoice (inv.txt).")

    other_sid = sessions.create_session()
    rag.ingest(other_sid, put_file(other_sid, "notes.txt", "hello there\nsome words"), fake_embeddings)
    unknown = rag.ask(other_sid, "summarize", fake_embeddings, FakeLLM(reply=""))
    assert unknown["answer"].startswith("This is the file notes.txt.")


@pytest.mark.parametrize(
    "name, start, expected",
    [
        ("Venkata_Sai_Nagalla AI Resume.pdf", "Venkata Sai Nagalla\nFull Stack Developer", "resume"),
        ("my_cv.pdf", "Some Person", "resume"),
        ("asset-inventory-20260916.pdf", "EAOVA HRMS\nAsset Inventory Report", "report"),
        ("DEVOPS.pdf", "AWS DevOps Cheat Sheet\n1. System Administration", "cheat sheet"),
        ("notes.txt", "hello there", None),
    ],
)
def test_guess_kind(name, start, expected):
    assert rag._guess_kind(name, start) == expected


def test_ask_keeps_the_list_that_follows_an_intro_ending_in_a_colon(sid, fake_embeddings):
    """The stop-at-blank-line rule must not cut 'The skills are:' away from its list."""

    class ListLLM:
        def invoke(self, prompt, stop=None, **kwargs):
            if stop:  # cut at the first blank line, like Ollama would
                return "The skills are:"
            return "The skills are:\n\n- Python\n- React\n\nUnrelated rambling."

    rag.ingest(sid, put_file(sid, "a.txt", "Skills: Python, React"), fake_embeddings)
    result = rag.ask(sid, "What skills are listed?", fake_embeddings, ListLLM())
    assert result["answer"] == "The skills are:\n- Python\n- React"


def test_overview_covers_each_file_in_the_session(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "one.txt", "First file title\nbody"), fake_embeddings)
    rag.ingest(sid, put_file(sid, "two.txt", "Second file title\nbody"), fake_embeddings)

    result = rag.ask(sid, "what is this document about", fake_embeddings, FakeLLM(reply="Report"))

    assert [s["file"] for s in result["sources"]] == ["one.txt", "two.txt"]
    assert "“First file title”" in result["answer"] and "“Second file title”" in result["answer"]


# ------------------------------------------------------- contact questions ---
@pytest.mark.parametrize(
    "question",
    [
        "what is mail id",
        "What is the email address?",
        "phone number?",
        "can you tell me how to contact the document owner any details how to contact like e mail like that",
        "Does it list a LinkedIn profile?",
    ],
)
def test_is_contact_question_true(question):
    assert rag.is_contact_question(question)


@pytest.mark.parametrize(
    "question", ["What is the refund policy?", "How many contacts are stored in the table?", "Who is the CEO?"]
)
def test_is_contact_question_false(question):
    assert not rag.is_contact_question(question)


RESUME_TEXT = (
    "Jane Doe\nData Analyst\n"
    "jane.doe@example.com  |  +1 555-010-0199  |  linkedin.com/in/janedoe\n"
    "Skills: SQL, Excel, Tableau"
)


def test_contact_question_returns_details_copied_exactly_without_calling_the_model(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "1a2b3c4d_Jane.txt", RESUME_TEXT), fake_embeddings, display_name="Jane.txt")

    result = rag.ask(
        sid, "can you tell me how to contact the document owner any details like email", fake_embeddings, fake_llm
    )

    assert result["answer"] == (
        "Contact details in “Jane.txt” (it begins with “Jane Doe”):\n"
        "Email: jane.doe@example.com\n"
        "Phone: +1 555-010-0199\n"
        "Links: linkedin.com/in/janedoe"
    )
    assert result["sources"] == [{"file": "Jane.txt", "page": None}]
    assert fake_llm.prompts == []  # copied by pattern, so the model is not involved


def test_contact_details_are_found_beyond_the_first_chunk(sid, fake_embeddings, fake_llm):
    text = "Annual Report\n" + "Some filler sentence. " * 60 + "\n\nContact: press@example.org or (555) 123-4567."
    rag.ingest(sid, put_file(sid, "report.txt", text), fake_embeddings)

    answer = rag.ask(sid, "what is the email", fake_embeddings, fake_llm)["answer"]

    assert "press@example.org" in answer and "(555) 123-4567" in answer


def test_long_id_numbers_and_dates_are_not_mistaken_for_phone_numbers(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "assets.txt", "Serial 20260916164848 on 2026-09-16, ticket 164848"), fake_embeddings)

    result = rag.ask(sid, "what is the phone number", fake_embeddings, fake_llm)

    # Nothing to copy, so it falls back to the normal model answer.
    assert result["answer"] == "fake answer"
    assert len(fake_llm.prompts) == 1


def test_contact_question_without_any_contact_details_falls_back_to_the_model(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "plain.txt", "Just a note about apples."), fake_embeddings)
    assert rag.ask(sid, "what is the email", fake_embeddings, fake_llm)["answer"] == "fake answer"


def test_contact_details_are_listed_per_file(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "a.txt", "Alpha Co\nwrite to alpha@example.com"), fake_embeddings)
    rag.ingest(sid, put_file(sid, "b.txt", "Beta Co\nwrite to beta@example.com"), fake_embeddings)

    result = rag.ask(sid, "what is the email", fake_embeddings, fake_llm)

    assert "alpha@example.com" in result["answer"] and "beta@example.com" in result["answer"]
    assert [s["file"] for s in result["sources"]] == ["a.txt", "b.txt"]


def test_long_document_only_looks_at_the_top_for_contact_details(sid, fake_embeddings, fake_llm, monkeypatch):
    """A long manual full of example addresses must not have those passed off as the owner's."""
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 100)  # treat this file as "long"
    text = "Cheat Sheet\nno contact here\n\n" + "Filler sentence. " * 80 + "\n\nRun: mail you@example.com\n"
    rag.ingest(sid, put_file(sid, "manual.txt", text), fake_embeddings)

    result = rag.ask(sid, "what is the email", fake_embeddings, fake_llm)

    assert "you@example.com" not in result["answer"]  # not extracted from deep inside the file
    assert result["answer"] == "fake answer"  # fell back to the normal model answer


def test_contact_source_page_is_where_the_details_were_found(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "doc.pdf", make_text_pdf("Write to help@example.com")), fake_embeddings)

    result = rag.ask(sid, "what is the email", fake_embeddings, fake_llm)

    assert "help@example.com" in result["answer"]
    assert result["sources"] == [{"file": "doc.pdf", "page": 1}]


def test_related_text_names_the_words_that_do_occur_in_the_document(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Git is used for version control. Apples are red."), fake_embeddings)
    llm = FakeLLM(reply="I don't know based on the documents.")

    result = rag.ask(sid, "what is git", fake_embeddings, llm)

    assert result["answered"] is False
    assert result["answer"] == (
        "I couldn't find a direct answer, but these parts of your document mention “git”:"
    )
    assert "Git is used for version control" in result["passages"][0]["text"]


def test_overview_of_pasted_text_is_described_as_pasted_text(sid, fake_embeddings):
    rag.ingest(
        sid,
        put_file(sid, "1a2b3c4d_Pasted-text-1.txt", "Meeting notes\nWe agreed to ship on Friday."),
        fake_embeddings,
        display_name="Pasted text 1",
    )

    answer = rag.ask(sid, "what is this about", fake_embeddings, FakeLLM(reply="Plans for a Friday release."))["answer"]

    assert answer == (
        "This is pasted text 1. It begins with “Meeting notes” and “We agreed to ship on Friday.”. "
        "Plans for a Friday release."
    )


# ---------------------------------------------- forgiving search (Part 1) ---
def make_long_doc(needle: str, fillers: int = 8) -> str:
    """A document of unrelated paragraphs plus one paragraph with `needle`."""
    paragraphs = [f"Section {n} covers unrelated topic number {n}. " + f"Filler sentence {n}. " * 12 for n in range(fillers)]
    paragraphs.insert(fillers // 2, needle)
    return "\n\n".join(paragraphs)


def test_a_misspelled_word_still_finds_the_right_passage(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)  # force the search path
    monkeypatch.setattr(config, "TOP_K", 1)  # the model gets ONE passage only
    doc = make_long_doc("Kubernetes is a container orchestrator that runs pods on clusters.")
    rag.ingest(sid, put_file(sid, "guide.txt", doc), fake_embeddings)

    result = rag.ask(sid, "what is kuberntes", fake_embeddings, fake_llm)  # note the typo

    assert "container orchestrator" in fake_llm.prompts[0]
    assert "Question: what is kubernetes" in fake_llm.prompts[0]  # the corrected question
    assert "Kubernetes is a container orchestrator" in result["passages"][0]["text"]


def test_a_single_word_question_finds_the_passage_that_uses_the_word(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    monkeypatch.setattr(config, "TOP_K", 1)
    doc = make_long_doc("Run git status, then git commit, then git push to share your work.")
    rag.ingest(sid, put_file(sid, "cheatsheet.txt", doc), fake_embeddings)

    rag.ask(sid, "git", fake_embeddings, fake_llm)

    assert "git commit" in fake_llm.prompts[0]


def test_passages_are_short_excerpts_centred_on_the_matching_word(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    long_paragraph = ("Unrelated words go here. " * 14) + "Terraform provisions cloud servers. " + ("More unrelated words. " * 6)
    rag.ingest(sid, put_file(sid, "iac.txt", long_paragraph), fake_embeddings)

    result = rag.ask(sid, "what is terraform", fake_embeddings, fake_llm)

    excerpts = [p["text"] for p in result["passages"]]
    assert any("Terraform provisions cloud servers" in text for text in excerpts)
    assert all(len(text) <= 330 for text in excerpts)


def test_normal_answers_also_carry_the_matching_text(sid, fake_embeddings, fake_llm):
    rag.ingest(sid, put_file(sid, "a.txt", "The vault code is 4321."), fake_embeddings)

    result = rag.ask(sid, "what is the vault code", fake_embeddings, fake_llm)

    assert result["answered"] is True
    assert result["answer"] == "fake answer"
    assert result["passages"] == [{"file": "a.txt", "page": None, "text": "The vault code is 4321."}]


def test_at_most_three_excerpts_are_shown(sid, fake_embeddings, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    rag.ingest(sid, put_file(sid, "a.txt", make_long_doc("Docker builds images.", fillers=12)), fake_embeddings)

    result = rag.ask(sid, "docker", fake_embeddings, fake_llm)

    assert 1 <= len(result["passages"]) <= 3


def test_fallback_still_works_when_the_question_is_only_filler_words(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red. Bananas are yellow."), fake_embeddings)

    result = rag.ask(sid, "what is this", fake_embeddings, FakeLLM(reply="I don't know based on the documents."))

    assert result["answered"] is False
    assert result["passages"]  # never empty-handed
    assert "may not be related" in result["answer"]


def test_a_typo_in_the_fallback_message_is_shown_corrected(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Kubernetes runs containers in pods."), fake_embeddings)

    result = rag.ask(sid, "what is kuberntes", fake_embeddings, FakeLLM(reply="I don't know based on the documents."))

    assert result["answer"].endswith("mention “kubernetes”:")


def test_typos_in_overview_and_contact_questions_are_understood(sid, fake_embeddings):
    text = "Jane Doe\nData Analyst\njane.doe@example.com  |  +1 555-010-0199\nSkills: SQL, Excel"
    rag.ingest(sid, put_file(sid, "Jane_Resume.txt", text), fake_embeddings, display_name="Jane_Resume.txt")

    overview = rag.ask(sid, "what is the documenbt about", fake_embeddings, FakeLLM(reply="A resume."))
    assert overview["answer"].startswith("This is a resume (Jane_Resume.txt).")

    contact = rag.ask(sid, "what is the emial address", fake_embeddings, FakeLLM())
    assert "jane.doe@example.com" in contact["answer"]
