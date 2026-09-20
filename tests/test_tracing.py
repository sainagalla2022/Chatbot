"""Tests for the trace of a question: what is recorded, logged and printed."""

import json
import logging

import pytest

from app import config, debug, rag, search, sessions, tracing
from tests.conftest import FakeLLM
from tests.test_rag import make_long_doc, put_file


@pytest.fixture
def sid() -> str:
    return sessions.create_session()


def ask_with_trace(sid, question, embeddings, llm):
    trace: dict = {}
    result = rag.ask(sid, question, embeddings, llm, trace=trace)
    return result, trace


# ------------------------------------------------- what a search trace holds ---
def test_search_trace_records_every_step(sid, fake_embeddings, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    monkeypatch.setattr(config, "TOP_K", 2)
    doc = make_long_doc("Kubernetes is a container orchestrator that runs pods on clusters.")
    rag.ingest(sid, put_file(sid, "guide.txt", doc), fake_embeddings)
    llm = FakeLLM(reply="It orchestrates containers.")

    result, trace = ask_with_trace(sid, "what is kuberntes", fake_embeddings, llm)

    assert trace["original_question"] == "what is kuberntes"
    assert trace["route"] == "search"
    assert trace["rewritten_query"] == "what is kubernetes"  # spelling matched to the document
    assert trace["keyword_terms"] == ["kubernetes"]
    assert trace["matched_terms"] == ["kubernetes"]
    assert trace["index"]["chunks"] >= 2 and trace["index"]["characters"] > 0
    assert trace["prompt"] == llm.prompts[0]  # the trace holds the exact prompt the model received
    assert trace["raw_answer"] == "It orchestrates containers."
    assert trace["answer"] == result["answer"] and trace["answered"] is True
    assert trace["fallback_reason"] is None
    assert set(trace["seconds"]) == {"search", "llm", "total"}


def test_candidates_carry_page_scores_and_the_sent_flag(sid, fake_embeddings, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    monkeypatch.setattr(config, "TOP_K", 2)
    doc = make_long_doc("Docker builds images and runs containers.")
    rag.ingest(sid, put_file(sid, "guide.txt", doc), fake_embeddings)

    _, trace = ask_with_trace(sid, "docker", fake_embeddings, FakeLLM())

    rows = trace["candidates"]
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["blend_score"] for row in rows] == sorted((r["blend_score"] for r in rows), reverse=True)
    assert [row["sent_to_llm"] for row in rows[:3]] == [True, True, False]  # TOP_K = 2
    assert trace["context"]["mode"] == "top_k" and trace["context"]["chunks_sent"] == 2
    assert trace["context"]["characters"] > 0
    top = rows[0]
    assert "Docker builds images" in top["preview"]
    assert top["keyword_rank"] == 1 and top["keyword_score"] > 0
    assert {"file", "page", "meaning_rank", "meaning_distance", "position"} <= set(top)


def test_whole_document_mode_marks_every_chunk_as_sent(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "small.txt", "Apples are red. " * 30), fake_embeddings)

    _, trace = ask_with_trace(sid, "what colour are apples", fake_embeddings, FakeLLM())

    assert trace["context"]["mode"] == "whole_document"
    assert trace["context"]["chunks_sent"] == trace["index"]["chunks"]
    assert all(row["sent_to_llm"] for row in trace["candidates"])


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("I don't know based on the documents.", "exact don't-know sentence"),
        ("The address is not available here.", 'phrase "not available"'),
        ("", "empty answer"),
    ],
)
def test_fallback_reason_says_why_the_answer_was_replaced(sid, fake_embeddings, reply, reason):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)

    result, trace = ask_with_trace(sid, "where is the warehouse", fake_embeddings, FakeLLM(reply=reply))

    assert trace["fallback_reason"] == reason
    assert trace["answered"] is False and result["answered"] is False


# ------------------------------------------------------ the other routes ---
def test_other_routes_are_recorded(sid, fake_embeddings):
    empty_sid = sessions.create_session()
    _, none = ask_with_trace(empty_sid, "hello", fake_embeddings, FakeLLM())
    assert none["route"] == "no_documents"

    text = "Jane Doe\nData Analyst\njane@example.com | +1 555-010-0199"
    rag.ingest(sid, put_file(sid, "Jane_Resume.txt", text), fake_embeddings, display_name="Jane_Resume.txt")
    _, contact = ask_with_trace(sid, "what is the emial", fake_embeddings, FakeLLM())
    assert contact["route"] == "contact"
    assert contact["after_trigger_typo_fix"] == "what is the email"

    _, overview = ask_with_trace(sid, "what is this document about", fake_embeddings, FakeLLM(reply="A resume."))
    assert overview["route"] == "overview"

    for trace in (none, contact, overview):  # every kind of trace can be turned into text
        assert "route" in tracing.format_trace(trace)
        json.dumps(trace)  # and into JSON


def test_trace_is_optional_and_does_not_change_the_answer(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "The vault code is 4321."), fake_embeddings)
    with_trace = rag.ask(sid, "what is the vault code", fake_embeddings, FakeLLM(), trace={})
    without = rag.ask(sid, "what is the vault code", fake_embeddings, FakeLLM())
    assert with_trace == without


# --------------------------------------------------------------- logging ---
def test_one_summary_line_is_logged_per_question_without_document_text(sid, fake_embeddings, caplog):
    rag.ingest(sid, put_file(sid, "a.txt", "The secret vault code is 4321."), fake_embeddings)

    with caplog.at_level(logging.INFO, logger="chatbot.rag"):
        rag.ask(sid, "what is the vault code", fake_embeddings, FakeLLM())

    lines = [r.getMessage() for r in caplog.records if r.name == "chatbot.rag"]
    assert len(lines) == 1
    assert 'question="what is the vault code"' in lines[0]
    assert "route=search" in lines[0] and "answered=True" in lines[0] and "chunks_sent=" in lines[0]
    assert "4321" not in caplog.text  # no document text (and so no prompt) unless debugging is on


def test_full_trace_is_logged_only_when_debugging_is_on(sid, fake_embeddings, caplog, monkeypatch):
    rag.ingest(sid, put_file(sid, "a.txt", "The secret vault code is 4321."), fake_embeddings)
    monkeypatch.setattr(config, "RAG_DEBUG", True)

    with caplog.at_level(logging.INFO, logger="chatbot.rag"):
        rag.ask(sid, "what is the vault code", fake_embeddings, FakeLLM())

    assert "PROMPT SENT TO THE MODEL" in caplog.text
    assert "4321" in caplog.text  # the chunk text is in the logged prompt


def test_setup_logging_is_idempotent():
    tracing.setup_logging()
    tracing.setup_logging()
    assert len(logging.getLogger("chatbot").handlers) == 1


# ------------------------------------------------------------ formatting ---
def test_format_trace_shows_the_important_parts(sid, fake_embeddings, monkeypatch):
    monkeypatch.setattr(config, "FULL_CONTEXT_MAX_CHARS", 0)
    monkeypatch.setattr(config, "TOP_K", 1)
    rag.ingest(sid, put_file(sid, "g.txt", make_long_doc("Git tracks changes to files.")), fake_embeddings)
    _, trace = ask_with_trace(sid, "waht is git", fake_embeddings, FakeLLM(reply="A tool."))

    text = tracing.format_trace(trace, max_rows=5)

    for expected in ["waht is git", "search query", "RETRIEVED CHUNKS", "* = sent to the model",
                     "PROMPT SENT TO THE MODEL", "RAW MODEL OUTPUT", "A tool.", "Git tracks changes"]:
        assert expected in text, expected
    assert "after typo fix" not in text  # nothing was changed in this question, so no such line
    row_one = next(line for line in text.splitlines() if line.startswith("  1"))
    assert row_one.startswith("  1*")  # the best chunk is marked as sent to the model
    assert len(text.split("RETRIEVED CHUNKS")[1].split("context")[0].splitlines()) <= 5 + 3  # max_rows=5 honoured


def test_format_trace_shows_a_typo_fix_when_one_happened(sid, fake_embeddings):
    text = "Jane Doe\njane@example.com | +1 555-010-0199"
    rag.ingest(sid, put_file(sid, "a.txt", text), fake_embeddings)
    _, trace = ask_with_trace(sid, "what is the emial", fake_embeddings, FakeLLM())

    formatted = tracing.format_trace(trace)

    assert "after typo fix    : what is the email" in formatted


def test_fuse_scores_matches_fuse():
    rankings = [[3, 1, 2], [2, 3]]
    scored = search.fuse_scores(rankings)
    assert [position for position, _ in scored] == search.fuse(rankings) == [3, 2, 1]
    assert scored[0][1] > scored[1][1] > scored[2][1] > 0


# ------------------------------------------------------------- the CLI ---
def test_list_and_find_session(sid, fake_embeddings):
    other = sessions.create_session()  # a session with no documents
    rag.ingest(sid, put_file(sid, "1a2b3c4d_report.txt", "Some report text."), fake_embeddings, display_name="report.txt")

    rows = {row["session"]: row for row in debug.list_sessions()}

    assert rows[sid]["indexed"] is True and rows[sid]["files"] == ["report.txt"]
    assert rows[other]["indexed"] is False
    assert debug.find_session(None) == sid  # the newest one that has an index
    assert debug.find_session(sid[:6]) == sid
    with pytest.raises(SystemExit):
        debug.find_session("zzzz")


def test_find_session_without_any_documents_explains_what_to_do():
    sessions.create_session()
    with pytest.raises(SystemExit, match="Upload a file"):
        debug.find_session(None)


def test_run_returns_the_result_and_its_trace(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)

    result, trace = debug.run("what colour are apples", sid, fake_embeddings, FakeLLM(reply="Red."))

    assert result["answer"] == "Red." and trace["route"] == "search"


def test_retrieval_only_mode_never_needs_a_real_model(sid, fake_embeddings):
    rag.ingest(sid, put_file(sid, "a.txt", "Apples are red."), fake_embeddings)
    _, trace = debug.run("apples", sid, fake_embeddings, debug._SkippedModel())
    assert trace["raw_answer"] == "[language model skipped: --retrieval-only]"
    assert trace["candidates"]  # the retrieval part is still fully recorded
