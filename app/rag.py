"""RAG: load documents, index them per session, and answer questions from them.

Flow:
    ingest():  file -> pages -> chunks (500 chars) -> embeddings -> session's FAISS index
    ask():     question -> top-3 similar chunks -> prompt -> TinyLlama -> answer + sources

The embedding model and the LLM are loaded ONCE (see load_embeddings / load_llm,
called from the FastAPI lifespan) and passed into ingest() / ask(). Passing them
in, instead of importing global objects, also lets the tests use fakes.
"""

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config, search, tracing
from app.sessions import index_dir, session_dir, uploads_dir

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
NO_ANSWER = "I don't know based on the documents."
# Phrases TinyLlama tends to use instead of the exact NO_ANSWER sentence.
_NOT_FOUND_HINTS = (
    "not mentioned", "does not contain", "doesn't contain", "no information",
    "not available", "not provided",
)  # fmt: skip
NO_DOCUMENTS_MESSAGE = (
    "No documents have been uploaded to this session yet. "
    "Please upload a PDF or TXT file first, then ask again."
)


# --------------------------------------------------------------------------
# Errors. The API layer turns these into HTTP status codes.
# --------------------------------------------------------------------------
class DocumentError(Exception):
    """A problem with the uploaded document (the client's fault -> HTTP 400)."""


class UnsupportedFileTypeError(DocumentError):
    """The file extension is not .pdf or .txt (-> HTTP 415)."""


class EmptyDocumentError(DocumentError):
    """The file is empty or has no text we can read."""


class CorruptDocumentError(DocumentError):
    """The file could not be parsed (broken PDF, not UTF-8 text, ...)."""


# --------------------------------------------------------------------------
# Model loading (call once at startup)
# --------------------------------------------------------------------------
def load_embeddings() -> Any:
    """Load the local sentence-transformers embedding model (CPU only)."""
    # Imported here so that merely importing this module stays fast for tests.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def load_llm() -> Any:
    """Create the TinyLlama client (talks to the local Ollama service)."""
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_MAX_TOKENS,  # stop after this many tokens, whatever happens
        # Give up if Ollama goes silent for this many seconds.
        client_kwargs={"timeout": config.LLM_TIMEOUT},
    )


# --------------------------------------------------------------------------
# One lock per session, so two uploads to the same session cannot write the
# FAISS index at the same time and corrupt it.
# --------------------------------------------------------------------------
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_lock(sid: str) -> threading.Lock:
    with _locks_guard:
        if sid not in _locks:
            _locks[sid] = threading.Lock()
        return _locks[sid]


def _index_exists(sid: str) -> bool:
    return (index_dir(sid) / "index.faiss").exists()


def _load_index(sid: str, embeddings: Any) -> FAISS:
    # allow_dangerous_deserialization=True is needed because FAISS.load_local uses
    # pickle. That is only safe when the files are trusted: here the app creates
    # every index file itself inside the session folder, and users can only upload
    # .pdf/.txt files into a different folder (uploads/), never into index/.
    return FAISS.load_local(
        str(index_dir(sid)), embeddings, allow_dangerous_deserialization=True
    )


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
def _load_pages(path: Path) -> list[Document]:
    """Read a PDF or TXT file into a list of page-level Documents."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return PyPDFLoader(str(path)).load()
        return TextLoader(str(path), encoding="utf-8").load()
    except UnicodeDecodeError as exc:
        raise CorruptDocumentError(
            "The text file is not valid UTF-8. Please save it as UTF-8 and try again."
        ) from exc
    except Exception as exc:  # pypdf raises several different error types
        raise CorruptDocumentError(
            f"Could not read this {suffix[1:].upper()} file. It may be corrupt."
        ) from exc


def ingest(
    sid: str,
    file_path: str | Path,
    embeddings: Any,
    display_name: str | None = None,
) -> int:
    """Add one uploaded file to the session's FAISS index. Returns the chunk count.

    `display_name` is the name shown in "sources". It defaults to the file's own
    name (never a full path).
    """
    path = Path(file_path).resolve()

    # The file must live in this session's uploads folder (also validates `sid`).
    if not path.is_relative_to(uploads_dir(sid).resolve()):
        raise DocumentError("File is not inside this session's uploads folder.")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError("Unsupported file type. Please upload a .pdf or .txt file.")
    if not path.is_file() or path.stat().st_size == 0:
        raise EmptyDocumentError("The file is empty.")

    pages = _load_pages(path)

    # Drop pages with no text. A scanned PDF has pages but no extractable text.
    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        if path.suffix.lower() == ".pdf":
            raise EmptyDocumentError(
                "No text could be extracted from this PDF. It may be a scanned image; "
                "this app has no OCR, so please upload a text-based PDF."
            )
        raise EmptyDocumentError("The file contains no text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    # Keep only what we need in the metadata: the file NAME and a 1-based page.
    name = Path(display_name or path.name).name
    for chunk in chunks:
        page = chunk.metadata.get("page")  # PyPDFLoader pages start at 0
        chunk.metadata = {"file": name, "page": page + 1 if page is not None else None}

    # The lock makes "load, add, save" one atomic step per session.
    with _get_lock(sid):
        if _index_exists(sid):
            store = _load_index(sid, embeddings)
            store.add_documents(chunks)
        else:
            store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(index_dir(sid)))

    return len(chunks)


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------
def build_prompt(question: str, chunks: list[Document]) -> str:
    """Build the prompt: strict instructions + retrieved context + the question."""
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    # Layout matters a lot for a 1B model. In testing, putting the context FIRST and
    # the rules LAST worked best; rules placed before the context were echoed back.
    return (
        f"---\n{context}\n---\n\n"
        f"Question: {question}\n\n"
        "Rules: Answer in one short sentence using ONLY the context between the "
        "lines above. If the context does not contain the answer, "
        f"write exactly: {NO_ANSWER}\n"
        "Answer:"
    )


_RANK_DEPTH = 20  # how many chunks each search contributes before they are blended


def _key(chunk: Document) -> tuple[Any, ...]:
    return (chunk.page_content, chunk.metadata.get("file"), chunk.metadata.get("page"))


@dataclass
class Ranking:
    """The result of ranking every chunk for one question."""

    order: list[int]  # chunk positions, best first
    match: search.KeywordMatch
    blend: dict[int, float]  # position -> blended score (higher is better)
    meaning: dict[int, tuple[int, float]]  # position -> (rank, distance); a smaller distance is closer
    keyword_rank: dict[int, int]  # position -> rank in the keyword search


def _rank_chunks(
    store: FAISS, chunks: list[Document], index: search.TextIndex, question: str
) -> Ranking:
    """Rank the chunks for a question, best first, by blending two searches.

    * meaning search (embeddings): finds related passages even when the words differ
    * keyword search: finds exact and close-spelling words, which is what short
      questions like "git" need
    """
    position_of = {_key(chunk): i for i, chunk in enumerate(chunks)}
    found = store.similarity_search_with_score(question, k=min(_RANK_DEPTH, len(chunks)))
    by_meaning: list[int] = []
    meaning: dict[int, tuple[int, float]] = {}
    for doc, distance in found:
        position = position_of.get(_key(doc))
        if position is not None:
            by_meaning.append(position)
            meaning.setdefault(position, (len(by_meaning), float(distance)))

    match = index.match(question)
    by_words = sorted(
        (i for i, score in enumerate(match.scores) if score > 0), key=lambda i: -match.scores[i]
    )[:_RANK_DEPTH]

    blended = search.fuse_scores([by_meaning, by_words])
    return Ranking(
        order=[position for position, _ in blended],
        match=match,
        blend=dict(blended),
        meaning=meaning,
        keyword_rank={position: rank for rank, position in enumerate(by_words, start=1)},
    )


def _passages(chunks: list[Document], match: search.KeywordMatch) -> list[dict[str, Any]]:
    """Short excerpts of the best chunks, centred on the matching word, to show the user."""
    return [
        {
            "file": chunk.metadata.get("file"),
            "page": chunk.metadata.get("page"),
            "text": search.snippet(chunk.page_content, match.words),
        }
        for chunk in chunks
    ]


def _sources(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The distinct (file, page) pairs of the passages, in order."""
    sources: list[dict[str, Any]] = []
    for passage in passages:
        source = {"file": passage["file"], "page": passage["page"]}
        if source not in sources:
            sources.append(source)
    return sources


def _related_text(match: search.KeywordMatch, passages: list[dict[str, Any]]) -> dict[str, Any]:
    """What to show when the model found no answer: the closest parts of the document.

    Never a dead end. If the question's words do appear in the document, say so;
    if not, be honest that these passages may not be related.
    """
    if match.matched_terms:
        shown = ", ".join(f"“{term}”" for term in match.matched_terms[:3])
        text = f"I couldn't find a direct answer, but these parts of your document mention {shown}:"
    else:
        text = (
            "I couldn't find anything about that in your document. "
            "These are the closest parts I found, and they may not be related:"
        )
    return {"answer": text, "sources": _sources(passages), "passages": passages, "answered": False}


# --------------------------------------------------------------------------
# Keeping answers short. TinyLlama does not know when to stop: left alone it keeps
# writing "Context: ..." blocks and repeating itself until the token cap.
# --------------------------------------------------------------------------
# Ollama ends the answer at the first blank line (or when the model starts
# inventing a new "Question:" / "Context:" section).
_ANSWER_STOPS = ["\n\n", "\nQuestion:", "\nContext:"]


def _generate_short(llm: Any, prompt: str, stops: list[str] | None = None) -> str:
    """Call the model and return one short paragraph (or an intro plus its list)."""
    text = str(llm.invoke(prompt, stop=stops or _ANSWER_STOPS)).strip()
    if stops is not None:
        return text  # a caller with its own stop rules wants the text exactly as given

    if not text or text.endswith(":"):
        # Either the model began with a blank line (so the stop rule cut off everything),
        # or it wrote an intro like "The skills are:" and put the list after a blank line.
        # Ask again without the stop rule and keep the first paragraph, plus the list
        # that follows an intro.
        paragraphs = [p.strip() for p in str(llm.invoke(prompt)).split("\n\n") if p.strip()]
        text = "\n".join(paragraphs[: 2 if text else 1])
    return text


def _trim_if_cut_off(answer: str) -> str:
    """If the token cap chopped the answer mid-sentence, cut back to the last full sentence.

    Only long answers that do not end in punctuation count as "cut off"; a short
    answer such as a list of skills is left alone.
    """
    if not answer or answer[-1] in ".!?\"')" or len(answer) < config.LLM_MAX_TOKENS * 2.5:
        return answer
    sentence_ends = [m.end() for m in re.finditer(r"[.!?][\"')]*(?=\s|$)", answer)]
    return answer[: sentence_ends[-1]] if sentence_ends else answer


# --------------------------------------------------------------------------
# Overview questions: "what is this document about?", "whose resume is this?"
# A small model answers these by reciting the contents. Instead we describe the
# document from its file name and its first lines, which we copy exactly.
# --------------------------------------------------------------------------
_OVERVIEW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\babout\b.*\b(document|doc|file|pdf|resume|cv|text)\b",
        r"\b(document|doc|file|pdf|resume|cv|text)\b.*\babout\b",
        r"\bwhat(?:'s| is)? (?:this|it|that) (?:about|for)\b",  # "what is this about" (pasted text)
        r"\bsummar",
        r"\boverview\b",
        r"\bwhose\b",
        r"\bbelongs?\b",
        r"\bowner\b",
        r"\bwho (is|are|wrote|made)\b.*\b(candidate|applicant|author|person|owner|this)\b",
        # "what is the candidate's name / job title": we copy those lines exactly
        r"\b(candidate|applicant|person|author)('s)?\s+(name|job title|title)\b",
    )
]
_OVERVIEW_MAX_WORDS = 12  # a long, specific question is not an overview request


def is_overview_question(question: str) -> bool:
    """True for short questions asking what a document is or whose it is."""
    if len(question.split()) > _OVERVIEW_MAX_WORDS:
        return False
    return any(pattern.search(question) for pattern in _OVERVIEW_PATTERNS)


# Words that tell us what kind of document it is, most specific first. Guessing from
# words is far more reliable than asking a 1B model to name the type.
_DOCUMENT_KINDS = (
    "resume", "cv", "cheat sheet", "invoice", "report", "manual", "handbook", "guide",
    "letter", "contract", "agreement", "policy", "syllabus", "certificate",
    "presentation", "article", "paper", "book",
)  # fmt: skip


def _guess_kind(name: str, start: str) -> str | None:
    """Guess the type of document from its file name first, then from its first lines."""
    for text in (name, start):
        text = re.sub(r"[_\-.]", " ", text.lower())  # "Sai_Resume.pdf" -> "sai resume pdf"
        for kind in _DOCUMENT_KINDS:
            if re.search(rf"\b{kind}\b", text):
                return "resume" if kind == "cv" else kind
    return None


def _title_lines(text: str, count: int) -> list[str]:
    """The first `count` non-empty lines, skipping lines that hold contact details."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and "@" not in line and "http" not in line.lower()
    ]
    return lines[:count]


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()


def _describe_document(name: str, first_chunk: Document, llm: Any) -> str:
    """Describe one file: its type, its name, how it begins, and a one-sentence summary."""
    start = first_chunk.page_content.strip()[:500]

    # 1) The kind of document, guessed from its name and first lines.
    kind = _guess_kind(name, start)
    if name.startswith("Pasted text"):  # text pasted into the page, not a file
        text = f"This is {name.lower()}."
    elif kind:
        article = "an" if kind[0] in "aeiou" else "a"
        text = f"This is {article} {kind} ({name})."
    else:
        text = f"This is the file {name}."

    # 2) How it begins, copied exactly (usually a name and a title). Contact lines are skipped.
    lines = _title_lines(start, 2)
    if lines:
        text += " It begins with " + " and ".join(f"“{line[:100]}”" for line in lines) + "."

    # 3) One short sentence about what it covers.
    summary = _first_sentence(
        _generate_short(
            llm,
            f"---\n{start}\n---\n\nIn one sentence, what is this document about?\nAnswer:",
        )
    )
    if summary:
        text += " " + summary
    return text


def _overview(store: FAISS, llm: Any) -> dict[str, Any]:
    """Answer an overview question for each file in the session (at most 3)."""
    first_chunks: dict[str, Document] = {}
    for chunk in store.docstore._dict.values():  # in the order they were added
        first_chunks.setdefault(chunk.metadata.get("file"), chunk)

    paragraphs, sources = [], []
    for name, chunk in list(first_chunks.items())[:3]:
        paragraphs.append(_describe_document(name, chunk, llm))
        sources.append({"file": name, "page": chunk.metadata.get("page")})
    return {"answer": "\n\n".join(paragraphs), "sources": sources}


# --------------------------------------------------------------------------
# Contact questions: "what is the email?", "how do I contact the owner?"
# Emails and phone numbers are copied out of the text by pattern, exactly as
# written. A 1B model cannot be trusted to retype them (it says "the email in the
# context" without giving it, or mistypes it).
# --------------------------------------------------------------------------
_CONTACT_WORD_RE = re.compile(
    r"\b(e-?mail|mail id|contact|phone|mobile|telephone|linkedin|github)\b", re.IGNORECASE
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone numbers need a "+", brackets or separators, so long ID numbers are not mistaken for phones.
_PHONE_RE = re.compile(r"\+\d[\d\s.\-()]{8,16}\d|\(\d{3}\)\s?\d{3}[\s.\-]\d{4}|\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b")
_LINK_RE = re.compile(r"(?:https?://|www\.)[^\s|,;)>\]]+|(?:linkedin\.com|github\.com)/[^\s|,;)>\]]+", re.IGNORECASE)


def is_contact_question(question: str) -> bool:
    """True for questions that ask for an email, phone number or other way to contact someone."""
    return len(question.split()) <= 30 and bool(_CONTACT_WORD_RE.search(question))


def _unique(values: list[str], limit: int = 3) -> list[str]:
    """Remove duplicates, keep the order, and keep at most `limit` items."""
    return list(dict.fromkeys(v.strip().rstrip(".,;") for v in values))[:limit]


def _extract_contacts(text: str) -> dict[str, list[str]]:
    """Find emails, phone numbers and links in `text`. Empty groups are left out."""
    found = {
        "Email": _unique(_EMAIL_RE.findall(text)),
        "Phone": _unique(_PHONE_RE.findall(text)),
        "Links": _unique(_LINK_RE.findall(text)),
    }
    return {label: values for label, values in found.items() if values}


def _contact_details(store: FAISS) -> dict[str, Any] | None:
    """List the contact details found in each file (at most 3), or None if there are none."""
    chunks_by_file: dict[str, list[Document]] = {}
    for chunk in store.docstore._dict.values():  # in the order they were added
        chunks_by_file.setdefault(chunk.metadata.get("file"), []).append(chunk)

    paragraphs, sources = [], []
    for name, chunks in list(chunks_by_file.items())[:3]:
        # A short file (a resume) is searched all the way through. In a long one
        # (a manual full of example addresses) only the top counts: that is where the
        # owner's details would be, and the rest would give sample values like you@example.com.
        is_short = sum(len(c.page_content) for c in chunks) <= config.FULL_CONTEXT_MAX_CHARS
        found: dict[str, list[str]] = {}
        source_chunk = None
        for chunk in chunks if is_short else chunks[:1]:
            contacts = _extract_contacts(chunk.page_content)
            if contacts and source_chunk is None:
                source_chunk = chunk  # the page we report as the source
            for label, values in contacts.items():
                found[label] = _unique(found.get(label, []) + values)
        if not found:
            continue

        title = _title_lines(chunks[0].page_content, 1)
        heading = f"Contact details in “{name}”"
        if title:
            heading += f" (it begins with “{title[0][:100]}”)"
        lines = [heading + ":"] + [f"{label}: {', '.join(values)}" for label, values in found.items()]
        paragraphs.append("\n".join(lines))
        sources.append({"file": name, "page": source_chunk.metadata.get("page")})

    if not paragraphs:
        return None
    return {"answer": "\n\n".join(paragraphs), "sources": sources}


def _not_found_reason(answer: str) -> str | None:
    """Why an answer counts as "the model found nothing" (None if it is a real answer)."""
    lowered = answer.lower()
    if not answer:
        return "empty answer"
    if NO_ANSWER.lower() in lowered:
        return "exact don't-know sentence"
    for hint in _NOT_FOUND_HINTS:
        if hint in lowered:
            return f'phrase "{hint}"'
    return None


def _trace_candidates(
    ranking: Ranking, chunks: list[Document], sent: set[int], limit: int = 30
) -> list[dict[str, Any]]:
    """The best-ranked chunks with every score, for the trace."""
    rows = []
    for rank, position in enumerate(ranking.order[:limit], start=1):
        chunk = chunks[position]
        meaning_rank, distance = ranking.meaning.get(position, (0, 0.0))
        rows.append(
            {
                "rank": rank,
                "position": position,
                "file": chunk.metadata.get("file"),
                "page": chunk.metadata.get("page"),
                "blend_score": ranking.blend[position],
                "meaning_rank": meaning_rank,  # 0 = not among the meaning search's candidates
                "meaning_distance": distance,
                "keyword_rank": ranking.keyword_rank.get(position, 0),
                "keyword_score": ranking.match.scores[position],
                "sent_to_llm": position in sent,
                "preview": " ".join(search.snippet(chunk.page_content, ranking.match.words, 90).split()),
            }
        )
    return rows


def _finish(trace: dict[str, Any], result: dict[str, Any], started: float) -> dict[str, Any]:
    """Complete the trace, log it, and hand back the result."""
    trace["answer"] = result["answer"]
    trace["answered"] = result.get("answered", True)
    trace["seconds"]["total"] = time.perf_counter() - started
    tracing.log_trace(trace, debug=config.RAG_DEBUG)
    return result


def ask(
    sid: str,
    question: str,
    embeddings: Any,
    llm: Any,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a question from the session's documents.

    Returns {"answer": str, "sources": [{"file", "page"}]}. Answers from the search also carry
    "passages" (short excerpts of the best-matching text) and "answered" (False when the
    model found no answer and the closest parts of the document are shown instead).

    Pass an empty dict as `trace` to receive a record of every step (see app/tracing.py).
    """
    trace = {} if trace is None else trace
    started = time.perf_counter()
    trace["original_question"] = question
    trace["seconds"] = {}

    session_dir(sid)  # validates the session ID (raises a clear error if bad)

    if not _index_exists(sid):
        trace["route"] = "no_documents"
        return _finish(trace, {"answer": NO_DOCUMENTS_MESSAGE, "sources": []}, started)

    # Take the lock while loading so we never read a half-written index.
    with _get_lock(sid):
        store = _load_index(sid, embeddings)
    all_chunks = list(store.docstore._dict.values())  # every chunk, in the order they were added
    trace["index"] = {
        "chunks": len(all_chunks),
        "characters": sum(len(chunk.page_content) for chunk in all_chunks),
    }

    question = search.fix_trigger_typos(question)  # "emial" -> "email", "documenbt" -> "document"
    trace["after_trigger_typo_fix"] = question

    if is_contact_question(question):
        contact = _contact_details(store)
        if contact:  # nothing found in the text? fall through to the normal search below
            trace["route"] = "contact"
            return _finish(trace, contact, started)

    if is_overview_question(question):
        trace["route"] = "overview"
        return _finish(trace, _overview(store, llm), started)

    trace["route"] = "search"
    searching = time.perf_counter()
    index = search.TextIndex([chunk.page_content for chunk in all_chunks])
    question = index.correct(question)  # "documenbt" -> "document", using the document's own words
    trace["rewritten_query"] = question
    ranking = _rank_chunks(store, all_chunks, index, question)
    best = [all_chunks[position] for position in ranking.order]
    passages = _passages(best[:3], ranking.match)  # the excerpts shown under the answer
    trace["seconds"]["search"] = time.perf_counter() - searching
    trace["keyword_terms"] = search.keyword_terms(question)
    trace["matched_terms"] = ranking.match.matched_terms
    trace["matched_words"] = ranking.match.words

    # Short documents (a resume, say) go to the model whole: searching could miss a chunk
    # whose wording doesn't resemble the question. Longer ones send only the best few chunks.
    if trace["index"]["characters"] <= config.FULL_CONTEXT_MAX_CHARS:
        context, mode = all_chunks, "whole_document"
        sent = set(range(len(all_chunks)))
    else:
        context, mode = best[: config.TOP_K], "top_k"
        sent = set(ranking.order[: config.TOP_K])
    prompt = build_prompt(question, context)
    trace["context"] = {
        "mode": mode,
        "chunks_sent": len(context),
        "characters": sum(len(chunk.page_content) for chunk in context),
    }
    trace["candidates"] = _trace_candidates(ranking, all_chunks, sent)
    trace["prompt"] = prompt

    generating = time.perf_counter()
    raw_answer = _generate_short(llm, prompt)
    trace["raw_answer"] = raw_answer
    trace["seconds"]["llm"] = time.perf_counter() - generating
    answer = _trim_if_cut_off(raw_answer)

    # TinyLlama often says the documents don't contain the answer, in its own words.
    # Don't leave the user with that: show the closest parts of the document instead.
    trace["fallback_reason"] = _not_found_reason(answer)
    if trace["fallback_reason"]:
        return _finish(trace, _related_text(ranking.match, passages), started)

    result = {"answer": answer, "sources": _sources(passages), "passages": passages, "answered": True}
    return _finish(trace, result, started)
