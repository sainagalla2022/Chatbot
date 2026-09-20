"""FastAPI app: ties sessions, RAG and SQL mode together behind a small HTTP API.

Notes for readers:
  * The embedding model and the LLM client are created ONCE, in `lifespan`, and
    stored on `app.state`. Endpoints just reuse them.
  * Endpoints are plain `def` (not `async def`) because they do blocking work
    (disk, embeddings, waiting for Ollama). FastAPI runs plain `def` endpoints in a
    thread pool, so one slow request does not freeze the others.
  * Errors are turned into clean HTTP responses by the handlers near the bottom.
    Stack traces are logged on the server and never sent to the client.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import ollama
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import config, rag, sessions, sql_agent, tracing
from app.rag import load_embeddings, load_llm  # imported by name so tests can replace them

logger = logging.getLogger("chatbot")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UPLOAD_CHUNK_BYTES = 1024 * 1024  # copy uploads in 1 MB pieces
MULTIPART_OVERHEAD_BYTES = 1024 * 1024  # form boundaries etc. on top of the file itself
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
PASTE_FILE_PREFIX = "Pasted-text-"

OLLAMA_DOWN_MESSAGE = (
    "The language model (Ollama) is not reachable. "
    "Start it with `ollama serve`, or check the Ollama service, then try again."
)


# --------------------------------------------------------------------------
# Startup / shutdown
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the heavy models once when the server starts."""
    tracing.setup_logging()  # makes the "chatbot.*" log lines (including one per question) visible
    sessions.sessions_root().mkdir(parents=True, exist_ok=True)
    removed = sessions.cleanup_old_sessions(config.SESSION_MAX_AGE_HOURS)
    logger.info("Removed %d old session(s)", removed)

    logger.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
    app.state.embeddings = load_embeddings()
    app.state.llm = load_llm()
    logger.info("Ready (LLM: %s)", config.OLLAMA_MODEL)
    yield


app = FastAPI(title="Local RAG + SQL Chatbot", lifespan=lifespan)


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # trim spaces before validating

    session_id: str = Field(max_length=64)
    question: str = Field(min_length=1, max_length=1000)


class SQLRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)


class PasteRequest(BaseModel):
    text: str  # the pasted text; its size is checked in the endpoint (in bytes)


class SessionResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    message: str
    chunks: int
    filename: str


class Source(BaseModel):
    file: str | None
    page: int | None


class Passage(BaseModel):
    """A short excerpt of the document that matched the question."""

    file: str | None
    page: int | None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    passages: list[Passage] = []  # the best-matching excerpts (empty for overview and contact answers)
    answered: bool = True  # False when the model found no answer and the closest excerpts are shown


class SQLResponse(BaseModel):
    sql: str
    answer: str
    rows: list[dict[str, Any]] | None = None  # absent when the query was blocked


# --------------------------------------------------------------------------
# Upload size limit, part 1: a cheap early check on the Content-Length header.
# (Part 2, in the endpoint, counts the real bytes while streaming.)
# --------------------------------------------------------------------------
@app.middleware("http")
async def reject_huge_uploads_early(request: Request, call_next):  # type: ignore[no-untyped-def]
    if request.method == "POST" and request.url.path.startswith(("/upload/", "/paste/")):
        declared = request.headers.get("content-length", "")
        limit = config.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
        if declared.isdigit() and int(declared) > limit:
            return JSONResponse(status_code=413, content={"detail": _too_large_message()})
    return await call_next(request)


def _too_large_message(what: str = "File") -> str:
    return f"{what} too large. The limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
def _ollama_reachable() -> bool:
    """Quick check that the Ollama server answers (2 second limit)."""
    try:
        httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ollama": _ollama_reachable(), "model": config.OLLAMA_MODEL}


@app.post("/session", response_model=SessionResponse)
def create_session() -> dict[str, str]:
    return {"session_id": sessions.create_session()}


@app.post("/upload/{session_id}", response_model=UploadResponse)
def upload(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    uploads = sessions.uploads_dir(session_id)  # 400 / 404 if the session is bad

    # Only the last path component of the client's filename is used ("../../x" -> "x").
    original = Path((file.filename or "").replace("\\", "/")).name
    if not original:
        raise HTTPException(status_code=400, detail="The upload has no filename.")
    if len(original) > 150:
        raise HTTPException(status_code=400, detail="The filename is too long.")
    if Path(original).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Please upload a .pdf or .txt file."
        )

    # A random prefix means two uploads with the same name never overwrite each other.
    saved = uploads / f"{uuid.uuid4().hex[:8]}_{original}"
    try:
        _save_upload_streaming(file, saved)
        chunks = rag.ingest(session_id, saved, app.state.embeddings, display_name=original)
    except Exception:
        saved.unlink(missing_ok=True)  # don't keep rejected / half-written files
        raise

    return {"message": f"Uploaded and indexed {original}.", "chunks": chunks, "filename": original}


def _save_upload_streaming(file: UploadFile, destination: Path) -> None:
    """Copy the upload to disk in chunks, stopping as soon as it exceeds the limit."""
    written = 0
    with destination.open("wb") as out:
        while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=_too_large_message())
            out.write(chunk)


@app.post("/paste/{session_id}", response_model=UploadResponse)
def paste(session_id: str, request: PasteRequest) -> dict[str, Any]:
    """Add pasted text to the session. It is saved as a small .txt file and indexed like an upload."""
    uploads = sessions.uploads_dir(session_id)  # 400 / 404 if the session is bad

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="There is no text to add. Paste some text first.")
    data = request.text.encode("utf-8", errors="replace")  # bytes, so the limit matches file uploads
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=_too_large_message("Text"))

    # Each paste becomes its own document: "Pasted text 1", "Pasted text 2", ...
    # (Saved names look like "1a2b3c4d_Pasted-text-1.txt"; the first 9 characters are the prefix.)
    number = sum(1 for p in uploads.iterdir() if p.name[9:].startswith(PASTE_FILE_PREFIX)) + 1
    display_name = f"Pasted text {number}"
    saved = uploads / f"{uuid.uuid4().hex[:8]}_{PASTE_FILE_PREFIX}{number}.txt"
    try:
        saved.write_bytes(data)
        chunks = rag.ingest(session_id, saved, app.state.embeddings, display_name=display_name)
    except Exception:
        saved.unlink(missing_ok=True)
        raise

    return {"message": f"Added {display_name}.", "chunks": chunks, "filename": display_name}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    return rag.ask(request.session_id, request.question, app.state.embeddings, app.state.llm)


@app.post("/debug/chat", include_in_schema=config.RAG_DEBUG)
def debug_chat(request: ChatRequest) -> dict[str, Any]:
    """Like /chat, but also returns the full trace (chunks, scores, the prompt).

    Only available when the server was started with RAG_DEBUG=1; otherwise it does not exist.
    """
    if not config.RAG_DEBUG:
        raise HTTPException(status_code=404, detail="Not Found")
    trace: dict[str, Any] = {}
    result = rag.ask(request.session_id, request.question, app.state.embeddings, app.state.llm, trace=trace)
    return {"result": result, "trace": trace}


@app.post("/sql", response_model=SQLResponse, response_model_exclude_none=True)
def sql(request: SQLRequest) -> dict[str, Any]:
    return sql_agent.ask_database(request.question, app.state.llm)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------------------------------
# Error handling: turn exceptions into clean JSON {"detail": "..."} responses.
# --------------------------------------------------------------------------
def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default is 422 and echoes the input back; we send a short 400 instead.
    problems = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
    return _error(400, "Invalid request. " + "; ".join(problems))


@app.exception_handler(sessions.SessionNotFoundError)
async def handle_unknown_session(request: Request, exc: sessions.SessionNotFoundError) -> JSONResponse:
    return _error(404, str(exc))


@app.exception_handler(sessions.InvalidSessionError)
async def handle_invalid_session(request: Request, exc: sessions.InvalidSessionError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(rag.UnsupportedFileTypeError)
async def handle_unsupported_type(request: Request, exc: rag.UnsupportedFileTypeError) -> JSONResponse:
    return _error(415, str(exc))


@app.exception_handler(rag.DocumentError)
async def handle_bad_document(request: Request, exc: rag.DocumentError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(sql_agent.DatabaseNotFoundError)
async def handle_missing_database(request: Request, exc: sql_agent.DatabaseNotFoundError) -> JSONResponse:
    return _error(400, str(exc))


async def handle_ollama_problem(request: Request, exc: Exception) -> JSONResponse:
    """Anything that goes wrong while talking to Ollama becomes a 503."""
    logger.warning("Ollama problem: %s", type(exc).__name__)
    if isinstance(exc, httpx.TimeoutException):
        return _error(503, f"The language model took too long to answer. {OLLAMA_DOWN_MESSAGE}")
    if isinstance(exc, ollama.ResponseError):
        return _error(
            503,
            f"Ollama returned an error. Check that the model is installed: "
            f"`ollama pull {config.OLLAMA_MODEL}`.",
        )
    return _error(503, OLLAMA_DOWN_MESSAGE)


# ConnectionError is what the ollama library raises when the server is not running.
for ollama_error in (ConnectionError, httpx.ConnectError, httpx.TimeoutException, ollama.ResponseError):
    app.add_exception_handler(ollama_error, handle_ollama_problem)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The full traceback goes to the server log only; the client gets a generic message.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "Internal server error.")
