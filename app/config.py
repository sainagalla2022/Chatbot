"""All settings in one place.

Every value can be overridden with an environment variable of the same name.
Other modules read these as `config.NAME` (not `from config import NAME`) so
that tests can change a value with monkeypatch and it takes effect everywhere.
"""

import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    return int(os.environ.get(name, default))


# --- Text splitting / retrieval -------------------------------------------
CHUNK_SIZE: int = _int_env("CHUNK_SIZE", 500)
CHUNK_OVERLAP: int = _int_env("CHUNK_OVERLAP", 50)
TOP_K: int = _int_env("TOP_K", 3)
# If ALL the text in a session's index is at most this many characters (a resume,
# a one-page letter), send all of it to the model instead of searching. Keep it
# small: TinyLlama's context window is only 2048 tokens (about 4 characters each).
FULL_CONTEXT_MAX_CHARS: int = _int_env("FULL_CONTEXT_MAX_CHARS", 4500)

# --- Uploads ----------------------------------------------------------------
MAX_UPLOAD_BYTES: int = _int_env("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)  # 10 MB

# --- Paths ------------------------------------------------------------------
SQLITE_PATH: Path = Path(os.environ.get("SQLITE_PATH", "./sample.db"))
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "./data"))

# --- Models -----------------------------------------------------------------
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "tinyllama")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_TIMEOUT: int = _int_env("LLM_TIMEOUT", 120)  # seconds per request
LLM_TEMPERATURE: float = 0.1
# Hard cap on how much text the model may write per answer. Without it a small
# model can get stuck repeating itself for minutes (the timeout above only limits
# the wait for the NEXT piece of text, not the total time).
LLM_MAX_TOKENS: int = _int_env("LLM_MAX_TOKENS", 200)
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Debugging --------------------------------------------------------------
# 1 = log the full trace of every question (retrieved chunks, scores, the exact prompt)
# and enable POST /debug/chat. Off by default: the prompt contains your document text.
RAG_DEBUG: bool = os.environ.get("RAG_DEBUG", "0") == "1"

# --- Sessions / SQL mode ----------------------------------------------------
SESSION_MAX_AGE_HOURS: int = _int_env("SESSION_MAX_AGE_HOURS", 24)
SQL_MAX_ROWS: int = _int_env("SQL_MAX_ROWS", 50)
