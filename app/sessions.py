"""Session isolation.

Each session gets its own folder:

    data/sessions/<32 hex chars>/uploads/   (the files the user uploaded)
    data/sessions/<32 hex chars>/index/     (that session's FAISS index)

Every path that touches a session goes through `session_dir()`, which is the
single place where the session ID is validated. That is what stops a request
from reading another session's files or escaping with "../".
"""

import re
import shutil
import time
import uuid
from pathlib import Path

from app import config

# A valid ID is exactly what uuid4().hex produces: 32 lowercase hex characters.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class SessionError(Exception):
    """Base class for session problems."""


class InvalidSessionError(SessionError):
    """The session ID is not in the expected format."""


class SessionNotFoundError(SessionError):
    """The session ID is valid but no such session exists."""


def sessions_root() -> Path:
    """Return the absolute folder that holds all sessions."""
    return (Path(config.DATA_DIR) / "sessions").resolve()


def create_session() -> str:
    """Create a new session (with its folders) and return its ID."""
    sid = uuid.uuid4().hex
    base = sessions_root() / sid
    (base / "uploads").mkdir(parents=True)
    (base / "index").mkdir(parents=True)
    return sid


def session_dir(sid: str) -> Path:
    """Return the folder of an existing session, or raise a clear error.

    Checks, in order: the ID format, that the resolved path stays inside
    data/sessions, and that the folder exists.
    """
    # Strict format check first. Anything with "/", "..", spaces or uppercase fails.
    # (isinstance guards against non-string input; fullmatch avoids the "$" newline quirk.)
    if not isinstance(sid, str) or not _SESSION_ID_RE.fullmatch(sid):
        raise InvalidSessionError("Invalid session ID.")

    root = sessions_root()
    path = (root / sid).resolve()

    # Belt and braces: even with a valid-looking ID, confirm we are inside root.
    if not path.is_relative_to(root):
        raise InvalidSessionError("Invalid session ID.")

    if not path.is_dir():
        raise SessionNotFoundError("Unknown session. Create one with POST /session.")
    return path


def uploads_dir(sid: str) -> Path:
    """Folder where this session's uploaded files are stored."""
    return session_dir(sid) / "uploads"


def index_dir(sid: str) -> Path:
    """Folder where this session's FAISS index is stored."""
    return session_dir(sid) / "index"


def delete_session(sid: str) -> None:
    """Delete one session and everything in it (only inside data/sessions)."""
    path = session_dir(sid)  # validates the ID and the location
    shutil.rmtree(path)


def cleanup_old_sessions(max_age_hours: float) -> int:
    """Delete sessions not modified for `max_age_hours`. Returns how many were removed."""
    root = sessions_root()
    if not root.is_dir():
        return 0

    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for child in root.iterdir():
        # Only touch real folders that look like our own session IDs.
        if child.is_symlink() or not child.is_dir():
            continue
        if not _SESSION_ID_RE.fullmatch(child.name):
            continue
        if child.stat().st_mtime < cutoff:
            shutil.rmtree(child)
            removed += 1
    return removed
