"""Tests for session creation, ID validation and isolation."""

import os
import time

import pytest

from app import config, sessions
from app.sessions import InvalidSessionError, SessionNotFoundError


def test_create_session_makes_expected_folders():
    sid = sessions.create_session()

    assert len(sid) == 32
    base = config.DATA_DIR / "sessions" / sid
    assert (base / "uploads").is_dir()
    assert (base / "index").is_dir()


def test_session_dir_returns_path_inside_sessions_root():
    sid = sessions.create_session()
    path = sessions.session_dir(sid)
    assert path.is_relative_to(sessions.sessions_root())


@pytest.mark.parametrize(
    "bad_id",
    [
        "../x",
        "a/b",
        "",
        "/etc/passwd",
        "..",
        "not-hex-at-all-not-hex-at-all-1234",  # right-ish length, wrong characters
        "ABCDEF" * 5 + "AB",  # 32 chars but uppercase
        "a" * 31,  # too short
        "a" * 33,  # too long
        ("a" * 32) + "\n",  # trailing newline must not slip past "$"
        "../" + "a" * 29,  # traversal disguised at the right length
    ],
)
def test_session_dir_rejects_invalid_ids(bad_id):
    with pytest.raises(InvalidSessionError):
        sessions.session_dir(bad_id)


def test_session_dir_rejects_non_string():
    with pytest.raises(InvalidSessionError):
        sessions.session_dir(None)  # type: ignore[arg-type]


def test_session_dir_unknown_but_valid_id_raises_not_found():
    with pytest.raises(SessionNotFoundError):
        sessions.session_dir("0" * 32)


def test_two_sessions_are_isolated():
    a = sessions.create_session()
    b = sessions.create_session()
    assert a != b

    dir_a = sessions.session_dir(a)
    dir_b = sessions.session_dir(b)
    assert dir_a != dir_b
    # Neither folder is inside the other.
    assert not dir_a.is_relative_to(dir_b)
    assert not dir_b.is_relative_to(dir_a)

    # A file written in A's uploads is invisible from B's uploads.
    (sessions.uploads_dir(a) / "secret.txt").write_text("only for A")
    assert list(sessions.uploads_dir(b).iterdir()) == []


def test_delete_session_removes_only_that_session():
    a = sessions.create_session()
    b = sessions.create_session()

    sessions.delete_session(a)

    with pytest.raises(SessionNotFoundError):
        sessions.session_dir(a)
    assert sessions.session_dir(b).is_dir()


def test_delete_session_rejects_traversal():
    # A folder outside data/sessions must survive a malicious delete attempt.
    outside = config.DATA_DIR / "keep_me"
    outside.mkdir(parents=True)
    sessions.create_session()  # make sure data/sessions exists

    with pytest.raises(InvalidSessionError):
        sessions.delete_session("../keep_me")
    assert outside.is_dir()


def test_cleanup_old_sessions_removes_only_old_ones():
    old = sessions.create_session()
    new = sessions.create_session()

    # Pretend the "old" session was last touched 48 hours ago.
    two_days_ago = time.time() - 48 * 3600
    os.utime(sessions.session_dir(old), (two_days_ago, two_days_ago))

    removed = sessions.cleanup_old_sessions(max_age_hours=24)

    assert removed == 1
    with pytest.raises(SessionNotFoundError):
        sessions.session_dir(old)
    assert sessions.session_dir(new).is_dir()


def test_cleanup_ignores_folders_that_are_not_sessions():
    sessions.create_session()
    stray = sessions.sessions_root() / "not_a_session"
    stray.mkdir()
    old = time.time() - 48 * 3600
    os.utime(stray, (old, old))

    assert sessions.cleanup_old_sessions(max_age_hours=24) == 0
    assert stray.is_dir()
