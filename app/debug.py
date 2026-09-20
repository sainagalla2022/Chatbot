"""Run ONE question through the pipeline and print exactly what happened.

    python -m app.debug --list                              # which sessions can I use?
    python -m app.debug "what is git"                       # newest session with documents
    python -m app.debug "list all docker commands" --session 2bd5844b --rows 20
    python -m app.debug "git" --retrieval-only              # search only, skip the language model
    OLLAMA_MODEL=qwen2.5:3b python -m app.debug "what is git"   # or: --model qwen2.5:3b

It prints the original and rewritten question, every retrieved chunk with its scores and
page, how many chunks went to the model, the exact prompt, and the raw model output.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app import config, rag, sessions, tracing  # noqa: E402


class _SkippedModel:
    """Stands in for the language model when only the search should be examined."""

    def invoke(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        return "[language model skipped: --retrieval-only]"


def list_sessions() -> list[dict[str, Any]]:
    """Every session on disk, newest first, with its files and whether it has an index."""
    root = sessions.sessions_root()
    if not root.is_dir():
        return []
    rows = []
    for folder in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not re.fullmatch(r"[0-9a-f]{32}", folder.name):
            continue
        uploads = folder / "uploads"
        files = sorted(f.name[9:] for f in uploads.iterdir()) if uploads.is_dir() else []  # drop "1a2b3c4d_"
        rows.append(
            {
                "session": folder.name,
                "indexed": (folder / "index" / "index.faiss").exists(),
                "files": files,
                "age_minutes": int((time.time() - folder.stat().st_mtime) / 60),
            }
        )
    return rows


def find_session(prefix: str | None) -> str:
    """Pick a session: the one starting with `prefix`, or the newest one that has an index."""
    indexed = [row["session"] for row in list_sessions() if row["indexed"]]
    if prefix:
        matches = [sid for sid in indexed if sid.startswith(prefix.lower())]
        if len(matches) != 1:
            raise SystemExit(
                f"'{prefix}' matches {len(matches)} indexed sessions. Run with --list and use more characters."
            )
        return matches[0]
    if not indexed:
        raise SystemExit("No session with documents found. Upload a file in the chat page first.")
    return indexed[0]


def run(question: str, sid: str, embeddings: Any, llm: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask one question and return (result, trace)."""
    trace: dict[str, Any] = {}
    result = rag.ask(sid, question, embeddings, llm, trace=trace)
    return result, trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show what the RAG pipeline does for one question.")
    parser.add_argument("question", nargs="?", help="the question to ask")
    parser.add_argument("--session", help="start of a session id (default: newest with documents)")
    parser.add_argument("--list", action="store_true", help="list the sessions and exit")
    parser.add_argument("--model", help="Ollama model to use (default: OLLAMA_MODEL or tinyllama)")
    parser.add_argument("--rows", type=int, default=12, help="how many retrieved chunks to print")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the language model")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON")
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)  # so the relative ./data folder is found from anywhere

    if args.list:
        for row in list_sessions():
            state = "indexed" if row["indexed"] else "empty  "
            print(f"{row['session'][:8]}  {state}  {row['age_minutes']:>5} min ago  {', '.join(row['files']) or '-'}")
        return 0
    if not args.question:
        parser.error("give a question, or use --list")

    if args.model:
        config.OLLAMA_MODEL = args.model
    sid = find_session(args.session)
    print(f"session {sid[:8]}…  model: {'(skipped)' if args.retrieval_only else config.OLLAMA_MODEL}", file=sys.stderr)

    embeddings = rag.load_embeddings()
    llm = _SkippedModel() if args.retrieval_only else rag.load_llm()
    _, trace = run(args.question, sid, embeddings, llm)

    print(json.dumps(trace, indent=2, default=str) if args.json else tracing.format_trace(trace, args.rows))
    return 0


if __name__ == "__main__":
    tracing.setup_logging()
    raise SystemExit(main())
