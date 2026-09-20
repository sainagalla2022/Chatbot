"""Tracing and logging for one question: what was searched, found and sent to the model.

`ask()` fills a plain dict (the "trace") while it works. This module turns that dict
into log lines and into readable text. It never changes how a question is answered.

Logging levels:
  * one summary line per question is always logged (no document text in it)
  * the full trace, including the prompt (which contains document text), is logged
    only when the environment variable RAG_DEBUG=1 is set
"""

import logging
from typing import Any

logger = logging.getLogger("chatbot.rag")


def setup_logging() -> None:
    """Make the app's own log messages ("chatbot.*") appear on the console."""
    app_logger = logging.getLogger("chatbot")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)


def summary_line(trace: dict[str, Any]) -> str:
    """One line per question. It has no document text, only the question and the numbers."""
    question = " ".join(str(trace.get("original_question", "")).split())[:100]
    parts = [f'question="{question}"', f"route={trace.get('route')}"]
    if "context" in trace:
        parts.append(f"chunks_sent={trace['context']['chunks_sent']}/{trace['index']['chunks']}")
    parts.append(f"answered={trace.get('answered')}")
    if trace.get("fallback_reason"):
        parts.append(f"fallback={trace['fallback_reason']}")
    parts.append(f"time={trace.get('seconds', {}).get('total', 0.0):.1f}s")
    return " ".join(parts)


def log_trace(trace: dict[str, Any], debug: bool) -> None:
    """Log the summary line, and the whole trace too when debugging is on."""
    logger.info(summary_line(trace))
    if debug:
        logger.info("\n%s", format_trace(trace, max_rows=30))


def format_trace(trace: dict[str, Any], max_rows: int = 12) -> str:
    """Render a trace as readable text (used by the log, the CLI and the debug endpoint)."""
    lines: list[str] = ["=" * 78]
    add = lines.append

    original = trace.get("original_question", "")
    add(f"question          : {original}")
    fixed = trace.get("after_trigger_typo_fix")
    if fixed is not None and fixed != original:
        add(f"after typo fix    : {fixed}")
    rewritten = trace.get("rewritten_query")
    if rewritten is not None:
        note = "" if rewritten == fixed else "   (rewritten: spelling matched to the document)"
        add(f"search query      : {rewritten}{note}")
    add(f"route             : {trace.get('route')}")

    if "keyword_terms" in trace:
        add(f"keywords          : {', '.join(trace['keyword_terms']) or '(none)'}")
        found = ", ".join(trace["matched_terms"]) or "(none)"
        add(f"found in document : {found}   spellings: {', '.join(trace['matched_words']) or '-'}")
    if "index" in trace:
        add(f"index             : {trace['index']['chunks']} chunks, {trace['index']['characters']:,} characters")

    candidates = trace.get("candidates") or []
    if candidates:
        shown = min(len(candidates), max_rows)
        add("")
        add(f"RETRIEVED CHUNKS (best {shown} of {len(candidates)} candidates; * = sent to the model)")
        add(f"{'#':>3}  {'page':>5}  {'blend':>7}  {'meaning r/dist':>15}  {'keyword r/score':>16}  text")
        for row in candidates[:max_rows]:
            mark = "*" if row["sent_to_llm"] else " "
            meaning = f"{row['meaning_rank']}/{row['meaning_distance']:.2f}" if row["meaning_rank"] else "-"
            keyword = f"{row['keyword_rank']}/{row['keyword_score']:.1f}" if row["keyword_rank"] else "-"
            page = f"p{row['page']}" if row["page"] is not None else "-"
            add(
                f"{row['rank']:>3}{mark} {page:>5}  {row['blend_score']:.4f}  {meaning:>15}  "
                f"{keyword:>16}  {row['preview']}"
            )

    context = trace.get("context")
    if context:
        add("")
        add(f"context           : {context['mode']}, {context['chunks_sent']} chunks, {context['characters']:,} characters")
    if "prompt" in trace:
        add("")
        add("PROMPT SENT TO THE MODEL")
        add("-" * 78)
        add(trace["prompt"])
        add("-" * 78)
    if "raw_answer" in trace:
        add("RAW MODEL OUTPUT")
        add(trace["raw_answer"] or "(empty)")
        add("-" * 78)

    add(f"answered          : {trace.get('answered')}   fallback: {trace.get('fallback_reason') or 'no'}")
    add(f"answer            : {trace.get('answer')}")
    seconds = trace.get("seconds", {})
    add("time              : " + ", ".join(f"{name} {value:.2f}s" for name, value in seconds.items()))
    add("=" * 78)
    return "\n".join(lines)
