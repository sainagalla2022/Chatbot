"""SQL mode: turn a plain-English question into ONE read-only SQLite SELECT.

Safety comes in layers, so that no single mistake can change data:
    1. The prompt tells TinyLlama to output a single SELECT.
    2. clean_sql() strips markdown fences and other clutter.
    3. is_safe() rejects anything that is not one plain SELECT.
    4. The database is opened READ-ONLY (SQLite URI mode=ro, plus query_only).
    5. Only the first SQL_MAX_ROWS rows are fetched.
Even if layers 1-3 were fooled, layer 4 makes SQLite itself refuse any write.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from app import config

BLOCKED_MESSAGE = "Blocked: only single SELECT queries are allowed."

# Words that must never appear (matched as whole words, any letter case).
# "load_extension" is not in the original list; it is added because it is the one
# SELECT-callable function that could load native code.
_BLOCKED_WORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
    "pragma", "replace", "vacuum", "truncate", "load_extension",
)  # fmt: skip
_BLOCKED_RE = re.compile(r"\b(" + "|".join(_BLOCKED_WORDS) + r")\b", re.IGNORECASE)
_STOP_SEQUENCES = ["\n\n", "Output:"]
_STARTS_WITH_SELECT_RE =re.compile(r"^(select|with)\b", re.IGNORECASE)
_HAS_SELECT_RE = re.compile(r"\bselect\b", re.IGNORECASE)


class DatabaseNotFoundError(Exception):
    """The SQLite file does not exist (run scripts/create_sample_db.py)."""


# --------------------------------------------------------------------------
# Read-only database access
# --------------------------------------------------------------------------
def _make_engine() -> Engine:
    """Create an engine whose connections are strictly read-only."""
    path = Path(config.SQLITE_PATH).resolve()
    if not path.is_file():
        raise DatabaseNotFoundError(
            f"Database file not found: {path.name}. "
            "Create the sample one with: python scripts/create_sample_db.py"
        )

    def connect() -> sqlite3.Connection:
        # mode=ro: SQLite opens the file read-only and refuses every write.
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        # A second, independent read-only switch (belt and braces).
        connection.execute("PRAGMA query_only = ON")
        return connection

    # "creator" lets us hand SQLAlchemy our own read-only connection.
    return create_engine("sqlite://", creator=connect)


def get_schema() -> str:
    """Describe the database for the prompt, one line per table: table(col, col, ...)."""
    engine = _make_engine()
    try:
        inspector = inspect(engine)
        lines = []
        for table in inspector.get_table_names():
            columns = ", ".join(col["name"] for col in inspector.get_columns(table))
            lines.append(f"{table}({columns})")
        return "\n".join(lines)
    finally:
        engine.dispose()


# --------------------------------------------------------------------------
# Prompt, cleaning, safety check
# --------------------------------------------------------------------------
def build_prompt(question: str, schema: str) -> str:
    """Ask the model for exactly one SQLite SELECT, with two worked examples."""
    return (
        "You write SQLite queries.\n"
        f"Database schema:\n{schema}\n\n"
        "Rules: output ONE SQLite SELECT statement and nothing else. "
        "Use only the tables and columns listed above. "
        "No explanation, no example output, no markdown.\n\n"
        "Question: How many customers are there?\n"
        "SQL: SELECT COUNT(*) FROM customers\n\n"
        "Question: What is the total order amount for each customer name?\n"
        "SQL: SELECT c.name, SUM(o.amount) AS total FROM customers c "
        "JOIN orders o ON o.customer_id = c.id GROUP BY c.name\n\n"
        f"Question: {question}\n"
        "SQL:"
    )


def clean_sql(text: str) -> str:
    """Strip markdown code fences, a leading 'sql', whitespace and a trailing ';'."""
    sql = text.strip()
    # If the model wrapped the query in ```sql ... ```, keep only what is inside.
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if fenced:
        sql = fenced.group(1)
    sql = sql.replace("```", "").strip()
    sql = re.sub(r"^sql\b\s*:?\s*", "", sql, flags=re.IGNORECASE)  # leading "sql"
    # TinyLlama likes to add "Output: ..." or an explanation after a blank line.
    # Keep only the first block; the rest is never executed.
    sql = re.split(r"\n\s*\n", sql.strip(), maxsplit=1)[0]
    return sql.strip().rstrip(";").strip()


def is_safe(sql: str) -> bool:
    """True only for a single, comment-free SELECT (or WITH ... SELECT) statement."""
    sql = sql.strip()
    if sql.endswith(";"):  # one trailing semicolon is fine
        sql = sql[:-1].rstrip()
    if not sql:
        return False
    if ";" in sql:  # a second statement, e.g. "SELECT 1; DROP TABLE x"
        return False
    if "--" in sql or "/*" in sql or "*/" in sql:  # comments can hide tricks
        return False
    if not _STARTS_WITH_SELECT_RE.match(sql):
        return False
    if sql.lower().startswith("with") and not _HAS_SELECT_RE.search(sql):
        return False
    if _BLOCKED_RE.search(sql):
        return False
    return True


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def _run_query(sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run a (already checked) query on the read-only connection; return columns, rows."""
    engine = _make_engine()
    try:
        with engine.connect() as connection:
            # exec_driver_sql sends the text straight to SQLite (no ":name" parsing).
            result = connection.exec_driver_sql(sql)
            columns = list(result.keys())
            rows = result.fetchmany(config.SQL_MAX_ROWS)  # never more than 50 rows
            return columns, [tuple(row) for row in rows]
    finally:
        engine.dispose()


def _explain(llm: Any, question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Ask the model to describe the result in one plain sentence."""
    prompt = (
        f"Question: {question}\n"
        f"SQL used: {sql}\n"
        f"Result rows (up to 10 shown): {rows[:10]}\n\n"
        "Explain the result in one plain English sentence.\n"
        "Answer:"
    )
    return str(llm.invoke(prompt, stop=_STOP_SEQUENCES)).strip()


def ask_database(question: str, llm: Any) -> dict[str, Any]:
    """Answer a question about the database.

    Returns {"sql", "rows", "answer"}, or {"sql", "answer"} if the query was blocked.
    """
    schema = get_schema()  # also raises DatabaseNotFoundError if there is no DB
    # `stop` makes Ollama end generation at a blank line, so the model can't ramble on
    # (saves time too: unbounded rambling once took 147 s).
    raw = llm.invoke(build_prompt(question, schema), stop=_STOP_SEQUENCES)
    sql = clean_sql(str(raw))

    if not is_safe(sql):
        return {"sql": sql, "answer": BLOCKED_MESSAGE}

    try:
        columns, raw_rows = _run_query(sql)
    except (DBAPIError, sqlite3.Error) as exc:
        # Show SQLite's own message (e.g. "no such column: x") in a friendly wrapper.
        reason = str(getattr(exc, "orig", exc))
        return {
            "sql": sql,
            "rows": [],
            "answer": f"Sorry, that query could not be run ({reason}). Try rephrasing the question.",
        }

    rows = [dict(zip(columns, row)) for row in raw_rows]
    if not rows:
        return {"sql": sql, "rows": [], "answer": "The query ran, but no rows matched."}

    return {"sql": sql, "rows": rows, "answer": _explain(llm, question, sql, rows)}
