"""Tests for the SQL agent: safety checks, cleaning, and read-only behaviour."""

import sqlite3

import pytest
from sqlalchemy.exc import DBAPIError

from app import config, sql_agent
from app.sql_agent import BLOCKED_MESSAGE, DatabaseNotFoundError, clean_sql, is_safe
from tests.conftest import FakeLLM


@pytest.fixture
def db_path():
    """A small database at the (temporary) SQLITE_PATH."""
    path = config.SQLITE_PATH
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER,
                             product TEXT, amount REAL, order_date TEXT,
                             created_at TEXT, updated_at TEXT);
        INSERT INTO customers VALUES (1, 'Alice', 'Denver'), (2, 'Bob', 'Austin');
        INSERT INTO orders VALUES
            (1, 1, 'Laptop', 1000.0, '2026-01-05', 'x', 'y'),
            (2, 1, 'Mouse', 20.0, '2026-02-05', 'x', 'y'),
            (3, 2, 'Monitor', 300.0, '2026-03-05', 'x', 'y');
        """
    )
    conn.commit()
    conn.close()
    return path


def count_orders(path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------- is_safe ---
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "select 1",
        "SELECT COUNT(*) FROM orders;",  # one trailing semicolon is fine
        "  SELECT name FROM customers WHERE city = 'Denver'  ",
        "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t",
        # Column names that merely CONTAIN blocked words must not be blocked.
        "SELECT created_at, updated_at FROM orders",
    ],
)
def test_is_safe_accepts_valid_selects(sql):
    assert is_safe(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "delete from orders",
        "DROP TABLE orders",
        "UPDATE orders SET amount = 0",
        "INSERT INTO orders VALUES (9, 1, 'x', 1, 'd', 'a', 'b')",
        "ALTER TABLE orders ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INT)",
        "ATTACH DATABASE 'other.db' AS other",
        "DETACH DATABASE other",
        "REPLACE INTO orders VALUES (1, 1, 'x', 1, 'd', 'a', 'b')",
        "VACUUM",
        "TRUNCATE TABLE orders",
        "PRAGMA table_info(orders)",
        "  pragma writable_schema = 1",
        # More than one statement
        "SELECT 1; DROP TABLE x",
        "SELECT * FROM orders; DELETE FROM orders",
        "SELECT 1;;",
        # Comment tricks
        "SELECT 1 -- ; DROP TABLE x",
        "SELECT 1 /* harmless */",
        "SELECT/**/1",
        "SELECT 1 /* ; DROP TABLE orders ; */",
        # Blocked word hidden after a legitimate start
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "SELECT * FROM orders WHERE id IN (SELECT id FROM orders) OR 1=1 UNION SELECT load_extension('x')",
        # Empty / not a SELECT at all
        "",
        "   ",
        ";",
        "EXPLAIN SELECT 1",
        "Sure! Here is your query: SELECT 1",
    ],
)
def test_is_safe_rejects_dangerous_or_malformed_sql(sql):
    assert is_safe(sql) is False


# ------------------------------------------------------------- clean_sql ---
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1;  \n", "SELECT 1"),
        ("```sql\nSELECT * FROM orders;\n```", "SELECT * FROM orders"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("sql\nSELECT 1", "SELECT 1"),
        ("SQL: SELECT 1", "SELECT 1"),
        ("```SQL SELECT 1;```", "SELECT 1"),
        # Prose after a blank line is dropped; multi-line SQL itself is kept.
        ("SELECT 1;\n\nOutput:\n| a |\n|---|", "SELECT 1"),
        ("SELECT name\nFROM customers\nWHERE city = 'Denver'", "SELECT name\nFROM customers\nWHERE city = 'Denver'"),
    ],
)
def test_clean_sql(raw, expected):
    assert clean_sql(raw) == expected


# ---------------------------------------------------------- schema/prompt ---
def test_get_schema_lists_tables_and_columns(db_path):
    schema = sql_agent.get_schema()
    lines = schema.splitlines()
    assert "customers(id, name, city)" in lines
    assert lines[1].startswith("orders(id, customer_id, product, amount")


def test_prompt_has_schema_rules_examples_and_question(db_path):
    prompt = sql_agent.build_prompt("How many orders?", sql_agent.get_schema())
    assert "customers(id, name, city)" in prompt
    assert "ONE SQLite SELECT" in prompt
    assert prompt.count("Question:") == 3  # 2 few-shot examples + the real question
    assert "How many orders?" in prompt


def test_missing_database_raises_clear_error():
    # SQLITE_PATH points at a file that does not exist (no db_path fixture).
    with pytest.raises(DatabaseNotFoundError, match="create_sample_db"):
        sql_agent.get_schema()


# ---------------------------------------------------------- ask_database ---
def test_delete_all_orders_is_blocked_and_rows_unchanged(db_path):
    llm = FakeLLM(reply="DELETE FROM orders")  # the model "obeys" the malicious request

    result = sql_agent.ask_database("Delete all orders", llm)

    assert result == {"sql": "DELETE FROM orders", "answer": BLOCKED_MESSAGE}
    assert count_orders(db_path) == 3
    assert len(llm.prompts) == 1  # only the SQL-writing call; nothing was executed/explained


def test_fenced_delete_is_also_blocked(db_path):
    llm = FakeLLM(reply="```sql\nDELETE FROM orders;\n```")
    result = sql_agent.ask_database("Delete all orders", llm)
    assert result["answer"] == BLOCKED_MESSAGE
    assert count_orders(db_path) == 3


def test_multi_statement_attack_is_blocked(db_path):
    llm = FakeLLM(reply="SELECT 1; DROP TABLE orders")
    result = sql_agent.ask_database("hi", llm)
    assert result["answer"] == BLOCKED_MESSAGE
    assert count_orders(db_path) == 3


def test_read_only_connection_refuses_writes_even_if_is_safe_is_bypassed(db_path, monkeypatch):
    """Layer 4: if the text checks were fooled, SQLite itself must still refuse."""
    monkeypatch.setattr(sql_agent, "is_safe", lambda sql: True)  # simulate a bypass
    llm = FakeLLM(reply="DELETE FROM orders")

    result = sql_agent.ask_database("Delete all orders", llm)

    assert "could not be run" in result["answer"]
    assert result["sql"] == "DELETE FROM orders"
    assert count_orders(db_path) == 3


def test_engine_connection_rejects_direct_writes(db_path):
    engine = sql_agent._make_engine()
    try:
        with engine.connect() as connection:
            with pytest.raises(DBAPIError):
                connection.exec_driver_sql("DELETE FROM orders")
    finally:
        engine.dispose()
    assert count_orders(db_path) == 3


def test_valid_question_returns_sql_rows_and_explanation(db_path):
    llm = FakeLLM(
        replies=[
            "```sql\nSELECT COUNT(*) AS total FROM orders;\n```",
            "There are 3 orders in total.",
        ]
    )

    result = sql_agent.ask_database("How many orders are there?", llm)

    assert result["sql"] == "SELECT COUNT(*) AS total FROM orders"
    assert result["rows"] == [{"total": 3}]
    assert result["answer"] == "There are 3 orders in total."
    # The explanation prompt must include the result the model should describe.
    assert "'total': 3" in llm.prompts[1]


def test_trailing_explanation_from_model_is_ignored(db_path):
    llm = FakeLLM(
        replies=[
            "SELECT COUNT(*) AS n FROM customers\n\nOutput:\n| n |\n| 2 |\n\nThis counts customers.",
            "There are 2 customers.",
        ]
    )

    result = sql_agent.ask_database("How many customers?", llm)

    assert result["sql"] == "SELECT COUNT(*) AS n FROM customers"
    assert result["rows"] == [{"n": 2}]


def test_sql_error_returns_friendly_message_with_query(db_path):
    llm = FakeLLM(reply="SELECT nonexistent_column FROM orders")

    result = sql_agent.ask_database("Show me stuff", llm)

    assert result["sql"] == "SELECT nonexistent_column FROM orders"
    assert result["rows"] == []
    assert "could not be run" in result["answer"]
    assert "nonexistent_column" in result["answer"]


def test_empty_result_does_not_call_llm_again(db_path):
    llm = FakeLLM(reply="SELECT * FROM orders WHERE amount > 999999")

    result = sql_agent.ask_database("Any huge orders?", llm)

    assert result["rows"] == []
    assert "no rows" in result["answer"].lower()
    assert len(llm.prompts) == 1


def test_results_are_capped_at_50_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO orders (customer_id, product, amount) VALUES (1, 'bulk', 1.0)",
        [()] * 100,
    )
    conn.commit()
    conn.close()
    llm = FakeLLM(replies=["SELECT * FROM orders", "Many orders."])

    result = sql_agent.ask_database("List all orders", llm)

    assert len(result["rows"]) == 50
