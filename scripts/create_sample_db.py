"""Create the sample SQLite database (customers + orders) used by SQL mode.

Run from the project root:

    python scripts/create_sample_db.py

The file location comes from SQLITE_PATH (default ./sample.db). Running the
script again resets the two tables to the original sample data; it never
deletes the file itself.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from sqlalchemy import create_engine, text  # noqa: E402

from app import config  # noqa: E402

CUSTOMERS = [
    (1, "Alice Johnson", "alice@example.com", "Denver", "2025-11-03"),
    (2, "Bob Smith", "bob@example.com", "Austin", "2025-11-19"),
    (3, "Carla Gomez", "carla@example.com", "Seattle", "2025-12-05"),
    (4, "David Lee", "david@example.com", "Boston", "2025-12-22"),
    (5, "Emma Brown", "emma@example.com", "Denver", "2026-01-10"),
    (6, "Farid Khan", "farid@example.com", "Chicago", "2026-01-28"),
]

# (id, customer_id, product, amount, order_date, status)
ORDERS = [
    (1, 1, "Laptop", 1200.00, "2026-01-05", "delivered"),
    (2, 2, "Headphones", 89.99, "2026-01-12", "delivered"),
    (3, 3, "Keyboard", 49.50, "2026-01-20", "delivered"),
    (4, 1, "Monitor", 310.00, "2026-02-02", "delivered"),
    (5, 4, "Mouse", 25.00, "2026-02-09", "delivered"),
    (6, 5, "Laptop", 1150.00, "2026-02-14", "delivered"),
    (7, 2, "Webcam", 75.25, "2026-02-27", "cancelled"),
    (8, 6, "Desk Lamp", 39.99, "2026-03-03", "delivered"),
    (9, 3, "Monitor", 295.00, "2026-03-11", "delivered"),
    (10, 1, "Headphones", 89.99, "2026-03-18", "delivered"),
    (11, 4, "Laptop", 1300.00, "2026-03-29", "delivered"),
    (12, 5, "Keyboard", 52.00, "2026-04-04", "delivered"),
    (13, 6, "Mouse", 22.50, "2026-04-15", "delivered"),
    (14, 2, "Monitor", 305.00, "2026-04-21", "delivered"),
    (15, 3, "Webcam", 78.00, "2026-05-06", "delivered"),
    (16, 1, "Desk Lamp", 41.00, "2026-05-13", "delivered"),
    (17, 5, "Headphones", 92.50, "2026-05-24", "shipped"),
    (18, 4, "Monitor", 320.00, "2026-06-02", "shipped"),
    (19, 6, "Laptop", 1250.00, "2026-06-10", "shipped"),
    (20, 2, "Keyboard", 47.99, "2026-06-18", "pending"),
]


def main() -> None:
    db_path = Path(config.SQLITE_PATH).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:  # one transaction: all or nothing
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS customers"))
        conn.execute(
            text(
                "CREATE TABLE customers ("
                " id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,"
                " city TEXT NOT NULL, signup_date TEXT NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE orders ("
                " id INTEGER PRIMARY KEY,"
                " customer_id INTEGER NOT NULL REFERENCES customers(id),"
                " product TEXT NOT NULL, amount REAL NOT NULL,"
                " order_date TEXT NOT NULL, status TEXT NOT NULL)"
            )
        )
        conn.execute(
            text("INSERT INTO customers VALUES (:id, :name, :email, :city, :signup)"),
            [dict(id=r[0], name=r[1], email=r[2], city=r[3], signup=r[4]) for r in CUSTOMERS],
        )
        conn.execute(
            text("INSERT INTO orders VALUES (:id, :cid, :product, :amount, :date, :status)"),
            [
                dict(id=r[0], cid=r[1], product=r[2], amount=r[3], date=r[4], status=r[5])
                for r in ORDERS
            ],
        )

    print(f"Created {db_path} with {len(CUSTOMERS)} customers and {len(ORDERS)} orders.")


if __name__ == "__main__":
    main()
