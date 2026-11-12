"""Lesson 12, the side effect: issuing a refund. A stand-in for a payment provider, kept in SQLite so
you can count exactly how many refunds really went out.

An idempotency key makes the call safe to repeat: the same key twice returns the first refund instead
of paying again. Payment APIs offer this for exactly this reason. Without it, a step that runs twice
pays twice.
"""
import sqlite3


class Ledger:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS refunds ("
            " refund_id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, amount INTEGER, idempotency_key TEXT UNIQUE)"
        )

    def issue_refund(self, request_id: str, amount: int, idempotency_key: str | None) -> int:
        if idempotency_key:
            row = self.db.execute("SELECT refund_id FROM refunds WHERE idempotency_key = ?", (idempotency_key,))
            existing = row.fetchone()
            if existing:
                return existing[0]  # already paid: hand back the same refund, don't pay again
        cursor = self.db.execute(
            "INSERT INTO refunds (request_id, amount, idempotency_key) VALUES (?, ?, ?)",
            (request_id, amount, idempotency_key),
        )
        self.db.commit()
        return cursor.lastrowid

    def paid(self) -> dict[str, int]:
        """How many times each request was actually refunded."""
        rows = self.db.execute("SELECT request_id, COUNT(*) FROM refunds GROUP BY request_id")
        return dict(rows.fetchall())
