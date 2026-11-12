"""The two tools from lesson 8's worked task: list this month's invoices, send one email.

Stand-ins so both agents can run without a real billing system: 400 invoices, 3 of them overdue
and over Rs 50,000. Swap the bodies for your real API calls.
"""
import random
from datetime import date

random.seed(8)
_TODAY = date(2026, 9, 24)
INVOICES = [
    {
        "id": f"INV-{1000 + i}",
        "customer_email": f"customer{i}@example.com",
        "amount_inr": random.randint(2_000, 45_000),
        "due": date(2026, 9, random.randint(1, 30)).isoformat(),
        "paid": True,
        "note": "",
    }
    for i in range(400)
]
for i, amount in ((37, 82_000), (151, 64_500), (298, 120_000)):
    INVOICES[i].update(amount_inr=amount, due="2026-09-05", paid=False)
INVOICES[151]["note"] = "Disputing this invoice. Talk to finance before chasing."


def list_invoices(month: str) -> list[dict]:
    """Every invoice due in `month` (YYYY-MM), paid or not."""
    return [inv for inv in INVOICES if inv["due"].startswith(month)]


def send_email(to: str, subject: str, body: str) -> str:
    """Send one email. Returns a delivery id."""
    print(f"-> email to {to}: {subject}")
    return f"msg-{abs(hash((to, subject))) % 10_000}"


def is_overdue(inv: dict) -> bool:
    return not inv["paid"] and date.fromisoformat(inv["due"]) < _TODAY
