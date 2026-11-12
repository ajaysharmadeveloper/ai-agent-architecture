"""Lesson 13, the world the tools act on: an orders table and a payment service.

The orders table stores prices in rupees. The payment service, like most payment APIs, takes amounts
in the smallest currency unit, paise (Rs 1 = 100 paise). Two services, two units: exactly the kind of
detail a tool schema has to carry, because the model cannot see this file.

The orders are an illustration. The decisions in compare.run.txt come from the real model.
"""
from dataclasses import dataclass, field

TODAY = "2026-10-21"


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    total_rupees: int
    status: str  # placed | shipped | delivered
    delivered_days_ago: int | None


ORDERS = {
    o.order_id: o
    for o in [
        Order("58213", "40917", "wireless headphones", 3499, "delivered", 12),
        Order("61145", "40917", "screen protector", 499, "delivered", 5),
        Order("52210", "33852", "two coffee mugs", 900, "delivered", 8),
        Order("47781", "29164", "winter jacket", 2799, "delivered", 41),
        Order("63308", "51120", "43 inch TV", 32999, "delivered", 2),
        Order("64410", "18843", "running shoes", 3999, "shipped", None),
        Order("60017", "40388", "desk lamp", 1299, "delivered", 3),
        Order("55926", "47205", "phone case", 699, "delivered", 19),
    ]
}

STATUS_CODE = {"placed": 1, "shipped": 2, "delivered": 4}


@dataclass
class World:
    """Everything the agent did, so the score is read off what happened, not what it said."""
    refunds: list[tuple[str, int]] = field(default_factory=list)  # (order_id, amount_paise)
    escalations: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def pay_refund(self, order_id: str, amount_paise: int) -> dict:
        order = ORDERS.get(order_id)
        if order is None:
            return {"ok": False, "error": "no such order"}
        self.refunds.append((order_id, amount_paise))
        return {"ok": True, "refund_id": f"rf_{len(self.refunds)}", "amount_paise": amount_paise}

    def escalate(self, order_id: str) -> None:
        self.escalations.append(order_id)
