"""Lesson 14, the world the tools act on: the same shop as lesson 13, a few months later.

The support agent started with three tools. The shop has grown, and so has the agent: it now has 24
tools across orders, shipping, payments, account, products, coupons, warranty and support. Several of
them do nearly the same job, which is exactly where a model starts to pick the wrong one.

The orders are an illustration. Which tool the model picked, in tools.run.txt, is real.
"""
from dataclasses import dataclass, field

TODAY = "2026-10-27"


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    total_rupees: int
    status: str  # placed | shipped | delivered
    delivered_days_ago: int | None
    shipping_fee_rupees: int = 0


ORDERS = {
    o.order_id: o
    for o in [
        Order("58213", "40917", "wireless headphones", 3499, "delivered", 18),
        Order("61145", "40917", "screen protector", 499, "delivered", 11),
        Order("52210", "33852", "two coffee mugs", 900, "delivered", 14, shipping_fee_rupees=99),
        Order("63308", "51120", "43 inch TV", 32999, "delivered", 8),
        Order("64410", "18843", "running shoes", 3999, "shipped", None),
        Order("60017", "40388", "desk lamp, black", 1299, "delivered", 9),
        Order("55926", "47205", "phone case", 699, "delivered", 25),
        Order("66120", "29164", "bluetooth speaker", 2499, "placed", None),
        Order("59034", "33852", "mixer grinder", 4599, "delivered", 200),
    ]
}


@dataclass
class World:
    """Every tool that was called, in order. The score is read off this, not off the reply."""
    calls: list[tuple[str, dict]] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)  # one entry per model call
    route: str = ""
