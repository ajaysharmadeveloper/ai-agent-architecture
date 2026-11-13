"""Lesson 17, the shop the support agent works for. The same four orders as lesson 16.

Plain data, no MCP in this file. The point of the lesson is that this code does not change when you put
it behind an MCP server: the server is a thin wrapper around functions you already had.
"""
from dataclasses import dataclass

TODAY = "2026-11-12"


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    price_inr: int
    status: str            # "shipped" or "delivered"
    delivered_on: str | None
    now_at: str | None = None   # where a shipped package is now (a delivered one has nothing to track)


ORDERS = {o.order_id: o for o in [
    Order("58213", "40917", "wireless headphones", 2499, "delivered", "2026-10-18"),
    Order("61145", "40917", "screen protector", 399, "delivered", "2026-10-25"),
    Order("64410", "18843", "running shoes", 3199, "shipped", None, "Andheri East delivery hub"),
    Order("64412", "18843", "water bottle", 549, "shipped", None, "Bhiwandi sorting centre"),
]}

REFUND_POLICY = """Refund policy
- A delivered order can be refunded in full within 30 days of delivery.
- A shipped order can't be refunded until it is delivered.
- The money goes back to the original payment method in 5 to 7 working days.
- One refund per order."""

REFUNDED: set[str] = set()   # orders refunded while the server is running


class ShopError(Exception):
    """The call can't work as asked. The message says what is true and what to do instead."""


def get_order(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        raise ShopError(f"There is no order {order_id}. Check the number with the customer.")
    return {"order_id": o.order_id, "item": o.item, "price_inr": o.price_inr, "status": o.status,
            "delivered_on": o.delivered_on}


def track_package(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        raise ShopError(f"There is no order {order_id}. Check the number with the customer.")
    if o.status == "delivered":
        raise ShopError(f"Order {order_id} has nothing in transit: it was delivered on {o.delivered_on}.")
    return {"order_id": order_id, "now_at": o.now_at}


def refund(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        raise ShopError(f"There is no order {order_id}. Check the number with the customer.")
    if o.status != "delivered":
        raise ShopError(f"Order {order_id} has not been delivered yet, so it can't be refunded.")
    if order_id in REFUNDED:
        raise ShopError(f"Order {order_id} was already refunded. Don't refund it again.")
    REFUNDED.add(order_id)
    return {"order_id": order_id, "refunded_inr": o.price_inr, "arrives_in": "5 to 7 working days"}
