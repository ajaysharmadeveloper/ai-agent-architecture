"""Lesson 16, three tools. Two read, one changes something (it moves money).

A tool raises ToolError when the call itself can never work as asked. That is a wrong decision, not a
bad moment: trying again with the same input gets the same answer, so it goes back to the model.
A ServiceError from shop.py is the outside service failing, which may pass if you try again.
"""
from shop import ORDERS, LAST_SCAN, WAIT, World
import time

ID = {"type": "string", "pattern": "^[0-9]{5}$", "description": "The 5-digit order number, like 58213."}


def _fn(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)}}}


TOOLS = [
    _fn("orders_get_order",
        "Look up one order by its order ID: the item, the price, the status (shipped or delivered) and the "
        "delivery date. Changes nothing.",
        {"order_id": ID}),
    _fn("shipping_track_package",
        "Get the courier's live tracking for one order: where the package is now and the expected delivery "
        "date. Changes nothing. Use this when the customer asks where an order is.",
        {"order_id": ID}),
    _fn("payments_refund",
        "Refund one delivered order in full to the customer's original payment method. Moves money. Call it "
        "once per order.",
        {"order_id": ID}),
]

READ_ONLY = {"orders_get_order", "shipping_track_package"}


class ToolError(Exception):
    """The call can't work as asked. `hint` says what is true and what to do instead."""

    def __init__(self, status: int, hint: str):
        super().__init__(hint)
        self.status, self.hint = status, hint


def _order(order_id: str):
    o = ORDERS.get(order_id)
    if o is None:
        raise ToolError(404, f"There is no order {order_id}. Check the number with the customer.")
    return o


def run_tool(world: World, name: str, args: dict, idempotency_key: str | None = None) -> dict:
    order_id = str(args.get("order_id", "")).strip()
    if name == "orders_get_order":
        time.sleep(WAIT["database"])
        o = _order(order_id)
        return {"order_id": o.order_id, "item": o.item, "price_inr": o.price_inr, "status": o.status,
                "delivered_on": o.delivered_on}
    if name == "shipping_track_package":
        o = _order(order_id)
        if o.status == "delivered":
            # the courier has nothing to track; asking again will never change that
            world.courier_calls += 1
            raise ToolError(404, f"Order {order_id} has no package in transit: it was delivered on "
                                 f"{o.delivered_on}. Do not track it again. Tell the customer it was delivered "
                                 f"on that date, or look it up with orders_get_order.")
        now_at, eta = world.courier_track(order_id)
        return {"order_id": order_id, "now_at": now_at, "expected_delivery": eta, "live": True}
    if name == "payments_refund":
        o = _order(order_id)
        if o.status != "delivered":
            raise ToolError(409, f"Order {order_id} has not been delivered yet, so it can't be refunded.")
        return {"ok": True, **world.payment_refund(order_id, o.price_inr, idempotency_key)}
    raise ToolError(404, f"There is no tool called {name}.")


def last_known_scan(order_id: str) -> dict | None:
    """The fallback for tracking: the last scan saved in the shop's own database. Older, but real."""
    if order_id not in LAST_SCAN:
        return None
    where, when = LAST_SCAN[order_id]
    return {"order_id": order_id, "live": False, "last_scan_at": where, "last_scan_on": when,
            "note": "Live tracking is down right now. This is the last scan in our own records, not live. "
                    "Tell the customer that."}
