"""Lesson 15, four tools: two that only read, two that change things.

READ_ONLY is the line this lesson is about. Reading the same facts in any order gives the same answer,
so reads can run at the same time. A call that changes something can change what the next call sees,
so those run one at a time, in order. The model does not decide this. Your code does.
"""
from shop import ORDERS, TRACKING, World

ID = {"type": "string", "pattern": "^[0-9]{5}$", "description": "The 5-digit order number, like 58213."}


def _fn(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)}}}


TOOLS = [
    _fn("orders_get_order",
        "Look up one order by its order ID. Returns the item, the status (placed, shipped or delivered) and "
        "how many days ago it was delivered. Changes nothing. Call it once per order. It gives the status word "
        "only, not where a package is now; for that use shipping_track_package.",
        {"order_id": ID}),
    _fn("shipping_track_package",
        "Get the courier's live tracking for one shipped order: where the package is now and the expected "
        "delivery date. Changes nothing. Call it once per order. Use this when the customer asks where an "
        "order is or when it will arrive.",
        {"order_id": ID}),
    _fn("account_update_address",
        "Change the customer's saved address. Every parcel booked after this goes to the new address. "
        "Returns the saved address.",
        {"customer_id": {"type": "string", "description": "The customer ID, like 40917."},
         "address": {"type": "string", "description": "The full new address, with the pincode."}}),
    _fn("orders_ship_replacement",
        "Book a courier to send a free replacement for one delivered order. The parcel goes to the "
        "customer's saved address at the moment of booking. Returns the address on the parcel.",
        {"order_id": ID}),
]

READ_ONLY = {"orders_get_order", "shipping_track_package"}


def run_tool(world: World, name: str, args: dict) -> dict:
    order_id = str(args.get("order_id", "")).strip()
    if name == "orders_get_order":
        world.wait(name)
        o = ORDERS.get(order_id)
        if o is None:
            return {"error": "no order with that ID"}
        return {"order_id": o.order_id, "item": o.item, "status": o.status,
                "delivered_days_ago": o.delivered_days_ago}
    if name == "shipping_track_package":
        world.wait(name)
        if order_id not in TRACKING:
            return {"error": "no package in transit for that order"}
        now_at, eta = TRACKING[order_id]
        return {"order_id": order_id, "now_at": now_at, "expected_delivery": eta}
    if name == "account_update_address":
        world.wait(name)  # the write lands when the database is done, not before
        with world.lock:
            world.address[str(args.get("customer_id", "40917"))] = str(args.get("address", ""))
        return {"ok": True, "saved_address": str(args.get("address", ""))}
    if name == "orders_ship_replacement":
        o = ORDERS.get(order_id)
        if o is None:
            return {"error": "no order with that ID"}
        with world.lock:
            label = world.address[o.customer_id]  # read the address when the courier is booked
        world.wait(name)
        world.shipped_to.append((order_id, label))
        return {"ok": True, "shipped_to": label}
    return {"error": f"no tool called {name}"}
