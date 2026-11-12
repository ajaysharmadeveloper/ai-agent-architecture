"""Lesson 13, the same three tools described two ways.

Both sets call the SAME code underneath (shop.py). The only difference is what the model is told:
the names, the descriptions, the parameter names, and the shape of what comes back. The model never
sees your Python. The schema is the only part of the tool it reads.
"""
from shop import ORDERS, STATUS_CODE, World


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Ollama's tool format: a JSON Schema for the inputs, wrapped in a function entry."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# ── The badly described set: what a first draft usually looks like ──────────────────────────────

BAD_TOOLS = [
    _fn("lookup", "Lookup.", {"q": {"type": "string"}}, ["q"]),
    _fn("refund", "Refund.", {"id": {"type": "string"}, "amt": {"type": "number"}}, ["id", "amt"]),
    _fn("handoff", "Handoff.", {"note": {"type": "string"}}, ["note"]),
]


def run_bad(world: World, name: str, args: dict) -> dict:
    if name == "lookup":
        order = ORDERS.get(str(args.get("q", "")).strip())
        if order is None:
            return {"err": 404}
        # the raw row, internal codes and all
        return {"oid": order.order_id, "cid": order.customer_id, "st": STATUS_CODE[order.status],
                "d": order.delivered_days_ago, "p": order.total_rupees}
    if name == "refund":
        # amt goes straight to the payment service, which counts in paise
        return world.pay_refund(str(args.get("id", "")).strip(), int(round(float(args.get("amt", 0)))))
    if name == "handoff":
        note = str(args.get("note", ""))
        hit = next((oid for oid in ORDERS if oid in note), "")
        world.escalate(hit)
        return {"ok": True}
    return {"err": "unknown tool"}


# ── The well described set: names, descriptions, parameters and returns designed for the model ──

GOOD_TOOLS = [
    _fn(
        "orders_get_order",
        "Look up one order by its order ID. Returns the item, the delivery status, how many days ago it "
        "was delivered, and the order total in rupees. Use this first, before deciding anything about an "
        "order. The order ID is the 5-digit number the customer quotes for their order, like 58213; it is "
        "not the customer ID. Returns only that one order, not the customer's other orders.",
        {"order_id": {"type": "string", "pattern": "^[0-9]{5}$",
                      "description": "The 5-digit order number, like 58213. Never the customer ID."}},
        ["order_id"],
    ),
    _fn(
        "payments_issue_refund",
        "Send money back to the customer for one order. This moves real money and cannot be undone. Use it "
        "only after orders_get_order shows the order was delivered 30 days ago or less and the refund is "
        "Rs 5000 or less; for anything else, use support_escalate_to_human instead. The amount can be less "
        "than the order total, for a partial refund. Returns the refund ID and the amount refunded in rupees.",
        {
            "order_id": {"type": "string", "pattern": "^[0-9]{5}$",
                         "description": "The 5-digit order number being refunded."},
            "amount_rupees": {"type": "integer", "minimum": 1,
                              "description": "How much to refund, in whole rupees, like 3499 for Rs 3,499."},
            "reason": {"type": "string", "enum": ["damaged", "not_as_described", "charged_twice", "other"],
                       "description": "Why the customer is getting money back."},
        },
        ["order_id", "amount_rupees", "reason"],
    ),
    _fn(
        "support_escalate_to_human",
        "Hand this order to a person on the support team instead of acting yourself. Use it when a refund "
        "would be over Rs 5000, when the order was delivered more than 30 days ago, or when the customer "
        "asks for something no other tool can do. A person replies within one working day. Returns a ticket ID.",
        {
            "order_id": {"type": "string", "pattern": "^[0-9]{5}$", "description": "The 5-digit order number."},
            "summary": {"type": "string", "description": "One or two sentences a person can act on."},
        },
        ["order_id", "summary"],
    ),
]


def run_good(world: World, name: str, args: dict) -> dict:
    if name == "orders_get_order":
        order = ORDERS.get(str(args.get("order_id", "")).strip())
        if order is None:
            return {"found": False, "order_id": args.get("order_id")}
        return {"found": True, "order_id": order.order_id, "item": order.item, "status": order.status,
                "delivered_days_ago": order.delivered_days_ago, "total_rupees": order.total_rupees}
    if name == "payments_issue_refund":
        rupees = int(args.get("amount_rupees", 0))
        result = world.pay_refund(str(args.get("order_id", "")).strip(), rupees * 100)  # the unit lives in code
        if not result["ok"]:
            return {"refunded": False, "order_id": args.get("order_id")}
        return {"refunded": True, "refund_id": result["refund_id"], "order_id": args["order_id"],
                "amount_rupees": rupees}
    if name == "support_escalate_to_human":
        world.escalate(str(args.get("order_id", "")).strip())
        return {"ticket_id": f"T-{len(world.escalations)}", "order_id": args.get("order_id")}
    return {"error": f"unknown tool {name}"}


def run_checked(world: World, name: str, args: dict) -> dict:
    """The good set, plus the policy checked in the tool's own code. The description tells the model
    the rule; this makes sure a refund that breaks it is never paid, whatever the model sends.
    It checks facts the code knows (the order's total, status and age), not the amount the model
    picked: a model can shrink a refund to fit a limit, it can't change what the order cost."""
    if name == "payments_issue_refund":
        order = ORDERS.get(str(args.get("order_id", "")).strip())
        if order and (order.total_rupees > 5000 or order.status != "delivered" or (order.delivered_days_ago or 0) > 30):
            return {"refunded": False, "order_id": order.order_id,
                    "why": "outside the refund policy; use support_escalate_to_human"}
    return run_good(world, name, args)


TOOL_SETS = {"bad": (BAD_TOOLS, run_bad), "good": (GOOD_TOOLS, run_good), "checked": (GOOD_TOOLS, run_checked)}
