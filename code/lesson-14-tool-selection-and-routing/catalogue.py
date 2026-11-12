"""Lesson 14, the agent's 24 tools, written down once and turned into three tool sets.

Every tool has:
  - a description, written the lesson 13 way: what it does, when to use it, what it returns.
  - a `not_this` line: the sentence that names its look-alike and says when to use that one instead.
  - a team: the part of the shop it belongs to. The router uses this.

The three sets the agent is tested with:
  crowded        all 24 tools, descriptions only. Each one is fine on its own; none mentions its twin.
  disambiguated  all 24 tools, description + not_this.
  routed_by_team a first step picks the part of the shop (orders, payments, ...), then the agent sees
                 only that team's tools, plus the two tools every step may need (look up an order, hand
                 it to a person). Same descriptions as `disambiguated`.
  routed_by_goal the same, but the groups are what the customer wants (money back, where is it, ...),
                 and look-alike tools share a group.

The code each tool runs is the same in every set (run_tool below). Only what the model is shown changes.
"""
from dataclasses import dataclass

from shop import ORDERS, World

ID = {"type": "string", "pattern": "^[0-9]{5}$", "description": "The 5-digit order number, like 58213."}
CUSTOMER = {"type": "string", "description": "The customer ID the customer gives, like 40917."}
RUPEES = {"type": "integer", "minimum": 1, "description": "Whole rupees, like 3499 for Rs 3,499."}
TEXT = {"type": "string"}


@dataclass
class Tool:
    name: str
    team: str
    description: str
    not_this: str
    params: dict
    changes_things: bool  # True = it moves money, books something or changes a record


CATALOGUE = [
    # ── orders ──────────────────────────────────────────────────────────────────────────────
    Tool("orders_get_order", "orders",
         "Look up one order by its order ID. Returns the item, the status (placed, shipped or delivered), "
         "how many days ago it was delivered, and the total in rupees.",
         "It gives the status word only, not where a package is now or when it will arrive; for that use "
         "shipping_track_package.",
         {"order_id": ID}, False),
    Tool("orders_cancel_order", "orders",
         "Cancel an order so it never ships. The customer's payment is released automatically, so no refund "
         "step is needed. Returns whether the cancellation went through.",
         "Only for an order whose status is placed. If the order has shipped or been delivered, use "
         "payments_issue_refund or orders_request_return instead.",
         {"order_id": ID, "reason": TEXT}, True),
    Tool("orders_request_return", "orders",
         "Book a courier pickup so the customer can send a delivered item back. Returns the pickup date.",
         "It does not pay any money back, and it is not for swapping to another size or colour; that is "
         "orders_request_exchange.",
         {"order_id": ID, "reason": TEXT}, True),
    Tool("orders_request_exchange", "orders",
         "Swap a delivered item for a different size or colour of the same product. Returns the new "
         "delivery date.",
         "Only when the customer wants the same product in another size or colour. If they want their money "
         "back, use payments_issue_refund.",
         {"order_id": ID, "new_variant": {"type": "string", "description": "The size or colour they want."}},
         True),
    Tool("orders_update_delivery_address", "orders",
         "Change where one order will be delivered. Returns the new delivery address.",
         "Only for one order that has not been delivered yet. It does not change the customer's saved "
         "address for future orders; that is account_update_address.",
         {"order_id": ID, "address": TEXT}, True),
    Tool("orders_send_invoice", "orders",
         "Email a copy of the GST invoice for an order to the customer, as it is now. Returns the email "
         "address it was sent to.",
         "It sends the invoice unchanged. To correct something printed on an invoice, use "
         "support_create_ticket.",
         {"order_id": ID}, True),
    # ── shipping ────────────────────────────────────────────────────────────────────────────
    Tool("shipping_track_package", "shipping",
         "Get the courier's live tracking for an order: where the package is now and the expected delivery "
         "date.",
         "Use this when the customer asks where their order is or when it will arrive.",
         {"order_id": ID}, False),
    Tool("shipping_get_delivery_options", "shipping",
         "List the delivery speeds and charges for a pincode, before an order is placed.",
         "Not for an order that already exists.",
         {"pincode": {"type": "string", "pattern": "^[0-9]{6}$"}}, False),
    # ── payments ────────────────────────────────────────────────────────────────────────────
    Tool("payments_issue_refund", "payments",
         "Send money back to the card or UPI the customer paid with, for one order. This moves real money "
         "and cannot be undone. Use it only for a delivered order, 30 days or less after delivery, with an "
         "order total of Rs 5000 or less; for anything else use support_escalate_to_human. Returns the "
         "refund ID.",
         "If the customer asks for wallet credit, use payments_issue_store_credit. If they want a faulty "
         "item repaired or replaced, use warranty_file_claim. If the order has not shipped yet, use "
         "orders_cancel_order.",
         {"order_id": ID, "amount_rupees": RUPEES,
          "reason": {"type": "string", "enum": ["damaged", "not_as_described", "charged_twice", "other"]}},
         True),
    Tool("payments_issue_store_credit", "payments",
         "Add credit to the customer's shop wallet, for one order. Returns the new wallet balance.",
         "Only when the customer asks for wallet or store credit. It is not money back to their bank or "
         "card; for that use payments_issue_refund.",
         {"order_id": ID, "amount_rupees": RUPEES}, True),
    Tool("payments_refund_shipping_fee", "payments",
         "Refund the delivery charge on one order. Returns the amount refunded.",
         "Only the delivery fee, never the item price. Use when a delivery charge was applied by mistake.",
         {"order_id": ID}, True),
    Tool("payments_get_payment_status", "payments",
         "Check whether the payment for an order went through, is pending or failed.",
         "It does not move any money.",
         {"order_id": ID}, False),
    # ── account ─────────────────────────────────────────────────────────────────────────────
    Tool("account_get_profile", "account",
         "Get the customer's name, phone, email and saved address.",
         "It changes nothing.",
         {"customer_id": CUSTOMER}, False),
    Tool("account_update_address", "account",
         "Change the customer's saved default address, used for all their future orders. Returns the saved "
         "address.",
         "It does not change where an order already on its way is delivered; for that use "
         "orders_update_delivery_address.",
         {"customer_id": CUSTOMER, "address": TEXT}, True),
    Tool("account_get_loyalty_points", "account",
         "Get the customer's loyalty points balance.",
         "It changes nothing.",
         {"customer_id": CUSTOMER}, False),
    # ── products ────────────────────────────────────────────────────────────────────────────
    Tool("products_search", "products",
         "Search the catalogue by words, like 'white desk lamp'. Returns matching products and prices.",
         "Not for anything about an existing order.",
         {"query": TEXT}, False),
    Tool("products_check_stock", "products",
         "Check whether a product is in stock in a given size or colour.",
         "It does not reserve or swap anything; to swap an item on an order use orders_request_exchange.",
         {"product": TEXT, "variant": TEXT}, False),
    # ── coupons ─────────────────────────────────────────────────────────────────────────────
    Tool("coupons_list_offers", "coupons",
         "List the coupons and offers running today.",
         "It changes nothing.",
         {}, False),
    Tool("coupons_apply_coupon", "coupons",
         "Apply a coupon code to an order that has not shipped yet.",
         "Not a refund. It only works before the order ships.",
         {"order_id": ID, "code": TEXT}, True),
    # ── warranty ────────────────────────────────────────────────────────────────────────────
    Tool("warranty_register_product", "warranty",
         "Register a delivered product for its maker's warranty.",
         "To report that a product has stopped working, use warranty_file_claim.",
         {"order_id": ID}, True),
    Tool("warranty_file_claim", "warranty",
         "Open a warranty claim so the maker repairs or replaces a faulty item. Returns the claim number.",
         "No money is refunded. If the customer wants their money back, use payments_issue_refund.",
         {"order_id": ID, "problem": TEXT}, True),
    # ── support ─────────────────────────────────────────────────────────────────────────────
    Tool("support_escalate_to_human", "support",
         "Pass the case to a person on the support team now, with a short summary. Use it for any refund the "
         "policy does not allow, and whenever the customer asks for a person. Returns the ticket number.",
         "A person picks this up today. For a non-urgent follow-up, use support_create_ticket.",
         {"order_id": ID, "summary": TEXT}, True),
    Tool("support_create_ticket", "support",
         "Log a non-urgent issue for the support team to fix later, like a correction to an invoice. Returns "
         "the ticket number.",
         "Not for refunds and not for anything a person has to decide today; that is "
         "support_escalate_to_human.",
         {"order_id": ID, "note": TEXT}, True),
    Tool("support_send_sms", "support",
         "Send the customer a text message.",
         "Only to send information. It does not change any order.",
         {"customer_id": CUSTOMER, "text": TEXT}, True),
]

BY_NAME = {t.name: t for t in CATALOGUE}
TEAMS = {
    "orders": "cancel an order that has not shipped, return or exchange an item, change one order's "
              "delivery address, send an invoice",
    "shipping": "where a package is, when it will arrive, delivery charges for a pincode",
    "payments": "refunds, wallet or store credit, delivery fee refunds, payment status",
    "account": "the customer's saved address, profile, loyalty points",
    "products": "searching the catalogue, stock",
    "coupons": "coupons and offers",
    "warranty": "a faulty product that should be repaired or replaced under warranty",
    "support": "passing a case to a person, non-urgent follow-ups like invoice corrections",
}
# The same tools grouped a second way: by what the customer is asking for, not by which part of the shop
# owns the tool. Look-alike tools sit in the SAME group on purpose, so the model still sees both and the
# not_this lines can do their job. A tool can sit in more than one group.
GOALS = {
    "money_back": ("the customer wants money back, wallet credit, a fee returned, or to cancel before it ships",
                   ["payments_issue_refund", "payments_issue_store_credit", "payments_refund_shipping_fee",
                    "orders_cancel_order", "warranty_file_claim", "payments_get_payment_status"]),
    "where_is_it": ("where an order is, when it arrives, or where it should be delivered",
                    ["shipping_track_package", "orders_update_delivery_address", "account_update_address"]),
    # v2: payments_issue_store_credit added. In the first run (tools.run.txt) "wrong colour, put the money
    # in my wallet" was routed here, found no wallet tool, and the agent did nothing, 3 of 3.
    "fix_the_item": ("send an item back, swap its size or colour, or get a faulty item repaired",
                     ["orders_request_return", "orders_request_exchange", "warranty_file_claim",
                      "products_check_stock", "payments_issue_refund", "payments_issue_store_credit"]),
    "my_account": ("the customer's saved address, profile or loyalty points",
                   ["account_update_address", "account_get_profile", "account_get_loyalty_points",
                    "orders_update_delivery_address"]),
    "paperwork": ("invoices, corrections to an invoice, non-urgent follow-ups",
                  ["orders_send_invoice", "support_create_ticket"]),
    "shopping": ("finding products, stock, coupons, delivery charges before ordering",
                 ["products_search", "products_check_stock", "coupons_list_offers", "coupons_apply_coupon",
                  "shipping_get_delivery_options"]),
    "talk_to_a_person": ("the customer asks for a person, or nothing else fits",
                         ["support_create_ticket", "support_send_sms"]),
}
ALWAYS = ["orders_get_order", "support_escalate_to_human"]  # every step may need these


def _schema(tool: Tool, with_not_this: bool) -> dict:
    description = tool.description + (" " + tool.not_this if with_not_this else "")
    return {"type": "function", "function": {
        "name": tool.name, "description": description,
        "parameters": {"type": "object", "properties": tool.params, "required": list(tool.params)}}}


def all_tools(with_not_this: bool) -> list[dict]:
    return [_schema(t, with_not_this) for t in CATALOGUE]


def group_tools(grouping: str, group: str) -> list[dict]:
    """The tools the agent sees after the router picked `group`, plus the two it always gets."""
    if grouping == "team":
        picked = [t.name for t in CATALOGUE if t.team == group]
    else:
        picked = GOALS[group][1]
    names = ALWAYS + [n for n in picked if n not in ALWAYS]
    return [_schema(BY_NAME[n], True) for n in names]


def run_tool(world: World, name: str, args: dict) -> dict:
    """What every tool does. Read tools return facts; the rest return a plausible confirmation.
    The score never looks at these replies, only at which tools were called."""
    world.calls.append((name, args))
    order = ORDERS.get(str(args.get("order_id", "")).strip())
    if name not in BY_NAME:
        return {"error": f"no tool called {name}"}
    if "order_id" in BY_NAME[name].params and order is None:
        return {"error": "no order with that ID"}
    if name == "orders_get_order":
        return {"order_id": order.order_id, "item": order.item, "status": order.status,
                "delivered_days_ago": order.delivered_days_ago, "total_rupees": order.total_rupees,
                "shipping_fee_rupees": order.shipping_fee_rupees}
    if name == "shipping_track_package":
        if order.status != "shipped":
            return {"status": order.status, "note": "no package in transit"}
        return {"status": "in transit", "now_at": "Bhiwandi sorting hub", "expected_delivery": "2026-10-29"}
    if name == "orders_cancel_order" and order.status != "placed":
        return {"ok": False, "error": f"cannot cancel, order is {order.status}"}
    return {"ok": True, "reference": f"{name.split('_')[0]}_{len(world.calls)}"}
