"""Lesson 13, offline tests: no model, no network. Run with `python -m pytest` or `python test_schemas.py`."""
from agent import handle
from compare import CASES, score
from schemas import BAD_TOOLS, GOOD_TOOLS, run_bad, run_checked, run_good
from shop import World


def scripted(*turns):
    """A stand-in model that replays fixed tool calls, then stops."""
    queue = list(turns)

    def chat(messages, tools):
        if not queue:
            return {"role": "assistant", "content": "done"}
        name, args = queue.pop(0)
        return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}

    return chat


def test_good_refund_converts_rupees_to_paise():
    world = handle("x", GOOD_TOOLS, run_good, scripted(("payments_issue_refund", {"order_id": "58213", "amount_rupees": 3499, "reason": "damaged"})))
    assert world.refunds == [("58213", 349900)]


def test_bad_refund_sends_amt_straight_to_paise():
    world = handle("x", BAD_TOOLS, run_bad, scripted(("refund", {"id": "58213", "amt": 3499})))
    assert world.refunds == [("58213", 3499)]  # Rs 34.99: the unit was never named
    assert score(CASES[0], world) == "wrong amount: Rs 34.99 instead of Rs 3,499"


def test_good_lookup_returns_named_fields_only():
    result = run_good(World(), "orders_get_order", {"order_id": "47781"})
    assert result == {"found": True, "order_id": "47781", "item": "winter jacket", "status": "delivered",
                      "delivered_days_ago": 41, "total_rupees": 2799}


def test_bad_lookup_returns_internal_codes():
    assert run_bad(World(), "lookup", {"q": "47781"}) == {"oid": "47781", "cid": "29164", "st": 4, "d": 41, "p": 2799}


def test_escalation_case_fails_if_anything_was_paid():
    world = World()
    world.pay_refund("63308", 3299900)
    world.escalate("63308")
    assert score(CASES[4], world) == "paid a refund the policy forbids"


def test_every_good_description_is_at_least_three_sentences():
    for tool in GOOD_TOOLS:
        assert tool["function"]["description"].count(". ") >= 2, tool["function"]["name"]


def test_checked_tool_refuses_a_refund_over_the_limit():
    world = handle("x", GOOD_TOOLS, run_checked, scripted(("payments_issue_refund", {"order_id": "63308", "amount_rupees": 32999, "reason": "damaged"})))
    assert world.refunds == []


def test_checked_tool_refuses_a_shrunk_refund_on_an_expensive_order():
    world = handle("x", GOOD_TOOLS, run_checked, scripted(("payments_issue_refund", {"order_id": "63308", "amount_rupees": 5000, "reason": "damaged"})))
    assert world.refunds == []


def test_checked_tool_still_pays_a_valid_refund():
    world = handle("x", GOOD_TOOLS, run_checked, scripted(("payments_issue_refund", {"order_id": "58213", "amount_rupees": 3499, "reason": "damaged"})))
    assert world.refunds == [("58213", 349900)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
