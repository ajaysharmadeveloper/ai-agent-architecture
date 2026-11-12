"""Lesson 14, offline tests: no model, no network. Run with `python -m pytest` or `python test_routing.py`."""
from agent import handle, route
from catalogue import ALWAYS, CATALOGUE, GOALS, all_tools, group_tools
from compare import CASES, score


def scripted(*turns):
    """A stand-in model that replays fixed tool calls, then stops."""
    queue = list(turns)

    def chat(messages, tools):
        if not queue:
            return {"role": "assistant", "content": "done"}, 100
        name, args = queue.pop(0)
        return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}, 100

    return chat


def answers(word):
    return lambda messages: (word, 50)


def names(tools):
    return [t["function"]["name"] for t in tools]


def test_catalogue_has_24_tools_with_unique_names():
    assert len(CATALOGUE) == 24
    assert len(set(names(all_tools(False)))) == 24


def test_every_twin_named_in_a_case_exists():
    for case in CASES:
        assert case.tool in names(all_tools(True)) and case.twin in names(all_tools(True))


def test_goal_groups_keep_each_cases_twins_together_where_the_cost_is_money():
    money = GOALS["money_back"][1]
    for pair in [("payments_issue_refund", "payments_issue_store_credit"),
                 ("payments_issue_refund", "orders_cancel_order"),
                 ("payments_issue_refund", "warranty_file_claim")]:
        assert set(pair) <= set(money)


def test_a_group_always_carries_lookup_and_escalate():
    for group in GOALS:
        assert names(group_tools("goal", group))[:2] == ALWAYS


def test_router_falls_back_to_a_person_on_an_unknown_answer():
    assert route("x", "goal", answers("billing"))[0] == "talk_to_a_person"
    assert route("x", "team", answers("Payments."))[0] == "payments"


def test_score_ok_needs_the_right_tool_on_the_right_id():
    world = handle(CASES[0].message, all_tools(True), scripted(
        ("orders_get_order", {"order_id": "58213"}),
        ("payments_issue_refund", {"order_id": "58213", "amount_rupees": 3499, "reason": "damaged"})))
    assert score(CASES[0], world) == "ok"


def test_score_names_the_twin_it_picked_instead():
    world = handle(CASES[0].message, all_tools(True), scripted(
        ("warranty_file_claim", {"order_id": "58213", "problem": "stopped working"})))
    assert score(CASES[0], world) == "picked warranty_file_claim"


def test_routed_run_records_the_router_call_tokens_first():
    world = handle(CASES[3].message, "goal", scripted(("shipping_track_package", {"order_id": "64410"})),
                   answers("where_is_it"))
    assert world.route == "where_is_it" and world.prompt_tokens[0] == 50
    assert score(CASES[3], world) == "ok"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
