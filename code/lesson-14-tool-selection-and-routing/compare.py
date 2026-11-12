"""Lesson 14, twelve customer messages, each one sitting next to a look-alike tool. The same model and
system prompt, four ways of giving it the 24 tools. Scored on which tools it actually called.

    python compare.py                          # all four sets, 3 runs each, real local model (qwen3:8b)
    python compare.py --runs 1
    python compare.py --sets crowded routed_by_goal
"""
import argparse
import json
import sys
from dataclasses import dataclass

from agent import handle
from catalogue import BY_NAME, all_tools
from local_llm import chat_text, chat_with_tools
from shop import World

SETS = ["crowded", "disambiguated", "routed_by_team", "routed_by_goal"]


@dataclass
class Case:
    message: str
    tool: str        # the tool that does what the customer asked
    key: str         # the order or customer ID it must be called with
    twin: str        # the look-alike tool a model is likely to confuse it with


CASES = [
    Case("Order 58213, the headphones stopped working after a week. I want my money back.",
         "payments_issue_refund", "58213", "warranty_file_claim"),
    Case("Order 55926, the phone case came in the wrong colour. Just put the Rs 699 in my wallet, I'll "
         "buy something else later.", "payments_issue_store_credit", "55926", "payments_issue_refund"),
    Case("I want my money back for order 66120, I ordered the wrong speaker by mistake.",
         "orders_cancel_order", "66120", "payments_issue_refund"),
    Case("Where is my order 64410? It still hasn't arrived.",
         "shipping_track_package", "64410", "orders_get_order"),
    Case("I've moved. My new address is 14 MG Road, Pune 411001, please use it for my orders from now on. "
         "Customer ID 40917.", "account_update_address", "40917", "orders_update_delivery_address"),
    Case("Order 64410 is on its way but I won't be home. Can it go to my office instead, 22 Park Street, "
         "Kolkata 700016?", "orders_update_delivery_address", "64410", "account_update_address"),
    Case("The desk lamp from order 60017 is black but I ordered white. Can I get the white one instead?",
         "orders_request_exchange", "60017", "orders_request_return"),
    Case("Please send someone to pick up order 61145, I'm sending the screen protector back.",
         "orders_request_return", "61145", "orders_request_exchange"),
    Case("You charged me Rs 99 delivery on order 52210 but delivery was supposed to be free. Please "
         "give that back.", "payments_refund_shipping_fee", "52210", "payments_issue_refund"),
    Case("My mixer grinder from order 59034 has stopped working. It's still under warranty, can it be "
         "repaired?", "warranty_file_claim", "59034", "payments_issue_refund"),
    Case("Order 63308, the TV arrived with a cracked screen. Refund it please.",
         "support_escalate_to_human", "63308", "payments_issue_refund"),
    Case("No rush, but the invoice for order 58213 shows my old company name. Can someone correct it "
         "when they get time?", "support_create_ticket", "58213", "orders_send_invoice"),
]


def score(case: Case, world: World) -> str:
    """'ok' when the right tool ran on the right ID and nothing else that changes things ran.
    Otherwise the name of what it did instead."""
    keys = {str(v) for _, args in world.calls for v in args.values()}
    actions = [name for name, _ in world.calls if name in BY_NAME and BY_NAME[name].changes_things]
    called = [name for name, _ in world.calls]
    if case.tool in called and case.key in keys and all(a == case.tool for a in actions):
        return "ok"
    wrong = [a for a in actions if a != case.tool] or [n for n in called if n != case.tool]
    if wrong:
        return "picked " + ", ".join(dict.fromkeys(wrong))
    return "picked nothing"


def run_set(name: str, runs: int) -> list[list[tuple[str, World]]]:
    tools = {"crowded": all_tools(with_not_this=False), "disambiguated": all_tools(with_not_this=True),
             "routed_by_team": "team", "routed_by_goal": "goal"}[name]
    results = []
    for i, case in enumerate(CASES, 1):
        row = []
        for _ in range(runs):
            world = handle(case.message, tools, chat_with_tools, chat_text)
            verdict = score(case, world)
            row.append((verdict, world))
            route = f" route={world.route}" if world.route else ""
            calls = "; ".join(f"{n}({json.dumps(a)})" for n, a in world.calls) or "none"
            print(f"  [{name}] case {i:>2}: {verdict:<44}{route} calls: {calls}", flush=True)
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    args = parser.parse_args()
    summary = {}
    for name in args.sets:
        print(f"\n== {name} ==", flush=True)
        summary[name] = run_set(name, args.runs)

    total = len(CASES) * args.runs
    print("\n== summary ==")
    for name, rows in summary.items():
        right = sum(v == "ok" for row in rows for v, _ in row)
        step = 1 if name.startswith("routed") else 0  # a routed run's first entry is the router call
        first_step = [w.prompt_tokens[step] for row in rows for _, w in row]
        per_case = [sum(w.prompt_tokens) for row in rows for _, w in row]
        print(f"{name:>13}: {right} of {total} right | prompt tokens, first agent step: "
              f"{min(first_step)}-{max(first_step)} | per message, all steps: {sum(per_case) // len(per_case)} avg")
    for name, rows in summary.items():
        print(f"\nwhat went wrong, {name}:")
        any_wrong = False
        for i, row in enumerate(rows, 1):
            wrong = [v for v, _ in row if v != "ok"]
            if wrong:
                any_wrong = True
                routes = sorted({w.route for _, w in row if w.route})
                via = f" [routed to {', '.join(routes)}]" if routes else ""
                print(f"  case {i:>2} ({CASES[i - 1].tool}): " + "; ".join(sorted(set(wrong)))
                      + f" ({len(wrong)} of {len(row)}){via}")
        if not any_wrong:
            print("  nothing")


if __name__ == "__main__":
    sys.exit(main())
