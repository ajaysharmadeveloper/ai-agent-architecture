"""Lesson 13, the before and after: eight customer messages, the same model, the same system prompt,
two tool sets. Scored on what actually happened in the world (money paid, tickets raised), not on what
the agent said.

    python compare.py                      # bad and good sets, 3 runs each, real local model (qwen3:8b)
    python compare.py --runs 1
    python compare.py --sets good checked  # the good set with and without the limit checked in code
"""
import argparse
import sys
from dataclasses import dataclass

from agent import handle
from local_llm import chat_with_tools
from schemas import TOOL_SETS
from shop import World


@dataclass
class Case:
    message: str
    expect: tuple  # ("refund", order_id, rupees) | ("escalate", order_id) | ("none",)


CASES = [
    Case("Order 58213, the headphones stopped working after a week. I want my money back.",
         ("refund", "58213", 3499)),
    Case("Hi, my customer ID is 40917. Order 61145 arrived with a cracked screen protector, please refund it.",
         ("refund", "61145", 499)),
    Case("Order 52210: two mugs came and one was broken. Each mug was Rs 450. Please refund the broken one.",
         ("refund", "52210", 450)),
    Case("Please refund order 47781, the jacket doesn't fit.", ("escalate", "47781")),
    Case("Order 63308, the TV arrived with a cracked screen. Refund it please.", ("escalate", "63308")),
    Case("Where is my order 64410? It still hasn't arrived.", ("none",)),
    Case("I was charged twice for order 60017. Please refund the extra Rs 1299.", ("refund", "60017", 1299)),
    Case("My phone case, order 55926, came in the wrong colour. I'd like a refund.", ("refund", "55926", 699)),
]


def score(case: Case, world: World) -> str:
    """'ok', or a short name for what went wrong."""
    kind = case.expect[0]
    if kind == "refund":
        _, order_id, rupees = case.expect
        if world.refunds == [(order_id, rupees * 100)]:
            return "ok"
        if not world.refunds:
            return "no refund paid"
        if len(world.refunds) > 1:
            return "paid more than once"
        paid_order, paid_paise = world.refunds[0]
        if paid_order != order_id:
            return "refunded the wrong order"
        return f"wrong amount: Rs {paid_paise / 100:,.2f} instead of Rs {rupees:,}"
    if kind == "escalate":
        if world.refunds:
            return "paid a refund the policy forbids"
        return "ok" if case.expect[1] in world.escalations else "never reached a person"
    if world.refunds:
        return "paid a refund nobody asked for"
    return "ok"


def run_set(name: str, runs: int, chat) -> list[list[str]]:
    tools, run_tool = TOOL_SETS[name]
    results = []
    for i, case in enumerate(CASES, 1):
        row = []
        for _ in range(runs):
            world = handle(case.message, tools, run_tool, chat)
            verdict = score(case, world)
            row.append(verdict)
            print(f"  [{name}] case {i}: {verdict:<45} calls: {'; '.join(world.calls) or 'none'}", flush=True)
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sets", nargs="+", default=["bad", "good"], choices=sorted(TOOL_SETS))
    args = parser.parse_args()
    summary = {}
    for name in args.sets:
        print(f"\n== {name} tool set ==", flush=True)
        summary[name] = run_set(name, args.runs, chat_with_tools)

    total = len(CASES) * args.runs
    print("\n== summary ==")
    for name, rows in summary.items():
        right = sum(v == "ok" for row in rows for v in row)
        print(f"{name:>5}: {right} of {total} handled correctly")
    for name, rows in summary.items():
        print(f"\nwhat went wrong, {name} set:")
        any_wrong = False
        for i, row in enumerate(rows, 1):
            wrong = [v for v in row if v != "ok"]
            if wrong:
                any_wrong = True
                print(f"  case {i}: " + "; ".join(sorted(set(wrong))) + f" ({len(wrong)} of {len(row)})")
        if not any_wrong:
            print("  nothing")


if __name__ == "__main__":
    sys.exit(main())
