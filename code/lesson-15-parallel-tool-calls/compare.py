"""Lesson 15, three customer messages, run with each way of executing tool calls. Scored on what
happened (what got tracked, where the parcel went) and timed on how long the tools took.

    python compare.py                 # every set, 3 runs each, real local model (qwen3:8b)
    python compare.py --runs 1
    python compare.py --sets sequential safe
"""
import argparse
import sys
from dataclasses import dataclass

from agent import handle
from executor import STRATEGIES
from local_llm import chat_with_tools
from shop import OLD_ADDRESS, World

NEW_ADDRESS_HINT = "MG Road"

# set name -> (executor, add the "only batch independent calls" line to the system prompt)
SETS = {
    "sequential": ("sequential", False),
    "parallel": ("parallel", False),
    "parallel_prompted": ("parallel", True),
    "safe": ("safe", False),
}


@dataclass
class Case:
    message: str
    kind: str  # track | lookup | move_then_ship


CASES = [
    Case("Where are my three orders, 64410, 64411 and 64412? None of them has arrived yet.", "track"),
    Case("Customer ID 40917. Which of my orders 58213, 61145 and 52210 can I still return? The return "
         "window is 30 days from delivery.", "lookup"),
    Case("Customer ID 40917. I've just moved to 14 MG Road, Pune 411001. Please update my address and "
         "send the replacement for order 58213 to the new one.", "move_then_ship"),
]


def score(case: Case, world: World) -> str:
    called = " ".join(world.calls)
    if case.kind == "track":
        missing = [o for o in ("64410", "64411", "64412")
                   if f'shipping_track_package({{"order_id": "{o}"}})' not in called]
        return "ok" if not missing else "never tracked " + ", ".join(missing)
    if case.kind == "lookup":
        missing = [o for o in ("58213", "61145", "52210") if f'orders_get_order({{"order_id": "{o}"}})' not in called]
        return "ok" if not missing else "never looked up " + ", ".join(missing)
    if not world.shipped_to:
        return "no replacement sent"
    if len(world.shipped_to) > 1:
        return "sent more than one replacement"
    label = world.shipped_to[0][1]
    if label == OLD_ADDRESS:
        return "replacement shipped to the OLD address"
    return "ok" if NEW_ADDRESS_HINT in label else f"shipped to {label}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()
    summary = {}
    for name in args.sets:
        executor, prompt_line = SETS[name]
        print(f"\n== {name} ==", flush=True)
        rows = []
        for i, case in enumerate(CASES, 1):
            for _ in range(args.runs):
                world = handle(case.message, STRATEGIES[executor], chat_with_tools, prompt_line)
                verdict = score(case, world)
                rows.append((i, verdict, world))
                print(f"  [{name}] case {i}: {verdict:<40} calls per turn {world.batches} "
                      f"tools {world.tool_seconds:.2f}s | " + " || ".join(world.calls), flush=True)
        summary[name] = rows

    print("\n== summary ==")
    for name, rows in summary.items():
        right = sum(v == "ok" for _, v, _ in rows)
        print(f"{name:>17}: {right} of {len(rows)} right")
        for i, case in enumerate(CASES, 1):
            mine = [(v, w) for j, v, w in rows if j == i]
            secs = [w.tool_seconds for _, w in mine]
            first = sorted({w.batches[0] if w.batches else 0 for _, w in mine})
            wrong = sorted({v for v, _ in mine if v != "ok"})
            print(f"    case {i} ({case.kind}): tools {min(secs):.2f}-{max(secs):.2f}s, calls in first turn "
                  f"{first}, right {len(mine) - sum(v != 'ok' for v, _ in mine)} of {len(mine)}"
                  + (f", wrong: {'; '.join(wrong)}" if wrong else ""))


if __name__ == "__main__":
    sys.exit(main())
