"""Lesson 16, four customer messages, each with a failure planted in it, run with each error policy.
Scored on what actually happened: how often the outside service was hit, how many refunds were paid,
and what the customer was told.

    python compare.py                    # every policy, 3 runs each, real local model (qwen3:8b)
    python compare.py --runs 1
    python compare.py --policies raw smart
"""
import argparse
import sys
import time
from dataclasses import dataclass

from agent import handle
from handler import POLICIES
from local_llm import chat_with_tools
from shop import World


@dataclass
class Case:
    message: str
    kind: str
    faults: dict


CASES = [
    Case("Where is my order 64410? It hasn't arrived yet.", "busy", {"courier": "busy"}),
    Case("Where is my order 58213? It's the wireless headphones.", "wrong_call", {}),
    Case("Where is my order 64412? It's a water bottle.", "down", {"courier": "down"}),
    Case("Customer 40917. Please refund order 58213, the headphones arrived broken.", "refund",
         {"payment": "lost_reply"}),
]


def score(case: Case, world: World) -> str:
    said = world.reply.lower()
    if case.kind == "busy":
        return "ok" if "andheri" in said else "customer not told where it is"
    if case.kind == "wrong_call":
        return "ok" if "deliver" in said and "18" in said else "customer not told it was delivered on the 18th"
    if case.kind == "down":
        if "delhi" in said:
            return "ok"
        return "customer told nothing useful"
    paid = len(world.refunds_paid)
    if paid == 0:
        return "no refund paid"
    if paid > 1:
        return f"refund PAID {paid} TIMES"
    if any(w in said for w in FAILED_WORDS):
        return "paid once, customer told it FAILED"
    return "ok"


# words that tell the customer the refund did not go through
FAILED_WORDS = ("issue", "unable", "couldn't", "could not", "failed", "error", "try again", "problem")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    args = parser.parse_args()
    summary = {}
    for name in args.policies:
        print(f"\n== {name} ==", flush=True)
        rows = []
        for i, case in enumerate(CASES, 1):
            for _ in range(args.runs):
                start = time.perf_counter()
                world = handle(case.message, POLICIES[name], chat_with_tools, case.faults)
                secs = time.perf_counter() - start
                verdict = score(case, world)
                rows.append((i, verdict, world))
                print(f"  [{name}] case {i} ({case.kind}): {verdict:<40} model asked {world.model_calls} | "
                      f"tries {world.attempts} | courier hit {world.courier_calls}x, refunds paid "
                      f"{len(world.refunds_paid)} | {secs:.1f}s", flush=True)
                print(f"      reply: {world.reply[:300]!r}", flush=True)
        summary[name] = rows

    print("\n== summary ==")
    for name, rows in summary.items():
        right = sum(v == "ok" for _, v, _ in rows)
        print(f"{name:>9}: {right} of {len(rows)} right")
        for i, case in enumerate(CASES, 1):
            mine = [(v, w) for j, v, w in rows if j == i]
            wrong = sorted({v for v, _ in mine if v != "ok"})
            courier = sorted({w.courier_calls for _, w in mine})
            paid = sorted({len(w.refunds_paid) for _, w in mine})
            asked = sorted({len(w.model_calls) for _, w in mine})
            print(f"    case {i} ({case.kind}): right {sum(v == 'ok' for v, _ in mine)} of {len(mine)}, "
                  f"courier hit {courier}, refunds paid {paid}, tool calls the model asked for {asked}"
                  + (f", wrong: {'; '.join(wrong)}" if wrong else ""))


if __name__ == "__main__":
    sys.exit(main())
