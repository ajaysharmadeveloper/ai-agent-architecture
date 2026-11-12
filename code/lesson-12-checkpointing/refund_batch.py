"""Lesson 12, Friday's refund batch: twelve requests, one model decision each, a real refund for every
approval. Long enough that a crash in the middle hurts.

Run: python refund_batch.py --fresh --crash-at 9   # crashes during request 9, right after paying it
     python refund_batch.py                         # same thread id: resumes at request 9, not 1
     python refund_batch.py --fresh --no-checkpoint --no-idempotency --crash-at 9
                                                    # then run it again: the shortcut, starts at 1
     python refund_batch.py --offline               # no model at all, a fixed rule decides

The model is qwen3:8b through Ollama (local_llm.py). Every step's decision is saved in the checkpoint,
so a resumed run never asks the model again about a request it already decided.
"""
import argparse
import os
import sys
import time
from pathlib import Path

from checkpoints import MemoryCheckpointer, SqliteCheckpointer
from payments import Ledger

HERE = Path(__file__).parent
POLICY = {"auto_limit": 5000, "days": 30}

# Friday's queue. Amounts in rupees; delivered_days_ago None means it never arrived.
REQUESTS = [
    {"id": "R1", "order": "58101", "amount": 1299, "delivered_days_ago": 4, "note": "Kettle arrived cracked."},
    {"id": "R2", "order": "58117", "amount": 8999, "delivered_days_ago": 2, "note": "Headphones stopped working."},
    {"id": "R3", "order": "58124", "amount": 499, "delivered_days_ago": 41, "note": "Changed my mind about the mug."},
    {"id": "R4", "order": "58133", "amount": 2450, "delivered_days_ago": None, "note": "Never delivered."},
    {"id": "R5", "order": "58140", "amount": 3200, "delivered_days_ago": 12, "note": "Wrong size shoes."},
    {"id": "R6", "order": "58152", "amount": 6100, "delivered_days_ago": 9, "note": "Mixer grinder sparks."},
    {"id": "R7", "order": "58166", "amount": 780, "delivered_days_ago": 1, "note": "Book pages missing."},
    {"id": "R8", "order": "58171", "amount": 1999, "delivered_days_ago": 35, "note": "Jacket faded after a wash."},
    {"id": "R9", "order": "58213", "amount": 3499, "delivered_days_ago": None, "note": "Not delivered by Friday."},
    {"id": "R10", "order": "58220", "amount": 950, "delivered_days_ago": 6, "note": "Charger is the wrong plug."},
    {"id": "R11", "order": "58234", "amount": 12500, "delivered_days_ago": 3, "note": "Phone screen dead on arrival."},
    {"id": "R12", "order": "58241", "amount": 640, "delivered_days_ago": 15, "note": "Bottle leaks."},
]

DECISION = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["APPROVE", "HUMAN"]}, "reason": {"type": "string"}},
    "required": ["decision", "reason"],
}


def llm_decide(request: dict, policy: dict) -> dict:
    import local_llm

    rules = (
        f"You process refund requests for an online store. APPROVE only if the amount is at most "
        f"Rs {policy['auto_limit']} AND (the order was never delivered OR it was delivered at most "
        f"{policy['days']} days ago). Otherwise send it to a HUMAN. Give a one-line reason. "
        + policy.get("extra_rule", "")
    )
    return local_llm.chat_json(
        [{"role": "system", "content": rules}, {"role": "user", "content": str(request)}], DECISION
    )


def rule_decide(request: dict, policy: dict) -> dict:
    """The offline stand-in: the same policy, as plain code."""
    days = request["delivered_days_ago"]
    ok = request["amount"] <= policy["auto_limit"] and (days is None or days <= policy["days"])
    return {"decision": "APPROVE" if ok else "HUMAN", "reason": "rule"}


def crash():
    sys.stdout.flush()  # so you can still read what it printed
    os._exit(1)  # like a kill -9 or a pulled plug: no cleanup, no except block, nothing saved


def run(thread_id, checkpointer, ledger, decide, crash_at=None, idempotent=True, crash=crash):
    found = checkpointer.latest(thread_id) if checkpointer else None
    if found:
        state = found.state
        print(f"resuming '{thread_id}' from its checkpoint: {found.step} of {len(REQUESTS)} done")
    else:
        state = {"policy": POLICY, "next": 0, "done": []}
        print(f"starting '{thread_id}' from request 1")

    for i in range(state["next"], len(REQUESTS)):
        request, started = REQUESTS[i], time.time()
        result = decide(request, state["policy"])
        refund_id = None
        if result["decision"] == "APPROVE":
            key = f"refund-{request['id']}" if idempotent else None
            refund_id = ledger.issue_refund(request["id"], request["amount"], key)
        if crash_at == i + 1:
            print(f"  step {i + 1}: {request['id']} paid, then the process died before saving")
            crash()
        state = {
            **state,
            "next": i + 1,
            "done": state["done"] + [{"id": request["id"], **result, "refund_id": refund_id}],
        }
        if checkpointer:
            checkpointer.save(thread_id, i + 1, state)
        print(f"  step {i + 1}: {request['id']} {result['decision']:8} {time.time() - started:5.1f}s  {result['reason'][:70]}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="friday-refunds")
    parser.add_argument("--crash-at", type=int)
    parser.add_argument("--fresh", action="store_true", help="delete the checkpoint and payment files first")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--memory", action="store_true", help="checkpoints in RAM: lost when the process dies")
    parser.add_argument("--no-idempotency", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.fresh:
        for name in ("checkpoints.db", "payments.db"):
            (HERE / name).unlink(missing_ok=True)
    if args.no_checkpoint:
        checkpointer = None
    elif args.memory:
        checkpointer = MemoryCheckpointer()
    else:
        checkpointer = SqliteCheckpointer(str(HERE / "checkpoints.db"))
    ledger = Ledger(str(HERE / "payments.db"))

    started = time.time()
    run(args.thread, checkpointer, ledger, rule_decide if args.offline else llm_decide,
        crash_at=args.crash_at, idempotent=not args.no_idempotency)
    paid = ledger.paid()
    twice = [rid for rid, n in paid.items() if n > 1]
    print(f"done in {time.time() - started:.0f}s. refunds paid: {sum(paid.values())} for {len(paid)} requests. "
          f"paid twice: {', '.join(twice) if twice else 'none'}")


if __name__ == "__main__":
    main()
