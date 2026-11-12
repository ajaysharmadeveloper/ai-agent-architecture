"""Lesson 12, time travel: look back through a finished run's checkpoints, and fork from one of them.

Run after refund_batch.py has finished a run:
    python time_travel.py                       # every checkpoint of the run, step by step
    python time_travel.py --show 3              # the full state at step 3: what was decided and why
    python time_travel.py --fork 2 --rule "If delivered_days_ago is more than 30, the decision is HUMAN."
        # copy step 2 into a new thread with one more rule, and run the rest as a DRY RUN

A fork re-runs every step after the checkpoint, model calls included, so the answers can come out
different. That's the point when debugging, and the danger with side effects: this fork never pays.
"""
import argparse
import json
from pathlib import Path

from checkpoints import SqliteCheckpointer
from refund_batch import llm_decide, rule_decide, run

HERE = Path(__file__).parent


class DryRunLedger:
    """Pays nothing. A fork is for looking, not for moving money."""

    def issue_refund(self, request_id, amount, idempotency_key):
        print(f"    (dry run: would refund {request_id}, Rs {amount})")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="friday-refunds")
    parser.add_argument("--show", type=int)
    parser.add_argument("--fork", type=int)
    parser.add_argument("--rule", default="")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    checkpointer = SqliteCheckpointer(str(HERE / "checkpoints.db"))

    if args.show is not None:
        print(json.dumps(checkpointer.at(args.thread, args.show).state, indent=2))
    elif args.fork is not None:
        old = checkpointer.at(args.thread, args.fork)
        new_thread = f"{args.thread}-fork-{args.fork}"
        checkpointer.fork(args.thread, args.fork, new_thread, {"policy": {**old.state["policy"], "extra_rule": args.rule}})
        run(new_thread, checkpointer, DryRunLedger(), rule_decide if args.offline else llm_decide)
        print(f"original '{args.thread}' still has {checkpointer.latest(args.thread).step} steps, unchanged")
    else:
        for checkpoint in reversed(checkpointer.history(args.thread)):
            last = checkpoint.state["done"][-1]
            print(f"step {checkpoint.step:2}: {last['id']:4} {last['decision']:8} {last['reason'][:60]}")


if __name__ == "__main__":
    main()
