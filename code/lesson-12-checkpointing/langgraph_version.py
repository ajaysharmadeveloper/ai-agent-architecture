"""Lesson 12, the same refund batch with LangGraph's own checkpointer instead of ours.

Needs: pip install langgraph langgraph-checkpoint-sqlite   (and Ollama with qwen3:8b, or --offline)
Run:   python langgraph_version.py --fresh --crash-at 9
       python langgraph_version.py            # same thread_id: picks up at request 9
       python langgraph_version.py --history  # get_state_history, newest first

One node handles one request and loops back to itself, so every request is one step, and LangGraph
saves a checkpoint after each one. durability="sync" writes it before the next step starts.
"""
import argparse
import operator
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from payments import Ledger
from refund_batch import POLICY, REQUESTS, crash, llm_decide, rule_decide

HERE = Path(__file__).parent


class State(TypedDict):
    policy: dict
    next: int
    done: Annotated[list, operator.add]


def build(ledger: Ledger, decide, crash_at: int | None):
    def process(state: State) -> dict:
        request = REQUESTS[state["next"]]
        result = decide(request, state["policy"])
        if result["decision"] == "APPROVE":
            ledger.issue_refund(request["id"], request["amount"], f"refund-{request['id']}")
        if crash_at == state["next"] + 1:
            print(f"  step {state['next'] + 1}: {request['id']} paid, then the process died before saving")
            crash()
        print(f"  step {state['next'] + 1}: {request['id']} {result['decision']}")
        return {"next": state["next"] + 1, "done": [{"id": request["id"], **result}]}

    graph = StateGraph(State)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_conditional_edges("process", lambda s: "process" if s["next"] < len(REQUESTS) else END)
    checkpointer = SqliteSaver(sqlite3.connect(str(HERE / "langgraph.db"), check_same_thread=False))
    return graph.compile(checkpointer=checkpointer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="friday-refunds")
    parser.add_argument("--crash-at", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.fresh:
        for name in ("langgraph.db", "payments.db"):
            (HERE / name).unlink(missing_ok=True)

    ledger = Ledger(str(HERE / "payments.db"))
    graph = build(ledger, rule_decide if args.offline else llm_decide, args.crash_at)
    config = {"configurable": {"thread_id": args.thread}}

    if args.history:
        for snapshot in graph.get_state_history(config):
            print(snapshot.metadata["step"], snapshot.values.get("next"), snapshot.next)
        return
    saved = graph.get_state(config)
    if saved.next:
        print(f"resuming '{args.thread}': {saved.values['next']} of {len(REQUESTS)} done")
        graph.invoke(None, config, recursion_limit=100, durability="sync")
    else:
        print(f"starting '{args.thread}' from request 1")
        graph.invoke({"policy": POLICY, "next": 0, "done": []}, config, recursion_limit=100, durability="sync")
    paid = ledger.paid()
    print(f"refunds paid: {sum(paid.values())} for {len(paid)} requests. "
          f"paid twice: {', '.join(r for r, n in paid.items() if n > 1) or 'none'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
