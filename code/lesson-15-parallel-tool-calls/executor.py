"""Lesson 15, three ways to run the tool calls a model asks for in one turn.

The model hands back a list of calls. The API does not run them, and neither does the model: your
loop does, and this file is the only place that decides how.

    sequential  one after another, in the order the model wrote them
    parallel    all at once
    safe        reads at the same time; anything that changes something alone, in the model's order
"""
import time
from concurrent.futures import ThreadPoolExecutor

from tools import READ_ONLY, run_tool


def _run_together(world, calls):
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(lambda c: run_tool(world, c[0], c[1]), calls))


def sequential(world, calls):
    return [run_tool(world, name, args) for name, args in calls]


def parallel(world, calls):
    return _run_together(world, calls)


def safe(world, calls):
    """Walk the list in order. Gather reads that sit next to each other and run them together; run each
    call that changes something on its own, after everything before it has finished."""
    results, reads = [], []

    def flush():
        if reads:
            results.extend(_run_together(world, reads))
            reads.clear()

    for name, args in calls:
        if name in READ_ONLY:
            reads.append((name, args))
        else:
            flush()
            results.append(run_tool(world, name, args))
    flush()
    return results


def timed(strategy, world, calls):
    start = time.perf_counter()
    results = strategy(world, calls)
    world.tool_seconds += time.perf_counter() - start
    return results


STRATEGIES = {"sequential": sequential, "parallel": parallel, "safe": safe}
