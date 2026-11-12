"""Lesson 16, three ways to handle a tool call that fails.

    raw        run it once; on any failure send the model the bare status, like "HTTP 503"
    retry_all  on any failure try the same call again, up to 3 tries, then send the bare status
    smart      the lesson's handler:
                 1. a wrong decision (the call can never work as asked) is not retried; the model gets
                    an error that says what is true and what to do instead
                 2. a transient failure (busy, timeout) is retried in code, with growing waits plus a
                    little randomness, up to 3 tries; the model never sees it if a retry works
                 3. a call that changes something is only retried with an idempotency key, so the
                    service can recognise the repeat and not do it twice
                 4. if the retries run out, use a fallback if the tool has one, and say it is a fallback
"""
import random
import time

from shop import ServiceError, World
from tools import READ_ONLY, ToolError, last_known_scan, run_tool

MAX_TRIES = 3
BASE_WAIT = 0.25   # seconds before the 2nd try; doubles each time, plus up to the same again at random


def backoff(attempt: int) -> float:
    """Exponential backoff with jitter: 0.25 s, 0.5 s, 1 s ... each plus a random share of itself."""
    wait = BASE_WAIT * 2 ** (attempt - 1)
    return wait + random.uniform(0, wait)


def _log(world: World, name: str, args: dict, outcome: str) -> None:
    world.attempts.append(f"{name}({args.get('order_id', '')}) -> {outcome}")


def raw(world: World, name: str, args: dict) -> dict:
    try:
        result = run_tool(world, name, args)
        _log(world, name, args, "ok")
        return result
    except (ToolError, ServiceError) as e:
        _log(world, name, args, f"HTTP {e.status}")
        return {"error": f"HTTP {e.status}"}


def retry_all(world: World, name: str, args: dict) -> dict:
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = run_tool(world, name, args)
            _log(world, name, args, "ok")
            return result
        except (ToolError, ServiceError) as e:
            _log(world, name, args, f"HTTP {e.status}")
            if attempt == MAX_TRIES:
                return {"error": f"HTTP {e.status}"}
            time.sleep(backoff(attempt))


def smart(world: World, name: str, args: dict) -> dict:
    writes = name not in READ_ONLY
    # the key names the ACTION (refund this order), not the attempt, so every repeat carries the same key
    key = f"{name}:{args.get('order_id', '')}" if writes else None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = run_tool(world, name, args, idempotency_key=key)
            _log(world, name, args, "ok")
            return result
        except ToolError as e:                      # a wrong decision: retrying can't fix it
            _log(world, name, args, f"HTTP {e.status}, sent back to the model")
            return {"error": e.hint}
        except ServiceError as e:                   # a bad moment: retrying might
            _log(world, name, args, f"HTTP {e.status}")
            if attempt < MAX_TRIES:
                time.sleep(backoff(attempt))
                continue
            if name == "shipping_track_package" and (fallback := last_known_scan(args.get("order_id", ""))):
                _log(world, name, args, "fallback: last scan from our database")
                return fallback
            return {"error": f"{e.message} Tried {MAX_TRIES} times. Tell the customer it is temporarily "
                             f"unavailable and that nothing was done twice. Do not call it again now."}


POLICIES = {"raw": raw, "retry_all": retry_all, "smart": smart}
