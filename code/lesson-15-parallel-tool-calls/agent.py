"""Lesson 15, the support agent loop. One change from lesson 14: a model turn can ask for several tool
calls at once, and the loop hands that whole list to an executor that decides how to run it.
"""
import json

from executor import timed
from shop import TODAY, World
from tools import TOOLS

SYSTEM = (
    "You are the support agent for an online shop. Today is " + TODAY + ". "
    "Use the tools to act. Do not promise something without doing it."
)
# Anthropic's docs suggest this line to keep dependent calls out of the same batch.
INDEPENDENT_ONLY = " Only batch tool calls that are independent of each other."

MAX_TURNS = 6


def handle(message: str, strategy, chat, prompt_line: bool = False) -> World:
    world = World()
    system = SYSTEM + (INDEPENDENT_ONLY if prompt_line else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
    for _ in range(MAX_TURNS):
        reply = chat(messages, TOOLS)
        messages.append(reply)
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in reply.get("tool_calls") or []]
        if not calls:
            break
        world.batches.append(len(calls))
        world.calls.append(" + ".join(f"{n}({json.dumps(a)})" for n, a in calls))
        results = timed(strategy, world, calls)
        for (name, _), result in zip(calls, results):  # every result goes back, in the order asked
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return world
