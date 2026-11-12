"""Lesson 16, the support agent loop. One change from lesson 15: every tool call goes through an error
policy, and whatever the policy returns, a result or an error, goes back to the model as the tool result.
"""
import json

from shop import TODAY, World
from tools import TOOLS

SYSTEM = (
    "You are the support agent for an online shop. Today is " + TODAY + ". "
    "Use the tools to act. Do not promise something without doing it, and never make up an order's location."
)

MAX_TURNS = 6


def handle(message: str, policy, chat, faults: dict) -> World:
    world = World(faults=dict(faults))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    for _ in range(MAX_TURNS):
        reply = chat(messages, TOOLS)
        messages.append(reply)
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in reply.get("tool_calls") or []]
        if not calls:
            world.reply = reply.get("content", "").strip()
            break
        for name, args in calls:
            world.model_calls.append(f"{name}({args.get('order_id', '')})")
            result = policy(world, name, args)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return world
