"""Lesson 13, the support agent: one tool-calling loop, used with either tool set.

The system prompt is the same for both sets, and it states the refund policy. So any difference in
what the agent does comes from the tool schemas alone.
"""
import json

from shop import TODAY, World

SYSTEM = (
    "You are the support agent for an online shop. Today is " + TODAY + ". "
    "Refund policy: a refund is allowed only for a delivered order, within 30 days of delivery, and only "
    "up to Rs 5000. Anything else goes to a person on the support team. "
    "Use the tools to act. Do not promise something without doing it."
)

MAX_TURNS = 6


def handle(message: str, tools: list[dict], run_tool, chat) -> World:
    """Run one customer message to the end. Returns the World: what was actually paid and escalated."""
    world = World()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    for _ in range(MAX_TURNS):
        reply = chat(messages, tools)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            world.calls.append(f"{name}({json.dumps(args)})")
            result = run_tool(world, name, args)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return world
