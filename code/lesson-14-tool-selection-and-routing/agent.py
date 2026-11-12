"""Lesson 14, the support agent: the lesson 13 loop, plus an optional routing step in front of it.

The system prompt is the same for every tool set and states the refund policy, so any difference in
what the agent does comes from which tools it was shown and how they were described.
"""
import json

from catalogue import GOALS, TEAMS, group_tools, run_tool
from shop import TODAY, World

SYSTEM = (
    "You are the support agent for an online shop. Today is " + TODAY + ". "
    "Refund policy: a refund is allowed only for a delivered order, within 30 days of delivery, and only "
    "for an order total up to Rs 5000. Anything else goes to a person on the support team. "
    "Use the tools to act. Do not promise something without doing it."
)

GROUPS = {
    "team": {name: does for name, does in TEAMS.items()},
    "goal": {name: does for name, (does, _) in GOALS.items()},
}
FALLBACK = {"team": "support", "goal": "talk_to_a_person"}  # an answer that is not a group goes to a person

MAX_TURNS = 6


def router_prompt(grouping: str) -> str:
    return ("You sort customer messages for an online shop's support desk. Pick the ONE group that should "
            "handle this message. Groups:\n"
            + "\n".join(f"- {name}: {does}" for name, does in GROUPS[grouping].items())
            + "\nReply with the group name only, exactly as written.")


def route(message: str, grouping: str, chat_text) -> tuple[str, int]:
    """Step one of routing: a plain model call, no tools, that names a group. Returns (group, prompt tokens)."""
    answer, tokens = chat_text([{"role": "system", "content": router_prompt(grouping)},
                                {"role": "user", "content": message}])
    word = answer.strip().strip(".`'\"").lower()
    return (word if word in GROUPS[grouping] else FALLBACK[grouping]), tokens


def handle(message: str, tools: list[dict] | str, chat, chat_text=None) -> World:
    """Run one customer message to the end. tools is a list of tool schemas, or "team" / "goal" to route
    first and show only the tools of the group the router picked."""
    world = World()
    if isinstance(tools, str):
        world.route, tokens = route(message, tools, chat_text)
        world.prompt_tokens.append(tokens)
        tools = group_tools(tools, world.route)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    for _ in range(MAX_TURNS):
        reply, tokens = chat(messages, tools)
        world.prompt_tokens.append(tokens)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            result = run_tool(world, name, args)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return world
