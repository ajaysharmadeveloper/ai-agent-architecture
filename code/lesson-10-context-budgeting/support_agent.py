"""Lesson 10, the budget in front of a real call. Needs ANTHROPIC_API_KEY. Run: python support_agent.py

build_step() fits every slot to its envelope with a rough estimate. Before sending, this counts the
real request with the token counting endpoint (free, an estimate from the API itself), refuses to
send if it is over budget, and reserves the reply's room with max_tokens.
"""
import anthropic

from context_budget import BUDGET, STEP_BUDGET, format_report
from demo_budget import run

MODEL = "claude-sonnet-5-5"
TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up one order by its id. Returns status, items and the latest tracking scan.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id, e.g. 58213"}},
            "required": ["order_id"],
        },
    }
]


def send(client, system, messages):
    counted = client.messages.count_tokens(model=MODEL, system=system, messages=messages, tools=TOOLS)
    limit = STEP_BUDGET - BUDGET["reply"]
    if counted.input_tokens > limit:
        raise RuntimeError(f"request is {counted.input_tokens} tokens, over the {limit} budget; tighten an envelope")
    print(f"counted by the API: {counted.input_tokens} input tokens (limit {limit})")
    return client.messages.create(model=MODEL, max_tokens=BUDGET["reply"], system=system, messages=messages, tools=TOOLS)


if __name__ == "__main__":
    _, report, system, messages = run()
    print(format_report(report))
    reply = send(anthropic.Anthropic(), system, messages)
    print(next((b.text for b in reply.content if b.type == "text"), reply.stop_reason))
