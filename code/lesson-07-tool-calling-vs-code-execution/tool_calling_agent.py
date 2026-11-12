"""Lesson 8, the TOOL CALLING version: one trip to the model per action.

The model never runs anything. It asks for a tool by name with JSON arguments, this loop runs
the function, appends the result, and calls the model again until it answers instead of asking.
"""
import json

from openai import OpenAI

from invoice_tools import list_invoices, send_email

MODEL = "gpt-4.1-mini"  # the working tier: pick the model the step needs, not the biggest one
TASK = (
    "Find every overdue invoice this month (2026-09) that is over Rs 50,000 "
    "and email each of those customers a reminder."
)

# 1. Describe each tool: its name, what it does, and what inputs it takes.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_invoices",
            "description": "List every invoice due in a month, paid or not.",
            "parameters": {
                "type": "object",
                "properties": {"month": {"type": "string", "description": "YYYY-MM"}},
                "required": ["month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send one email to one customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]
FUNCTIONS = {"list_invoices": list_invoices, "send_email": send_email}


def run() -> None:
    client = OpenAI()
    messages = [{"role": "user", "content": TASK}]
    trips = 0
    # 2. The loop: the model answers or asks for a tool. Run it, append the result, ask again.
    while True:
        trips += 1
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = reply.choices[0].message
        if not msg.tool_calls:
            print(msg.content)
            break
        messages.append(msg)
        for call in msg.tool_calls:
            result = FUNCTIONS[call.function.name](**json.loads(call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    print(f"round trips to the model: {trips}")


if __name__ == "__main__":
    run()
