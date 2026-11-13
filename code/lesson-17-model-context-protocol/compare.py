"""Lesson 17, the same agent two ways: tools wired in by hand (lessons 13 to 16), and tools from the MCP
server. Same model, same messages. Prints what the model was given and what it did, both ways.

    python compare.py --runs 3
"""
import argparse
import asyncio
import json
import sys

import shop
from agent import SERVERS, Host, handle, system_prompt
from local_llm import chat_with_tools

MESSAGES = [
    "Where is my order 64410?",
    "My headphones, order 58213, stopped working. I want my money back, please refund it.",
    "Can I get a refund on order 64412?",
]


class HandWired:
    """Lessons 13 to 16: the tool list and the functions live inside this one agent."""

    ID = {"type": "string", "pattern": "^[0-9]{5}$", "description": "The 5-digit order number, like 58213."}
    FNS = {"orders_get_order": shop.get_order, "shipping_track_package": shop.track_package,
           "payments_refund": shop.refund}

    def __init__(self, mcp_tools: list[dict]):
        # Same names and descriptions as the server, so the only difference is where they come from.
        self.tools = [{"type": "function", "function": {
            "name": t["function"]["name"], "description": t["function"]["description"],
            "parameters": {"type": "object", "properties": {"order_id": self.ID}, "required": ["order_id"]}}}
            for t in mcp_tools]
        self.context = "\n\n" + shop.REFUND_POLICY

    async def call(self, name: str, args: dict) -> str:
        try:
            return json.dumps(self.FNS[name](args["order_id"]))
        except shop.ShopError as e:
            return json.dumps({"error": str(e)})


def canonical(tools) -> str:
    """The tool list without the `title` labels the MCP SDK adds to every schema."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "title"}
        return [strip(v) for v in x] if isinstance(x, list) else x
    return json.dumps(strip(tools), sort_keys=True)


def tokens_without_tools(host) -> int:
    messages = [{"role": "system", "content": system_prompt(host)}, {"role": "user", "content": MESSAGES[0]}]
    return chat_with_tools(messages, [])["prompt_eval_count"]


async def run(runs: int):
    async with Host(SERVERS) as mcp_host:
        hand = HandWired(mcp_host.tools)
        same = canonical(hand.tools) == canonical(mcp_host.tools)
        print(f"tool list, hand-wired vs MCP: {'the same, apart from the title labels the SDK adds' if same else 'DIFFERENT'}")
        print(f"MCP tool list as the model receives it:\n{json.dumps(mcp_host.tools, indent=2)}\n")
        print(f"prompt tokens, no tools at all: {tokens_without_tools(mcp_host)}")

    def report(label, i, r, out):
        print(f"[{label}] case {i} run {r} | prompt tokens {out['prompt_tokens']} | "
              f"calls {out['calls']} | reply: {out['reply']!r}")
        sys.stdout.flush()

    for i, msg in enumerate(MESSAGES, 1):
        for r in range(1, runs + 1):
            shop.REFUNDED.clear()                     # hand-wired: the refunds live in this process
            report("hand-wired", i, r, await handle(hand, msg))
            async with Host(SERVERS) as fresh:        # mcp: a fresh server process, so no refunds yet
                report("mcp", i, r, await handle(fresh, msg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    asyncio.run(run(p.parse_args().runs))
