"""Lesson 17, our own agent as an MCP host.

The model never sees MCP. It sees the same tool list it saw in lessons 7 and 13. What changed is where
that list comes from: the host asks each server for its tools (tools/list), hands them to the model, and
sends each call the model makes to the server that owns that tool (tools/call).

    python agent.py "Where is my order 64410?"
"""
import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters

from local_llm import chat_with_tools
from shop import TODAY

# One entry per MCP server this host connects to. Add a server = add a line. No tool code here.
SERVERS = {
    "support": StdioServerParameters(command=sys.executable, args=["support_server.py"]),
}

MAX_TURNS = 6


def as_model_tool(tool) -> dict:
    """An MCP tool definition, turned into the tool format the model's API expects."""
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}


def result_text(result) -> str:
    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
    return json.dumps({"error": text}) if result.is_error else text


class Host:
    """Holds one MCP client per server, and a map from tool name to the client that owns it."""

    def __init__(self, servers: dict):
        self.servers, self.owner, self.tools, self.context = servers, {}, [], ""

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        for name, params in self.servers.items():
            client = await self._stack.enter_async_context(Client(params))   # one client per server
            for tool in (await client.list_tools()).tools:                    # tools/list
                self.owner[tool.name] = client
                self.tools.append(as_model_tool(tool))
            # Resources are the APPLICATION's to use: this host reads every one and puts it in the prompt.
            for res in (await client.list_resources()).resources:            # resources/list
                body = (await client.read_resource(res.uri)).contents[0].text  # resources/read
                self.context += f"\n\n{body}"
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def call(self, name: str, args: dict) -> str:
        if name not in self.owner:
            return json.dumps({"error": f"There is no tool called {name}."})
        return result_text(await self.owner[name].call_tool(name, args))   # tools/call


def system_prompt(host: Host) -> str:
    return ("You are the support agent for an online shop. Today is " + TODAY + ". "
            "Use the tools to act. Do not promise something without doing it." + host.context)


async def handle(host: Host, message: str) -> dict:
    messages = [{"role": "system", "content": system_prompt(host)}, {"role": "user", "content": message}]
    calls, first_prompt_tokens = [], None
    for _ in range(MAX_TURNS):
        response = chat_with_tools(messages, host.tools)
        if first_prompt_tokens is None:
            first_prompt_tokens = response["prompt_eval_count"]   # what the model read before its first step
        reply = response["message"]
        messages.append(reply)
        if not reply.get("tool_calls"):
            return {"calls": calls, "reply": reply.get("content", "").strip(), "prompt_tokens": first_prompt_tokens}
        for c in reply["tool_calls"]:
            name, args = c["function"]["name"], c["function"]["arguments"]
            calls.append(f"{name}({args.get('order_id', '')})")
            messages.append({"role": "tool", "tool_name": name, "content": await host.call(name, args)})
    return {"calls": calls, "reply": "(ran out of turns)", "prompt_tokens": first_prompt_tokens}


async def main(message: str):
    async with Host(SERVERS) as host:
        print(f"tools from MCP: {[t['function']['name'] for t in host.tools]}")
        out = await handle(host, message)
        print(f"calls: {out['calls']}\nreply: {out['reply']}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Where is my order 64410?"))
