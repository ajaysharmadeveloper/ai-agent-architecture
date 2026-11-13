# Lesson 17: MCP, the support tools as one server

The support agent from lessons 13 to 16, with its tools moved behind an MCP server. You write the server
once. Our own agent uses it, and Claude Code uses the same file, with no change to the server.

| File | What it is |
|---|---|
| `shop.py` | The shop: four orders, the refund policy. Plain Python, no MCP. |
| `support_server.py` | The MCP server: 3 tools, 1 resource (the refund policy), 1 prompt. |
| `agent.py` | Our agent as an MCP host. One client per server, `tools/list` in, `tools/call` out. |
| `local_llm.py` | The model, `qwen3:8b` through Ollama. Runs on your machine, no API key. |
| `compare.py` | The same agent with tools wired in by hand, then from MCP. Same model, same messages. |
| `test_server.py` | Checks for the server. No model needed. |

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen3:8b

python agent.py "Where is my order 64410?"
python compare.py --runs 3
python -m pytest test_server.py -q
```

## Plug the same server into Claude Code

```bash
claude mcp add support -- python /full/path/to/support_server.py
```

Then ask Claude Code "Where is my order 64410?".

⚠ This is SDK v2. Older tutorials write `from mcp.server.fastmcp import FastMCP`; on v2 that import fails.
It is `from mcp.server.mcpserver import MCPServer` now.
