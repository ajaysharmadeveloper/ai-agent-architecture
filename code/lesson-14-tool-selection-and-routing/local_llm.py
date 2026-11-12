"""Lesson 14, a local model through Ollama: no API key, nothing leaves your machine.

Needs Ollama running (https://ollama.com) and one model:
    ollama pull qwen3:8b

Standard library only, so there is nothing else to install.
"""
import json
import urllib.request

OLLAMA = "http://localhost:11434"
CHAT_MODEL = "qwen3:8b"


def _post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def chat_with_tools(messages: list[dict], tools: list[dict]) -> tuple[dict, int]:
    """One model turn. Returns the assistant message (which may carry tool_calls) and the prompt size
    in tokens, as Ollama counted it. The tool definitions are part of that prompt on every turn."""
    body = {"model": CHAT_MODEL, "messages": messages, "tools": tools, "stream": False, "think": False}
    reply = _post("/api/chat", body)
    return reply["message"], reply["prompt_eval_count"]


def chat_text(messages: list[dict]) -> tuple[str, int]:
    """One plain model turn with no tools. Used by the router."""
    body = {"model": CHAT_MODEL, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0}}
    reply = _post("/api/chat", body)
    return reply["message"]["content"], reply["prompt_eval_count"]
