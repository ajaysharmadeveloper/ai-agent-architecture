"""Lesson 12, a local model through Ollama: no API key, nothing leaves your machine.

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


def chat_json(messages: list[dict], schema: dict) -> dict:
    """One model call whose reply must match a JSON schema (Ollama's structured outputs)."""
    body = {"model": CHAT_MODEL, "messages": messages, "format": schema, "stream": False, "think": False}
    return json.loads(_post("/api/chat", body)["message"]["content"])
