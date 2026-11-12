"""Lesson 11, a local model through Ollama: no API key, nothing leaves your machine.

Needs Ollama running (https://ollama.com) and two models:
    ollama pull embeddinggemma   # turns text into a vector
    ollama pull qwen3:8b         # the agent's model, it can call tools

Standard library only, so there is nothing else to install.
"""
import json
import urllib.request

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen3:8b"


def _post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def embed(text: str) -> dict:
    """A real embedding: a list of numbers where texts that mean similar things land close together.
    Returned as {position: value} so the same cosine() works for it and for the offline stand-in."""
    vector = _post("/api/embed", {"model": EMBED_MODEL, "input": text})["embeddings"][0]
    return dict(enumerate(vector))


def as_ollama_tools(tools: list[dict]) -> list[dict]:
    """Our tool definitions, in the shape Ollama's /api/chat expects."""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ]


def chat(messages: list[dict], tools: list[dict]) -> dict:
    """One step of the agent: returns the model's message, which may ask for tool calls."""
    body = {"model": CHAT_MODEL, "messages": messages, "tools": as_ollama_tools(tools), "stream": False, "think": False}
    return _post("/api/chat", body)["message"]
