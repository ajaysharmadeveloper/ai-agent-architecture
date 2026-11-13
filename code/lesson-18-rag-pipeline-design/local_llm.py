"""Lesson 18, a local model through Ollama: no API key, nothing leaves your machine.

Needs Ollama running (https://ollama.com) and two models:
    ollama pull embeddinggemma   # turns text into a vector
    ollama pull qwen3:8b         # writes the answer, and scores chunks in the rerank step

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
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def embed(texts: list[str]) -> list[list[float]]:
    """One vector per text. Texts that mean similar things land close together.
    /api/embed returns unit-length vectors, so a dot product is the cosine similarity."""
    return _post("/api/embed", {"model": EMBED_MODEL, "input": texts})["embeddings"]


def chat(system: str, user: str) -> dict:
    """One model turn. Returns {"text": the reply, "prompt_tokens": how many tokens the model had to read}."""
    body = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 7, "num_ctx": 16384},
    }
    response = _post("/api/chat", body)
    return {"text": response["message"]["content"].strip(), "prompt_tokens": response["prompt_eval_count"]}
