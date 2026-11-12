"""Lesson 11, the three memory tools in front of a real model, running locally through Ollama.
No API key. Needs: ollama pull qwen3:8b && ollama pull embeddinggemma. Run: python support_agent.py

The profile is small and always needed, so it goes straight into the system prompt. The event log
and the meaning search are tools, so the model reads them only when the question needs them, and
each tool's description tells the model which kind of question it answers.
"""
import json

import local_llm
from demo_memory import fixed
from memory_stores import MEMORY_TOOLS, handle_tool

RULES = "You are a support agent for an online store. Use your memory tools; never guess a fact."


def ask(memory, question: str) -> str:
    system = f"{RULES}\n\nWhat is true about this customer right now:\n{memory.profile.as_text()}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for _ in range(5):  # a hard turn limit, from the ReAct lesson
        reply = local_llm.chat(messages, MEMORY_TOOLS)
        if not reply.get("tool_calls"):
            return reply["content"]
        messages.append(reply)
        for call in reply["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            print(f"  tool: {name}({args})")
            output = handle_tool(memory, name, args)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output)})
    raise RuntimeError("no answer after 5 turns")


if __name__ == "__main__":
    memory = fixed(local_llm.embed)
    for question in (
        "Where should the replacement for 58213 go?",
        "How many refunds has she asked for so far?",
        "Her box arrived crushed. Has anything like that happened before?",
    ):
        print(question)
        print(" ", ask(memory, question))
