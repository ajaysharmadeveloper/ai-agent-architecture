"""Lesson 9, the support agent from lesson 8, now with long-term memory across sessions.

    load the notes  ->  the same agent loop as before  ->  the write step  ->  save the notes

Needs: pip install anthropic, and ANTHROPIC_API_KEY set.
"""
import json

import anthropic

from long_term_memory import WRITE_PROMPT, apply_actions, build_system, load_memory, save_memory

MODEL = "claude-sonnet-5-5"
WRITE_MODEL = "claude-haiku-4-5-20251001"  # the write step is a small sorting job
INSTRUCTIONS = "You are a support agent for an online store."

client = anthropic.Anthropic()


def run_session(user_id: str, user_turns: list[str]) -> list[dict]:
    notes = load_memory(user_id)  # read step, before the customer types anything
    messages: list[dict] = []
    for text in user_turns:
        messages.append({"role": "user", "content": text})
        reply = client.messages.create(
            model=MODEL, max_tokens=800, system=build_system(INSTRUCTIONS, notes), messages=messages
        )
        messages.append({"role": "assistant", "content": reply.content[0].text})
        print("agent:", reply.content[0].text)

    # write step, once the chat has ended
    result = client.messages.create(
        model=WRITE_MODEL,
        max_tokens=800,
        system=WRITE_PROMPT,
        messages=[{"role": "user", "content": f"Existing notes: {json.dumps(notes)}\n\nConversation: {json.dumps(messages)}"}],
    )
    notes = apply_actions(notes, json.loads(result.content[0].text))
    save_memory(user_id, notes)
    return notes


if __name__ == "__main__":
    run_session("cust-1042", ["Order 58213 is late. I've just moved to 14 Lake Road, Pune. Message only please, don't call."])
    run_session("cust-1042", ["Hi, any news on my delivery?"])
    run_session("cust-1042", ["I've moved again, I'm at 7 Hill Street, Nashik now."])
