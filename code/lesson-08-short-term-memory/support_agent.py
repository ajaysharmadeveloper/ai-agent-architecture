"""Lesson 8, a support agent with three layers of short-term memory.

    pinned facts  +  a summary of the old middle  +  the last few turns, word for word

Needs: pip install anthropic, and ANTHROPIC_API_KEY set.
"""
import anthropic

from memory import SAVE_FACT_TOOL, build_system, save_facts, split_into_turns, trim_by_turns

MODEL = "claude-sonnet-5"
INSTRUCTIONS = "You are a support agent for an online store. Never refund more than 5000 rupees without a manager."
SUMMARISE_AFTER_TURNS = 20
KEEP_RECENT_TURNS = 6

client = anthropic.Anthropic()
messages: list[dict] = []  # the short-term memory: the whole conversation, resent every turn
pinned: dict = {}  # filled by the model through save_fact, never trimmed
summary: str | None = None


def summarise(old: list[dict], previous: str | None) -> str:
    reply = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="Summarise this support conversation. Keep decisions and open questions.",
        messages=[{"role": "user", "content": f"Earlier summary: {previous}\n\nConversation: {old}"}],
    )
    return reply.content[0].text


def chat(user_text: str) -> str:
    global messages, summary
    messages.append({"role": "user", "content": user_text})

    turns = split_into_turns(messages)
    if len(turns) > SUMMARISE_AFTER_TURNS:
        old = [m for turn in turns[:-KEEP_RECENT_TURNS] for m in turn]
        summary = summarise(old, summary)
        messages = [m for turn in turns[-KEEP_RECENT_TURNS:] for m in turn]

    while True:
        reply = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=build_system(INSTRUCTIONS, pinned, summary),
            tools=[SAVE_FACT_TOOL],
            messages=trim_by_turns(messages, max_tokens=50_000),
        )
        content = [block.model_dump(exclude_none=True) for block in reply.content]
        messages.append({"role": "assistant", "content": content})
        if reply.stop_reason != "tool_use":
            return "".join(b["text"] for b in content if b["type"] == "text")
        messages.append({"role": "user", "content": save_facts(pinned, content)})


if __name__ == "__main__":
    print(chat("My order is 58213 and it's late. Please don't call me, I'm in meetings all day. Message only."))
