"""Lesson 8, short-term memory. The pure parts: no model calls, no SDK, so they run and test offline.

Messages use the Anthropic Messages API shape: {"role": "user" | "assistant", "content": ...}.
A tool result travels back as a USER message whose content holds "tool_result" blocks, and it must
directly follow the assistant message that asked for the tool.
"""
import json


def rough_tokens(message: dict) -> int:
    """A rough count, about four characters per token. Use the token counting API for exact numbers."""
    return len(json.dumps(message["content"])) // 4 + 1


def is_tool_result(message: dict) -> bool:
    content = message["content"]
    return (
        message["role"] == "user"
        and isinstance(content, list)
        and any(block.get("type") == "tool_result" for block in content)
    )


def split_into_turns(messages: list[dict]) -> list[list[dict]]:
    """A turn starts at a real user message. Tool calls and their results stay inside it."""
    turns: list[list[dict]] = []
    for message in messages:
        if message["role"] == "user" and not is_tool_result(message):
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim_by_turns(messages: list[dict], max_tokens: int, count=rough_tokens) -> list[dict]:
    """Drop the OLDEST whole turns until the list fits.

    Never cuts between a tool call and its result, and the list always starts with a user message.
    """
    turns = split_into_turns(messages)
    while len(turns) > 1 and sum(count(m) for turn in turns for m in turn) > max_tokens:
        turns.pop(0)
    return [m for turn in turns for m in turn]


def build_system(instructions: str, pinned: dict, summary: str | None) -> str:
    """Pinned facts go near the top of every call. Nothing trims them and nothing summarises them."""
    parts = [instructions, "Facts that must stay exact:"]
    parts += [f"- {key}: {value}" for key, value in pinned.items()]
    if summary:
        parts += ["Summary of the earlier conversation:", summary]
    return "\n".join(parts)


SAVE_FACT_TOOL = {
    "name": "save_fact",
    "description": "Save a fact the user told you that must stay exact for the rest of the chat, "
    "like an order ID or how they want to be contacted.",
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"],
    },
}


def save_facts(pinned: dict, content: list[dict]) -> list[dict]:
    """Apply every save_fact call in an assistant reply to the pinned block.

    Returns the tool_result blocks to send back, one per call, with the same ID,
    so every tool call keeps its answer.
    """
    results = []
    for block in content:
        if block.get("type") == "tool_use" and block.get("name") == "save_fact":
            pinned[block["input"]["key"]] = block["input"]["value"]
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": "saved"})
    return results
