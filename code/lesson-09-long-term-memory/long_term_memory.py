"""Lesson 9, long-term memory. The pure parts: no model calls, no SDK, so they run and test offline.

One JSON file per user holds that user's notes. Each note is {"id": "n1", "text": "..."}.
Read at the start of a session, written on purpose at the end.
"""
import json
from pathlib import Path

MEMORY_DIR = Path("memories")

WRITE_PROMPT = """You maintain a support agent's notes about one customer.
You get the existing notes and the conversation that just ended.

Only keep a fact if ALL four answers are yes:
1. Will it matter in a future conversation?
2. Is it stable, true for months and not just today?
3. Is it something we cannot look up somewhere better, like the orders database?
4. Did the customer say it, rather than us guessing it?

For every fact that passes, pick one action:
- add: it is new
- update: it adds detail to an existing note (give that note's id)
- delete: it contradicts an existing note (give the old note's id), then add the new fact
- none: it is already known

Reply with JSON only: a list of {"action": ..., "id": ..., "text": ...}."""


def load_memory(user_id: str, folder: Path = MEMORY_DIR) -> list[dict]:
    path = folder / f"{user_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_memory(user_id: str, notes: list[dict], folder: Path = MEMORY_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{user_id}.json").write_text(json.dumps(notes, indent=2))


def build_system(instructions: str, notes: list[dict]) -> str:
    """The read step: the user's notes go into the system prompt before they type a word."""
    if not notes:
        return instructions
    lines = [instructions, "What you already know about this customer:"]
    lines += [f"- {note['text']}" for note in notes]
    return "\n".join(lines)


def apply_actions(notes: list[dict], actions: list[dict]) -> list[dict]:
    """The update step. Never appends blindly: every candidate fact is an add, update, delete or none."""
    notes = [dict(note) for note in notes]
    next_id = max((int(note["id"][1:]) for note in notes), default=0) + 1
    for action in actions:
        kind = action["action"]
        if kind == "add":
            notes.append({"id": f"n{next_id}", "text": action["text"]})
            next_id += 1
        elif kind == "update":
            note = next(n for n in notes if n["id"] == action["id"])
            note["text"] = action["text"]
        elif kind == "delete":
            notes = [n for n in notes if n["id"] != action["id"]]
        elif kind != "none":
            raise ValueError(f"unknown action: {kind}")
    return notes
