"""Lesson 9, the three sessions from the video, run offline.

The actions below are the kind of reply the write step returns for each session, written out by hand
so this runs with no API key. Run: python demo_three_sessions.py
"""
import tempfile
from pathlib import Path

from long_term_memory import apply_actions, build_system, load_memory, save_memory

SESSION_ACTIONS = {
    "session 1": [
        {"action": "add", "text": "Contact by message only, never call"},
        {"action": "add", "text": "Address: 14 Lake Road, Pune"},
        {"action": "add", "text": "Promised a replacement for order 58213 if not delivered by Friday"},
    ],
    "session 2": [
        {"action": "none", "id": "n1", "text": "Contact by message only, never call"},
    ],
    "session 3": [
        {"action": "delete", "id": "n2", "text": "Address: 14 Lake Road, Pune"},
        {"action": "add", "text": "Address: 7 Hill Street, Nashik"},
    ],
}

if __name__ == "__main__":
    folder = Path(tempfile.mkdtemp())
    for session, actions in SESSION_ACTIONS.items():
        notes = load_memory("cust-1042", folder)
        print(f"--- {session}, system prompt the agent starts with ---")
        print(build_system("You are a support agent for an online store.", notes))
        save_memory("cust-1042", apply_actions(notes, actions), folder)
    print("--- the file after session 3 ---")
    print((folder / "cust-1042.json").read_text())
