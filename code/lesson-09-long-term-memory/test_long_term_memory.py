"""Offline tests for long_term_memory.py. Run: python -m unittest test_long_term_memory.py"""
import tempfile
import unittest
from pathlib import Path

from demo_three_sessions import SESSION_ACTIONS
from long_term_memory import WRITE_PROMPT, apply_actions, build_system, load_memory, save_memory


class LongTermMemoryTest(unittest.TestCase):
    def test_unknown_user_starts_empty(self):
        self.assertEqual(load_memory("nobody", Path(tempfile.mkdtemp())), [])

    def test_save_then_load_round_trips(self):
        folder = Path(tempfile.mkdtemp())
        notes = [{"id": "n1", "text": "Contact by message only"}]
        save_memory("cust-1", notes, folder)
        self.assertEqual(load_memory("cust-1", folder), notes)

    def test_notes_go_into_the_system_prompt(self):
        system = build_system("You are a support agent.", [{"id": "n1", "text": "Contact by message only"}])
        self.assertIn("- Contact by message only", system)
        self.assertEqual(build_system("Plain.", []), "Plain.")

    def test_write_prompt_holds_the_four_questions(self):
        for question in ("future conversation", "stable", "look up", "say it"):
            self.assertIn(question, WRITE_PROMPT)

    def test_three_sessions_end_with_one_current_address(self):
        notes = []
        after = {}
        for session, actions in SESSION_ACTIONS.items():
            notes = apply_actions(notes, actions)
            after[session] = [n["text"] for n in notes]
        self.assertEqual(len(after["session 1"]), 3)
        self.assertEqual(after["session 2"], after["session 1"])
        addresses = [t for t in after["session 3"] if t.startswith("Address")]
        self.assertEqual(addresses, ["Address: 7 Hill Street, Nashik"])
        self.assertEqual(len(after["session 3"]), 3)

    def test_update_changes_a_note_in_place(self):
        notes = apply_actions([{"id": "n1", "text": "Prefers messages"}], [{"action": "update", "id": "n1", "text": "Prefers WhatsApp messages"}])
        self.assertEqual(notes, [{"id": "n1", "text": "Prefers WhatsApp messages"}])

    def test_unknown_action_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_actions([], [{"action": "append", "text": "x"}])


if __name__ == "__main__":
    unittest.main()
