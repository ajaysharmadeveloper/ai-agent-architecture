"""Offline tests for memory.py. Run: python -m unittest test_memory.py"""
import unittest

from memory import build_system, save_facts, split_into_turns, trim_by_turns


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def tool_call(name):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": name, "input": {}}]}


def tool_result(text):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}]}


CONVERSATION = [
    user("order 58213, message only"),
    tool_call("lookup_order"),
    tool_result("x" * 400),
    assistant("It ships tomorrow."),
    user("can I change the address"),
    assistant("Yes, send it."),
    user("12 MG Road"),
    assistant("Done."),
]


class TurnsTest(unittest.TestCase):
    def test_tool_call_and_result_stay_in_one_turn(self):
        turns = split_into_turns(CONVERSATION)
        self.assertEqual(len(turns), 3)
        self.assertEqual(len(turns[0]), 4)

    def test_trim_drops_the_oldest_whole_turn(self):
        trimmed = trim_by_turns(CONVERSATION, max_tokens=60)
        self.assertEqual(trimmed[0], user("can I change the address"))
        self.assertNotIn(tool_result("x" * 400), trimmed)

    def test_trim_never_starts_on_a_tool_result_or_orphans_a_call(self):
        for limit in range(1, 400):
            trimmed = trim_by_turns(CONVERSATION, max_tokens=limit)
            self.assertEqual(trimmed[0]["role"], "user")
            self.assertIsInstance(trimmed[0]["content"], str)
            for i, message in enumerate(trimmed):
                if message["role"] == "assistant" and isinstance(message["content"], list):
                    self.assertEqual(trimmed[i + 1]["content"][0]["type"], "tool_result")

    def test_trimming_loses_the_first_message(self):
        # The point of the lesson: the order number was in turn one.
        trimmed = trim_by_turns(CONVERSATION, max_tokens=60)
        self.assertFalse(any("58213" in str(m["content"]) for m in trimmed))

    def test_pinned_facts_survive_in_the_system_prompt(self):
        system = build_system("Be helpful.", {"order_id": "58213"}, summary="Late order, address changed.")
        self.assertIn("order_id: 58213", system)
        self.assertLess(system.index("58213"), system.index("Late order"))

    def test_save_fact_fills_the_pin_and_answers_every_call(self):
        pinned = {}
        reply = [
            {"type": "text", "text": "Noted."},
            {"type": "tool_use", "id": "a1", "name": "save_fact", "input": {"key": "order_id", "value": "58213"}},
            {"type": "tool_use", "id": "a2", "name": "save_fact", "input": {"key": "contact", "value": "message only"}},
        ]
        results = save_facts(pinned, reply)
        self.assertEqual(pinned, {"order_id": "58213", "contact": "message only"})
        self.assertEqual([r["tool_use_id"] for r in results], ["a1", "a2"])


if __name__ == "__main__":
    unittest.main()
