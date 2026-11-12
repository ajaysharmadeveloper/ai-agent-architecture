"""Offline tests for memory_stores.py. Run: python -m unittest test_memory_stores.py"""
import unittest

from demo_memory import run
from memory_stores import MEMORY_TOOLS, EventLog, Memory, Profile, handle_tool


class MemoryStoresTest(unittest.TestCase):
    def test_a_profile_field_has_one_current_value(self):
        profile = Profile()
        profile.set("address", "14 Lake Road, Pune", at="2026-07-02")
        profile.set("address", "7 Hill View, Bengaluru", at="2026-09-30")
        self.assertEqual(profile.get("address"), "7 Hill View, Bengaluru")
        self.assertIsNone(profile.get("birthday"))

    def test_the_event_log_only_appends_in_time_order(self):
        log = EventLog()
        log.append("2026-08-19", "refund", "a")
        with self.assertRaises(ValueError):
            log.append("2026-07-02", "refund", "older than the last event")

    def test_the_shortcut_puts_the_old_address_on_top(self):
        a, _, _ = run()
        self.assertIn("Pune", a["address"][0]["text"])
        self.assertIn("Bengaluru", a["address"][1]["text"])

    def test_counting_search_hits_undercounts(self):
        a, b, _ = run()
        self.assertEqual(len(a["refunds"]), 3)
        self.assertEqual(b["refunds"], 4)

    def test_meaning_search_finds_what_an_exact_word_search_cannot(self):
        _, b, memory = run()
        self.assertIn("cracked", b["damage"][0]["text"])
        self.assertEqual(memory.log.find_word("crushed"), [])

    def test_the_fix_answers_each_question_from_its_own_store(self):
        _, b, memory = run()
        self.assertEqual(b["address"], "7 Hill View, Bengaluru")
        self.assertEqual(memory.log.latest("promise").text[:30], "Promised a replacement for ord")

    def test_every_memory_tool_has_a_handler(self):
        memory = Memory()
        memory.remember_fact("address", "x", at="2026-01-01")
        memory.remember_event("2026-01-01", "refund", "a refund")
        args = {"get_fact": {"field": "address"}, "recent_events": {}, "search_past": {"query": "refund"}}
        for tool in MEMORY_TOOLS:
            handle_tool(memory, tool["name"], args[tool["name"]])
        with self.assertRaises(ValueError):
            handle_tool(memory, "forget_everything", {})


if __name__ == "__main__":
    unittest.main()
