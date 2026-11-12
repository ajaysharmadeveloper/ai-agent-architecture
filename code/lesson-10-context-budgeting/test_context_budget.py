"""Offline tests for context_budget.py. Run: python -m unittest test_context_budget.py"""
import unittest

from context_budget import (
    BUDGET,
    STEP_BUDGET,
    build_step,
    check_budget,
    estimate_tokens,
    fit_docs,
    fit_history,
    fit_tool_results,
)
from demo_budget import run


class ContextBudgetTest(unittest.TestCase):
    def test_the_allowances_fit_the_step_budget(self):
        check_budget(BUDGET)
        with self.assertRaises(ValueError):
            check_budget({**BUDGET, "history": STEP_BUDGET})

    def test_a_fixed_slot_over_its_allowance_is_an_error_not_a_trim(self):
        with self.assertRaises(ValueError):
            build_step("rule " * 5000, "tools", "notes", [], [], [], "hi")

    def test_docs_best_first_and_a_weak_match_never_gets_in(self):
        docs = [{"score": 0.4, "text": "x" * 40}, {"score": 0.9, "text": "y" * 40}, {"score": 0.7, "text": "z" * 4000}]
        kept = fit_docs(docs, allowance=100)
        self.assertEqual([d["score"] for d in kept], [0.9])

    def test_old_tool_results_become_placeholders_and_the_newest_stay(self):
        results = [{"tool": "lookup_order", "text": f"result {i} " * 200} for i in range(6)]
        fitted = fit_tool_results(results, allowance=6000, keep=3)
        self.assertTrue(all(r["text"].startswith("[cleared") for r in fitted[:3]))
        self.assertIn("result 5", fitted[-1]["text"])

    def test_one_huge_result_is_cut_and_says_where_the_rest_is(self):
        fitted = fit_tool_results([{"tool": "lookup_order", "text": "z" * 100_000}], allowance=6000, keep=3)
        self.assertLessEqual(estimate_tokens(fitted[0]["text"]), 6000)
        self.assertIn("call lookup_order again", fitted[0]["text"])

    def test_history_keeps_whole_newest_turns(self):
        turns = [[{"role": "user", "content": f"q{i} " * 50}, {"role": "assistant", "content": f"a{i} " * 50}] for i in range(10)]
        kept = fit_history(turns, allowance=300)
        self.assertTrue(all(len(t) == 2 for t in kept))
        self.assertEqual(kept[-1][0]["content"][:2], "q9")

    def test_twenty_steps_stay_inside_the_budget_and_the_question_is_last(self):
        unbudgeted, report, _, messages = run()
        self.assertGreater(unbudgeted[-1], STEP_BUDGET)
        for name, used in report.items():
            self.assertLessEqual(used, BUDGET[name])
        self.assertLessEqual(sum(report.values()), STEP_BUDGET - BUDGET["reply"])
        self.assertTrue(messages[-1]["content"].endswith("When will my replacement ship?"))


if __name__ == "__main__":
    unittest.main()
