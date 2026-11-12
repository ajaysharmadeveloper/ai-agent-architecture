"""Offline tests for the checkpointing lesson. Run: python -m unittest test_checkpoints.py"""
import unittest

from checkpoints import MemoryCheckpointer, SqliteCheckpointer
from payments import Ledger
from refund_batch import REQUESTS, rule_decide, run


class Crashed(Exception):
    pass


def crash():
    raise Crashed  # the test stand-in for the process dying


def quiet_run(*args, **kwargs):
    return run(*args, decide=rule_decide, crash=crash, **kwargs)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.checkpoints = SqliteCheckpointer(":memory:")
        self.ledger = Ledger(":memory:")

    def test_a_resumed_run_starts_after_the_last_saved_step(self):
        with self.assertRaises(Crashed):
            quiet_run("t", self.checkpoints, self.ledger, crash_at=9)
        self.assertEqual(self.checkpoints.latest("t").step, 8)
        state = quiet_run("t", self.checkpoints, self.ledger)
        self.assertEqual(state["next"], len(REQUESTS))
        self.assertEqual([d["id"] for d in state["done"]], [r["id"] for r in REQUESTS])

    def test_the_step_in_flight_runs_again_and_the_key_stops_a_second_payment(self):
        with self.assertRaises(Crashed):
            quiet_run("t", self.checkpoints, self.ledger, crash_at=9)
        quiet_run("t", self.checkpoints, self.ledger)
        self.assertTrue(all(n == 1 for n in self.ledger.paid().values()))

    def test_without_the_key_the_step_in_flight_pays_twice(self):
        with self.assertRaises(Crashed):
            quiet_run("t", self.checkpoints, self.ledger, crash_at=9, idempotent=False)
        quiet_run("t", self.checkpoints, self.ledger, idempotent=False)
        self.assertEqual(self.ledger.paid()["R9"], 2)

    def test_without_checkpoints_a_restart_pays_everything_again(self):
        with self.assertRaises(Crashed):
            quiet_run("t", None, self.ledger, crash_at=9, idempotent=False)
        quiet_run("t", None, self.ledger, idempotent=False)
        twice = [rid for rid, n in self.ledger.paid().items() if n > 1]
        self.assertGreater(len(twice), 1)

    def test_a_memory_checkpointer_is_gone_in_a_new_process(self):
        with self.assertRaises(Crashed):
            quiet_run("t", MemoryCheckpointer(), self.ledger, crash_at=9)
        self.assertIsNone(MemoryCheckpointer().latest("t"))

    def test_a_fork_leaves_the_original_run_untouched(self):
        quiet_run("t", self.checkpoints, self.ledger)
        before = self.checkpoints.at("t", 5).state
        self.checkpoints.fork("t", 5, "t-fork", {"policy": {"auto_limit": 1, "days": 30}})
        self.assertEqual(self.checkpoints.at("t", 5).state, before)
        self.assertEqual(self.checkpoints.latest("t-fork").state["policy"]["auto_limit"], 1)

    def test_history_is_newest_first(self):
        quiet_run("t", self.checkpoints, self.ledger)
        steps = [c.step for c in self.checkpoints.history("t")]
        self.assertEqual(steps, sorted(steps, reverse=True))


if __name__ == "__main__":
    unittest.main()
