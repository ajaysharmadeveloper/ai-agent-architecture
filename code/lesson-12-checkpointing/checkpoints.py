"""Lesson 12, checkpointing. The pure part: no model, no network, so it runs and tests offline.

A checkpoint is the whole state of a run, saved after a step finishes, under the run's thread id.
To resume, load the newest checkpoint for that thread id and carry on from the step after it.

  SqliteCheckpointer   a file on disk: survives the process dying. (Postgres does the same job for
                       many servers; the shape is identical.)
  MemoryCheckpointer   a dict in RAM: fine for tests, gone the moment the process dies.

History is never rewritten. A fork copies an old checkpoint into a NEW thread id, so the original
run stays exactly as it was.
"""
import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    thread_id: str
    step: int  # how many steps were finished when this was saved
    state: dict


class SqliteCheckpointer:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            " thread_id TEXT, step INTEGER, state TEXT, PRIMARY KEY (thread_id, step))"
        )

    def save(self, thread_id: str, step: int, state: dict) -> None:
        """Written and committed before the next step starts, so a crash can't lose a finished step."""
        self.db.execute("INSERT INTO checkpoints VALUES (?, ?, ?)", (thread_id, step, json.dumps(state)))
        self.db.commit()

    def history(self, thread_id: str) -> list[Checkpoint]:
        """Every checkpoint for one run, newest first."""
        rows = self.db.execute(
            "SELECT step, state FROM checkpoints WHERE thread_id = ? ORDER BY step DESC", (thread_id,)
        )
        return [Checkpoint(thread_id, step, json.loads(state)) for step, state in rows]

    def latest(self, thread_id: str) -> Checkpoint | None:
        found = self.history(thread_id)
        return found[0] if found else None

    def at(self, thread_id: str, step: int) -> Checkpoint:
        return next(c for c in self.history(thread_id) if c.step == step)

    def fork(self, thread_id: str, step: int, new_thread_id: str, changes: dict) -> Checkpoint:
        """Copy the checkpoint at `step` into a new thread, with some state changed.
        The original thread is not touched."""
        old = self.at(thread_id, step)
        forked = Checkpoint(new_thread_id, step, {**old.state, **changes})
        self.save(forked.thread_id, forked.step, forked.state)
        return forked


class MemoryCheckpointer(SqliteCheckpointer):
    """Same API, kept in RAM. A new process starts with nothing."""

    def __init__(self):
        super().__init__(":memory:")
