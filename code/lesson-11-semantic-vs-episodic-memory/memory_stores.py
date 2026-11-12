"""Lesson 11, vector memory vs structured memory. The pure parts: no model calls, no SDK, no network,
so they run and test offline.

An agent asks its memory three different kinds of question about a user, and each one has its own
store:

  Profile      "what is true about them right now?"       one value per field, overwritten on change
  EventLog     "what happened, when, and how many times?"  append only, never edited, kept in order
  VectorIndex  "has anything LIKE this happened before?"   search by meaning over the events' text

Facts about the user (semantic memory) live in the Profile. Things that happened (episodic memory)
live in the EventLog, and the VectorIndex is how you search those events by meaning. Vector search
is a way to SEARCH, not a place to keep facts.

embed() below is a deliberately tiny stand-in for a real embedding model, so the tests run offline
with no model at all. It maps a few words onto shared concepts, which is the job a real model does
across every word it knows. For the real thing, pass local_llm.embed (a local model through Ollama)
to VectorIndex: python demo_memory.py --local
"""
import math
from collections import Counter
from dataclasses import dataclass


class Profile:
    """Facts about one user. One current value per field: a new value replaces the old one."""

    def __init__(self):
        self.fields: dict[str, dict] = {}

    def set(self, field: str, value: str, at: str) -> None:
        self.fields[field] = {"value": value, "at": at}

    def get(self, field: str) -> str | None:
        """None means "we don't know", which the agent should say rather than guess."""
        entry = self.fields.get(field)
        return entry["value"] if entry else None

    def as_text(self) -> str:
        return "\n".join(f"{field}: {entry['value']}" for field, entry in self.fields.items())


@dataclass(frozen=True)
class Event:
    at: str  # ISO date, so string order is time order
    kind: str  # refund, complaint, promise, ...
    text: str


class EventLog:
    """Things that happened to one user. Append only: history is never rewritten, only added to."""

    def __init__(self):
        self.events: list[Event] = []

    def append(self, at: str, kind: str, text: str) -> Event:
        if self.events and at < self.events[-1].at:
            raise ValueError(f"event at {at} is older than the last one at {self.events[-1].at}")
        event = Event(at, kind, text)
        self.events.append(event)
        return event

    def latest(self, kind: str) -> Event | None:
        return next((e for e in reversed(self.events) if e.kind == kind), None)

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)

    def find_word(self, word: str) -> list[Event]:
        """Exact word match: finds "crushed" only where someone typed "crushed"."""
        return [e for e in self.events if word.lower() in e.text.lower().split()]


# The stand-in for meaning. A real embedding model knows that crushed, cracked and damaged are close
# without anyone writing it down.
CONCEPTS = {
    "lives": "home", "live": "home", "address": "home", "deliver": "home", "deliveries": "home", "delivery": "home",
    "flat": "home", "house": "home",
    "crushed": "damage", "cracked": "damage", "damaged": "damage", "broken": "damage", "smashed": "damage",
    "refund": "money_back", "refunds": "money_back", "money": "money_back", "back": "money_back",
}
STOP = {"the", "a", "an", "and", "of", "for", "on", "at", "to", "is", "was", "it", "this", "has", "have",
        "any", "there", "be", "in", "with", "we", "they", "their", "what", "who", "how", "many"}


def embed(text: str) -> Counter:
    words = [w.strip(".,?!:").lower() for w in text.split()]
    return Counter(CONCEPTS.get(w, w) for w in words if w and w not in STOP)


def cosine(a: dict, b: dict) -> float:
    dot = sum(a[k] * b[k] for k in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class VectorIndex:
    """Search by meaning. Returns the k closest texts. It ranks by similarity and nothing else:
    no field says which text is newer, which one is current, or how many matches exist past k."""

    def __init__(self, embed=embed):
        self.embed = embed
        self.items: list[tuple[dict, str, str]] = []

    def add(self, text: str, at: str) -> None:
        self.items.append((self.embed(text), text, at))

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embed(query)
        scored = [{"score": round(cosine(q, vec), 3), "text": text, "at": at} for vec, text, at in self.items]
        scored = [s for s in scored if s["score"] > 0]
        return sorted(scored, key=lambda s: s["score"], reverse=True)[:k]


class Memory:
    """One user's memory: the three stores, and the one rule for writing to them."""

    def __init__(self, embed=embed):
        self.profile = Profile()
        self.log = EventLog()
        self.index = VectorIndex(embed)

    def remember_fact(self, field: str, value: str, at: str) -> None:
        """A fact about the user: overwrite the field. The old value is not "also true"."""
        self.profile.set(field, value, at)

    def remember_event(self, at: str, kind: str, text: str) -> None:
        """Something that happened: append it, and index its text so it can be found by meaning."""
        self.log.append(at, kind, text)
        self.index.add(text, at)


# The agent reads its memory through three tools. Each description says which QUESTION the tool
# answers, because the description is what the model reads when it picks one.
MEMORY_TOOLS = [
    {
        "name": "get_fact",
        "description": "Get one current fact about the customer, such as address or contact_preference. "
        "Use this for anything that is true about them right now. Returns null if unknown.",
        "input_schema": {
            "type": "object",
            "properties": {"field": {"type": "string", "description": "e.g. address, contact_preference"}},
            "required": ["field"],
        },
    },
    {
        "name": "recent_events",
        "description": "List what happened with the customer, newest first, optionally only one kind "
        "(refund, complaint, promise). Use this for when, how many, and what we promised last time.",
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
        },
    },
    {
        "name": "search_past",
        "description": "Search past events by meaning. Use this only to ask whether anything LIKE a "
        "situation has happened before, when you don't know the exact words or kind.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
            "required": ["query"],
        },
    },
]


def handle_tool(memory: Memory, name: str, args: dict):
    if name == "get_fact":
        return memory.profile.get(args["field"])
    if name == "recent_events":
        kind, limit = args.get("kind"), args.get("limit", 5)
        events = [e for e in reversed(memory.log.events) if kind is None or e.kind == kind]
        return {"total": len(events), "events": [e.__dict__ for e in events[:limit]]}
    if name == "search_past":
        return memory.index.search(args["query"], args.get("k", 3))
    raise ValueError(f"unknown memory tool: {name}")
