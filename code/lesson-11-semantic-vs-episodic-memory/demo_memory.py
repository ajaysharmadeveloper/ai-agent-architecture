"""Lesson 11, the support customer's four sessions.

Run: python demo_memory.py            real embeddings from a local model (embeddinggemma via Ollama)
     python demo_memory.py --offline  no model at all, the tiny stand-in from memory_stores.py

Version A, the shortcut: every note goes into one vector index, and every question is a search.
Version B, the fix: facts into the profile, events into the log, and the log's text into the index.

The two embedders give different numbers. The point does not depend on the numbers: a similarity
score ranks by meaning, and nothing in it says which note is current or how many matches exist past k.
"""
import sys

from memory_stores import Memory, VectorIndex, embed, handle_tool

# What the agent learned, session by session.
FACTS = [
    ("2026-07-02", "address", "14 Lake Road, Pune", "Customer lives at 14 Lake Road, Pune, deliveries go there."),
    ("2026-07-02", "contact_preference", "message only, never call", "Prefers a message, please never call."),
    ("2026-09-30", "address", "7 Hill View, Bengaluru", "Has shifted to a new flat: 7 Hill View, Bengaluru."),
]
EVENTS = [
    ("2026-07-02", "refund", "Asked for a refund on order 51007, the kettle arrived cracked."),
    ("2026-08-19", "refund", "Asked for a refund on order 54410, the shoes were the wrong size."),
    ("2026-09-30", "refund", "Asked for a refund on order 56002, it came two weeks late."),
    ("2026-10-14", "refund", "Asked for a refund on order 58213, not delivered by Friday."),
    ("2026-10-14", "promise", "Promised a replacement for order 58213 if not delivered by Friday."),
]
QUESTIONS = {
    "address": "What is the customer's delivery address?",
    "refunds": "How many refunds has this customer asked for?",
    "damage": "The box arrived crushed. Has this happened to them before?",
}
CURRENT_ADDRESS = "7 Hill View, Bengaluru"


def shortcut(embed=embed) -> VectorIndex:
    """Version A: append every note, facts and events alike, to one vector index."""
    index = VectorIndex(embed)
    notes = [(at, text) for at, _, _, text in FACTS] + [(at, text) for at, _, text in EVENTS]
    for at, text in sorted(notes):
        index.add(text, at)
    return index


def fixed(embed=embed) -> Memory:
    """Version B: each kind of memory in the store built for its question."""
    memory = Memory(embed)
    items = [(at, "fact", (field, value)) for at, field, value, _ in FACTS]
    items += [(at, "event", (kind, text)) for at, kind, text in EVENTS]
    for at, what, payload in sorted(items):
        if what == "fact":
            memory.remember_fact(*payload, at=at)
        else:
            memory.remember_event(at, *payload)
    return memory


def run(embed=embed):
    index, memory = shortcut(embed), fixed(embed)
    a = {name: index.search(q, k=3) for name, q in QUESTIONS.items()}
    everything = index.search(QUESTIONS["address"], k=len(index.items))
    a["address_rank"] = next(i for i, hit in enumerate(everything, 1) if CURRENT_ADDRESS in hit["text"])
    a["notes"] = len(index.items)
    b = {
        "address": handle_tool(memory, "get_fact", {"field": "address"}),
        "refunds": handle_tool(memory, "recent_events", {"kind": "refund"})["total"],
        "damage": handle_tool(memory, "search_past", {"query": QUESTIONS["damage"], "k": 1}),
    }
    return a, b, memory


if __name__ == "__main__":
    if "--offline" in sys.argv:
        a, b, memory = run()
        print("embedder: the offline stand-in")
    else:
        import local_llm

        a, b, memory = run(local_llm.embed)
        print(f"embedder: {local_llm.EMBED_MODEL}, running locally through Ollama")

    print("\nA. EVERYTHING IN ONE VECTOR INDEX, top 3 for each question")
    for name, question in QUESTIONS.items():
        print(f"\n  {question}")
        for hit in a[name]:
            print(f"    {hit['score']:.3f}  {hit['at']}  {hit['text']}")
    print(f"\n  -> address: the current address ranks {a['address_rank']} of {a['notes']} notes. "
          f"Top hit: {a['address'][0]['text']}")
    print(f"  -> refunds: the search returned {len(a['refunds'])} refund notes. The log holds {b['refunds']}.")
    print(f"  -> crushed box: top hit, with no word in common: {a['damage'][0]['text']}")

    print("\nB. EACH QUESTION TO ITS OWN STORE")
    print(f"  get_fact(address)             -> {b['address']}")
    print(f"  recent_events(kind=refund)    -> total {b['refunds']}")
    print(f"  search_past(crushed box, k=1) -> {b['damage'][0]['text']}")
    print("\n  profile, loaded into the system prompt:")
    print("    " + memory.profile.as_text().replace("\n", "\n    "))
