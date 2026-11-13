"""Measure every stage of the pipeline on the same 22 questions.

    python evaluate.py            # search stages only (fast)
    python evaluate.py --answers  # also asks the model to answer, five ways

Part 1 asks: did the search put the chunk that holds the fact in its top 3? And was it first?
Part 2 asks: did the model's answer state the fact?
Every line of the summary is computed from the results. Nothing is typed in by hand.
"""
import sys
import time

import local_llm
from chunking import Chunk, fixed_size_chunks, heading_chunks, load_docs
from pipeline import ANSWER_SYSTEM, TOP_K, Pipeline
from questions import QUESTIONS

NO_DOCS_SYSTEM = (
    "You are the support assistant for an online shop. Answer the customer's question in one or two "
    "sentences. If you don't know the shop's rule, say you don't know."
)


def position_of_fact(pipeline: Pipeline, found: list[int], fact: str) -> int | None:
    """1, 2 or 3 when a returned chunk contains the fact, None when none of them does."""
    for position, i in enumerate(found, start=1):
        if fact in pipeline.chunks[i].text:
            return position
    return None


def whole_document_chunks(docs: dict[str, str]) -> list[Chunk]:
    return [Chunk(f"{source}#0", source, " ".join(text.split())) for source, text in docs.items()]


def search_stages(docs: dict[str, str]) -> list[tuple[str, Pipeline, str]]:
    by_section = Pipeline(heading_chunks(docs))
    return [
        ("chunking  every 25 words          + meaning", Pipeline(fixed_size_chunks(docs, 25)), "vector"),
        ("chunking  every 80 words          + meaning", Pipeline(fixed_size_chunks(docs, 80)), "vector"),
        ("chunking  80 words, 20 overlap    + meaning", Pipeline(fixed_size_chunks(docs, 80, overlap=20)), "vector"),
        ("chunking  one per section         + meaning", by_section, "vector"),
        ("chunking  one per whole document  + meaning", Pipeline(whole_document_chunks(docs)), "vector"),
        ("search    one per section         + keyword", by_section, "keyword"),
        ("search    one per section         + hybrid", by_section, "hybrid"),
        ("search    one per section         + hybrid + rerank", by_section, "rerank"),
    ]


def run_search(stages) -> dict[str, list[int | None]]:
    print(f"PART 1  search: is the fact in the top {TOP_K}, and where\n")
    for n, item in enumerate(QUESTIONS, start=1):
        print(f"  Q{n:<2} {item['q']}")
    print()
    results = {}
    for name, pipeline, mode in stages:
        started = time.perf_counter()
        positions = [position_of_fact(pipeline, pipeline.retrieve(item["q"], mode), item["fact"]) for item in QUESTIONS]
        seconds = (time.perf_counter() - started) / len(QUESTIONS)
        results[name] = positions
        marks = " ".join(str(p) if p else "-" for p in positions)
        found = sum(p is not None for p in positions)
        first = sum(p == 1 for p in positions)
        print(f"  {name:52} {marks}   found {found:>2}/{len(QUESTIONS)}   first {first:>2}/{len(QUESTIONS)}   {len(pipeline.chunks):>3} chunks   {seconds:.2f} s per question")
    return results


def states_the_fact(answer: str, says: list[str]) -> bool:
    return any(way in answer.lower() for way in says)


def run_answers(stages, docs: dict[str, str]) -> None:
    print("\nPART 2  answers: does the reply state the fact\n")
    tiny, by_section, whole_docs = stages[0][1], stages[3][1], stages[4][1]
    whole_help_centre = "\n\n".join(docs.values())
    ways = [
        ("no documents", lambda q: local_llm.chat(NO_DOCS_SYSTEM, q)),
        ("25-word chunks + meaning search", lambda q: _as_reply(tiny.answer(q, "vector"))),
        ("sections + meaning search", lambda q: _as_reply(by_section.answer(q, "vector"))),
        ("sections + hybrid + rerank", lambda q: _as_reply(by_section.answer(q, "rerank"))),
        ("whole documents + meaning search", lambda q: _as_reply(whole_docs.answer(q, "vector"))),
        ("whole help centre in the prompt", lambda q: local_llm.chat(ANSWER_SYSTEM, f"Help centre:\n\n{whole_help_centre}\n\nCustomer question: {q}")),
    ]
    summary = []
    for name, ask in ways:
        right, tokens = 0, 0
        print(f"  --- {name}")
        for n, item in enumerate(QUESTIONS, start=1):
            reply = ask(item["q"])
            ok = states_the_fact(reply["text"], item["says"])
            right += ok
            tokens += reply["prompt_tokens"]
            print(f"  Q{n:<2} {'RIGHT' if ok else 'WRONG'}  {reply['text'].replace(chr(10), ' ')}")
        summary.append((name, right, round(tokens / len(QUESTIONS))))
        print()
    print("  summary")
    for name, right, tokens in summary:
        print(f"  {name:34} right {right:>2}/{len(QUESTIONS)}   average tokens read {tokens}")


def _as_reply(result: dict) -> dict:
    return {"text": result["answer"], "prompt_tokens": result["prompt_tokens"]}


def verdict(results: dict[str, list[int | None]]) -> None:
    tiny, fixed, overlapped, section, whole, keyword, hybrid, reranked = list(results)
    found = {name: sum(p is not None for p in positions) for name, positions in results.items()}
    first = {name: sum(p == 1 for p in positions) for name, positions in results.items()}
    total = len(QUESTIONS)

    def line(label: str, name: str) -> None:
        print(f"  {label:40} found {found[name]:>2}/{total}   first {first[name]:>2}/{total}")

    print("\n  what the search numbers say")
    for label, name in (("25-word chunks", tiny), ("80-word chunks", fixed), ("80-word chunks with overlap", overlapped),
                        ("one chunk per section", section), ("one chunk per document", whole),
                        ("keyword search alone", keyword), ("hybrid", hybrid), ("hybrid + rerank", reranked)):
        line(label, name)
    for a, b, label in ((section, keyword, "meaning search missed, keyword search found"),
                        (keyword, section, "keyword search missed, meaning search found"),
                        (section, reranked, "meaning search missed, full pipeline found"),
                        (tiny, section, "25-word chunks missed, sections found")):
        which = [f"Q{n}" for n, (x, y) in enumerate(zip(results[a], results[b]), start=1) if x is None and y is not None]
        print(f"  {label}: {', '.join(which) or 'none'}")
    not_first = [f"Q{n}" for n, p in enumerate(results[section], start=1) if p != 1]
    print(f"  meaning search, sections, fact not in first place: {', '.join(not_first) or 'none'}")


if __name__ == "__main__":
    docs = load_docs()
    print(f"{len(docs)} documents, {sum(len(t.split()) for t in docs.values())} words, models: {local_llm.EMBED_MODEL} + {local_llm.CHAT_MODEL}\n")
    stages = search_stages(docs)
    results = run_search(stages)
    verdict(results)
    if "--answers" in sys.argv:
        run_answers(stages, docs)
