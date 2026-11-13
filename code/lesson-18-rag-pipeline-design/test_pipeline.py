"""Checks that run anywhere: no Ollama, no model. A tiny stand-in turns text into a vector by counting words.

    python -m pytest test_pipeline.py -q
"""
import math
import zlib
from collections import Counter

from chunking import fixed_size_chunks, heading_chunks, load_docs
from pipeline import Pipeline, rerank
from questions import QUESTIONS
from search import KeywordIndex, fuse, tokens

DOCS = load_docs()


def toy_embed(texts: list[str]) -> list[list[float]]:
    """A stand-in for the embedding model: 64 buckets of word counts, scaled to unit length."""
    vectors = []
    for text in texts:
        buckets = [0.0] * 64
        for word, count in Counter(tokens(text)).items():
            buckets[zlib.crc32(word.encode()) % 64] += count
        length = math.sqrt(sum(x * x for x in buckets)) or 1.0
        vectors.append([x / length for x in buckets])
    return vectors


def test_every_fact_appears_exactly_once_in_the_help_centre():
    whole = " ".join(" ".join(text.split()) for text in DOCS.values())
    for item in QUESTIONS:
        assert whole.count(item["fact"]) == 1, item["fact"]


def test_fixed_size_chunks_lose_no_words():
    for source, text in DOCS.items():
        pieces = [c.text for c in fixed_size_chunks({source: text}, 80)]
        assert " ".join(pieces).split() == text.split()


def test_overlapping_chunks_repeat_the_last_words_of_the_one_before():
    chunks = fixed_size_chunks({"doc": " ".join(str(n) for n in range(200))}, 80, overlap=20)
    assert chunks[1].text.split()[:20] == chunks[0].text.split()[-20:]


def test_a_section_chunk_carries_its_title_and_heading():
    chunks = heading_chunks(DOCS)
    cod = next(c for c in chunks if "7 to 10 working days" in c.text)
    assert cod.text.startswith("Returns and refunds > Cash on delivery orders\n")


def test_25_word_chunks_cut_some_facts_in_half():
    tiny = fixed_size_chunks(DOCS, 25)
    cut = [item["fact"] for item in QUESTIONS if not any(item["fact"] in c.text for c in tiny)]
    assert cut, "expected at least one fact to straddle a 25-word boundary"
    by_section = heading_chunks(DOCS)
    assert all(any(item["fact"] in c.text for c in by_section) for item in QUESTIONS)


def test_an_error_code_stays_one_token():
    assert "pay-4027" in tokens("My payment failed with PAY-4027.")


def test_keyword_search_puts_the_exact_code_first():
    chunks = heading_chunks(DOCS)
    best = KeywordIndex(chunks).search("My payment failed with PAY-4027. What should I do?", 1)[0]
    assert "wait 30 minutes" in chunks[best].text


def test_fusion_prefers_what_both_searches_agree_on():
    assert fuse([[1, 2, 3], [3, 2, 9]], k=2)[0] in (2, 3)
    assert fuse([[1, 2, 3], [3, 2, 9]], k=4)[-1] == 9      # 9 is last in one list and missing from the other


def test_rerank_moves_the_best_scored_chunk_to_the_top():
    chunks = heading_chunks(DOCS)
    target = next(i for i, c in enumerate(chunks) if "wait 30 minutes" in c.text)
    candidates = [i for i in range(5) if i != target] + [target]

    def scoring_chat(system: str, user: str) -> dict:
        return {"text": "9" if "wait 30 minutes" in user else "2", "prompt_tokens": 0}

    assert rerank("PAY-4027?", chunks, candidates, k=3, chat=scoring_chat)[0] == target


def test_the_pipeline_returns_k_chunks_and_builds_a_prompt_from_them():
    seen = {}

    def fake_chat(system: str, user: str) -> dict:
        if "Reply with the number only" in system:
            return {"text": "5", "prompt_tokens": 0}
        seen["prompt"] = user
        return {"text": "ok", "prompt_tokens": 123}

    pipeline = Pipeline(embed=toy_embed, chat=fake_chat)
    result = pipeline.answer("How many times will the courier try?")
    assert len(result["chunks"]) == 3 and result["prompt_tokens"] == 123
    assert all(chunk.text in seen["prompt"] for chunk in result["chunks"])
