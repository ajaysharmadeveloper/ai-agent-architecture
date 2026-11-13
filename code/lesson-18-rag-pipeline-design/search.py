"""Steps 2 to 4 of the pipeline: find the chunks that might hold the answer.

- VectorIndex: search by meaning. Embed every chunk once, embed the question, rank by cosine similarity.
- KeywordIndex: search by exact words, with BM25, the ranking function search engines have used for decades.
- fuse: hybrid search. Merge the two rankings with reciprocal rank fusion, which only looks at positions,
  so it doesn't matter that the two searches score on different scales.

Each search returns chunk positions (indexes into the chunk list), best first.
"""
import math
import re
from collections import Counter

from chunking import Chunk

# "AX-220", "PAY-4027" and "5,000" each stay one token, so an exact code or amount can be matched exactly.
TOKEN = re.compile(r"[a-z0-9]+(?:[-,][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embed):
        self.embed = embed
        self.vectors = embed([c.text for c in chunks])      # the indexing step, done once

    def search(self, question: str, k: int) -> list[int]:
        q = self.embed([question])[0]
        scores = [sum(a * b for a, b in zip(q, v)) for v in self.vectors]   # unit vectors: dot = cosine
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


class KeywordIndex:
    """BM25 with the usual constants (k1 = 1.5, b = 0.75)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / len(self.lengths)
        in_how_many = Counter(word for c in self.counts for word in c)
        n = len(chunks)
        # A word that appears in few chunks is worth more than a word that appears in most of them.
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in in_how_many.items()}

    def search(self, question: str, k: int) -> list[int]:
        words = [w for w in tokens(question) if w in self.idf]
        scores = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for w in words:
                tf = counts[w]
                score += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.average_length))
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in ranked if scores[i] > 0][:k]


def fuse(rankings: list[list[int]], k: int, rrf_k: int = 60) -> list[int]:
    """Reciprocal rank fusion: a chunk scores 1 / (60 + its position) in every ranking it appears in.
    A chunk both searches like beats a chunk only one of them likes."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1 / (rrf_k + position)
    return sorted(scores, key=lambda i: -scores[i])[:k]
