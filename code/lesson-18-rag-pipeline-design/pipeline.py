"""The whole path from a document to an answer.

    chunk -> embed -> search (meaning + keywords) -> fuse -> rerank -> build the prompt -> answer

Try it:
    python pipeline.py "I paid cash on delivery. How do I get my refund?"
"""
import re
import sys

import local_llm
from chunking import Chunk, heading_chunks, load_docs
from search import KeywordIndex, VectorIndex, fuse

CANDIDATES = 10   # how many chunks the two searches hand to the reranker
TOP_K = 3         # how many chunks reach the model

ANSWER_SYSTEM = (
    "You are the support assistant for an online shop. Answer the customer's question using only the "
    "help-centre passages you are given. If the passages don't contain the answer, say you don't know. "
    "Answer in one or two sentences."
)

RERANK_SYSTEM = (
    "You judge search results for a shop's help centre. You get a customer question and one passage. "
    "Reply with a single whole number from 0 to 10: 10 means the passage directly answers the question, "
    "0 means it is about something else. Reply with the number only."
)


def rerank(question: str, chunks: list[Chunk], candidates: list[int], k: int, chat=local_llm.chat) -> list[int]:
    """Read the question and each candidate together, score each one, keep the best k.
    Production systems use a small dedicated reranking model for this. Here the local chat model does the
    same job, so there is nothing extra to install. Ties keep the order the search gave them."""
    scored = []
    for position, i in enumerate(candidates):
        reply = chat(RERANK_SYSTEM, f"Question: {question}\n\nPassage:\n{chunks[i].text}")["text"]
        number = re.search(r"\d+", reply)
        scored.append((-(int(number.group()) if number else 0), position, i))
    return [i for _, _, i in sorted(scored)[:k]]


class Pipeline:
    def __init__(self, chunks: list[Chunk] | None = None, embed=local_llm.embed, chat=local_llm.chat):
        self.chunks = chunks if chunks is not None else heading_chunks(load_docs())
        self.chat = chat
        self.vectors = VectorIndex(self.chunks, embed)
        self.keywords = KeywordIndex(self.chunks)

    def retrieve(self, question: str, mode: str = "rerank", k: int = TOP_K) -> list[int]:
        """mode: "vector", "keyword", "hybrid" or "rerank" (hybrid, then reranked)."""
        if mode == "vector":
            return self.vectors.search(question, k)
        if mode == "keyword":
            return self.keywords.search(question, k)
        both = [self.vectors.search(question, CANDIDATES), self.keywords.search(question, CANDIDATES)]
        if mode == "hybrid":
            return fuse(both, k)
        return rerank(question, self.chunks, fuse(both, CANDIDATES), k, self.chat)

    def answer(self, question: str, mode: str = "rerank") -> dict:
        found = self.retrieve(question, mode)
        passages = "\n\n".join(f"[{n}] {self.chunks[i].text}" for n, i in enumerate(found, start=1))
        reply = self.chat(ANSWER_SYSTEM, f"Help-centre passages:\n\n{passages}\n\nCustomer question: {question}")
        return {"answer": reply["text"], "prompt_tokens": reply["prompt_tokens"], "chunks": [self.chunks[i] for i in found]}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "I paid cash on delivery. How do I get my refund?"
    pipeline = Pipeline()
    print(f"indexed {len(pipeline.chunks)} chunks from {len(load_docs())} documents\n")
    for mode in ("vector", "keyword", "hybrid", "rerank"):
        ids = [pipeline.chunks[i].chunk_id for i in pipeline.retrieve(question, mode)]
        print(f"{mode:8} top {TOP_K}: {ids}")
    result = pipeline.answer(question)
    print(f"\nthe model read {result['prompt_tokens']} tokens")
    print(f"answer: {result['answer']}")
