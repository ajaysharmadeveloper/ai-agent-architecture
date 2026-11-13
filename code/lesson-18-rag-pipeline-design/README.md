# Lesson 18: a RAG pipeline for the shop's help centre

The support agent's shop from lessons 13 to 17 now has a help centre: 10 markdown pages, 57 sections. This
is the whole path from those pages to an answer, in plain Python, with no vector database and no API key.

    chunk -> embed -> search (meaning + keywords) -> fuse -> rerank -> build the prompt -> answer

| File | What it is |
|---|---|
| `help_center/` | The documents: returns, shipping, warranty, payments, error codes, products and more. |
| `chunking.py` | Step 1. Cut every N words, or one chunk per section with its title and heading on line one. |
| `search.py` | Meaning search (embeddings + cosine), keyword search (BM25), and reciprocal rank fusion. |
| `pipeline.py` | The pipeline: search, fuse, rerank, build the prompt, answer. |
| `questions.py` | 22 customer questions, each with the fact that answers it. |
| `evaluate.py` | Runs every chunking and search option on the same 22 questions and prints the numbers. |
| `local_llm.py` | `embeddinggemma` and `qwen3:8b` through Ollama. Standard library only. |
| `test_pipeline.py` | 10 checks that need no model. |

## Run it

```bash
ollama pull embeddinggemma
ollama pull qwen3:8b

python pipeline.py "I paid cash on delivery. How do I get my refund?"
python evaluate.py              # search numbers, about 2 minutes
python evaluate.py --answers    # plus the answers, about 9 minutes on a laptop
python -m pytest test_pipeline.py -q
```

There is nothing to `pip install` except `pytest` for the tests.

## What the run showed (22 questions, on a laptop)

| Setup | Right chunk in top 3 | Right chunk first | Answers right | Tokens read |
|---|---|---|---|---|
| No documents | | | 1 | 72 |
| 25-word chunks, meaning search | 14 | 8 | 12 | 198 |
| 80-word chunks, meaning search | 22 | 19 | | |
| 80-word chunks, 20 words of overlap | 22 | 16 | | |
| One chunk per section, meaning search | 22 | 20 | 22 | 301 |
| One chunk per section, keyword search | 19 | 17 | | |
| One chunk per section, hybrid | 22 | 20 | | |
| One chunk per section, hybrid + rerank | 22 | 21 | 22 | 312 |
| Whole documents as chunks | 22 | 21 | 22 | 1231 |
| Whole help centre in the prompt | | | 21 | 3869 |

Reranking took 4.75 seconds a question here, because the chat model does the scoring. A dedicated
reranking model is much faster. Your numbers will differ a little from run to run and model to model.
