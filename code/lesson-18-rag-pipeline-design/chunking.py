"""Step 1 of the pipeline: cut each document into chunks.

Two ways to cut, so you can compare them on the same questions:

- fixed_size_chunks: every N words, wherever that lands. Simple, and it cuts through the middle of a rule.
- heading_chunks: one chunk per section, and each chunk carries its document title and section heading,
  so a chunk still says what it is about after it has been pulled out of its page.
"""
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "help_center"


@dataclass
class Chunk:
    chunk_id: str      # "returns-and-refunds#3"
    source: str        # the file it came from
    text: str          # what gets embedded, searched and shown to the model


def load_docs(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    return {path.stem: path.read_text() for path in sorted(docs_dir.glob("*.md"))}


def fixed_size_chunks(docs: dict[str, str], words_per_chunk: int = 80, overlap: int = 0) -> list[Chunk]:
    """Every `words_per_chunk` words, stepping forward by words_per_chunk - overlap."""
    chunks = []
    step = words_per_chunk - overlap
    for source, text in docs.items():
        words = text.split()
        for n, start in enumerate(range(0, len(words), step)):
            piece = words[start:start + words_per_chunk]
            chunks.append(Chunk(f"{source}#{n}", source, " ".join(piece)))
            if start + words_per_chunk >= len(words):
                break
    return chunks


def heading_chunks(docs: dict[str, str]) -> list[Chunk]:
    """One chunk per `## section`. The first line of every chunk is "Document title > Section heading"."""
    chunks = []
    for source, text in docs.items():
        title = source
        heading, body, n = None, [], 0

        def close():
            nonlocal n
            if heading and body:
                chunks.append(Chunk(f"{source}#{n}", source, f"{title} > {heading}\n" + " ".join(body)))
                n += 1

        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                close()
                heading, body = line[3:].strip(), []
            elif line.strip():
                body.append(line.strip())
        close()
    return chunks
