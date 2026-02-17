"""Chunk English medical textbooks into pretraining JSONL for LLaMA-Factory."""

import json
import re
from pathlib import Path

INPUT_DIR = Path("data_clean/data_clean/textbooks/en")
OUTPUT_PATH = Path("data/textbooks_pretrain.jsonl")

CHUNK_SIZE = 4096  # characters (~1024 tokens)
OVERLAP_SIZE = 512  # characters (~128 tokens)
MIN_CHUNK_SIZE = 200  # skip very small trailing chunks


def split_into_paragraphs(text: str) -> list[str]:
    """Split text on double newlines (paragraph boundaries)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, finalize current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > CHUNK_SIZE:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous
            overlap = current_chunk[-OVERLAP_SIZE:] if len(current_chunk) > OVERLAP_SIZE else current_chunk
            current_chunk = overlap + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    textbook_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"Found {len(textbook_files)} textbooks")

    total_chunks = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for filepath in textbook_files:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text)
            for chunk in chunks:
                fout.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
            print(f"  {filepath.name}: {len(chunks)} chunks")
            total_chunks += len(chunks)

    print(f"Wrote {total_chunks} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
