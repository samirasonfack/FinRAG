from __future__ import annotations

import os


def chunk_text(text: str) -> list[str]:
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    t = text.strip()
    if not t:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(t[i:end])
        if end >= n:
            break
        i += step
    return chunks
