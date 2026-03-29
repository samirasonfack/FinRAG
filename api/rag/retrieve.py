from __future__ import annotations

import os

from rag.store import query_chunks

_EMPTY_HINT = (
    "Aucun extrait trouvé. Vérifie que des PDF/TXT sont dans data/reports/ "
    "et lance POST /api/rag/reindex (ou `python -m rag.ingest`)."
)


def format_hits(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return _EMPTY_HINT
    parts: list[str] = []
    for i, (doc, meta) in enumerate(rows, 1):
        src = meta.get("source", "?")
        parts.append(f"--- Extrait {i} (source: {src}) ---\n{doc.strip()}")
    return "\n\n".join(parts)


def format_context_for_llm(query: str, k: int | None = None) -> str:
    """Texte brut des extraits (tool RAG ou construction de bloc)."""
    k = k if k is not None else int(os.getenv("RAG_TOP_K", "5"))
    return format_hits(query_chunks(query, k=k))


def build_rag_augmentation_block(query: str, k: int | None = None) -> tuple[str, int]:
    """
    Un seul passage retrieve : retourne (bloc system à injecter, nombre d’extraits).
    """
    k = k if k is not None else int(os.getenv("RAG_TOP_K", "5"))
    rows = query_chunks(query, k=k)
    ctx = format_hits(rows)
    block = (
        "=== Retrieved excerpts from indexed financial reports (source in each header) ===\n"
        f"{ctx}\n"
        "=== End excerpts ===\n"
        "Use these excerpts to answer when they are relevant; cite the source filename. "
        "If they are insufficient or unrelated, say so clearly."
    )
    return block, len(rows)
