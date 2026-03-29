from __future__ import annotations

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import COLLECTION_NAME, chroma_dir, reports_dir

_ef: SentenceTransformerEmbeddingFunction | None = None
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None  # type: ignore[valid-type]


def _embedding_fn() -> SentenceTransformerEmbeddingFunction:
    global _ef
    if _ef is None:
        model = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _ef = SentenceTransformerEmbeddingFunction(model_name=model)
    return _ef


def _chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        chroma_dir().mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(chroma_dir()))
    return _client


def get_collection() -> chromadb.Collection:
    """Collection courante (créée au premier ingest)."""
    global _collection
    if _collection is None:
        _collection = _chroma_client().get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn(),
        )
    return _collection


def _invalidate_collection() -> None:
    global _collection
    _collection = None


def rebuild_index() -> dict[str, int | list[str]]:
    """
    Supprime l’index, relit tous les PDF/TXT dans data/reports, chunk, embed, stocke.
    """
    global _collection
    client = _chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None

    coll = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn(),
    )
    _collection = coll

    rdir = reports_dir()
    rdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(rdir.glob("*.pdf")) + sorted(rdir.glob("*.txt"))
    skipped: list[str] = []
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, str | int]] = []

    from rag.chunking import chunk_text
    from rag.extract import extract_text

    for path in paths:
        text = extract_text(path)
        if not text.strip():
            skipped.append(path.name)
            continue
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_ids.append(f"{path.stem}_{path.suffix}_{i}")
            all_docs.append(ch)
            all_meta.append({"source": path.name, "chunk_index": i})

    if all_ids:
        coll.add(ids=all_ids, documents=all_docs, metadatas=all_meta)

    return {
        "chunks": len(all_ids),
        "files_scanned": len(paths),
        "skipped_empty": skipped,
    }


def query_chunks(query: str, k: int | None = None) -> list[tuple[str, dict]]:
    k = k if k is not None else int(os.getenv("RAG_TOP_K", "5"))
    try:
        coll = get_collection()
    except Exception:
        return []
    if coll.count() == 0:
        return []
    res = coll.query(query_texts=[query], n_results=min(k, max(1, coll.count())))
    out: list[tuple[str, dict]] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0] if res.get("distances") else [None] * len(docs)
    for i, doc in enumerate(docs):
        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
        if i < len(dists) and dists[i] is not None:
            meta["distance"] = dists[i]
        out.append((doc, meta))
    return out


def collection_stats() -> dict[str, int | str]:
    try:
        coll = get_collection()
        n = coll.count()
    except Exception:
        n = 0
    return {"collection": COLLECTION_NAME, "chunks": n}
