"""
Injecte le contexte RAG dans les messages : embed de la requête → chunks → bloc system pour le LLM.
"""

from __future__ import annotations

from rag.retrieve import build_rag_augmentation_block


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def apply_rag_to_messages(
    messages: list[dict[str, str]],
    *,
    rag_top_k: int | None = None,
) -> tuple[list[dict[str, str]], int]:
    """
    Fusionne un bloc system (extraits + consignes) avec les system existants.
    La requête de retrieval = dernier message user (question actuelle).
    Retourne (messages enrichis, nombre d’extraits renvoyés par Chroma).
    """
    q = _last_user_content(messages).strip()
    if not q:
        return messages, 0

    block, hits = build_rag_augmentation_block(q, k=rag_top_k)
    non_sys = [m for m in messages if m.get("role") != "system"]
    sys_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    full_sys = "\n\n".join([*sys_parts, block]) if sys_parts else block
    return [{"role": "system", "content": full_sys}] + non_sys, hits
