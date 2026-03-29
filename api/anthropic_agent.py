"""
Agent avec tool-use : implémenté via l’API Messages d’Anthropic uniquement.
Les autres providers n’ont pas le même format d’outils ; utiliser POST /api/chat pour du chat générique.
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from fastapi import HTTPException

MAX_AGENT_STEPS = 12

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_financial_docs",
        "description": (
            "Recherche sémantique dans les rapports indexés (data/reports/, après reindex). "
            "Passe une requête en langage naturel pour récupérer les extraits pertinents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Requête en langage naturel"},
            },
            "required": ["query"],
        },
    },
]


def _text_from_blocks(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _client() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY manquant pour l’agent tool-use (Anthropic).",
        )
    return Anthropic(api_key=key)


def _run_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "search_financial_docs":
        from rag.retrieve import format_context_for_llm

        q = str(tool_input.get("query", "")).strip()
        if not q:
            return "Requête vide : précise ta question pour la recherche documentaire."
        return format_context_for_llm(q)
    return f"Outil inconnu : {name}"


def run_anthropic_tool_agent(prompt: str, *, max_tokens: int) -> dict[str, Any]:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = _client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    last_text = ""

    for _ in range(MAX_AGENT_STEPS):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=TOOLS,
            messages=messages,
        )
        last_text = _text_from_blocks(response.content) or last_text
        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            out = _run_tool(block.name, dict(block.input))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                }
            )
        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    return {"content": last_text, "model": model, "provider": "anthropic"}
