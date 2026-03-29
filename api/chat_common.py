"""Schémas et helpers communs à tous les providers (POST /api/chat)."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from providers import Provider


def resolve_provider(override: Provider | None) -> Provider:
    raw = (override or os.getenv("LLM_PROVIDER", "anthropic")).strip().lower()
    allowed: tuple[Provider, ...] = ("anthropic", "openai", "ollama", "gemini")
    if raw not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"LLM_PROVIDER invalide : {raw!r}. Attendu : {', '.join(allowed)}.",
        )
    return raw  # type: ignore[return-value]


def coerce_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if content is None:
            continue
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
                elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                    texts.append(str(getattr(block, "text", "")))
            content = "".join(texts)
        out.append({"role": str(role), "content": str(content)})
    return out


class ChatBody(BaseModel):
    """`prompt` OU `messages`. Optionnel : `provider`, `use_rag` (retrieve + injecte en system)."""

    prompt: str | None = Field(default=None, min_length=1)
    messages: list[dict[str, Any]] | None = None
    provider: Literal["anthropic", "openai", "ollama", "gemini"] | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=128_000)
    use_rag: bool = False
    rag_top_k: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def one_of_prompt_or_messages(self) -> "ChatBody":
        if bool(self.prompt) == bool(self.messages):
            raise ValueError("Fournis exactement l’un des deux : `prompt` ou `messages`.")
        return self

    def resolved_messages(self) -> list[dict[str, str]]:
        if self.prompt is not None:
            return [{"role": "user", "content": self.prompt}]
        assert self.messages is not None
        return coerce_messages(self.messages)


class PromptBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_rag: bool = False
    rag_top_k: int | None = Field(default=None, ge=1, le=50)
