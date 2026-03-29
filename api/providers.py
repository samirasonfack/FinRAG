"""Appels LLM unifiés (chat texte seul, sans tool loop)."""

from __future__ import annotations

import os
from typing import Any, Literal

from anthropic import Anthropic
from fastapi import HTTPException
from google.api_core import exceptions as google_api_exceptions
from openai import OpenAI

Provider = Literal["anthropic", "openai", "ollama", "gemini"]


def _split_system(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    system: str | None = None
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system = (system + "\n\n" if system else "") + (m.get("content") or "")
        else:
            rest.append(m)
    return system, rest


def chat_anthropic(
    messages: list[dict[str, str]], model: str, max_tokens: int = 4096
) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY manquant pour le provider anthropic.",
        )
    system, rest = _split_system(messages)
    client = Anthropic(api_key=key)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": rest,
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def chat_openai_compatible(
    messages: list[dict[str, str]],
    model: str,
    *,
    api_key: str,
    base_url: str | None,
    max_tokens: int = 4096,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    # OpenAI attend system|user|assistant dans messages
    openai_msgs: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            continue
        openai_msgs.append({"role": role, "content": m.get("content", "")})
    if not openai_msgs:
        raise HTTPException(status_code=400, detail="Aucun message valide.")
    resp = client.chat.completions.create(
        model=model,
        messages=openai_msgs,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    content = choice.message.content
    return content or ""


def chat_ollama(messages: list[dict[str, str]], model: str, max_tokens: int = 4096) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    return chat_openai_compatible(
        messages,
        model,
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        base_url=base,
        max_tokens=max_tokens,
    )


def chat_openai(messages: list[dict[str, str]], model: str, max_tokens: int = 4096) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY manquant pour le provider openai.",
        )
    base = os.getenv("OPENAI_BASE_URL") or None
    if base == "":
        base = None
    return chat_openai_compatible(
        messages, model, api_key=key, base_url=base, max_tokens=max_tokens
    )


def chat_gemini(messages: list[dict[str, str]], model: str, max_tokens: int = 4096) -> str:
    import google.generativeai as genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY (ou GEMINI_API_KEY) manquant pour gemini.",
        )
    system, rest = _split_system(messages)
    if not rest:
        raise HTTPException(status_code=400, detail="Il faut au moins un message user/assistant.")
    genai.configure(api_key=key)
    gm = genai.GenerativeModel(
        model,
        system_instruction=system if system else None,
    )
    history: list[dict[str, Any]] = []
    for m in rest[:-1]:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})
    last = rest[-1]
    if last.get("role") != "user":
        raise HTTPException(
            status_code=400,
            detail="Pour Gemini, le dernier message doit être du rôle 'user'.",
        )
    chat = gm.start_chat(history=history)
    try:
        resp = chat.send_message(
            last.get("content", ""),
            generation_config={"max_output_tokens": max_tokens},
        )
    except google_api_exceptions.ResourceExhausted as e:
        raise HTTPException(
            status_code=429,
            detail=(
                "Quota Gemini API dépassé (souvent free tier / limite minute ou jour). "
                "Voir https://ai.google.dev/gemini-api/docs/rate-limits — "
                f"réessayer plus tard ou changer GEMINI_MODEL / activer la facturation. Détail : {getattr(e, 'message', None) or str(e)}"
            ),
        ) from e
    except google_api_exceptions.GoogleAPIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur API Google Gemini : {getattr(e, 'message', None) or str(e)}",
        ) from e

    try:
        out = (resp.text or "").strip()
    except ValueError:
        out = ""
    if not out:
        raise HTTPException(
            status_code=502,
            detail="Gemini n'a renvoyé aucun texte (filtre de sécurité, modèle indisponible ou réponse vide).",
        )
    return out


def complete_chat(
    provider: Provider,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
) -> tuple[str, str, str]:
    """
    Retourne (texte, provider utilisé, id modèle résolu).
    """
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        text = chat_anthropic(messages, model, max_tokens)
        return text, provider, model
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        text = chat_openai(messages, model, max_tokens)
        return text, provider, model
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        text = chat_ollama(messages, model, max_tokens)
        return text, provider, model
    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        text = chat_gemini(messages, model, max_tokens)
        return text, provider, model
    raise HTTPException(status_code=500, detail=f"Provider inconnu : {provider}")
