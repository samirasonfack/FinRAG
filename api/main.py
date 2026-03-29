import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from anthropic_agent import run_anthropic_tool_agent
from chat_common import ChatBody, PromptBody, resolve_provider
from providers import complete_chat

load_dotenv()

app = FastAPI(
    title="LLM chat API",
    description=(
        "POST /api/chat : chat multi-provider ; `use_rag: true` embed la dernière question user, "
        "retrieve des chunks dans Chroma, injecte les extraits en message system puis appelle le LLM. "
        "POST /api/agent : tool-use Anthropic (recherche explicite par l’outil). "
        "Ingest : POST /api/rag/reindex."
    ),
)

_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TOKENS_CHAT = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(body: ChatBody) -> dict[str, Any]:
    provider = resolve_provider(body.provider)
    messages = body.resolved_messages()
    if not messages:
        raise HTTPException(status_code=400, detail="Aucun message après normalisation.")
    rag_hits: int | None = None
    if body.use_rag:
        from rag.augment import apply_rag_to_messages

        messages, rag_hits = apply_rag_to_messages(
            messages,
            rag_top_k=body.rag_top_k,
        )
    max_tok = body.max_tokens if body.max_tokens is not None else MAX_TOKENS_CHAT
    text, used, model = complete_chat(provider, messages, max_tokens=max_tok)
    out: dict[str, Any] = {
        "content": text,
        "provider": used,
        "model": model,
        "use_rag": body.use_rag,
    }
    if body.use_rag:
        out["rag_hits"] = rag_hits
    return out


@app.post("/api/claude")
def legacy_prompt_chat(body: PromptBody) -> dict[str, Any]:
    """Alias : POST /api/chat avec `prompt` (+ optionnel `use_rag` / `rag_top_k`)."""
    return chat(
        ChatBody(
            prompt=body.prompt,
            use_rag=body.use_rag,
            rag_top_k=body.rag_top_k,
        )
    )


@app.post("/api/agent")
def agent_with_tools(body: PromptBody) -> dict[str, Any]:
    """
    Agent avec outils (recherche RAG dans les rapports indexés). **Uniquement Anthropic**.
    """
    return run_anthropic_tool_agent(body.prompt, max_tokens=MAX_TOKENS_CHAT)


@app.post("/api/rag/reindex")
def rag_reindex() -> dict[str, Any]:
    """Relit `data/reports/` (PDF/TXT), découpe, embed, reconstruit l’index Chroma."""
    from rag.store import rebuild_index

    return rebuild_index()


@app.get("/api/rag/stats")
def rag_stats() -> dict[str, Any]:
    from rag.store import collection_stats

    return collection_stats()


@app.get("/api/rag/search")
def rag_search(q: str, k: int = 5) -> dict[str, Any]:
    """Test retrieval sans agent (aperçu des extraits)."""
    from rag.retrieve import format_hits
    from rag.store import query_chunks

    if not q.strip():
        raise HTTPException(status_code=400, detail="Paramètre `q` requis.")
    rows = query_chunks(q.strip(), k=k)
    return {"context": format_hits(rows), "hits": len(rows)}
