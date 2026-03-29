# LLM Chat API with RAG

A production-ready multi-provider LLM API with a full RAG (Retrieval-Augmented Generation) pipeline built from scratch — no LangChain, no magic.

Supports **Anthropic (Claude)**, **OpenAI (GPT)**, **Google (Gemini)** and **Ollama** (local models).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                     │
│                                                             │
│  POST /api/chat          POST /api/agent   POST /api/rag/reindex
│  GET  /api/rag/stats     GET  /api/rag/search               │
└────────┬────────────────────────┬────────────────┬──────────┘
         │                        │                │
         ▼                        ▼                ▼
  ┌─────────────┐        ┌──────────────┐   ┌───────────────┐
  │ chat_common │        │anthropic_    │   │  RAG Pipeline │
  │ .py         │        │agent.py      │   │               │
  │ (routing,   │        │(tool-use     │   │ extract.py    │
  │  validation)│        │ agent)       │   │ chunking.py   │
  └──────┬──────┘        └──────────────┘   │ store.py      │
         │                                  │ retrieve.py   │
         ▼                                  │ augment.py    │
  ┌─────────────┐                           └──────┬────────┘
  │ providers.py│                                  │
  │             │                           ┌──────▼────────┐
  │ - Anthropic │                           │   ChromaDB    │
  │ - OpenAI    │◄──────────────────────────│  (local disk) │
  │ - Gemini    │   augmented messages      └───────────────┘
  │ - Ollama    │
  └─────────────┘
```

---

## RAG Pipeline (Retrieval-Augmented Generation)

```
INGESTION (once — POST /api/rag/reindex)
─────────────────────────────────────────

PDF / TXT files
     │
     ▼  extract.py
  Raw text  (pypdf)
     │
     ▼  chunking.py
  Chunks  [chunk_0, chunk_1, ...]
  (sliding window: 1200 chars, overlap 200)
     │
     ▼  store.py
  Embeddings  (all-MiniLM-L6-v2 via SentenceTransformers)
     │
     ▼
  ChromaDB  (persisted to chroma_db/)


QUERY (each request with use_rag: true)
────────────────────────────────────────

  User question
       │
       ▼  augment.py → retrieve.py → store.py
  ChromaDB similarity search
  → top-k most relevant chunks
       │
       ▼  augment.py
  Injected into system message:
  "=== Retrieved excerpts === ... Use these to answer."
       │
       ▼  providers.py
  LLM generates answer grounded in real document content
```

---

## Project Structure

```
claude-agent/
├── api/
│   ├── main.py               # FastAPI app — all routes
│   ├── providers.py          # Multi-provider LLM calls (Anthropic/OpenAI/Gemini/Ollama)
│   ├── chat_common.py        # Shared schemas and provider resolution
│   ├── anthropic_agent.py    # Tool-use agent (Anthropic only)
│   ├── requirements.txt
│   ├── .env                  # API keys and config (not committed)
│   ├── .env.example          # Template
│   ├── data/
│   │   └── reports/          # Drop your PDF/TXT files here
│   ├── chroma_db/            # ChromaDB vector index (auto-generated)
│   └── rag/
│       ├── config.py         # Paths (reports dir, chroma dir)
│       ├── extract.py        # PDF/TXT text extraction (pypdf)
│       ├── chunking.py       # Sliding window chunker
│       ├── store.py          # ChromaDB client + embed + index + query
│       ├── retrieve.py       # Format chunks for LLM injection
│       ├── augment.py        # Inject RAG context into messages
│       └── ingest.py         # CLI: python -m rag.ingest
└── backend/
    └── agent.js              # Legacy Node.js backend (Anthropic)
```

---

## Quickstart

### 1. Install dependencies

```bash
cd api
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Mac/Linux
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set your API keys and LLM_PROVIDER
```

### 3. Start the API

```bash
.\.venv\Scripts\uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) — interactive Swagger UI.

---

## Using the RAG

### Step 1 — Add documents

Drop PDF or TXT files into `api/data/reports/`.

### Step 2 — Index

```
POST /api/rag/reindex
```

Returns:
```json
{"chunks": 627, "files_scanned": 1, "skipped_empty": []}
```

### Step 3 — Query with RAG

```
POST /api/chat
{
  "prompt": "What are Apple's main privacy and security measures?",
  "use_rag": true
}
```

Returns:
```json
{
  "content": "According to the 2025 proxy statement (source: 01 Apple, Inc...)...",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "use_rag": true,
  "rag_hits": 5
}
```

### Test retrieval only (no LLM)

```
GET /api/rag/search?q=CEO compensation&k=5
```

Returns raw chunks from ChromaDB — useful for debugging retrieval quality.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Multi-provider chat (+ optional RAG) |
| `POST` | `/api/claude` | Alias for `/api/chat` with `prompt` |
| `POST` | `/api/agent` | Tool-use agent (Anthropic only) |
| `POST` | `/api/rag/reindex` | Rebuild ChromaDB index from `data/reports/` |
| `GET` | `/api/rag/stats` | Number of indexed chunks |
| `GET` | `/api/rag/search` | Test retrieval without LLM |

---

## Configuration (`.env`)

```ini
# Provider: anthropic | openai | ollama | gemini
LLM_PROVIDER=gemini

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Google Gemini (key from https://aistudio.google.com/apikey)
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash

# Ollama (local)
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=llama3.2

# RAG
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=5
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Uvicorn |
| LLM providers | Anthropic, OpenAI, Google Gemini, Ollama |
| Vector store | ChromaDB (local persistent) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| PDF extraction | pypdf |
| Validation | Pydantic v2 |

---

## Why RAG instead of sending the full document?

| | Full document to LLM | RAG |
|---|---|---|
| 1 document | Works fine | Works fine |
| 100+ documents | Context window exceeded | Handles easily |
| Cost per query | High — full doc tokenized | Low — 5 chunks only |
| Speed | Slow | Fast |
| Verifiability | Hard to trace | `rag_hits` + sources per chunk |
