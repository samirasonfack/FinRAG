# FinRAG: Financial Analysis & Research Assistant 📈🤖

FinRAG is a **production-ready RAG (Retrieval-Augmented Generation) pipeline** designed as an AI research assistant for financial analysts. By ingesting complex financial documents (proxy statements, annual reports, earnings transcripts), FinRAG provides accurate, context-aware answers to investment queries — grounded in real documents, not hallucinations.

> Built from scratch — no LangChain, no magic. Every component is transparent and replaceable.

---

## 🚀 Features

- **Multi-Provider LLM** — Switch between Anthropic (Claude), OpenAI (GPT), Google (Gemini), or Ollama (local) via a single env variable
- **RAG Pipeline from Scratch** — PDF extraction → chunking → vector embeddings → semantic search → LLM augmentation
- **Source Citations** — Every answer references the exact source document retrieved from ChromaDB
- **Tool-Use Agent** — Anthropic-powered agent that autonomously decides when to search the document index
- **REST API** — Clean FastAPI backend with interactive Swagger UI at `/docs`
- **Verifiable Retrieval** — `/api/rag/search` endpoint lets you inspect raw chunks before the LLM touches them

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **LLM Providers** | Anthropic Claude · OpenAI GPT · Google Gemini · Ollama |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (local persistent) |
| **Embeddings** | [SentenceTransformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` via HuggingFace |
| **PDF Extraction** | [pypdf](https://pypdf.readthedocs.io/) |
| **Validation** | Pydantic v2 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)                        │
│                                                             │
│  POST /api/chat      POST /api/agent    POST /api/rag/reindex│
│  GET  /api/rag/stats GET  /api/rag/search                   │
└────────┬─────────────────────┬──────────────────┬───────────┘
         │                     │                  │
         ▼                     ▼                  ▼
  ┌─────────────┐    ┌──────────────────┐  ┌─────────────────┐
  │ chat_common │    │ anthropic_agent  │  │  RAG Pipeline   │
  │ (routing &  │    │ (tool-use agent) │  │                 │
  │  validation)│    └──────────────────┘  │  extract.py     │
  └──────┬──────┘                          │  chunking.py    │
         │                                 │  store.py       │
         ▼                                 │  retrieve.py    │
  ┌─────────────┐                          │  augment.py     │
  │ providers   │◄─────────────────────────┤                 │
  │             │    augmented messages    └────────┬────────┘
  │ • Anthropic │                                   │
  │ • OpenAI    │                            ┌──────▼───────┐
  │ • Gemini    │                            │   ChromaDB   │
  │ • Ollama    │                            │  (disk)      │
  └─────────────┘                            └──────────────┘
```

---

## 🔄 RAG Pipeline

```
INGESTION  (once — POST /api/rag/reindex)
──────────────────────────────────────────────────────
PDF / TXT  →  extract.py (pypdf)  →  Raw text
           →  chunking.py (1200 chars, overlap 200)  →  Chunks
           →  store.py + all-MiniLM-L6-v2  →  Vectors
           →  ChromaDB (persisted to chroma_db/)

QUERY  (each request with use_rag: true)
──────────────────────────────────────────────────────
User question  →  ChromaDB similarity search  →  Top-k chunks
               →  augment.py (inject into system message)
               →  LLM  →  Answer grounded in real document
```

---

## ⚙️ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/samirasonfack/FinRAG.git
cd FinRAG
```

### 2. Install dependencies

```bash
cd api
python -m venv .venv

# Windows
.\.venv\Scripts\pip install -r requirements.txt

# Mac / Linux
source .venv/bin/activate && pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — add your API keys and set LLM_PROVIDER
```

### 4. Start the API

```bash
# Windows
.\.venv\Scripts\uvicorn main:app --reload --host 127.0.0.1 --port 8080

# Mac / Linux
uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) — interactive Swagger UI.

---

## 📂 Using the RAG

### Step 1 — Add documents

Drop PDF or TXT files into `api/data/reports/`.

```
api/data/reports/
├── Apple_10K_2024.pdf
├── Tesla_Earnings_Q3.pdf
└── ...
```

### Step 2 — Index documents

```
POST /api/rag/reindex
```

```json
{"chunks": 627, "files_scanned": 1, "skipped_empty": []}
```

### Step 3 — Ask questions with RAG

```
POST /api/chat
{
  "prompt": "What are Apple's main privacy and security measures?",
  "use_rag": true
}
```

```json
{
  "content": "According to the 2025 proxy statement...",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "use_rag": true,
  "rag_hits": 5
}
```

### Step 4 — Verify retrieval (no LLM)

```
GET /api/rag/search?q=CEO compensation&k=5
```

Returns raw chunks from ChromaDB — inspect retrieval quality before involving the LLM.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Multi-provider chat (+ optional RAG) |
| `POST` | `/api/claude` | Alias — chat with `prompt` field |
| `POST` | `/api/agent` | Tool-use agent (Anthropic only) |
| `POST` | `/api/rag/reindex` | Rebuild ChromaDB index from `data/reports/` |
| `GET` | `/api/rag/stats` | Number of indexed chunks |
| `GET` | `/api/rag/search` | Test retrieval without LLM |

---

## 🔧 Configuration

```ini
# .env — copy from .env.example

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

# RAG tuning
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=5
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 📁 Project Structure

```
FinRAG/
├── README.md
├── package.json
├── backend/
│   └── agent.js              # Legacy Node.js Anthropic backend
└── api/
    ├── main.py               # FastAPI app — all routes
    ├── providers.py          # Multi-provider LLM calls
    ├── chat_common.py        # Shared schemas & provider resolution
    ├── anthropic_agent.py    # Tool-use agent (Anthropic)
    ├── requirements.txt
    ├── .env.example
    ├── data/
    │   └── reports/          # Drop your PDF/TXT files here
    ├── chroma_db/            # ChromaDB index (auto-generated, not committed)
    └── rag/
        ├── config.py         # Paths configuration
        ├── extract.py        # PDF/TXT text extraction
        ├── chunking.py       # Sliding window text chunker
        ├── store.py          # ChromaDB client — index & query
        ├── retrieve.py       # Format chunks for LLM injection
        ├── augment.py        # Inject RAG context into messages
        └── ingest.py         # CLI indexer: python -m rag.ingest
```

---

## 💡 Why RAG instead of sending the full document?

| | Full document to LLM | RAG |
|---|---|---|
| 1 document | ✅ Works | ✅ Works |
| 100+ documents | ❌ Context window exceeded | ✅ Scales easily |
| Cost per query | 🔴 High — full doc tokenized every time | 🟢 Low — 5 chunks only |
| Speed | 🔴 Slow on large docs | 🟢 Fast |
| Sensitive data | 🔴 Full doc sent to external API | 🟢 Only relevant chunks leave your system |
| Verifiability | 🟡 Hard to trace | 🟢 `rag_hits` + source per chunk |

---

## 🗺️ Roadmap

- [ ] Reranking with cross-encoders (Cohere Rerank)
- [ ] Hybrid search (semantic + keyword BM25)
- [ ] Streamlit / React chat UI
- [ ] Docker deployment
- [ ] Multi-document agent with LangGraph
- [ ] `yfinance` + `sec-edgar-downloader` auto-ingestion
