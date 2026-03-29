from __future__ import annotations

import os
from pathlib import Path

# Répertoire du package api/ (parent de rag/)
API_ROOT = Path(__file__).resolve().parent.parent


def reports_dir() -> Path:
    rel = os.getenv("RAG_REPORTS_DIR", "data/reports")
    p = Path(rel)
    return p if p.is_absolute() else (API_ROOT / p)


def chroma_dir() -> Path:
    rel = os.getenv("RAG_CHROMA_DIR", "chroma_db")
    p = Path(rel)
    return p if p.is_absolute() else (API_ROOT / p)


COLLECTION_NAME = "financial_reports"
