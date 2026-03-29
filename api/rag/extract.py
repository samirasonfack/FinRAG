from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _extract_pdf(path)
    if suf == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)
