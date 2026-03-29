from __future__ import annotations

"""
Indexer les rapports : depuis le dossier api/, exécuter :
  python -m rag.ingest
"""

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from rag.store import rebuild_index

    stats = rebuild_index()
    print(
        f"Fichiers scannés: {stats['files_scanned']}, "
        f"chunks indexés: {stats['chunks']}"
    )
    if stats["skipped_empty"]:
        print("Fichiers sans texte (PDF scanné ou vide):", ", ".join(stats["skipped_empty"]))


if __name__ == "__main__":
    main()
