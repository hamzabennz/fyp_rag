#!/usr/bin/env python3
"""
Standalone script to wipe all persistent RAG storage (ChromaDB + SQLite).

Usage:
    python clear_db.py
"""

import sys
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHROMA_PERSIST_DIR, DOC_DB_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # --- Clear ChromaDB (delete the entire persist directory) ---
    chroma_path = Path(CHROMA_PERSIST_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logger.info(f"✓ Deleted ChromaDB directory: {chroma_path}")
    else:
        logger.info(f"ChromaDB directory not found (already clean): {chroma_path}")

    # --- Clear SQLite (delete the .db file) ---
    # DOC_DB_URL format: "sqlite:///./data/rag_docs.db"
    db_file = Path(DOC_DB_URL.replace("sqlite:///", ""))
    if db_file.exists():
        db_file.unlink()
        logger.info(f"✓ Deleted SQLite database: {db_file}")
    else:
        logger.info(f"SQLite database not found (already clean): {db_file}")

    logger.info("✓ All persistent storage cleared. Ready for fresh ingestion.")


if __name__ == "__main__":
    main()
