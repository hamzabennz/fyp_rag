#!/usr/bin/env python3
"""
Batch directory ingestion - loads the pipeline ONCE and ingests all files.

Usage:
    python ingest_dir.py --dir ./trump_docs --strategy semantic --device cuda
    python ingest_dir.py --dir ./trump_docs --pattern "*.txt" --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, ChunkingStrategy
from config import CHROMA_PERSIST_DIR, DOC_DB_URL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest all files in a directory into the RAG system (single pipeline load)"
    )
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--pattern", default="*.txt", help="File glob pattern (default: *.txt)")
    parser.add_argument(
        "--strategy", choices=["semantic", "layout", "hybrid"], default="semantic"
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument(
        "--resource-type", default="default", help="Resource type / collection name"
    )
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        logger.error(f"Directory not found: {doc_dir}")
        sys.exit(1)

    files = sorted(doc_dir.glob(args.pattern))
    if not files:
        logger.error(f"No files matching '{args.pattern}' found in {doc_dir}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"Found {len(files)} files in {doc_dir}")
    logger.info(f"Strategy: {args.strategy} | Device: {args.device} | Collection: {args.resource_type}")
    logger.info("=" * 70)

    # Load pipeline ONCE
    strategy_map = {
        "semantic": ChunkingStrategy.SEMANTIC,
        "layout": ChunkingStrategy.LAYOUT,
        "hybrid": ChunkingStrategy.HYBRID,
    }
    pipeline = RAGPipeline(
        chunking_strategy=strategy_map[args.strategy],
        device=args.device,
        use_persistent_storage=True,
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        sqlite_db_url=DOC_DB_URL,
    )
    logger.info("✓ Pipeline loaded — starting ingestion\n")

    total_chunks = 0
    skipped = 0

    for i, file_path in enumerate(files, 1):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                logger.warning(f"[{i}/{len(files)}] SKIP (empty): {file_path.name}")
                skipped += 1
                continue

            num_chunks = pipeline.add_document(
                content=content,
                source=file_path.name,
                metadata={"filename": file_path.name, "file_size": file_path.stat().st_size},
                resource_type=args.resource_type,
            )
            total_chunks += num_chunks
            logger.info(f"[{i}/{len(files)}] ✓ {file_path.name} → {num_chunks} chunks (total: {total_chunks})")

        except Exception as e:
            logger.error(f"[{i}/{len(files)}] ERROR {file_path.name}: {e}")
            skipped += 1

    logger.info("\n" + "=" * 70)
    logger.info(f"✓ Done! Ingested {len(files) - skipped}/{len(files)} files")
    logger.info(f"  Total chunks stored: {total_chunks}")
    logger.info(f"  Skipped: {skipped}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
