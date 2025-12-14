#!/usr/bin/env python3
"""
Document Ingestion Script

Ingest documents into the RAG system with persistent storage.

Usage:
    python ingest.py --source "doc.txt" --text "Your document content here"
    python ingest.py --source "doc.txt" --file path/to/document.txt
    python ingest.py --source "doc.txt" --file path/to/document.txt --strategy semantic
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, ChunkingStrategy
from config import RAGConfig, CHROMA_PERSIST_DIR, DOC_DB_URL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    
    parser.add_argument(
        "--source",
        required=True,
        help="Document source identifier (e.g., filename, URL)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        help="Document text content"
    )
    group.add_argument(
        "--file",
        help="Path to document file"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["semantic", "layout", "hybrid"],
        default="semantic",
        help="Chunking strategy (default: semantic)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_PERSIST_DIR,
        help=f"ChromaDB persist directory (default: {CHROMA_PERSIST_DIR})"
    )
    
    parser.add_argument(
        "--db-url",
        default=DOC_DB_URL,
        help=f"SQLite database URL (default: {DOC_DB_URL})"
    )
    
    parser.add_argument(
        "--metadata",
        help="Additional metadata as JSON string"
    )
    
    args = parser.parse_args()
    
    # Get document content
    if args.text:
        content = args.text
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Loaded document from: {args.file}")
    
    # Parse metadata
    metadata = {}
    if args.metadata:
        import json
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in --metadata")
            sys.exit(1)
    
    # Add file info to metadata if applicable
    if args.file:
        file_path = Path(args.file)
        metadata.update({
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
        })
    
    # Map strategy
    strategy_map = {
        "semantic": ChunkingStrategy.SEMANTIC,
        "layout": ChunkingStrategy.LAYOUT,
        "hybrid": ChunkingStrategy.HYBRID,
    }
    
    # Initialize RAG pipeline with persistent storage
    logger.info(f"Initializing RAG pipeline with {args.strategy} chunking...")
    logger.info(f"Chroma directory: {args.chroma_dir}")
    logger.info(f"Database URL: {args.db_url}")
    
    pipeline = RAGPipeline(
        chunking_strategy=strategy_map[args.strategy],
        device=args.device,
        use_persistent_storage=True,
        chroma_persist_dir=args.chroma_dir,
        sqlite_db_url=args.db_url,
    )
    
    # Ingest document
    logger.info(f"Ingesting document: {args.source}")
    logger.info(f"Content length: {len(content)} characters")
    
    num_chunks = pipeline.add_document(
        content=content,
        source=args.source,
        metadata=metadata,
    )
    
    logger.info("=" * 60)
    logger.info(f"✓ Successfully ingested document: {args.source}")
    logger.info(f"  Chunks created: {num_chunks}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info("=" * 60)
    
    # Show stats
    if pipeline.use_persistent_storage:
        doc_stats = pipeline.doc_store.get_stats()
        vector_stats = pipeline.vector_store.get_stats()
        
        logger.info("\nStorage Statistics:")
        logger.info(f"  Total documents: {doc_stats['total_documents']}")
        logger.info(f"  Total chunks: {vector_stats['total_chunks']}")


if __name__ == "__main__":
    main()
