#!/usr/bin/env python3
"""
Query Script

Query the RAG system with persistent storage.

Usage:
    python query.py --q "What is machine learning?"
    python query.py --q "Your question" --top-k 10
    python query.py --q "Your question" --source "specific_doc.txt"
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
    parser = argparse.ArgumentParser(description="Query RAG system")
    
    parser.add_argument(
        "--q", "--query",
        dest="query",
        required=True,
        help="Query text"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--source",
        help="Filter by document source"
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
        "--show-text",
        action="store_true",
        help="Show full chunk text in results"
    )
    
    parser.add_argument(
        "--resources",
        nargs="+",
        help="Resource types to search (e.g., --resources emails sms transactions)"
    )
    
    args = parser.parse_args()
    
    # Map strategy
    strategy_map = {
        "semantic": ChunkingStrategy.SEMANTIC,
        "layout": ChunkingStrategy.LAYOUT,
        "hybrid": ChunkingStrategy.HYBRID,
    }
    
    # Initialize RAG pipeline with persistent storage
    logger.info(f"Initializing RAG pipeline...")
    logger.info(f"Chroma directory: {args.chroma_dir}")
    logger.info(f"Database URL: {args.db_url}")
    
    pipeline = RAGPipeline(
        chunking_strategy=strategy_map[args.strategy],
        device=args.device,
        use_persistent_storage=True,
        chroma_persist_dir=args.chroma_dir,
        sqlite_db_url=args.db_url,
    )
    
    # Query
    logger.info(f"\nQuerying: {args.query}")
    logger.info(f"Top-K: {args.top_k}")
    if args.source:
        logger.info(f"Source filter: {args.source}")
    if args.resources:
        logger.info(f"Resource types: {', '.join(args.resources)}")
    
    results = pipeline.retrieve_persistent(
        query=args.query,
        top_k=args.top_k,
        filter_source=args.source,
        resource_types=args.resources,
    )
    
    # Display results
    logger.info("=" * 80)
    logger.info(f"QUERY RESULTS ({len(results)} results)")
    logger.info("=" * 80)
    
    if not results:
        logger.warning("No results found!")
        
        # Show storage stats
        if pipeline.use_persistent_storage:
            doc_stats = pipeline.doc_store.get_stats()
            vector_stats = pipeline.vector_store.get_stats()
            
            logger.info("\nStorage Statistics:")
            logger.info(f"  Total documents: {doc_stats['total_documents']}")
            logger.info(f"  Total chunks: {vector_stats['total_chunks']}")
            
            if doc_stats['total_documents'] == 0:
                logger.info("\n💡 Tip: No documents found. Ingest documents first using:")
                logger.info("   python ingest.py --source 'doc.txt' --text 'Your content'")
    else:
        for i, result in enumerate(results, 1):
            logger.info(f"\n[{i}] Score: {result['similarity_score']:.4f}")
            logger.info(f"    Resource: {result.get('resource_type', 'unknown')}")
            logger.info(f"    Source: {result['metadata']['source']}")
            logger.info(f"    Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}")
            
            if args.show_text:
                text = result['text']
                # Truncate if too long
                if len(text) > 200:
                    text = text[:200] + "..."
                logger.info(f"    Text: {text}")
            else:
                # Show preview
                preview = result['text'][:100]
                if len(result['text']) > 100:
                    preview += "..."
                logger.info(f"    Preview: {preview}")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
