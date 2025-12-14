"""
Test Persistent Storage

Quick test to verify ChromaDB + SQLite integration.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, ChunkingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_persistent_storage():
    """Test the persistent storage functionality."""
    
    logger.info("=" * 80)
    logger.info("Testing Persistent Storage (ChromaDB + SQLite)")
    logger.info("=" * 80)
    
    # Sample documents
    doc1 = """
    Machine Learning Fundamentals
    
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    
    Key Concepts:
    - Supervised Learning: Learning from labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through trial and error
    """
    
    doc2 = """
    Deep Learning Overview
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    These networks can learn complex patterns and representations from large amounts of data.
    
    Applications include:
    - Computer Vision: Image recognition, object detection
    - Natural Language Processing: Text analysis, machine translation
    - Speech Recognition: Voice assistants, transcription
    """
    
    # Initialize pipeline with persistent storage
    logger.info("\n1. Initializing RAG Pipeline with persistent storage...")
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device="cuda",  # Use your available device
        use_persistent_storage=True,
        chroma_persist_dir="./data/chroma",
        sqlite_db_url="sqlite:///./data/rag_docs.db",
    )
    
    # Ingest documents
    logger.info("\n2. Ingesting documents...")
    num_chunks1 = pipeline.add_document(doc1, source="ml_fundamentals.txt")
    logger.info(f"   Document 1: {num_chunks1} chunks")
    
    num_chunks2 = pipeline.add_document(doc2, source="deep_learning.txt")
    logger.info(f"   Document 2: {num_chunks2} chunks")
    
    # Show storage stats
    if pipeline.use_persistent_storage:
        doc_stats = pipeline.doc_store.get_stats()
        vector_stats = pipeline.vector_store.get_stats()
        
        logger.info("\n3. Storage Statistics:")
        logger.info(f"   Documents in DB: {doc_stats['total_documents']}")
        logger.info(f"   Chunks in Vector Store: {vector_stats['total_chunks']}")
    
    # Test retrieval from persistent storage
    logger.info("\n4. Testing retrieval from persistent storage...")
    
    queries = [
        "What is supervised learning?",
        "Tell me about deep learning applications",
    ]
    
    for query in queries:
        logger.info(f"\n   Query: {query}")
        results = pipeline.retrieve_persistent(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"   [{i}] Score: {result['similarity_score']:.4f} | Source: {result['metadata']['source']}")
            preview = result['text'][:100].replace('\n', ' ')
            logger.info(f"       Preview: {preview}...")
    
    # Test persistence by creating new pipeline instance
    logger.info("\n5. Testing persistence - creating new pipeline instance...")
    new_pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device="cuda",
        use_persistent_storage=True,
        chroma_persist_dir="./data/chroma",
        sqlite_db_url="sqlite:///./data/rag_docs.db",
    )
    
    # Verify data persisted
    if new_pipeline.use_persistent_storage:
        doc_stats = new_pipeline.doc_store.get_stats()
        vector_stats = new_pipeline.vector_store.get_stats()
        
        logger.info(f"   Documents in new instance: {doc_stats['total_documents']}")
        logger.info(f"   Chunks in new instance: {vector_stats['total_chunks']}")
        
        # Query from new instance
        query = "What is machine learning?"
        results = new_pipeline.retrieve_persistent(query, top_k=2)
        logger.info(f"\n   Query from new instance: {query}")
        logger.info(f"   Retrieved {len(results)} results ✓")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Persistent Storage Test Complete!")
    logger.info("=" * 80)
    
    # List all documents
    if pipeline.use_persistent_storage:
        docs = pipeline.doc_store.list_documents()
        logger.info("\nDocuments in database:")
        for doc in docs:
            logger.info(f"  - {doc['source']}: {doc['chunk_count']} chunks")


if __name__ == "__main__":
    test_persistent_storage()
