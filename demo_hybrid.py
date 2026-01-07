"""
Demo: Hybrid Retrieval with BM25 and Vector Embeddings

This script demonstrates the new hybrid retrieval capabilities that combine:
1. Vector embedding search (semantic similarity)
2. BM25 keyword search (lexical matching)

The system supports three retrieval modes:
- SEMANTIC: Vector embeddings only
- BM25: Keyword search only
- HYBRID: Combined approach using weighted fusion or RRF
"""

import logging
from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Sample documents
documents = {
    "ai_overview.txt": """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning uses neural networks with multiple layers to process information.
    Natural Language Processing (NLP) allows computers to understand human language.
    """,
    
    "ml_basics.txt": """
    Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.
    Supervised learning requires labeled training data to make predictions.
    Unsupervised learning finds patterns in unlabeled data.
    Neural networks are inspired by biological neurons in the brain.
    """,
    
    "python_guide.txt": """
    Python is a high-level programming language known for its simplicity.
    Python supports object-oriented, functional, and procedural programming paradigms.
    Popular Python libraries include NumPy, Pandas, and TensorFlow.
    Python is widely used in data science and machine learning applications.
    """
}


def demo_semantic_only():
    """Demo: Semantic retrieval using vector embeddings only."""
    print("\n" + "="*80)
    print("DEMO 1: SEMANTIC RETRIEVAL (Vector Embeddings Only)")
    print("="*80)
    
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        retrieval_mode=RetrievalMode.SEMANTIC,
        device="cpu"
    )
    
    # Add documents
    for source, content in documents.items():
        pipeline.add_document(content, source)
    
    # Test query
    query = "What is neural network architecture?"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = pipeline.retrieve(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        score = result["similarity_score"]
        print(f"\n[Result {i}] Score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:200]}...")


def demo_bm25_only():
    """Demo: BM25 keyword search only."""
    print("\n" + "="*80)
    print("DEMO 2: BM25 KEYWORD SEARCH (Lexical Matching Only)")
    print("="*80)
    
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        retrieval_mode=RetrievalMode.BM25,
        device="cpu"
    )
    
    # Add documents
    for source, content in documents.items():
        pipeline.add_document(content, source)
    
    # Test query with specific keywords
    query = "Python programming language libraries"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = pipeline.retrieve(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        score = result["bm25_score"]
        print(f"\n[Result {i}] BM25 Score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:200]}...")


def demo_hybrid_weighted():
    """Demo: Hybrid retrieval with weighted score fusion."""
    print("\n" + "="*80)
    print("DEMO 3: HYBRID RETRIEVAL (Weighted Fusion)")
    print("="*80)
    
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        retrieval_mode=RetrievalMode.HYBRID,
        hybrid_semantic_weight=0.6,  # 60% weight on semantic
        hybrid_bm25_weight=0.4,       # 40% weight on BM25
        hybrid_fusion_method="weighted",
        device="cpu"
    )
    
    # Add documents
    for source, content in documents.items():
        pipeline.add_document(content, source)
    
    # Test query
    query = "neural networks machine learning"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = pipeline.retrieve(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        combined_score = result["combined_score"]
        score_details = result["score_details"]
        
        print(f"\n[Result {i}] Combined Score: {combined_score:.4f}")
        print(f"  - Semantic Score: {score_details['semantic']:.4f}")
        print(f"  - BM25 Score: {score_details['bm25']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:200]}...")


def demo_hybrid_rrf():
    """Demo: Hybrid retrieval with Reciprocal Rank Fusion."""
    print("\n" + "="*80)
    print("DEMO 4: HYBRID RETRIEVAL (Reciprocal Rank Fusion)")
    print("="*80)
    
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        retrieval_mode=RetrievalMode.HYBRID,
        hybrid_fusion_method="rrf",  # Use RRF instead of weighted
        device="cpu"
    )
    
    # Add documents
    for source, content in documents.items():
        pipeline.add_document(content, source)
    
    # Test query
    query = "supervised learning labeled data"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = pipeline.retrieve(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        combined_score = result["combined_score"]
        score_details = result["score_details"]
        
        print(f"\n[Result {i}] RRF Score: {combined_score:.4f}")
        print(f"  - Semantic Score: {score_details['semantic']:.4f} (Rank: {score_details.get('semantic_rank', 'N/A')})")
        print(f"  - BM25 Score: {score_details['bm25']:.4f} (Rank: {score_details.get('bm25_rank', 'N/A')})")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:200]}...")


def demo_comparison():
    """Demo: Compare all three retrieval modes on the same query."""
    print("\n" + "="*80)
    print("DEMO 5: COMPARISON OF ALL RETRIEVAL MODES")
    print("="*80)
    
    query = "Python machine learning"
    print(f"\nQuery: {query}")
    
    # Create pipelines for each mode
    modes = {
        "Semantic": RetrievalMode.SEMANTIC,
        "BM25": RetrievalMode.BM25,
        "Hybrid": RetrievalMode.HYBRID
    }
    
    for mode_name, mode in modes.items():
        print(f"\n--- {mode_name} Mode ---")
        
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            retrieval_mode=mode,
            device="cpu"
        )
        
        # Add documents
        for source, content in documents.items():
            pipeline.add_document(content, source)
        
        # Get top result
        results = pipeline.retrieve(query, top_k=1)
        
        if results:
            result = results[0]
            chunk = result["chunk"]
            print(f"Top Result Source: {chunk['source']}")
            print(f"Text: {chunk['text'][:150]}...")
            
            # Print score based on mode
            if mode == RetrievalMode.SEMANTIC:
                print(f"Similarity Score: {result['similarity_score']:.4f}")
            elif mode == RetrievalMode.BM25:
                print(f"BM25 Score: {result['bm25_score']:.4f}")
            else:
                print(f"Combined Score: {result['combined_score']:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL SYSTEM DEMO")
    print("Combining Vector Embeddings and BM25 Keyword Search")
    print("="*80)
    
    # Run all demos
    demo_semantic_only()
    demo_bm25_only()
    demo_hybrid_weighted()
    demo_hybrid_rrf()
    demo_comparison()
    
    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80)
