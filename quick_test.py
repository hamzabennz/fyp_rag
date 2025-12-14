#!/usr/bin/env python3
"""
Quick test script to verify RAG pipeline is working correctly
"""

import logging
from src.rag_pipeline import RAGPipeline, ChunkingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_pipeline():
    """Test the RAG pipeline with a simple example."""
    
    print("=" * 80)
    print("🦛 Testing RAG Pipeline with CUDA")
    print("=" * 80)
    
    # Test document
    document = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can analyze data and make predictions.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Each type serves
    different purposes and uses different approaches.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data.
    It tries to find patterns in the data to make predictions on new, unseen data.
    Common applications include classification and regression tasks.
    
    Applications
    
    Machine learning powers many modern applications including recommendation
    systems, natural language processing, computer vision, and predictive analytics.
    """
    
    print("\n1️⃣ Testing Layout Chunking...")
    print("-" * 80)
    try:
        pipeline_layout = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cuda"
        )
        
        num_chunks = pipeline_layout.add_document(document, source="test_layout.txt")
        print(f"✅ Layout chunking: Created {num_chunks} chunks")
        
        # Test retrieval
        results = pipeline_layout.retrieve("What is supervised learning?", top_k=2)
        print(f"✅ Retrieval: Found {len(results)} relevant chunks")
        if results:
            print(f"   Top result score: {results[0]['similarity_score']:.4f}")
    except Exception as e:
        print(f"❌ Layout chunking failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2️⃣ Testing Semantic Chunking...")
    print("-" * 80)
    try:
        pipeline_semantic = RAGPipeline(
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            device="cuda"
        )
        
        num_chunks = pipeline_semantic.add_document(document, source="test_semantic.txt")
        print(f"✅ Semantic chunking: Created {num_chunks} chunks")
        
        # Test retrieval
        results = pipeline_semantic.retrieve("What are machine learning applications?", top_k=2)
        print(f"✅ Retrieval: Found {len(results)} relevant chunks")
        if results:
            print(f"   Top result score: {results[0]['similarity_score']:.4f}")
    except Exception as e:
        print(f"❌ Semantic chunking failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3️⃣ Testing Hybrid Chunking...")
    print("-" * 80)
    try:
        pipeline_hybrid = RAGPipeline(
            chunking_strategy=ChunkingStrategy.HYBRID,
            device="cuda"
        )
        
        num_chunks = pipeline_hybrid.add_document(document, source="test_hybrid.txt")
        print(f"✅ Hybrid chunking: Created {num_chunks} chunks")
        
        # Test retrieval
        results = pipeline_hybrid.retrieve("Tell me about machine learning types", top_k=2)
        print(f"✅ Retrieval: Found {len(results)} relevant chunks")
        if results:
            print(f"   Top result score: {results[0]['similarity_score']:.4f}")
    except Exception as e:
        print(f"❌ Hybrid chunking failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✨ Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_pipeline()
