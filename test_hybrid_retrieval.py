"""
Simple test script to verify BM25 and Hybrid Retrieval functionality
This can be run after installing dependencies: pip install rank-bm25
"""

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.bm25_retriever import BM25Retriever
        print("✓ BM25Retriever imported successfully")
    except Exception as e:
        print(f"✗ BM25Retriever import failed: {e}")
        return False
    
    try:
        from src.hybrid_retriever import HybridRetriever
        print("✓ HybridRetriever imported successfully")
    except Exception as e:
        print(f"✗ HybridRetriever import failed: {e}")
        return False
    
    try:
        from src.rag_pipeline import RAGPipeline, RetrievalMode
        print("✓ RAGPipeline with RetrievalMode imported successfully")
    except Exception as e:
        print(f"✗ RAGPipeline import failed: {e}")
        return False
    
    return True


def test_bm25_retriever():
    """Test BM25Retriever functionality."""
    print("\nTesting BM25Retriever...")
    
    try:
        from src.bm25_retriever import BM25Retriever
        
        # Create retriever
        retriever = BM25Retriever(k1=1.5, b=0.75)
        
        # Add test chunks
        chunks = [
            {"chunk_id": "1", "text": "Machine learning is a subset of artificial intelligence"},
            {"chunk_id": "2", "text": "Python is a popular programming language"},
            {"chunk_id": "3", "text": "Neural networks are used in deep learning"},
        ]
        retriever.add_chunks(chunks)
        
        # Test retrieval
        results = retriever.retrieve("machine learning", top_k=2)
        
        if len(results) > 0:
            print(f"✓ BM25 retrieval successful: found {len(results)} results")
            return True
        else:
            print("✗ BM25 retrieval returned no results")
            return False
            
    except Exception as e:
        print(f"✗ BM25Retriever test failed: {e}")
        return False


def test_hybrid_retriever():
    """Test HybridRetriever functionality."""
    print("\nTesting HybridRetriever...")
    
    try:
        from src.hybrid_retriever import HybridRetriever
        
        # Create retriever
        retriever = HybridRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            fusion_method="weighted"
        )
        
        # Add test chunks
        chunks = [
            {"chunk_id": "1", "text": "Machine learning is a subset of artificial intelligence"},
            {"chunk_id": "2", "text": "Python is a popular programming language"},
            {"chunk_id": "3", "text": "Neural networks are used in deep learning"},
        ]
        retriever.add_chunks(chunks)
        
        # Test retrieval
        results = retriever.retrieve("machine learning", top_k=2)
        
        if len(results) > 0:
            chunk, score, details = results[0]
            print(f"✓ Hybrid retrieval successful: found {len(results)} results")
            print(f"  First result score details: {details}")
            return True
        else:
            print("✗ Hybrid retrieval returned no results")
            return False
            
    except Exception as e:
        print(f"✗ HybridRetriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline_modes():
    """Test RAGPipeline with different retrieval modes."""
    print("\nTesting RAGPipeline with different modes...")
    
    try:
        from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode
        
        test_content = """
        Machine learning is a subset of artificial intelligence.
        Python is widely used for data science and machine learning.
        Neural networks are the foundation of deep learning.
        """
        
        # Test each mode
        modes = [RetrievalMode.SEMANTIC, RetrievalMode.BM25, RetrievalMode.HYBRID]
        
        for mode in modes:
            print(f"\n  Testing {mode.value} mode...")
            
            pipeline = RAGPipeline(
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                retrieval_mode=mode,
                device="cpu"
            )
            
            # Add document
            chunks_count = pipeline.add_document(test_content, "test_doc.txt")
            print(f"    - Added document with {chunks_count} chunks")
            
            # Test retrieval
            results = pipeline.retrieve("machine learning", top_k=2)
            
            if len(results) > 0:
                print(f"    ✓ {mode.value} mode working: {len(results)} results")
            else:
                print(f"    ✗ {mode.value} mode returned no results")
                return False
        
        print("\n✓ All RAGPipeline modes working successfully")
        return True
        
    except Exception as e:
        print(f"✗ RAGPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("HYBRID RETRIEVAL SYSTEM TEST")
    print("="*70)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_bm25_retriever()
    all_passed &= test_hybrid_retriever()
    all_passed &= test_rag_pipeline_modes()
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        print("\nMake sure you have installed: pip install rank-bm25")
    print("="*70)
