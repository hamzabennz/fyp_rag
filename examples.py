"""
Example usage of the RAG pipeline
"""

from src.rag_pipeline import RAGPipeline, ChunkingStrategy


def example_semantic_chunking():
    """Example using semantic chunking strategy."""
    print("\n=== Semantic Chunking Example ===\n")

    # Initialize pipeline with semantic chunking
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device="cpu"  # Use 'mps' on macOS, 'cuda' on GPU
    )

    # Example document
    document = """
    Machine Learning and Natural Language Processing

    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can analyze data and make predictions.

    Natural Language Processing (NLP) is a branch of AI that helps computers
    understand, interpret, and generate human language. NLP combines computational
    linguistics with machine learning to process text and speech data.

    Key Applications

    Machine learning and NLP have numerous applications in real-world scenarios.
    These include sentiment analysis, machine translation, question answering systems,
    and chatbots. Companies use these technologies to improve customer service
    and automate business processes.

    Text summarization is another important application where NLP algorithms
    condense long documents into shorter, meaningful summaries while preserving
    the main information and key points.
    """

    # Add document to pipeline
    num_chunks = pipeline.add_document(document, source="ml_nlp_article.txt")
    print(f"Document processed into {num_chunks} chunks\n")

    # Retrieve relevant chunks for a query
    query = "What are the applications of machine learning and NLP?"
    results = pipeline.retrieve(query, top_k=3, include_context=False)

    print(f"Query: {query}\n")
    print("Top Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Chunk: {result['chunk']['text'][:200]}...")


def example_layout_chunking():
    """Example using layout-based chunking strategy."""
    print("\n=== Layout Chunking Example ===\n")

    # Initialize pipeline with layout chunking
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.LAYOUT,
        layout_min_words=30,
        device="cpu"
    )

    # Example document with structure
    document = """# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) combines retrieval and generation capabilities.
This approach allows systems to access external knowledge bases for more accurate responses.

## Key Components

### Document Retrieval
Document retrieval is the first step in RAG systems. It involves finding relevant
documents or chunks that match the user's query using similarity search.

### Generation
After retrieval, the system generates responses based on both the query and
the retrieved context. This improves answer accuracy and relevance.

## Benefits

RAG systems provide several advantages:
- Better accuracy through external knowledge
- Reduced hallucination in language models
- Lower computational costs compared to fine-tuning
- Easy knowledge base updates without retraining
"""

    # Add document
    num_chunks = pipeline.add_document(document, source="rag_guide.md")
    print(f"Document processed into {num_chunks} layout-based chunks\n")

    # Retrieve
    query = "What are the benefits of RAG systems?"
    results = pipeline.retrieve(query, top_k=2)

    print(f"Query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['similarity_score']:.4f}")
        print(f"   Text: {result['chunk']['text'][:150]}...\n")


def example_hybrid_chunking():
    """Example using hybrid chunking strategy (layout + semantic)."""
    print("\n=== Hybrid Chunking Example ===\n")

    # Initialize with hybrid strategy
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.HYBRID,
        device="cpu"
    )

    document = """
    Deep Learning Fundamentals

    Deep learning is a subset of machine learning inspired by biological neural networks.
    It uses artificial neural networks with multiple layers to process and learn from data.

    Neural Networks Basics
    A neural network consists of interconnected nodes called neurons. These neurons are
    organized in layers: input, hidden, and output layers. Each connection has a weight
    that is adjusted during training.

    Applications of Deep Learning
    Deep learning has revolutionized many fields. In computer vision, it powers image
    recognition and object detection systems. In natural language processing, transformers
    like BERT and GPT have achieved state-of-the-art results in various tasks.

    The success of deep learning stems from its ability to learn hierarchical representations
    of data, automatically discovering patterns and features without manual engineering.
    """

    num_chunks = pipeline.add_document(document, source="deep_learning.txt")
    print(f"Document chunked with hybrid strategy: {num_chunks} chunks\n")

    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline Statistics:")
    print(f"  Strategy: {stats['chunking_strategy']}")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Chunks: {stats['total_chunks']}\n")

    # Retrieve with context
    query = "How do neural networks work?"
    results = pipeline.retrieve(query, top_k=2, include_context=True, context_window=1)

    print(f"Query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity_score']:.4f}")
        print(f"   Main Chunk: {result['chunk']['text'][:150]}...")
        if result['context']:
            print(f"   Context chunks available: {len(result['context'])}")
        print()


def example_multiple_documents():
    """Example with multiple documents."""
    print("\n=== Multiple Documents Example ===\n")

    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device="cpu"
    )

    # Add multiple documents
    documents = {
        "python_basics.txt": "Python is a high-level programming language known for its simplicity...",
        "java_guide.txt": "Java is an object-oriented programming language used for building applications...",
        "javascript_intro.txt": "JavaScript is a versatile language that powers web development...",
    }

    for source, content in documents.items():
        pipeline.add_document(content, source=source)

    # Query across all documents
    query = "What makes Python unique?"
    results = pipeline.retrieve(query, top_k=2)

    print(f"Query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Source: {result['chunk']['source']}")
        print(f"   Score: {result['similarity_score']:.4f}")
        print()

    # Get chunks from specific source
    python_chunks = pipeline.get_chunks_by_source("python_basics.txt")
    print(f"Chunks from python_basics.txt: {len(python_chunks)}")


if __name__ == "__main__":
    print("RAG Pipeline Examples")
    print("=" * 50)

    try:
        example_semantic_chunking()
    except ImportError as e:
        print(f"Note: Semantic chunking requires CHONKIE: {e}")

    try:
        example_layout_chunking()
    except Exception as e:
        print(f"Error in layout chunking example: {e}")

    try:
        example_hybrid_chunking()
    except Exception as e:
        print(f"Error in hybrid chunking example: {e}")

    try:
        example_multiple_documents()
    except Exception as e:
        print(f"Error in multiple documents example: {e}")
