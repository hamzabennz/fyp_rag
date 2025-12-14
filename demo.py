"""
Main demo script for RAG pipeline
"""

import logging
from src.rag_pipeline import RAGPipeline, ChunkingStrategy
from config import DEFAULT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_rag_system():
    """Demonstrate the RAG system with example documents."""
    
    logger.info("=" * 80)
    logger.info("FYP RAG System Demo - Semantic and Layout Chunking")
    logger.info("=" * 80)
    
    # Example documents about machine learning
    documents = {
        "ml_intro.txt": """
            Introduction to Machine Learning

            Machine learning is a subset of artificial intelligence that focuses on
            the development of algorithms and statistical models that enable computer
            systems to improve their performance on tasks through experience without
            being explicitly programmed.

            Core Concepts
            Machine learning systems learn from data by identifying patterns and
            relationships. The quality of learning depends heavily on the quality
            and quantity of training data available. Different algorithms are
            suited for different types of problems.

            Applications
            Machine learning powers many modern applications including:
            - Recommendation systems (Netflix, Amazon)
            - Natural language processing (ChatGPT, translation)
            - Computer vision (facial recognition, autonomous vehicles)
            - Predictive analytics (fraud detection, forecasting)
        """,
        
        "deep_learning.txt": """
            Deep Learning Fundamentals

            Deep learning is a specialized branch of machine learning inspired by
            the structure and function of biological neural networks. It uses
            artificial neural networks with multiple layers (hence "deep") to
            progressively extract higher-level features from raw input.

            Neural Network Architecture
            A neural network consists of:
            - Input layer: Receives raw data
            - Hidden layers: Process and transform data
            - Output layer: Produces predictions

            Key Advantages
            Deep learning excels at:
            - Learning complex non-linear relationships
            - Automatic feature extraction
            - Handling high-dimensional data
            - End-to-end learning from raw inputs
        """,
        
        "nlp_guide.txt": """
            Natural Language Processing

            Natural Language Processing (NLP) is a branch of artificial intelligence
            that helps computers understand, interpret, and generate human language
            in a meaningful and useful way.

            NLP Techniques
            Modern NLP builds on several key techniques:
            - Tokenization: Breaking text into words and sentences
            - Embeddings: Converting text to numerical vectors
            - Transformers: Neural architectures for sequence processing

            Real-world Applications
            NLP technology enables:
            - Sentiment analysis for social media monitoring
            - Machine translation between languages
            - Question answering systems
            - Chatbots and conversational AI
        """,
    }
    
    # Initialize pipeline with semantic chunking
    logger.info("\nInitializing RAG Pipeline with Semantic Chunking...")
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device=DEFAULT_CONFIG.device
    )
    
    # Add documents
    logger.info("\nAdding documents to pipeline...")
    total_chunks = 0
    for source, content in documents.items():
        num_chunks = pipeline.add_document(content.strip(), source=source)
        total_chunks += num_chunks
        logger.info(f"  {source}: {num_chunks} chunks")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    logger.info(f"\nPipeline Statistics:")
    logger.info(f"  Total documents: {stats['total_documents']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Embedding dimension: {stats['retriever_stats']['embedding_dimension']}")
    
    # Example queries
    queries = [
        "What are the applications of machine learning?",
        "How do neural networks work?",
        "What is natural language processing used for?",
        "Tell me about deep learning",
        "What are transformers in NLP?",
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL DEMONSTRATIONS")
    logger.info("=" * 80)
    
    for query in queries:
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'─' * 80}")
        
        # Retrieve relevant chunks
        results = pipeline.retrieve(
            query,
            top_k=3,
            include_context=False
        )
        
        if not results:
            logger.info("No results found")
            continue
        
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            score = result['similarity_score']
            
            logger.info(f"\n  {i}. Source: {chunk['source']}")
            logger.info(f"     Similarity Score: {score:.4f}")
            logger.info(f"     Text Preview: {chunk['text'][:150]}...")
    
    # Demonstrate layout chunking
    logger.info("\n" + "=" * 80)
    logger.info("LAYOUT CHUNKING DEMONSTRATION")
    logger.info("=" * 80)
    
    layout_pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.LAYOUT,
        device=DEFAULT_CONFIG.device
    )
    
    structured_doc = """# Machine Learning Overview

    Fundamentals Section
    Machine learning is a powerful approach to solving problems.
    It learns patterns from data automatically.

    Advanced Techniques
    Deep learning uses neural networks with multiple layers.
    These networks can learn complex non-linear relationships.

    Real-world Benefits:
    - Improved accuracy over traditional methods
    - Reduced manual feature engineering
    - Better generalization to new data
    - Scalable to large datasets
    """
    
    logger.info("\nProcessing structured document...")
    num_chunks = layout_pipeline.add_document(structured_doc, source="structured.txt")
    logger.info(f"Created {num_chunks} layout-based chunks")
    
    chunks = layout_pipeline.get_chunks_by_source("structured.txt")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"\n  Chunk {i}:")
        logger.info(f"    Words: {chunk['word_count']}")
        logger.info(f"    Is Header: {chunk.get('is_header', False)}")
        logger.info(f"    Text: {chunk['text'][:100]}...")
    
    # Export chunks example
    logger.info("\n" + "=" * 80)
    logger.info("CHUNK EXPORT/IMPORT DEMONSTRATION")
    logger.info("=" * 80)
    
    exported_chunks = pipeline.export_chunks()
    logger.info(f"\nExported {len(exported_chunks)} chunks")
    
    # Create new pipeline and import
    new_pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        device=DEFAULT_CONFIG.device
    )
    new_pipeline.import_chunks(exported_chunks)
    logger.info(f"Imported {len(new_pipeline.processed_chunks)} chunks into new pipeline")
    
    # Test query on imported pipeline
    query = "What is machine learning?"
    results = new_pipeline.retrieve(query, top_k=2)
    logger.info(f"\nQuery on imported chunks: '{query}'")
    logger.info(f"Found {len(results)} results")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo Complete!")
    logger.info("=" * 80)


def demo_hybrid_chunking():
    """Demonstrate hybrid chunking strategy."""
    
    logger.info("\n" + "=" * 80)
    logger.info("HYBRID CHUNKING DEMONSTRATION")
    logger.info("=" * 80)
    
    pipeline = RAGPipeline(
        chunking_strategy=ChunkingStrategy.HYBRID,
        device=DEFAULT_CONFIG.device
    )
    
    document = """
    # Artificial Intelligence

    Introduction
    Artificial intelligence has become a transformative technology in modern society.
    It encompasses machine learning, deep learning, and many other approaches.

    ## Machine Learning
    Machine learning enables systems to learn from data and improve performance
    without explicit programming. It powers many real-world applications.

    ## Deep Learning
    Deep learning uses neural networks to automatically discover representations
    needed for detection or classification. It has revolutionized computer vision
    and natural language processing.

    ## Applications
    AI applications span multiple domains:
    - Healthcare: Diagnostics and drug discovery
    - Finance: Risk assessment and trading
    - Transportation: Autonomous vehicles
    """
    
    logger.info("\nProcessing document with hybrid strategy...")
    num_chunks = pipeline.add_document(document.strip(), source="hybrid_demo.txt")
    logger.info(f"Created {num_chunks} hybrid chunks")
    
    # Query
    query = "Tell me about deep learning applications"
    results = pipeline.retrieve(query, top_k=2)
    
    logger.info(f"\nQuery: {query}")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Score: {result['similarity_score']:.4f}")
        logger.info(f"    Text: {result['chunk']['text'][:120]}...")


if __name__ == "__main__":
    try:
        demo_rag_system()
        demo_hybrid_chunking()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
