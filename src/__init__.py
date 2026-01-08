"""
FYP RAG - Retrieval-Augmented Generation with Advanced Chunking
"""

from .semantic_chunker import SemanticChunker
from .layout_chunker import LayoutChunker
from .retriever import EmbeddingRetriever
from .bm25_retriever import BM25Retriever
from .rag_pipeline import RAGPipeline, ChunkingStrategy
from .vector_store import ChromaVectorStore
from .doc_store import SQLiteDocStore

__version__ = "0.1.0"
__all__ = [
    "SemanticChunker",
    "LayoutChunker",
    "EmbeddingRetriever",
    "BM25Retriever",
    "RAGPipeline",
    "ChunkingStrategy",
    "ChromaVectorStore",
    "SQLiteDocStore",
]
