"""
RAG (Retrieval-Augmented Generation) Pipeline

Integrates semantic chunking, layout chunking, and embedding-based retrieval
to provide a complete document processing and retrieval system.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from .semantic_chunker import SemanticChunker
from .layout_chunker import LayoutChunker
from .retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Chunking strategy options."""
    SEMANTIC = "semantic"
    LAYOUT = "layout"
    HYBRID = "hybrid"


class RAGPipeline:
    """
    Complete RAG pipeline combining chunking and retrieval.
    Supports multiple chunking strategies and embedding-based retrieval.
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        semantic_model: str = "minishlab/potion-base-8M",
        semantic_threshold: float = 0.75,
        semantic_chunk_size: int = 1536,
        semantic_window: int = 3,
        layout_min_words: int = 30,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunking_strategy: Which chunking strategy to use (SEMANTIC, LAYOUT, HYBRID)
            semantic_model: Model name for semantic chunking
            semantic_threshold: Similarity threshold for semantic chunking
            semantic_chunk_size: Chunk size for semantic chunking (tokens)
            semantic_window: Similarity window for semantic chunking (sentences)
            layout_min_words: Minimum words for layout chunking
            embedding_model: Model for embedding generation
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        self.chunking_strategy = chunking_strategy
        self.device = device

        # Initialize chunkers based on strategy
        if chunking_strategy in (ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID):
            self.semantic_chunker = SemanticChunker(
                model_name=semantic_model,
                similarity_threshold=semantic_threshold,
                chunk_size=semantic_chunk_size,
                similarity_window=semantic_window,
                device=device,
            )
        else:
            self.semantic_chunker = None

        if chunking_strategy in (ChunkingStrategy.LAYOUT, ChunkingStrategy.HYBRID):
            self.layout_chunker = LayoutChunker(min_chunk_length=layout_min_words)
        else:
            self.layout_chunker = None

        # Initialize retriever
        self.retriever = EmbeddingRetriever(model_name=embedding_model, device=device)

        # Store processed documents
        self.documents = {}  # source -> document content
        self.processed_chunks = []  # all chunks across documents

        logger.info(
            f"RAGPipeline initialized with strategy={chunking_strategy.value}"
        )

    def add_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a document to the pipeline for processing.

        Args:
            content: Document content
            source: Document identifier (e.g., filename, URL)
            metadata: Additional metadata about the document

        Returns:
            Number of chunks created
        """
        if not content or not content.strip():
            logger.warning(f"Empty content for document: {source}")
            return 0

        # Store document
        self.documents[source] = {
            "content": content,
            "metadata": metadata or {},
        }

        # Chunk document based on strategy
        chunks = self._chunk_document(content, source)

        # Add chunks to retriever
        self.retriever.add_chunks(chunks)
        self.processed_chunks.extend(chunks)

        logger.info(f"Added document '{source}' with {len(chunks)} chunks")
        return len(chunks)

    def _chunk_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Chunk a document based on the selected strategy.

        Args:
            content: Document content
            source: Document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []

        if self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            chunks = self.semantic_chunker.chunk_with_metadata(content, source)

        elif self.chunking_strategy == ChunkingStrategy.LAYOUT:
            chunks = self.layout_chunker.chunk_with_metadata(content, source)

        elif self.chunking_strategy == ChunkingStrategy.HYBRID:
            # Apply layout chunking first, then semantic chunking on each layout chunk
            layout_chunks = self.layout_chunker.chunk_with_metadata(content, source)

            for layout_chunk in layout_chunks:
                # Apply semantic chunking to each layout chunk
                semantic_chunks = self.semantic_chunker.chunk_with_metadata(
                    layout_chunk["text"],
                    source=f"{source}_layout_{layout_chunk['chunk_index']}"
                )
                chunks.extend(semantic_chunks)

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
        include_context: bool = False,
        context_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            include_context: Whether to include adjacent chunks as context
            context_window: Number of adjacent chunks to include

        Returns:
            List of result dictionaries with chunks and scores
        """
        if not self.processed_chunks:
            logger.warning("No documents have been added to the pipeline")
            return []

        if include_context:
            results = self.retriever.retrieve_with_context(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                context_window=context_window,
            )
        else:
            raw_results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            results = [
                {
                    "chunk": chunk,
                    "similarity_score": score,
                    "context": [],
                }
                for chunk, score in raw_results
            ]

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
    ) -> List[tuple]:
        """
        Retrieve chunks with similarity scores.

        Args:
            query: Query text
            top_k: Number of top results
            similarity_threshold: Minimum similarity score

        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        return self.retriever.retrieve(query, top_k, similarity_threshold)

    def get_chunks_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific document source.

        Args:
            source: Document source identifier

        Returns:
            List of chunks from that source
        """
        return [chunk for chunk in self.processed_chunks if chunk.get("source") == source]

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "chunking_strategy": self.chunking_strategy.value,
            "total_documents": len(self.documents),
            "total_chunks": len(self.processed_chunks),
            "retriever_stats": self.retriever.get_stats(),
        }

    def clear(self) -> None:
        """Clear all documents and chunks from the pipeline."""
        self.documents = {}
        self.processed_chunks = []
        self.retriever.clear()
        logger.info("RAG pipeline cleared")

    def export_chunks(self) -> List[Dict[str, Any]]:
        """
        Export all chunks for external use.

        Returns:
            List of all chunks
        """
        return self.processed_chunks.copy()

    def import_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Import pre-processed chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        self.processed_chunks = chunks
        self.retriever.add_chunks(chunks)
        logger.info(f"Imported {len(chunks)} chunks")
