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
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .vector_store import ChromaVectorStore
from .doc_store import SQLiteDocStore

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Chunking strategy options."""
    SEMANTIC = "semantic"
    LAYOUT = "layout"
    HYBRID = "hybrid"


class RetrievalMode(Enum):
    """Retrieval mode options."""
    SEMANTIC = "semantic"  # Vector embedding only
    BM25 = "bm25"  # Keyword search only
    HYBRID = "hybrid"  # Combined semantic + BM25


class RAGPipeline:
    """
    Complete RAG pipeline combining chunking and retrieval.
    Supports multiple chunking strategies and retrieval modes:
    - Semantic (vector embedding)
    - BM25 (keyword search)
    - Hybrid (combined semantic + BM25)
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        semantic_model: str = "minishlab/potion-base-8M",
        semantic_threshold: float = 0.75,
        semantic_chunk_size: int = 1536,
        semantic_window: int = 3,
        semantic_min_sentences: int = 2,
        layout_min_words: int = 30,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        hybrid_semantic_weight: float = 0.5,
        hybrid_bm25_weight: float = 0.5,
        hybrid_fusion_method: str = "weighted",
        use_persistent_storage: bool = False,
        chroma_persist_dir: str = "./data/chroma",
        sqlite_db_url: str = "sqlite:///./data/rag_docs.db",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunking_strategy: Which chunking strategy to use (SEMANTIC, LAYOUT, HYBRID)
            retrieval_mode: Which retrieval mode to use (SEMANTIC, BM25, HYBRID)
            semantic_model: Model name for semantic chunking
            semantic_threshold: Similarity threshold for semantic chunking
            semantic_chunk_size: Chunk size for semantic chunking (tokens)
            semantic_window: Similarity window for semantic chunking (sentences)
            semantic_min_sentences: Minimum sentences per chunk (prevents single-sentence chunks)
            layout_min_words: Minimum words for layout chunking
            embedding_model: Model for embedding generation
            device: Device to run models on ('cpu', 'cuda', 'mps')
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (length normalization)
            hybrid_semantic_weight: Weight for semantic scores in hybrid mode
            hybrid_bm25_weight: Weight for BM25 scores in hybrid mode
            hybrid_fusion_method: Fusion method for hybrid mode ("weighted" or "rrf")
            use_persistent_storage: Enable persistent storage (ChromaDB + SQLite)
            chroma_persist_dir: Directory for ChromaDB persistent storage
            sqlite_db_url: SQLite database URL for document metadata
        """
        self.chunking_strategy = chunking_strategy
        self.retrieval_mode = retrieval_mode
        self.device = device

        # Initialize chunkers based on strategy
        if chunking_strategy in (ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID):
            self.semantic_chunker = SemanticChunker(
                embedding_model=semantic_model,
                threshold=semantic_threshold,
                chunk_size=semantic_chunk_size,
                similarity_window=semantic_window,
                min_sentences_per_chunk=semantic_min_sentences,
            )
        else:
            self.semantic_chunker = None

        if chunking_strategy in (ChunkingStrategy.LAYOUT, ChunkingStrategy.HYBRID):
            self.layout_chunker = LayoutChunker(min_chunk_length=layout_min_words)
        else:
            self.layout_chunker = None

        # Initialize retriever(s) based on retrieval mode
        if retrieval_mode == RetrievalMode.SEMANTIC:
            self.retriever = EmbeddingRetriever(model_name=embedding_model, device=device)
            self.bm25_retriever = None
            self.hybrid_retriever = None
            logger.info("Using SEMANTIC retrieval mode (vector embeddings only)")
        
        elif retrieval_mode == RetrievalMode.BM25:
            self.retriever = None
            self.bm25_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
            self.hybrid_retriever = None
            logger.info("Using BM25 retrieval mode (keyword search only)")
        
        elif retrieval_mode == RetrievalMode.HYBRID:
            self.retriever = None
            self.bm25_retriever = None
            self.hybrid_retriever = HybridRetriever(
                embedding_model=embedding_model,
                device=device,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                semantic_weight=hybrid_semantic_weight,
                bm25_weight=hybrid_bm25_weight,
                fusion_method=hybrid_fusion_method,
            )
            logger.info(f"Using HYBRID retrieval mode (semantic + BM25, fusion={hybrid_fusion_method})")
        
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

        # Initialize persistent storage (optional)
        self.use_persistent_storage = use_persistent_storage
        if use_persistent_storage:
            self.vector_store = ChromaVectorStore(
                collection_name="rag_chunks",
                persist_directory=chroma_persist_dir,
            )
            self.doc_store = SQLiteDocStore(db_url=sqlite_db_url)
            logger.info("Persistent storage enabled (Chroma + SQLite)")
        else:
            self.vector_store = None
            self.doc_store = None
            logger.info("Using in-memory storage only")

        # Store processed documents
        self.documents = {}  # source -> document content
        self.processed_chunks = []  # all chunks across documents

        logger.info(
            f"RAGPipeline initialized: chunking={chunking_strategy.value}, "
            f"retrieval={retrieval_mode.value}"
        )

    def add_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        resource_type: str = "default",
    ) -> int:
        """
        Add a document to the pipeline for processing.

        Args:
            content: Document content
            source: Document identifier (e.g., filename, URL)
            metadata: Additional metadata about the document
            resource_type: Type of resource (e.g., 'emails', 'sms', 'transactions')

        Returns:
            Number of chunks created
        """
        if not content or not content.strip():
            logger.warning(f"Empty content for document: {source}")
            return 0

        # Check if document has changed (for persistent storage)
        if self.use_persistent_storage:
            if self.doc_store.exists(source) and not self.doc_store.has_changed(source, content):
                logger.info(f"Document '{source}' unchanged, skipping")
                doc = self.doc_store.get_document(source)
                return doc["chunk_count"]

        # Store document
        doc_metadata = metadata or {}
        doc_metadata["resource_type"] = resource_type
        
        self.documents[source] = {
            "content": content,
            "metadata": doc_metadata,
        }

        # Chunk document based on strategy
        chunks = self._chunk_document(content, source)

        # Add chunks to retriever(s) (in-memory)
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            self.retriever.add_chunks(chunks)
        elif self.retrieval_mode == RetrievalMode.BM25:
            self.bm25_retriever.add_chunks(chunks)
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            self.hybrid_retriever.add_chunks(chunks)
        
        self.processed_chunks.extend(chunks)

        # Add to persistent storage if enabled
        if self.use_persistent_storage:
            # Get embeddings (only if using semantic or hybrid mode)
            if self.retrieval_mode == RetrievalMode.SEMANTIC:
                embeddings = self.retriever.chunk_embeddings[-len(chunks):]
            elif self.retrieval_mode == RetrievalMode.HYBRID:
                embeddings = self.hybrid_retriever.embedding_retriever.chunk_embeddings[-len(chunks):]
            else:
                # For BM25-only mode, we still store embeddings if needed
                # Create a temporary embedding retriever
                from .retriever import EmbeddingRetriever
                temp_retriever = EmbeddingRetriever(device=self.device)
                temp_retriever.add_chunks(chunks)
                embeddings = temp_retriever.chunk_embeddings
            
            # Store in vector DB
            self.vector_store.add_chunks(chunks, embeddings, resource_type=resource_type)
            
            # Store metadata in doc DB
            self.doc_store.add_document(
                source=source,
                content=content,
                chunk_count=len(chunks),
                metadata=metadata,
            )

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
        Retrieve relevant chunks for a query using the configured retrieval mode.

        Args:
            query: Query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1) for semantic/hybrid mode
            include_context: Whether to include adjacent chunks as context (semantic mode only)
            context_window: Number of adjacent chunks to include

        Returns:
            List of result dictionaries with chunks and scores
        """
        if not self.processed_chunks:
            logger.warning("No documents have been added to the pipeline")
            return []

        # Route to appropriate retrieval method
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            return self._retrieve_semantic(
                query, top_k, similarity_threshold, include_context, context_window
            )
        elif self.retrieval_mode == RetrievalMode.BM25:
            return self._retrieve_bm25(query, top_k)
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            return self._retrieve_hybrid(query, top_k, similarity_threshold)
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

    def _retrieve_semantic(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
        include_context: bool = False,
        context_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Retrieve using semantic (embedding) search only."""
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
                    "retrieval_method": "semantic",
                    "context": [],
                }
                for chunk, score in raw_results
            ]

        return results

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve using BM25 keyword search only."""
        raw_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )
        results = [
            {
                "chunk": chunk,
                "bm25_score": score,
                "retrieval_method": "bm25",
                "context": [],
            }
            for chunk, score in raw_results
        ]

        return results

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve using hybrid search (semantic + BM25)."""
        raw_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        results = [
            {
                "chunk": chunk,
                "combined_score": combined_score,
                "score_details": score_details,
                "retrieval_method": "hybrid",
                "context": [],
            }
            for chunk, combined_score, score_details in raw_results
        ]

        return results

    def retrieve_persistent(
        self,
        query: str,
        top_k: int = 5,
        filter_source: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from persistent storage (ChromaDB).
        Note: This uses semantic search only, regardless of retrieval mode.

        Args:
            query: Query text
            top_k: Number of results
            filter_source: Optional source filter

        Returns:
            List of result dictionaries
        """
        if not self.use_persistent_storage:
            logger.warning("Persistent storage not enabled")
            return []

        # Get embedding model
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            embedding_model = self.retriever.model
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            embedding_model = self.hybrid_retriever.embedding_retriever.model
        else:
            # For BM25 mode, create temporary embedding model
            from .retriever import EmbeddingRetriever
            temp_retriever = EmbeddingRetriever(device=self.device)
            embedding_model = temp_retriever.model

        # Generate query embedding
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()

        # Search vector store
        filter_dict = {"source": filter_source} if filter_source else None
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
    ) -> List[tuple]:
        """
        Retrieve chunks with similarity scores.
        Note: For hybrid mode, returns tuples with score_details dict.

        Args:
            query: Query text
            top_k: Number of top results
            similarity_threshold: Minimum similarity score

        Returns:
            List of tuples (chunk_dict, score) or (chunk_dict, combined_score, score_details)
        """
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            return self.retriever.retrieve(query, top_k, similarity_threshold)
        elif self.retrieval_mode == RetrievalMode.BM25:
            return self.bm25_retriever.retrieve(query, top_k)
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            return self.hybrid_retriever.retrieve(query, top_k, similarity_threshold)

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
        stats = {
            "chunking_strategy": self.chunking_strategy.value,
            "retrieval_mode": self.retrieval_mode.value,
            "total_documents": len(self.documents),
            "total_chunks": len(self.processed_chunks),
        }

        # Add retriever-specific stats
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            stats["retriever_stats"] = self.retriever.get_stats()
        elif self.retrieval_mode == RetrievalMode.BM25:
            stats["bm25_stats"] = self.bm25_retriever.get_stats()
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            stats["hybrid_stats"] = self.hybrid_retriever.get_stats()

        return stats

    def clear(self) -> None:
        """Clear all documents and chunks from the pipeline."""
        self.documents = {}
        self.processed_chunks = []
        
        # Clear appropriate retriever(s)
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            self.retriever.clear()
        elif self.retrieval_mode == RetrievalMode.BM25:
            self.bm25_retriever.clear()
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            self.hybrid_retriever.clear()
        
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
        
        # Add to appropriate retriever(s)
        if self.retrieval_mode == RetrievalMode.SEMANTIC:
            self.retriever.add_chunks(chunks)
        elif self.retrieval_mode == RetrievalMode.BM25:
            self.bm25_retriever.add_chunks(chunks)
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            self.hybrid_retriever.add_chunks(chunks)
        
        logger.info(f"Imported {len(chunks)} chunks")
