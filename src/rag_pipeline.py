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
from .vector_store import ChromaVectorStore
from .doc_store import SQLiteDocStore

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
        semantic_min_sentences: int = 2,
        layout_min_words: int = 30,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_persistent_storage: bool = False,
        chroma_persist_dir: str = "./data/chroma",
        sqlite_db_url: str = "sqlite:///./data/rag_docs.db",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunking_strategy: Which chunking strategy to use (SEMANTIC, LAYOUT, HYBRID)
            semantic_model: Model name for semantic chunking
            semantic_threshold: Similarity threshold for semantic chunking
            semantic_chunk_size: Chunk size for semantic chunking (tokens)
            semantic_window: Similarity window for semantic chunking (sentences)
            semantic_min_sentences: Minimum sentences per chunk (prevents single-sentence chunks)
            layout_min_words: Minimum words for layout chunking
            embedding_model: Model for embedding generation
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        self.chunking_strategy = chunking_strategy
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

        # Initialize retrievers
        self.retriever = EmbeddingRetriever(model_name=embedding_model, device=device)
        self.bm25_retriever = BM25Retriever()

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
            f"RAGPipeline initialized with strategy={chunking_strategy.value}"
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
            if self.doc_store.exists(source):
                doc = self.doc_store.get_document(source)
                content_unchanged = not self.doc_store.has_changed(source, content)
                
                # Check if resource_type changed
                existing_resource_type = doc.get("metadata", {}).get("resource_type", "default")
                resource_type_unchanged = existing_resource_type == resource_type
                
                if content_unchanged and resource_type_unchanged:
                    logger.info(f"Document '{source}' unchanged, skipping")
                    return doc["chunk_count"]
                elif content_unchanged and not resource_type_unchanged:
                    logger.info(
                        f"Document '{source}' content unchanged but resource_type changed "
                        f"({existing_resource_type} -> {resource_type}), re-processing..."
                    )
                    # Delete old chunks from old collection
                    self.vector_store.delete_by_source(source, resource_type=existing_resource_type)

        # Store document
        doc_metadata = metadata or {}
        doc_metadata["resource_type"] = resource_type
        
        self.documents[source] = {
            "content": content,
            "metadata": doc_metadata,
        }

        # Chunk document based on strategy
        chunks = self._chunk_document(content, source)

        # Add chunks to retrievers (in-memory)
        self.retriever.add_chunks(chunks)
        self.bm25_retriever.add_chunks(chunks)
        self.processed_chunks.extend(chunks)

        # Add to persistent storage if enabled
        if self.use_persistent_storage:
            # Get embeddings from retriever
            embeddings = self.retriever.chunk_embeddings[-len(chunks):]
            
            # Store in vector DB with resource_type
            self.vector_store.add_chunks(chunks, embeddings, resource_type=resource_type)
            
            # Store metadata in doc DB
            self.doc_store.add_document(
                source=source,
                content=content,
                chunk_count=len(chunks),
                metadata=doc_metadata,
            )

            logger.info(f"Added document '{source}' with {len(chunks)} chunks to collection '{resource_type}'")
        else:
            logger.info(f"Added document '{source}' with {len(chunks)} chunks (in-memory only)")
        
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

    def _hybrid_rerank(
        self,
        embedding_results: List[tuple],
        bm25_results: List[tuple],
        top_k: int,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Combine and re-rank results from embedding and BM25 using Reciprocal Rank Fusion.

        Args:
            embedding_results: Results from embedding retriever (chunk, score)
            bm25_results: Results from BM25 retriever (chunk, score)
            top_k: Number of results to return
            similarity_threshold: Optional threshold for filtering

        Returns:
            Combined and re-ranked results
        """
        # Build RRF scores
        rrf_k = 60  # Standard RRF constant
        chunk_scores = {}  # chunk_id -> {chunk, embedding_score, bm25_score, rrf_score}
        
        # Process embedding results
        for rank, (chunk, score) in enumerate(embedding_results, 1):
            chunk_id = chunk.get("chunk_id", id(chunk))
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "embedding_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            chunk_scores[chunk_id]["embedding_score"] = score
            chunk_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        
        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results, 1):
            chunk_id = chunk.get("chunk_id", id(chunk))
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "embedding_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            chunk_scores[chunk_id]["bm25_score"] = score
            chunk_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Apply threshold if specified (using max of embedding and BM25 scores)
        if similarity_threshold is not None:
            sorted_chunks = [
                c for c in sorted_chunks
                if max(c["embedding_score"], c["bm25_score"]) >= similarity_threshold
            ]
        
        # Take top-k
        sorted_chunks = sorted_chunks[:top_k]
        
        # Format results
        results = []
        for item in sorted_chunks:
            results.append({
                "chunk": item["chunk"],
                "similarity_score": item["rrf_score"],
                "embedding_score": item["embedding_score"],
                "bm25_score": item["bm25_score"],
                "context": [],
                "retrieval_method": "hybrid",
            })
        
        logger.info(f"Hybrid retrieval: combined {len(embedding_results)} embedding + {len(bm25_results)} BM25 results -> {len(results)} final results")
        return results

    def _filter_chunks_by_resources(
        self,
        chunks: List[Dict[str, Any]],
        resource_types: Optional[List[str]] = None,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks by resource types and/or source.

        Args:
            chunks: List of chunks to filter
            resource_types: List of resource types to include
            filter_source: Optional source filter

        Returns:
            Filtered chunks
        """
        filtered = chunks
        
        if resource_types:
            filtered = [
                c for c in filtered
                if c.get("metadata", {}).get("resource_type", "default") in resource_types
            ]
        
        if filter_source:
            filtered = [
                c for c in filtered
                if c.get("source") == filter_source
            ]
        
        return filtered

    def _hybrid_rerank_persistent(
        self,
        embedding_results: List[Dict[str, Any]],
        bm25_results: List[tuple],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Combine and re-rank results from embedding (persistent) and BM25 using RRF.

        Args:
            embedding_results: Results from vector store (dicts)
            bm25_results: Results from BM25 retriever (chunk, score tuples)
            top_k: Number of results to return

        Returns:
            Combined and re-ranked results
        """
        rrf_k = 60
        chunk_scores = {}
        
        # Process embedding results (from ChromaDB)
        for rank, result in enumerate(embedding_results, 1):
            chunk_id = result.get("chunk_id", "")
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "result": result,
                    "embedding_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            chunk_scores[chunk_id]["embedding_score"] = result.get("similarity_score", 0.0)
            chunk_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        
        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results, 1):
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id not in chunk_scores:
                # Create result dict for BM25-only result
                chunk_scores[chunk_id] = {
                    "result": {
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "metadata": chunk,
                        "similarity_score": 0.0,
                        "resource_type": chunk.get("metadata", {}).get("resource_type", "default"),
                    },
                    "embedding_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            chunk_scores[chunk_id]["bm25_score"] = score
            chunk_scores[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)
        
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for item in sorted_chunks:
            result = item["result"].copy()
            result["similarity_score"] = item["rrf_score"]
            result["embedding_score"] = item["embedding_score"]
            result["bm25_score"] = item["bm25_score"]
            result["retrieval_method"] = "hybrid"
            results.append(result)
        
        logger.info(f"Hybrid persistent retrieval: {len(results)} results")
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
        include_context: bool = False,
        context_window: int = 1,
        retrieval_mode: str = "embedding",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            include_context: Whether to include adjacent chunks as context
            context_window: Number of adjacent chunks to include
            retrieval_mode: Retrieval mode ('embedding', 'bm25', or 'hybrid')

        Returns:
            List of result dictionaries with chunks and scores
        """
        if not self.processed_chunks:
            logger.warning("No documents have been added to the pipeline")
            return []

        # Choose retrieval method based on mode
        if retrieval_mode == "bm25":
            raw_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=similarity_threshold,
            )
            results = [
                {
                    "chunk": chunk,
                    "similarity_score": score,
                    "context": [],
                    "retrieval_method": "bm25",
                }
                for chunk, score in raw_results
            ]
        elif retrieval_mode == "hybrid":
            # Get results from both methods
            embedding_results = self.retriever.retrieve(
                query=query,
                top_k=top_k * 2,  # Get more candidates
                similarity_threshold=None,
            )
            bm25_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=top_k * 2,  # Get more candidates
                score_threshold=None,
            )
            
            # Combine and re-rank using RRF (Reciprocal Rank Fusion)
            results = self._hybrid_rerank(
                embedding_results, bm25_results, top_k, similarity_threshold
            )
        else:  # embedding mode (default)
            if include_context:
                results = self.retriever.retrieve_with_context(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    context_window=context_window,
                )
                for r in results:
                    r["retrieval_method"] = "embedding"
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
                        "retrieval_method": "embedding",
                    }
                    for chunk, score in raw_results
                ]

        return results

    def retrieve_persistent(
        self,
        query: str,
        top_k: int = 5,
        filter_source: str = None,
        resource_types: Optional[List[str]] = None,
        retrieval_mode: str = "embedding",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from persistent storage (ChromaDB).
        Can search across specific resource types or all resources.

        Args:
            query: Query text
            top_k: Number of results
            filter_source: Optional source filter
            resource_types: List of resource types to search (e.g., ['emails', 'sms']).
                          If None, searches all available collections.
            retrieval_mode: Retrieval mode ('embedding', 'bm25', or 'hybrid')

        Returns:
            List of result dictionaries
        """
        if not self.use_persistent_storage:
            logger.warning("Persistent storage not enabled")
        if not self.use_persistent_storage:
            logger.warning("Persistent storage not enabled")
            return []

        if retrieval_mode == "embedding":
            # Standard embedding-based retrieval from vector store
            query_embedding = self.retriever.model.encode([query], convert_to_numpy=True)[0].tolist()
            filter_dict = {"source": filter_source} if filter_source else None
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
                resource_types=resource_types,
            )
            # Add retrieval method to results
            for r in results:
                r["retrieval_method"] = "embedding"
            
        elif retrieval_mode == "bm25":
            # BM25 retrieval: use in-memory BM25 index
            # Filter chunks by resource types if specified
            filtered_chunks = self._filter_chunks_by_resources(
                self.processed_chunks, resource_types, filter_source
            )
            
            # Temporarily create BM25 retriever with filtered chunks
            temp_bm25 = BM25Retriever()
            temp_bm25.add_chunks(filtered_chunks)
            
            raw_results = temp_bm25.retrieve(query=query, top_k=top_k)
            
            # Format to match vector store results
            results = []
            for chunk, score in raw_results:
                results.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "text": chunk.get("text", ""),
                    "metadata": chunk,
                    "similarity_score": score,
                    "resource_type": chunk.get("metadata", {}).get("resource_type", "default"),
                    "retrieval_method": "bm25",
                })
            
        elif retrieval_mode == "hybrid":
            # Hybrid: combine embedding and BM25
            # Get embedding results
            query_embedding = self.retriever.model.encode([query], convert_to_numpy=True)[0].tolist()
            filter_dict = {"source": filter_source} if filter_source else None
            embedding_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more candidates
                filter_dict=filter_dict,
                resource_types=resource_types,
            )
            
            # Get BM25 results
            filtered_chunks = self._filter_chunks_by_resources(
                self.processed_chunks, resource_types, filter_source
            )
            temp_bm25 = BM25Retriever()
            temp_bm25.add_chunks(filtered_chunks)
            bm25_raw = temp_bm25.retrieve(query=query, top_k=top_k * 2)
            
            # Re-rank using RRF
            results = self._hybrid_rerank_persistent(
                embedding_results, bm25_raw, top_k
            )
        else:
            logger.error(f"Invalid retrieval_mode: {retrieval_mode}")
            return []

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
        self.bm25_retriever.clear()
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
