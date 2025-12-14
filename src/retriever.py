"""
Embedding-based Retrieval Module

Handles embedding generation and similarity-based document retrieval.
Uses sentence-transformers for efficient semantic similarity search.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """
    Retriever using embedding-based similarity search.
    Generates embeddings for chunks and queries to find semantically similar content.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize the embedding retriever.

        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run embeddings on ('cpu', 'cuda', 'mps')
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not found. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

        # Storage for embeddings
        self.chunk_embeddings = []
        self.chunks = []

        logger.info(f"EmbeddingRetriever initialized with model={model_name}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the retriever and generate embeddings.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Store chunks and embeddings
        self.chunks.extend(chunks)
        self.chunk_embeddings.extend(embeddings)

        logger.info(f"Total chunks in retriever: {len(self.chunks)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve most similar chunks for a query.

        Args:
            query: Query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold (0-1)

        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        if not self.chunks:
            logger.warning("No chunks in retriever. Add chunks first with add_chunks()")
            return []

        # Generate embedding for query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Calculate similarity scores (cosine similarity)
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.chunk_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            # Apply threshold if specified
            if similarity_threshold is not None and score < similarity_threshold:
                continue

            results.append((self.chunks[idx], score))

        logger.info(f"Retrieved {len(results)} chunks for query")
        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None,
        context_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with surrounding context.

        Args:
            query: Query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            context_window: Number of adjacent chunks to include

        Returns:
            List of result dictionaries with chunk and context
        """
        results = self.retrieve(query, top_k, similarity_threshold)

        result_dicts = []
        for chunk, score in results:
            chunk_index = chunk.get("chunk_index", -1)
            source = chunk.get("source", "unknown")

            # Find adjacent chunks from same source
            context_chunks = []
            if chunk_index >= 0:
                for offset in range(-context_window, context_window + 1):
                    if offset == 0:
                        continue
                    neighbor_idx = chunk_index + offset
                    if 0 <= neighbor_idx < len(self.chunks):
                        neighbor = self.chunks[neighbor_idx]
                        if neighbor.get("source") == source:
                            context_chunks.append({
                                "text": neighbor["text"],
                                "relative_position": offset,
                            })

            result_dicts.append({
                "chunk": chunk,
                "similarity_score": score,
                "context": context_chunks,
            })

        return result_dicts

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.

        Returns:
            Dictionary with retriever statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "model_name": self.model_name,
            "device": self.device,
        }

    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        self.chunks = []
        self.chunk_embeddings = []
        logger.info("Retriever cleared")
