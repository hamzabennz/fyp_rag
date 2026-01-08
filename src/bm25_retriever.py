"""
BM25 Keyword-based Retrieval Module

Handles keyword-based document retrieval using BM25 algorithm.
Uses rank_bm25 library for efficient keyword matching.
"""

from typing import List, Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Retriever using BM25 keyword-based search.
    Provides fast keyword matching for documents using BM25 algorithm.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 retriever.

        Args:
            k1: Controls term frequency saturation (default: 1.5)
            b: Controls document length normalization (default: 0.75)
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 not found. "
                "Install with: pip install rank-bm25"
            )

        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

        logger.info(f"BM25Retriever initialized with k1={k1}, b={b}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase words)
        """
        # Simple tokenization: lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the retriever and build BM25 index.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return

        from rank_bm25 import BM25Okapi

        # Store chunks
        self.chunks.extend(chunks)

        # Tokenize all texts
        logger.info(f"Tokenizing {len(chunks)} chunks for BM25 index...")
        new_tokenized = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.tokenized_corpus.extend(new_tokenized)

        # Rebuild BM25 index with all chunks
        logger.info(f"Building BM25 index for {len(self.tokenized_corpus)} chunks...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(f"Total chunks in BM25 retriever: {len(self.chunks)}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve most relevant chunks for a query using BM25.

        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum BM25 score threshold (optional)

        Returns:
            List of tuples (chunk_dict, bm25_score)
        """
        if not self.chunks:
            logger.warning("No chunks in retriever. Add chunks first with add_chunks()")
            return []

        if self.bm25 is None:
            logger.error("BM25 index not initialized")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])

            # Apply threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            results.append((self.chunks[idx], score))

        logger.info(f"Retrieved {len(results)} chunks for query using BM25")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.

        Returns:
            Dictionary with retriever statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "index_size": len(self.tokenized_corpus),
            "k1": self.k1,
            "b": self.b,
        }

    def clear(self) -> None:
        """Clear all stored chunks and index."""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
        logger.info("BM25 Retriever cleared")
