"""
BM25 Keyword-Based Retrieval Module

Implements BM25 (Best Matching 25) algorithm for keyword-based document retrieval.
Complements embedding-based retrieval with traditional information retrieval techniques.
"""

from typing import List, Dict, Tuple, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25-based keyword retriever for lexical matching.
    Uses the BM25 algorithm to rank documents based on term frequency and inverse document frequency.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize the BM25 retriever.

        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls length normalization (0=no normalization, 1=full normalization)
            epsilon: Floor value for IDF to avoid negative scores
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 not found. "
                "Install with: pip install rank-bm25"
            )

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

        logger.info(f"BM25Retriever initialized with k1={k1}, b={b}, epsilon={epsilon}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split by whitespace.
        Can be enhanced with more sophisticated tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization - can be improved with nltk or spacy
        return text.lower().split()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the BM25 index.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return

        # Store chunks
        self.chunks.extend(chunks)

        # Tokenize all texts
        texts = [chunk["text"] for chunk in self.chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in texts]

        # Initialize BM25 with tokenized corpus
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

        logger.info(f"BM25 index updated with {len(self.chunks)} total chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve most relevant chunks for a query using BM25.

        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum BM25 score threshold

        Returns:
            List of tuples (chunk_dict, bm25_score)
        """
        if not self.chunks or self.bm25 is None:
            logger.warning("No chunks in BM25 index. Add chunks first with add_chunks()")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
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

        logger.info(f"BM25 retrieved {len(results)} chunks for query")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 retriever.

        Returns:
            Dictionary with retriever statistics
        """
        avg_doc_len = np.mean([len(doc) for doc in self.tokenized_corpus]) if self.tokenized_corpus else 0
        
        return {
            "total_chunks": len(self.chunks),
            "average_doc_length": float(avg_doc_len),
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
        }

    def clear(self) -> None:
        """Clear all stored chunks and BM25 index."""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
        logger.info("BM25 retriever cleared")
