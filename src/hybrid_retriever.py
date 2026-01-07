"""
Hybrid Retrieval Module

Combines embedding-based (semantic) retrieval with BM25 (keyword) retrieval
to provide better search results using both semantic and lexical matching.
"""

from typing import List, Dict, Tuple, Any, Optional
import logging
import numpy as np

from .retriever import EmbeddingRetriever
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining semantic (embedding) and lexical (BM25) search.
    Uses weighted combination or reciprocal rank fusion to merge results.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        fusion_method: str = "weighted",  # "weighted" or "rrf" (reciprocal rank fusion)
    ):
        """
        Initialize the hybrid retriever.

        Args:
            embedding_model: Name of the sentence-transformer model
            device: Device to run embeddings on ('cpu', 'cuda', 'mps')
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            fusion_method: Method to combine results ("weighted" or "rrf")
        """
        # Initialize both retrievers
        self.embedding_retriever = EmbeddingRetriever(
            model_name=embedding_model,
            device=device
        )
        self.bm25_retriever = BM25Retriever(
            k1=bm25_k1,
            b=bm25_b
        )

        # Fusion parameters
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method

        logger.info(
            f"HybridRetriever initialized: fusion={fusion_method}, "
            f"weights=(semantic={semantic_weight}, bm25={bm25_weight})"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to both retrievers.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return

        # Add to both retrievers
        self.embedding_retriever.add_chunks(chunks)
        self.bm25_retriever.add_chunks(chunks)

        logger.info(f"Added {len(chunks)} chunks to hybrid retriever")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        bm25_threshold: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Query text
            top_k: Number of final results to return
            semantic_top_k: Number of results to fetch from semantic search (default: top_k * 2)
            bm25_top_k: Number of results to fetch from BM25 search (default: top_k * 2)
            similarity_threshold: Minimum semantic similarity score
            bm25_threshold: Minimum BM25 score

        Returns:
            List of tuples (chunk_dict, combined_score, score_details)
            where score_details = {"semantic": score, "bm25": score, "combined": score}
        """
        if not self.embedding_retriever.chunks:
            logger.warning("No chunks in retrievers. Add chunks first with add_chunks()")
            return []

        # Default to fetching more results for better fusion
        semantic_top_k = semantic_top_k or (top_k * 2)
        bm25_top_k = bm25_top_k or (top_k * 2)

        # Get results from both retrievers
        semantic_results = self.embedding_retriever.retrieve(
            query=query,
            top_k=semantic_top_k,
            similarity_threshold=similarity_threshold,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=bm25_top_k,
            score_threshold=bm25_threshold,
        )

        # Combine results using selected fusion method
        if self.fusion_method == "weighted":
            combined_results = self._weighted_fusion(semantic_results, bm25_results)
        elif self.fusion_method == "rrf":
            combined_results = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        final_results = combined_results[:top_k]

        logger.info(
            f"Hybrid retrieval: {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
            f"→ {len(final_results)} final results"
        )

        return final_results

    def _weighted_fusion(
        self,
        semantic_results: List[Tuple[Dict[str, Any], float]],
        bm25_results: List[Tuple[Dict[str, Any], float]],
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """
        Combine results using weighted score fusion.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search

        Returns:
            Combined results with normalized scores
        """
        # Normalize scores to [0, 1] range
        semantic_scores = {}
        if semantic_results:
            max_sem = max(score for _, score in semantic_results)
            min_sem = min(score for _, score in semantic_results)
            denom_sem = max_sem - min_sem if max_sem > min_sem else 1.0
            
            for chunk, score in semantic_results:
                chunk_id = chunk.get("chunk_id", id(chunk))
                normalized = (score - min_sem) / denom_sem
                semantic_scores[chunk_id] = (chunk, normalized)

        bm25_scores = {}
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            denom_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            
            for chunk, score in bm25_results:
                chunk_id = chunk.get("chunk_id", id(chunk))
                normalized = (score - min_bm25) / denom_bm25
                bm25_scores[chunk_id] = (chunk, normalized)

        # Combine scores
        combined = {}
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

        for chunk_id in all_chunk_ids:
            sem_chunk, sem_score = semantic_scores.get(chunk_id, (None, 0.0))
            bm25_chunk, bm25_score = bm25_scores.get(chunk_id, (None, 0.0))
            
            chunk = sem_chunk if sem_chunk is not None else bm25_chunk
            
            # Weighted combination
            combined_score = (
                self.semantic_weight * sem_score +
                self.bm25_weight * bm25_score
            )

            score_details = {
                "semantic": float(sem_score),
                "bm25": float(bm25_score),
                "combined": float(combined_score),
            }

            combined[chunk_id] = (chunk, combined_score, score_details)

        return list(combined.values())

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Dict[str, Any], float]],
        bm25_results: List[Tuple[Dict[str, Any], float]],
        k: int = 60,
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        RRF(d) = sum(1 / (k + rank(d))) for each ranker

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: Constant for RRF (typically 60)

        Returns:
            Combined results with RRF scores
        """
        # Build rank maps
        semantic_ranks = {
            chunk.get("chunk_id", id(chunk)): (chunk, rank + 1, score)
            for rank, (chunk, score) in enumerate(semantic_results)
        }

        bm25_ranks = {
            chunk.get("chunk_id", id(chunk)): (chunk, rank + 1, score)
            for rank, (chunk, score) in enumerate(bm25_results)
        }

        # Calculate RRF scores
        combined = {}
        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        for chunk_id in all_chunk_ids:
            sem_data = semantic_ranks.get(chunk_id)
            bm25_data = bm25_ranks.get(chunk_id)

            sem_rank = sem_data[1] if sem_data else float('inf')
            bm25_rank = bm25_data[1] if bm25_data else float('inf')

            # RRF score
            rrf_score = 0.0
            if sem_rank != float('inf'):
                rrf_score += 1.0 / (k + sem_rank)
            if bm25_rank != float('inf'):
                rrf_score += 1.0 / (k + bm25_rank)

            # Get chunk
            chunk = sem_data[0] if sem_data else bm25_data[0]

            # Get original scores
            sem_score = sem_data[2] if sem_data else 0.0
            bm25_score = bm25_data[2] if bm25_data else 0.0

            score_details = {
                "semantic": float(sem_score),
                "bm25": float(bm25_score),
                "combined": float(rrf_score),
                "semantic_rank": sem_rank if sem_rank != float('inf') else None,
                "bm25_rank": bm25_rank if bm25_rank != float('inf') else None,
            }

            combined[chunk_id] = (chunk, rrf_score, score_details)

        return list(combined.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid retriever.

        Returns:
            Dictionary with retriever statistics
        """
        return {
            "total_chunks": len(self.embedding_retriever.chunks),
            "embedding_stats": self.embedding_retriever.get_stats(),
            "bm25_stats": self.bm25_retriever.get_stats(),
            "fusion_method": self.fusion_method,
            "weights": {
                "semantic": self.semantic_weight,
                "bm25": self.bm25_weight,
            }
        }

    def clear(self) -> None:
        """Clear all stored chunks from both retrievers."""
        self.embedding_retriever.clear()
        self.bm25_retriever.clear()
        logger.info("Hybrid retriever cleared")
