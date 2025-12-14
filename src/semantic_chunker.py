"""
Semantic Chunking Module

Uses CHONKIE library for semantic-aware text chunking based on similarity.

Configuration (as per requirements):
- Model: minishlab/potion-base-8M
- Similarity threshold: 0.75
- Chunk size: 1536 tokens
- Double-pass merge: Enabled (via skip_window and filtering)
- Device: mps (Metal Performance Shaders for macOS) / cuda / cpu

The CHONKIE SemanticChunker uses advanced peak detection with Savitzky-Golay filtering
for smoother boundary detection and direct window embedding calculation for more accurate
semantic similarity computation.
"""

from typing import List, Dict, Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunking using CHONKIE library.
    Chunks text based on semantic similarity between sentences/paragraphs.
    
    Uses the official CHONKIE SemanticChunker with peak detection and
    Savitzky-Golay filtering for optimal boundary detection.
    """

    def __init__(
        self,
        embedding_model: str = "minishlab/potion-base-8M",
        threshold: float = 0.75,
        chunk_size: int = 1536,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Optional[List[str]] = None,
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 1,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
    ):
        """
        Initialize the semantic chunker using CHONKIE library.

        Configuration matches the project requirements:
        - Model: minishlab/potion-base-8M
        - Similarity threshold: 0.75
        - Chunk size: 1536 tokens
        - Double-pass merge: Enabled via skip_window and filtering

        Args:
            embedding_model: Name of the embedding model for semantic similarity
            threshold: Threshold for detecting split points (0-1)
            chunk_size: Target chunk size in tokens
            similarity_window: Number of sentences to consider for similarity (default: 3)
            min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
            min_characters_per_sentence: Minimum characters per sentence (default: 24)
            delim: Sentence delimiters (default: [". ", "! ", "? ", "\\n"])
            include_delim: Whether to include delimiters in prev/next chunk
            skip_window: Number of chunks to skip for double-pass merging (default: 1)
            filter_window: Window size for Savitzky-Golay filter (default: 5)
            filter_polyorder: Polynomial order for Savitzky-Golay filter (default: 3)
            filter_tolerance: Tolerance for filtering split indices (default: 0.2)
        """
        try:
            from chonkie import SemanticChunker as ChonkieChunker
        except ImportError:
            raise ImportError(
                "CHONKIE library not found. Install with: pip install chonkie[semantic]"
            )

        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.skip_window = skip_window
        
        # Set default delimiters if not provided
        if delim is None:
            delim = [". ", "! ", "? ", "\n"]

        # Initialize CHONKIE semantic chunker with correct API
        self.chunker = ChonkieChunker(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=delim,
            include_delim=include_delim,
            skip_window=skip_window,
            filter_window=filter_window,
            filter_polyorder=filter_polyorder,
            filter_tolerance=filter_tolerance,
        )

        logger.info(
            f"SemanticChunker initialized with embedding_model={embedding_model}, "
            f"threshold={threshold}, chunk_size={chunk_size}, "
            f"similarity_window={similarity_window}, skip_window={skip_window}"
        )

    def chunk(self, text: str) -> List[str]:
        """
        Chunk text based on semantic similarity using CHONKIE.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to semantic chunker")
            return []

        try:
            # CHONKIE chunker returns Chunk objects
            chunks = self.chunker.chunk(text)
            # Extract text from Chunk objects
            chunk_texts = [chunk.text for chunk in chunks]
            logger.info(f"Chunked text into {len(chunk_texts)} semantic chunks")
            return chunk_texts
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            raise

    def chunk_with_metadata(self, text: str, source: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text and return with metadata.

        Args:
            text: Input text to be chunked
            source: Source identifier for the text (e.g., filename, URL)

        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = self.chunk(text)
        chunked_data = []

        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{source or 'doc'}_{i}" if source else f"chunk_{i}",
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
            })

        return chunked_data
