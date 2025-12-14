"""
Layout Chunking Module

Chunks text based on document structure and layout heuristics.
Configuration:
- Minimum chunk length: 30 words
- Chunk boundary heuristic rules:
  - Headers are merged with subsequent text block to preserve continuity
  - Subsequent headers are concatenated into a single header
  - Paragraphs shorter than minimum length are merged with previous paragraphs
  - Paragraphs ending with colons are merged with previous paragraphs
    unless the latter begins with a header marker
"""

from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class LayoutChunker:
    """
    Layout-based chunking using document structure and heuristic rules.
    """

    def __init__(self, min_chunk_length: int = 30):
        """
        Initialize the layout chunker.

        Args:
            min_chunk_length: Minimum word count for a chunk (default: 30)
        """
        self.min_chunk_length = min_chunk_length
        logger.info(f"LayoutChunker initialized with min_chunk_length={min_chunk_length} words")

    def _is_header(self, text: str) -> bool:
        """
        Check if a line is a header (starts with # or other header markers).

        Args:
            text: Text to check

        Returns:
            True if text is a header, False otherwise
        """
        text = text.strip()
        # Markdown headers
        if re.match(r"^#{1,6}\s+", text):
            return True
        # Common header patterns (numbered, bold, etc.)
        if re.match(r"^(\d+\.?\s+|[A-Z\*_]+:?)\s*", text):
            return True
        return False

    def _is_ends_with_colon(self, text: str) -> bool:
        """
        Check if paragraph ends with a colon.

        Args:
            text: Text to check

        Returns:
            True if text ends with colon, False otherwise
        """
        return text.rstrip().endswith(":")

    def _word_count(self, text: str) -> int:
        """
        Count words in text.

        Args:
            text: Text to count

        Returns:
            Word count
        """
        return len(text.split())

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs (separated by blank lines).

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split by multiple newlines or blank lines
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _apply_merge_heuristics(self, paragraphs: List[str]) -> List[str]:
        """
        Apply heuristic rules to merge related paragraphs.

        Args:
            paragraphs: List of paragraphs

        Returns:
            Merged paragraphs based on heuristics
        """
        if not paragraphs:
            return []

        merged = []
        i = 0

        while i < len(paragraphs):
            current = paragraphs[i]

            # Rule 1: Headers are merged with subsequent text block
            if self._is_header(current):
                # Collect consecutive headers
                header_group = [current]
                while i + 1 < len(paragraphs) and self._is_header(paragraphs[i + 1]):
                    # Rule 2: Subsequent headers are concatenated
                    header_group.append(paragraphs[i + 1])
                    i += 1

                merged_header = " ".join(header_group)

                # Merge with next non-header paragraph if available
                if i + 1 < len(paragraphs) and not self._is_header(paragraphs[i + 1]):
                    current = merged_header + "\n" + paragraphs[i + 1]
                    i += 1
                else:
                    current = merged_header

            # Rule 3: Merge short paragraphs with previous ones
            if merged and self._word_count(current) < self.min_chunk_length:
                if not self._is_header(current):
                    merged[-1] = merged[-1] + "\n" + current
                    i += 1
                    continue

            # Rule 4: Paragraphs ending with colons are merged with previous
            # (unless the previous begins with a header marker)
            if (
                merged
                and self._is_ends_with_colon(paragraphs[i])
                and not self._is_header(merged[-1].split("\n")[0])
            ):
                merged[-1] = merged[-1] + "\n" + current
                i += 1
                continue

            merged.append(current)
            i += 1

        return merged

    def chunk(self, text: str) -> List[str]:
        """
        Chunk text based on layout and heuristic rules.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to layout chunker")
            return []

        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)

        # Apply merge heuristics
        chunks = self._apply_merge_heuristics(paragraphs)

        logger.info(f"Chunked text into {len(chunks)} layout-based chunks")
        return chunks

    def chunk_with_metadata(self, text: str, source: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text and return with metadata.

        Args:
            text: Input text to be chunked
            source: Source identifier for the text

        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = self.chunk(text)
        chunked_data = []

        for i, chunk in enumerate(chunks):
            word_count = self._word_count(chunk)
            chunked_data.append({
                "chunk_id": f"{source or 'doc'}_{i}" if source else f"chunk_{i}",
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "word_count": word_count,
                "is_header": self._is_header(chunk.split("\n")[0]),
            })

        return chunked_data
