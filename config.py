"""
Configuration for RAG Pipeline
"""

import os
from dataclasses import dataclass
from typing import Optional

# Device selection
DEVICE = os.getenv("RAG_DEVICE", "cpu")  # 'cpu', 'cuda', 'mps'

# Semantic Chunking Configuration
SEMANTIC_CONFIG = {
    "embedding_model": "minishlab/potion-base-8M",
    "threshold": 0.75,
    "chunk_size": 1536,  # tokens
    "tokenizer": "gpt2",
}

# Layout Chunking Configuration
LAYOUT_CONFIG = {
    "min_chunk_length": 30,  # words
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": DEVICE,
    "batch_size": 32,
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.0,  # No threshold by default
    "include_context": False,
    "context_window": 1,
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "chunking_strategy": "semantic",  # 'semantic', 'layout', 'hybrid'
    "device": DEVICE,
}


@dataclass
class RAGConfig:
    """Configuration class for RAG Pipeline."""

    # Chunking
    chunking_strategy: str = "semantic"
    semantic_embedding_model: str = "minishlab/potion-base-8M"
    semantic_threshold: float = 0.75
    semantic_chunk_size: int = 1536
    semantic_tokenizer: str = "gpt2"
    layout_min_words: int = 30

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_threshold: Optional[float] = None

    # Device
    device: str = "cpu"

    # Logging
    log_level: str = "INFO"

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "chunking_strategy": self.chunking_strategy,
            "semantic_embedding_model": self.semantic_embedding_model,
            "semantic_threshold": self.semantic_threshold,
            "semantic_chunk_size": self.semantic_chunk_size,
            "semantic_tokenizer": self.semantic_tokenizer,
            "layout_min_words": self.layout_min_words,
            "embedding_model": self.embedding_model,
            "retrieval_top_k": self.retrieval_top_k,
            "device": self.device,
        }


# Default configuration instance
DEFAULT_CONFIG = RAGConfig()
