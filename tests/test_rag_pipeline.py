"""
Unit tests for the RAG pipeline
"""

import pytest
from src.layout_chunker import LayoutChunker
from src.retriever import EmbeddingRetriever
from src.rag_pipeline import RAGPipeline, ChunkingStrategy


class TestLayoutChunker:
    """Test cases for LayoutChunker."""

    def test_initialization(self):
        """Test chunker initialization."""
        chunker = LayoutChunker(min_chunk_length=30)
        assert chunker.min_chunk_length == 30

    def test_word_count(self):
        """Test word counting."""
        chunker = LayoutChunker()
        text = "This is a test sentence with five words"
        assert chunker._word_count(text) == 8

    def test_is_header(self):
        """Test header detection."""
        chunker = LayoutChunker()
        assert chunker._is_header("# Main Title") == True
        assert chunker._is_header("## Subtitle") == True
        assert chunker._is_header("Regular paragraph") == False

    def test_is_ends_with_colon(self):
        """Test colon detection."""
        chunker = LayoutChunker()
        assert chunker._is_ends_with_colon("This ends with colon:") == True
        assert chunker._is_ends_with_colon("This does not") == False

    def test_split_paragraphs(self):
        """Test paragraph splitting."""
        chunker = LayoutChunker()
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        paragraphs = chunker._split_into_paragraphs(text)
        assert len(paragraphs) == 3
        assert "Paragraph 1" in paragraphs[0]

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = LayoutChunker()
        result = chunker.chunk("")
        assert result == []

    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        chunker = LayoutChunker(min_chunk_length=5)
        text = "This is a simple test. It has two sentences."
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        assert isinstance(chunks[0], str)

    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        chunker = LayoutChunker()
        text = "Test paragraph one.\n\nTest paragraph two."
        result = chunker.chunk_with_metadata(text, source="test.txt")
        
        assert len(result) > 0
        assert "chunk_id" in result[0]
        assert "text" in result[0]
        assert "source" in result[0]
        assert result[0]["source"] == "test.txt"


class TestEmbeddingRetriever:
    """Test cases for EmbeddingRetriever."""

    def test_initialization(self):
        """Test retriever initialization."""
        retriever = EmbeddingRetriever(device="cpu")
        assert retriever.chunks == []
        assert len(retriever.chunk_embeddings) == 0

    def test_add_chunks(self):
        """Test adding chunks to retriever."""
        retriever = EmbeddingRetriever(device="cpu")
        chunks = [
            {"text": "Python is a programming language", "chunk_id": "0"},
            {"text": "Java is also a programming language", "chunk_id": "1"},
        ]
        retriever.add_chunks(chunks)
        assert len(retriever.chunks) == 2
        assert len(retriever.chunk_embeddings) == 2

    def test_retrieve_empty(self):
        """Test retrieval on empty retriever."""
        retriever = EmbeddingRetriever(device="cpu")
        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_similar(self):
        """Test retrieving similar chunks."""
        retriever = EmbeddingRetriever(device="cpu")
        chunks = [
            {"text": "The cat is sleeping", "chunk_id": "0"},
            {"text": "The dog is running", "chunk_id": "1"},
            {"text": "A feline rests peacefully", "chunk_id": "2"},
        ]
        retriever.add_chunks(chunks)
        
        results = retriever.retrieve("cat sleeping", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_get_stats(self):
        """Test getting retriever statistics."""
        retriever = EmbeddingRetriever(device="cpu")
        stats = retriever.get_stats()
        assert "total_chunks" in stats
        assert "embedding_dimension" in stats
        assert "model_name" in stats
        assert stats["total_chunks"] == 0

    def test_clear(self):
        """Test clearing retriever."""
        retriever = EmbeddingRetriever(device="cpu")
        chunks = [{"text": "Test", "chunk_id": "0"}]
        retriever.add_chunks(chunks)
        assert len(retriever.chunks) == 1
        
        retriever.clear()
        assert len(retriever.chunks) == 0
        assert len(retriever.chunk_embeddings) == 0


class TestRAGPipeline:
    """Test cases for RAGPipeline."""

    def test_initialization_semantic(self):
        """Test initializing pipeline with semantic strategy."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            device="cpu"
        )
        assert pipeline.chunking_strategy == ChunkingStrategy.SEMANTIC
        assert pipeline.semantic_chunker is not None

    def test_initialization_layout(self):
        """Test initializing pipeline with layout strategy."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        assert pipeline.chunking_strategy == ChunkingStrategy.LAYOUT
        assert pipeline.layout_chunker is not None

    def test_initialization_hybrid(self):
        """Test initializing pipeline with hybrid strategy."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.HYBRID,
            device="cpu"
        )
        assert pipeline.semantic_chunker is not None
        assert pipeline.layout_chunker is not None

    def test_add_document_layout(self):
        """Test adding document with layout chunking."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        document = "# Title\n\nParagraph 1.\n\nParagraph 2."
        num_chunks = pipeline.add_document(document, source="test.txt")
        
        assert num_chunks > 0
        assert len(pipeline.documents) == 1
        assert len(pipeline.processed_chunks) == num_chunks

    def test_retrieve_layout(self):
        """Test retrieval from pipeline."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        document = "Machine learning is a subset of artificial intelligence. " \
                   "It enables systems to learn from data."
        pipeline.add_document(document, source="ml.txt")
        
        results = pipeline.retrieve("What is machine learning?", top_k=1)
        assert len(results) > 0
        assert "chunk" in results[0]
        assert "similarity_score" in results[0]

    def test_get_chunks_by_source(self):
        """Test getting chunks by source."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        pipeline.add_document("Document 1 content", source="doc1.txt")
        pipeline.add_document("Document 2 content", source="doc2.txt")
        
        doc1_chunks = pipeline.get_chunks_by_source("doc1.txt")
        assert len(doc1_chunks) > 0
        assert all(c["source"] == "doc1.txt" for c in doc1_chunks)

    def test_get_pipeline_stats(self):
        """Test getting pipeline statistics."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        pipeline.add_document("Test document", source="test.txt")
        stats = pipeline.get_pipeline_stats()
        
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] > 0
        assert stats["chunking_strategy"] == "layout"

    def test_clear_pipeline(self):
        """Test clearing the pipeline."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        pipeline.add_document("Test", source="test.txt")
        assert len(pipeline.documents) > 0
        
        pipeline.clear()
        assert len(pipeline.documents) == 0
        assert len(pipeline.processed_chunks) == 0

    def test_export_import_chunks(self):
        """Test exporting and importing chunks."""
        pipeline = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        
        pipeline.add_document("Test document content", source="test.txt")
        exported = pipeline.export_chunks()
        assert len(exported) > 0
        
        # Create new pipeline and import
        pipeline2 = RAGPipeline(
            chunking_strategy=ChunkingStrategy.LAYOUT,
            device="cpu"
        )
        pipeline2.import_chunks(exported)
        assert len(pipeline2.processed_chunks) == len(exported)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
