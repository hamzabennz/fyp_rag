# FYP RAG - Retrieval-Augmented Generation with Advanced Chunking

A comprehensive implementation of Retrieval-Augmented Generation (RAG) with advanced chunking strategies including **semantic chunking** and **layout-based chunking**.

## Features

### 1. **Semantic Chunking** 
- Uses CHONKIE library with `minishlab/potion-base-8M` model
- Similarity threshold: 0.75
- Similarity window: 3 sentences
- Chunk size: 1536 tokens
- Advanced peak detection with Savitzky-Golay filtering for smoother boundary detection
- Direct window embedding calculation for accurate semantic similarity

### 2. **Layout Chunking**
- Structure-aware text chunking
- Minimum chunk length: 30 words
- Smart heuristics for:
  - Merging headers with subsequent text blocks
  - Concatenating consecutive headers
  - Merging short paragraphs with previous ones
  - Handling paragraphs ending with colons

### 3. **Hybrid Chunking**
- Combines layout and semantic chunking
- First applies layout structure recognition
- Then applies semantic chunking to each layout chunk
- Best of both worlds: structural awareness + semantic similarity

### 4. **Embedding-based Retrieval**
- Uses `sentence-transformers` for efficient similarity search
- Fast cosine similarity matching
- Supports threshold-based filtering
- Optional context window retrieval with adjacent chunks

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repository-url>
cd fyp_rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `chonkie>=0.1.0` - Semantic chunking
- `sentence-transformers>=2.2.0` - Embedding generation
- `torch>=2.0.0` - Neural network backend
- `numpy>=1.24.0` - Numerical operations

## Quick Start

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy

# Initialize pipeline with semantic chunking
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    device="cpu"  # Use 'mps' on macOS, 'cuda' on GPU
)

# Add a document
document = """
Your long document text here...
Multiple paragraphs and sections...
"""
num_chunks = pipeline.add_document(document, source="document.txt")
print(f"Document chunked into {num_chunks} pieces")

# Retrieve relevant chunks
query = "What is your question?"
results = pipeline.retrieve(query, top_k=5)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Similarity: {result['similarity_score']:.4f}")
    print(f"   {result['chunk']['text'][:200]}...")
```

### Chunking Strategies

#### Semantic Chunking
```python
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    semantic_embedding_model="minishlab/potion-base-8M",
    semantic_threshold=0.75,
    semantic_chunk_size=1536,
    semantic_tokenizer="gpt2"
)
```

#### Layout Chunking
```python
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.LAYOUT,
    layout_min_words=30
)
```

#### Hybrid Chunking
```python
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.HYBRID
)
```

## API Documentation

### RAGPipeline

Main class for the complete RAG system.

#### Initialization Parameters
- `chunking_strategy` (ChunkingStrategy): Strategy to use
- `semantic_embedding_model` (str): Embedding model for semantic chunking
- `semantic_threshold` (float): Threshold for semantic similarity (0-1)
- `semantic_chunk_size` (int): Target chunk size in tokens
- `semantic_tokenizer` (str): Tokenizer for semantic chunking
- `layout_min_words` (int): Minimum words for layout chunks
- `embedding_model` (str): Model for generating embeddings
- `device` (str): Device for computation ('cpu', 'cuda', 'mps')

#### Key Methods

**`add_document(content, source, metadata=None)`**
- Add a document for processing
- Returns: Number of chunks created

**`retrieve(query, top_k=5, similarity_threshold=None, include_context=False)`**
- Retrieve relevant chunks for a query
- Returns: List of result dictionaries with chunks and scores

**`get_chunks_by_source(source)`**
- Get all chunks from a specific document
- Returns: List of chunks

**`get_pipeline_stats()`**
- Get pipeline statistics
- Returns: Dictionary with stats

**`clear()`**
- Clear all documents and chunks

**`export_chunks()`** / **`import_chunks(chunks)`**
- Export/import chunks for external use

### SemanticChunker

```python
from src.semantic_chunker import SemanticChunker

chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-8M",
    threshold=0.75,
    chunk_size=1536,
    tokenizer="gpt2"
)

chunks = chunker.chunk(text)
chunks_with_metadata = chunker.chunk_with_metadata(text, source="doc.txt")
```

### LayoutChunker

```python
from src.layout_chunker import LayoutChunker

chunker = LayoutChunker(min_chunk_length=30)

chunks = chunker.chunk(text)
chunks_with_metadata = chunker.chunk_with_metadata(text, source="doc.txt")
```

### EmbeddingRetriever

```python
from src.retriever import EmbeddingRetriever

retriever = EmbeddingRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Add chunks
retriever.add_chunks(chunk_list)

# Retrieve
results = retriever.retrieve(query, top_k=5)
results_with_context = retriever.retrieve_with_context(
    query, top_k=5, context_window=1
)
```

## Configuration Details

### Semantic Chunking Configuration
```python
# Model: minishlab/potion-base-8M
# Similarity threshold: 0.75
# Chunk size: 1536 tokens
# Similarity window: 3 sentences
# Device: mps (macOS), cuda (GPU), cpu (fallback)
```

### Layout Chunking Configuration
```python
# Minimum chunk length: 30 words
# Heuristics:
#   - Headers merged with subsequent text
#   - Consecutive headers concatenated
#   - Short paragraphs merged with previous
#   - Paragraphs ending with ':' merged unless preceded by header
```

### Embedding Model
```python
# Default: sentence-transformers/all-MiniLM-L6-v2
# Dimension: 384
# Fast and efficient for most tasks
# Alternative: all-mpnet-base-v2 (larger, more accurate)
```

## Examples

See [examples.py](examples.py) for complete working examples:

```bash
python examples.py
```

### Example Scenarios

1. **Single Document Retrieval**
2. **Layout-based Structure Preservation**
3. **Hybrid Multi-level Chunking**
4. **Multiple Document Management**

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_rag_pipeline.py -v
```

## Performance Considerations

### Memory Usage
- Embedding dimension: 384 (default) or 768 (larger models)
- Memory = (num_chunks × embedding_dim) × 4 bytes
- Example: 1000 chunks × 384 = ~1.5 MB

### Speed
- Semantic chunking: ~50-200 ms per 1000 tokens (depends on model)
- Layout chunking: ~10-50 ms per 1000 tokens
- Retrieval: ~1-5 ms per query

### Device Selection
- **CPU**: Suitable for small to medium documents
- **GPU (CUDA)**: Recommended for large-scale processing
- **MPS (macOS)**: Apple Silicon acceleration

## Architecture

```
RAGPipeline
├── SemanticChunker (CHONKIE-based)
├── LayoutChunker (Structure-aware)
└── EmbeddingRetriever (Similarity search)
    ├── Chunk Storage
    └── Embedding Storage
```

## Workflow

1. **Document Ingestion**: Add documents to the pipeline
2. **Chunking**: Split documents using selected strategy
3. **Embedding**: Generate embeddings for each chunk
4. **Indexing**: Store chunks and embeddings
5. **Retrieval**: Find relevant chunks for queries
6. **Re-ranking**: Optional similarity threshold filtering

## Advanced Usage

### Custom Embedding Model
```python
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

### Threshold-based Filtering
```python
results = pipeline.retrieve(
    query,
    top_k=10,
    similarity_threshold=0.5  # Only return > 0.5 similarity
)
```

### Context Window Retrieval
```python
results = pipeline.retrieve(
    query,
    top_k=5,
    include_context=True,
    context_window=2  # Include 2 chunks before and after
)
```

### Multiple Documents
```python
pipeline.add_document(content1, source="doc1.txt")
pipeline.add_document(content2, source="doc2.txt")

# Get chunks from specific document
chunks = pipeline.get_chunks_by_source("doc1.txt")
```

## Troubleshooting

### CHONKIE Not Found
```bash
pip install chonkie
```

### Out of Memory
- Use 'cpu' device or smaller embedding model
- Process documents in batches
- Reduce chunk size or number of embeddings

### Slow Retrieval
- Use a smaller embedding model
- Reduce number of chunks
- Use GPU acceleration (CUDA/MPS)

## References

- CHONKIE: Advanced semantic chunking library
- Sentence-Transformers: State-of-the-art sentence embeddings
- RAG Framework: Retrieval-Augmented Generation techniques

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Author

FYP Project - RAG Implementation with Advanced Chunking
