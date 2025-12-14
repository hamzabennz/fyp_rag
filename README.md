# FYP RAG - Retrieval-Augmented Generation with Advanced Chunking

Production-ready RAG system with semantic/layout chunking, persistent storage (ChromaDB + SQLite), and embedding-based retrieval.

## Features

- **Semantic Chunking**: CHONKIE library with `minishlab/potion-base-8M` (threshold: 0.75, chunk size: 1536 tokens)
- **Layout Chunking**: Structure-aware chunking with smart heuristics for headers and paragraphs (min: 30 words)
- **Hybrid Mode**: Combines layout structure + semantic similarity
- **Persistent Storage**: ChromaDB for vectors, SQLite for document metadata
- **Embedding Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity
- **GPU Support**: CUDA/MPS acceleration

## Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies**: `chonkie`, `chromadb`, `sqlalchemy`, `sentence-transformers`, `torch`

## Quick Start

### CLI Usage (Recommended)

**1. Ingest Documents**
```bash
# From file
python ingest.py --source "doc.txt" --file path/to/document.txt --device cuda

# From text
python ingest.py --source "doc.txt" --text "Your content here" --device cuda

# Choose strategy (semantic, layout, hybrid)
python ingest.py --source "doc.txt" --file doc.txt --strategy semantic --device cuda
```

**2. Query Documents**
```bash
# Basic query
python query.py --q "What is machine learning?" --device cuda

# Advanced options
python query.py --q "Your question" --top-k 10 --show-text --device cuda

# Filter by source
python query.py --q "Your question" --source "specific_doc.txt" --device cuda
```

### Python API

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy

# Initialize with persistent storage
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    device="cuda",
    use_persistent_storage=True,
    chroma_persist_dir="./data/chroma",
    sqlite_db_url="sqlite:///./data/rag_docs.db"
)

# Ingest document
pipeline.add_document(content, source="doc.txt")

# Query (persistent storage)
results = pipeline.retrieve_persistent(query, top_k=5)

# Query (in-memory)
results = pipeline.retrieve(query, top_k=5)
```

## Configuration

**Environment Variables** (optional):
```bash
export CHROMA_PERSIST_DIR="./data/chroma"
export DOC_DB_URL="sqlite:///./data/rag_docs.db"
export RAG_DEVICE="cuda"  # or "cpu", "mps"
```

**Chunking Strategies**:
- `semantic`: CHONKIE-based (threshold=0.75, chunk_size=1536)
- `layout`: Structure-aware (min_words=30)
- `hybrid`: Layout + semantic combined

## Deployment (Kubeflow/Docker)

**Directory Structure**:
```
/home/jovyan/fyp_rag/
├── data/
│   ├── chroma/      # ChromaDB persistent storage
│   └── rag_docs.db  # SQLite document metadata
├── src/
├── ingest.py
└── query.py
```

**Docker Volume Mount**:
```bash
docker run -v /path/to/data:/app/data your-image
```

**Kubeflow PVC**:
Mount persistent volume to `/home/jovyan/fyp_rag/data/` for persistent storage across notebook restarts.

## Testing

```bash
# Test persistent storage
python test_persistent.py

# Run unit tests
pytest tests/ -v
```

## How It Works

1. **Ingestion**: Documents are chunked using semantic/layout/hybrid strategy
2. **Embedding**: Each chunk is encoded into 384-dim vectors using sentence-transformers
3. **Storage**: Vectors stored in ChromaDB, metadata in SQLite
4. **Query**: Query text encoded → ChromaDB similarity search → Top-K results returned
5. **Persistence**: Data persists across sessions via file-based storage

## Architecture

```
┌─────────────────────────────────────────┐
│         RAGPipeline                     │
├─────────────────────────────────────────┤
│  SemanticChunker (CHONKIE)              │
│  LayoutChunker (Heuristics)             │
│  EmbeddingRetriever (sentence-transformers) │
├─────────────────────────────────────────┤
│  ChromaVectorStore (vectors)            │
│  SQLiteDocStore (metadata)              │
└─────────────────────────────────────────┘
```

## Project Structure

```
fyp_rag/
├── src/
│   ├── semantic_chunker.py    # CHONKIE wrapper
│   ├── layout_chunker.py      # Structure-aware chunking
│   ├── retriever.py           # Embedding-based retrieval
│   ├── vector_store.py        # ChromaDB integration
│   ├── doc_store.py           # SQLite metadata store
│   └── rag_pipeline.py        # Main pipeline
├── ingest.py                  # CLI ingestion script
├── query.py                   # CLI query script
├── config.py                  # Configuration
└── requirements.txt
