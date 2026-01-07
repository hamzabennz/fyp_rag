# Hybrid Retrieval System Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Document Ingestion                         │  │
│  │  Input: Raw Text → Chunking Strategy → Chunks                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                ↓                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Retrieval Mode Selection                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                ↓                                     │
│         ┌──────────────────────┼──────────────────────┐            │
│         ↓                      ↓                      ↓             │
│  ┏━━━━━━━━━━━┓          ┏━━━━━━━━━━━┓         ┏━━━━━━━━━━━━━━━┓    │
│  ┃ SEMANTIC  ┃          ┃   BM25    ┃         ┃    HYBRID     ┃    │
│  ┃   MODE    ┃          ┃   MODE    ┃         ┃     MODE      ┃    │
│  ┗━━━━━━━━━━━┛          ┗━━━━━━━━━━━┛         ┗━━━━━━━━━━━━━━━┛    │
│       ↓                      ↓                      ↓               │
│  ┌─────────┐           ┌─────────┐           ┌─────────────────┐  │
│  │ Sentence│           │  BM25   │           │  Both Retrievers│  │
│  │Transform│           │Retriever│           │   + Fusion      │  │
│  └─────────┘           └─────────┘           └─────────────────┘  │
│       ↓                      ↓                      ↓               │
│  Vector Search         Keyword Search       Combined Results       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Document Ingestion Flow
```
Raw Document
    ↓
Chunking (Semantic/Layout/Hybrid)
    ↓
Chunks with Metadata
    ↓
┌────────────────┬─────────────────────┐
↓                ↓                     ↓
EmbeddingRetriever  BM25Retriever   ChromaDB (persistent)
```

### 2. Query Processing Flow

#### SEMANTIC Mode
```
User Query
    ↓
Encode to Embedding (Sentence Transformer)
    ↓
Cosine Similarity Search
    ↓
Top-K Results with Similarity Scores
```

#### BM25 Mode
```
User Query
    ↓
Tokenize Query
    ↓
BM25 Scoring (TF-IDF + Length Norm)
    ↓
Top-K Results with BM25 Scores
```

#### HYBRID Mode
```
User Query
    ↓
┌──────────────────┬──────────────────┐
↓                  ↓                  ↓
Semantic Search    BM25 Search
    ↓                  ↓
Semantic Results   BM25 Results
    ↓                  ↓
└──────────────────┴──────────────────┘
              ↓
    Score Fusion (Weighted or RRF)
              ↓
    Combined Top-K Results
```

## Component Details

### Core Components

```
src/
├── rag_pipeline.py         [Main Pipeline - Orchestrates Everything]
│   ├── RetrievalMode enum (SEMANTIC, BM25, HYBRID)
│   ├── add_document()
│   └── retrieve()
│
├── retriever.py            [Semantic/Vector Retrieval]
│   ├── SentenceTransformer model
│   ├── Embedding generation
│   └── Cosine similarity search
│
├── bm25_retriever.py       [Keyword Retrieval] ⭐ NEW
│   ├── rank-bm25 library
│   ├── Tokenization
│   └── BM25 scoring
│
├── hybrid_retriever.py     [Combined Retrieval] ⭐ NEW
│   ├── Both retrievers
│   ├── Score normalization
│   └── Fusion methods (weighted/RRF)
│
├── vector_store.py         [Persistent Vector Storage]
│   └── ChromaDB integration
│
└── doc_store.py            [Persistent Document Metadata]
    └── SQLite integration
```

## Score Fusion Methods

### Weighted Fusion
```
For each document:
1. Normalize semantic score to [0, 1]
2. Normalize BM25 score to [0, 1]
3. Combined = (w1 × semantic) + (w2 × bm25)
4. Sort by combined score
```

### Reciprocal Rank Fusion (RRF)
```
For each document:
1. Get rank in semantic results
2. Get rank in BM25 results
3. RRF_score = (1/(k + rank_semantic)) + (1/(k + rank_bm25))
4. Sort by RRF score
```

## Storage Architecture

### In-Memory Storage
```
RAGPipeline
├── documents: Dict[source → content]
├── processed_chunks: List[chunks]
└── retriever(s):
    ├── EmbeddingRetriever
    │   ├── chunks: List
    │   └── embeddings: List[np.array]
    └── BM25Retriever
        ├── chunks: List
        └── bm25_index: BM25Okapi
```

### Persistent Storage
```
ChromaDB (Vector Store)
├── Collection per resource type
├── Embeddings (384-1536 dims)
├── Document texts
└── Metadata

SQLite (Doc Store)
├── Documents table
│   ├── source
│   ├── content_hash
│   ├── chunk_count
│   └── metadata JSON
└── Indexes on source, timestamp
```

## Performance Characteristics

### Speed Comparison
```
BM25:        ████░░░░░░  (Fastest - no neural network)
SEMANTIC:    ██████░░░░  (Moderate - embedding inference)
HYBRID:      ████████░░  (Slowest - runs both)
```

### Accuracy Comparison
```
BM25:        ███████░░░  (Good for keywords)
SEMANTIC:    ████████░░  (Good for concepts)
HYBRID:      ██████████  (Best overall)
```

## Example Result Structure

### SEMANTIC Mode Result
```json
{
    "chunk": {
        "chunk_id": "doc1_chunk_0",
        "text": "Machine learning is...",
        "source": "doc1.txt",
        "metadata": {...}
    },
    "similarity_score": 0.8734,
    "retrieval_method": "semantic",
    "context": []
}
```

### BM25 Mode Result
```json
{
    "chunk": {
        "chunk_id": "doc1_chunk_0",
        "text": "Machine learning is...",
        "source": "doc1.txt",
        "metadata": {...}
    },
    "bm25_score": 12.456,
    "retrieval_method": "bm25",
    "context": []
}
```

### HYBRID Mode Result
```json
{
    "chunk": {
        "chunk_id": "doc1_chunk_0",
        "text": "Machine learning is...",
        "source": "doc1.txt",
        "metadata": {...}
    },
    "combined_score": 0.7891,
    "score_details": {
        "semantic": 0.8734,
        "bm25": 0.7048,
        "combined": 0.7891
    },
    "retrieval_method": "hybrid",
    "context": []
}
```

## Configuration Parameters Summary

### Pipeline Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| retrieval_mode | RetrievalMode | HYBRID | SEMANTIC, BM25, or HYBRID |
| embedding_model | str | all-MiniLM-L6-v2 | Sentence transformer model |
| device | str | "cpu" | "cpu", "cuda", or "mps" |

### BM25 Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| bm25_k1 | float | 1.5 | Term frequency saturation (1.2-2.0) |
| bm25_b | float | 0.75 | Length normalization (0-1) |

### Hybrid Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hybrid_semantic_weight | float | 0.5 | Weight for semantic scores |
| hybrid_bm25_weight | float | 0.5 | Weight for BM25 scores |
| hybrid_fusion_method | str | "weighted" | "weighted" or "rrf" |

## Decision Tree: Which Mode to Use?

```
Start
  │
  ├─ Need exact keyword matches? ──Yes──→ BM25 Mode
  │   (IDs, technical terms, etc.)
  │
  ├─ Need semantic understanding? ──Yes──→ SEMANTIC Mode
  │   (concepts, paraphrasing)
  │
  ├─ Not sure / Want both? ───────Yes──→ HYBRID Mode ⭐ RECOMMENDED
  │
  └─ Speed is critical? ──────────Yes──→ BM25 Mode
      (real-time applications)
```

## Migration Path from Old System

### Old System (Semantic Only)
```python
pipeline = RAGPipeline()  # Default was semantic
results = pipeline.retrieve(query)
```

### New System (Explicit Mode)
```python
# Same as before (backward compatible)
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.SEMANTIC)
results = pipeline.retrieve(query)

# Or use new hybrid mode
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.HYBRID)
results = pipeline.retrieve(query)
```

## Dependencies

### Required Packages
```
chonkie>=0.1.0              (Chunking)
sentence-transformers>=2.2.0 (Embeddings)
chromadb>=1.3.0             (Vector Store)
rank-bm25>=0.2.2            (BM25 Search) ⭐ NEW
sqlalchemy>=2.0.0           (Doc Store)
torch>=2.0.0                (Neural Networks)
```

## Testing Strategy

```
test_hybrid_retrieval.py
├── test_imports()           (Verify all modules load)
├── test_bm25_retriever()    (BM25 functionality)
├── test_hybrid_retriever()  (Hybrid functionality)
└── test_rag_pipeline_modes()(All three modes)
```
