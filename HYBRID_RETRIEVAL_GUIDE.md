# Hybrid Retrieval System Documentation

## Overview

This RAG system now supports **three retrieval modes** that can be used to search through your documents:

1. **SEMANTIC** - Vector embedding search (semantic similarity)
2. **BM25** - Keyword search (lexical matching)
3. **HYBRID** - Combined approach (semantic + BM25)

## ChromaDB and Keyword Search

**Important Finding:** ChromaDB does NOT natively support BM25 or keyword search. ChromaDB is designed specifically for vector similarity search using embeddings.

To add keyword search capabilities, we've implemented a separate BM25 retriever using the `rank-bm25` library that works alongside ChromaDB's vector search.

## Retrieval Modes

### 1. SEMANTIC Mode (Vector Embeddings)

Uses sentence transformers to create dense vector embeddings and finds semantically similar documents.

**Pros:**
- Understands context and meaning
- Works well with paraphrases and synonyms
- Good for conceptual queries

**Cons:**
- May miss exact keyword matches
- Computationally more expensive

**Example:**
```python
from src.rag_pipeline import RAGPipeline, RetrievalMode

pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.SEMANTIC,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

pipeline.add_document(content, "doc1.txt")
results = pipeline.retrieve("What is machine learning?", top_k=5)
```

### 2. BM25 Mode (Keyword Search)

Uses the BM25 algorithm (Best Matching 25) for traditional information retrieval based on term frequency and inverse document frequency.

**Pros:**
- Fast and efficient
- Excellent for exact keyword matches
- Works well with technical terms and proper nouns

**Cons:**
- Doesn't understand semantics
- Requires exact or similar words to match
- Sensitive to vocabulary mismatch

**Example:**
```python
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.BM25,
    bm25_k1=1.5,  # Term frequency saturation
    bm25_b=0.75,  # Length normalization
    device="cpu"
)

pipeline.add_document(content, "doc1.txt")
results = pipeline.retrieve("Python programming", top_k=5)
```

### 3. HYBRID Mode (Combined)

Combines both semantic and BM25 search using score fusion techniques.

**Pros:**
- Best of both worlds
- Catches both semantic matches and keyword matches
- More robust retrieval

**Cons:**
- Slightly slower (runs both searches)
- Requires tuning fusion parameters

**Example:**
```python
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.6,  # 60% semantic
    hybrid_bm25_weight=0.4,      # 40% BM25
    hybrid_fusion_method="weighted",  # or "rrf"
    device="cpu"
)

pipeline.add_document(content, "doc1.txt")
results = pipeline.retrieve("neural networks", top_k=5)
```

## Fusion Methods

### Weighted Fusion

Combines normalized scores from both retrievers using weighted average:

```
combined_score = (semantic_weight × semantic_score) + (bm25_weight × bm25_score)
```

**When to use:** When you want direct control over the importance of each retriever.

### Reciprocal Rank Fusion (RRF)

Combines results based on their ranks rather than raw scores:

```
RRF_score = Σ(1 / (k + rank))
```

**When to use:** When score scales are very different or when you want a rank-based approach.

## Configuration Parameters

### Chunking Parameters
- `chunking_strategy`: SEMANTIC, LAYOUT, or HYBRID
- `semantic_threshold`: Similarity threshold for semantic chunking (0-1)
- `semantic_chunk_size`: Target chunk size in tokens
- `layout_min_words`: Minimum words for layout chunking

### Retrieval Parameters
- `retrieval_mode`: SEMANTIC, BM25, or HYBRID
- `embedding_model`: Model for semantic embeddings
- `device`: "cpu", "cuda", or "mps"

### BM25 Parameters
- `bm25_k1`: Controls term frequency saturation (typical: 1.2-2.0)
  - Higher values = less saturation, more importance to term frequency
- `bm25_b`: Controls length normalization (0-1)
  - 0 = no normalization, 1 = full normalization
  - Typical value: 0.75

### Hybrid Parameters
- `hybrid_semantic_weight`: Weight for semantic scores (0-1)
- `hybrid_bm25_weight`: Weight for BM25 scores (0-1)
- `hybrid_fusion_method`: "weighted" or "rrf"

## Usage Examples

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode

# Create pipeline with hybrid retrieval
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_fusion_method="weighted"
)

# Add documents
pipeline.add_document(
    content="Your document text here...",
    source="document1.txt",
    metadata={"author": "John Doe"}
)

# Search with hybrid retrieval
results = pipeline.retrieve(
    query="machine learning algorithms",
    top_k=5
)

# Process results
for result in results:
    chunk = result["chunk"]
    score_details = result["score_details"]
    
    print(f"Text: {chunk['text']}")
    print(f"Semantic Score: {score_details['semantic']}")
    print(f"BM25 Score: {score_details['bm25']}")
    print(f"Combined Score: {score_details['combined']}")
```

### Comparing Retrieval Modes

```python
# Test different modes on same query
query = "neural networks"

for mode in [RetrievalMode.SEMANTIC, RetrievalMode.BM25, RetrievalMode.HYBRID]:
    pipeline = RAGPipeline(retrieval_mode=mode)
    pipeline.add_document(content, "doc.txt")
    
    results = pipeline.retrieve(query, top_k=3)
    print(f"\n{mode.value} Results:")
    for result in results:
        print(f"  - {result['chunk']['text'][:100]}...")
```

### Advanced: Custom Weights

```python
# Emphasize keyword matching for technical queries
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.3,  # 30% semantic
    hybrid_bm25_weight=0.7,      # 70% BM25
    hybrid_fusion_method="weighted"
)

# Emphasize semantic understanding for conceptual queries
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.8,  # 80% semantic
    hybrid_bm25_weight=0.2,      # 20% BM25
    hybrid_fusion_method="weighted"
)
```

## Installation

Update your requirements:

```bash
pip install rank-bm25>=0.2.2
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Performance Considerations

### Speed
- **BM25**: Fastest (no neural network inference)
- **SEMANTIC**: Moderate (requires embedding generation)
- **HYBRID**: Slowest (runs both searches)

### Accuracy
- **BM25**: Best for exact keyword matches
- **SEMANTIC**: Best for conceptual understanding
- **HYBRID**: Best overall accuracy

### Memory
- All modes have similar memory footprint
- Hybrid mode stores both BM25 index and embeddings

## Best Practices

1. **Use HYBRID mode by default** for best results
2. **Use SEMANTIC mode** when:
   - Queries are conceptual or paraphrased
   - Dealing with synonyms and related concepts
   - User queries use different vocabulary than documents

3. **Use BM25 mode** when:
   - Exact keyword matching is critical
   - Dealing with technical terms, IDs, or proper nouns
   - Speed is more important than semantic understanding

4. **Tune fusion weights** based on your use case:
   - Technical documentation: Higher BM25 weight
   - Conceptual content: Higher semantic weight
   - Mixed content: Equal weights (0.5/0.5)

5. **Try RRF fusion** if:
   - Score scales are very different
   - You want more stable results
   - Weighted fusion isn't working well

## Troubleshooting

### BM25 returns unexpected results
- Check if query terms appear in documents
- Adjust `bm25_k1` and `bm25_b` parameters
- Consider using HYBRID mode

### Semantic search misses obvious matches
- Verify embedding model is appropriate for your domain
- Try increasing `top_k` to see more results
- Consider using HYBRID mode

### Hybrid results seem biased
- Adjust fusion weights
- Try switching between "weighted" and "rrf" fusion
- Check individual semantic and BM25 scores in results

## API Reference

See the module docstrings for detailed API documentation:
- `src/bm25_retriever.py` - BM25 implementation
- `src/hybrid_retriever.py` - Hybrid retrieval with fusion
- `src/rag_pipeline.py` - Main pipeline with all modes
