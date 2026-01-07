# ChromaDB and Keyword Search - Research Summary

## Question: Does ChromaDB support keyword search?

**Short Answer: NO**

ChromaDB is designed specifically for **vector similarity search** and does not natively support traditional keyword search or BM25 algorithms.

## ChromaDB Capabilities

ChromaDB focuses on:
- ✅ Vector embeddings storage
- ✅ Cosine similarity search
- ✅ Euclidean distance search
- ✅ Inner product search
- ✅ Metadata filtering
- ❌ BM25 keyword search
- ❌ Full-text search
- ❌ TF-IDF search

## What ChromaDB DOES Offer

### 1. Vector Similarity Search
```python
collection.query(
    query_embeddings=[embedding],
    n_results=10
)
```
This finds documents with similar semantic meaning based on embedding vectors.

### 2. Metadata Filtering
```python
collection.query(
    query_embeddings=[embedding],
    where={"source": "document.txt"},  # Filter by metadata
    n_results=10
)
```
This allows filtering results by metadata fields, but NOT by keyword matching.

### 3. Document Storage
ChromaDB stores:
- Document text
- Embedding vectors
- Metadata (key-value pairs)
- Document IDs

## Why Not Keyword Search?

ChromaDB's architecture is optimized for:
1. **Dense vector operations** (embeddings are typically 384-1536 dimensions)
2. **Approximate nearest neighbor (ANN) search** using HNSW or other indexing
3. **Semantic similarity** rather than lexical matching

Keyword search requires:
1. **Inverted index** (word → document mappings)
2. **Term frequency counting**
3. **Document length normalization**
4. **Statistical ranking** (TF-IDF, BM25)

These are fundamentally different approaches with different data structures.

## Solution: Hybrid Retrieval System

Since ChromaDB doesn't support keyword search, we've implemented a **hybrid retrieval system** that combines:

### 1. ChromaDB for Vector Search
```python
from src.vector_store import ChromaVectorStore

vector_store = ChromaVectorStore()
vector_store.add_chunks(chunks, embeddings)
results = vector_store.search(query_embedding, top_k=5)
```

### 2. BM25 for Keyword Search
```python
from src.bm25_retriever import BM25Retriever

bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.add_chunks(chunks)
results = bm25.retrieve(query, top_k=5)
```

### 3. Combined Hybrid Approach
```python
from src.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever(
    semantic_weight=0.6,
    bm25_weight=0.4,
    fusion_method="weighted"
)
hybrid.add_chunks(chunks)
results = hybrid.retrieve(query, top_k=5)
```

## BM25 Algorithm Details

**BM25 (Best Matching 25)** is a ranking function used in information retrieval to estimate the relevance of documents to a query.

### Formula
```
Score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
```

Where:
- `D` = document
- `Q` = query
- `qi` = query terms
- `f(qi,D)` = term frequency of qi in D
- `|D|` = length of document D
- `avgdl` = average document length
- `k1` = term frequency saturation parameter (typically 1.2-2.0)
- `b` = length normalization parameter (0-1, typically 0.75)
- `IDF(qi)` = inverse document frequency of term qi

### Parameters

**k1** (Term Frequency Saturation)
- Controls how quickly term frequency saturates
- Higher values = more importance to term frequency
- Typical range: 1.2 - 2.0
- Default: 1.5

**b** (Length Normalization)
- Controls document length normalization
- 0 = no normalization
- 1 = full normalization
- Typical value: 0.75

### Advantages of BM25
- ✅ Fast computation (no neural network inference)
- ✅ Excellent for exact keyword matches
- ✅ Works well with technical terms and proper nouns
- ✅ Interpretable scores
- ✅ No training required

### Disadvantages of BM25
- ❌ Doesn't understand semantics
- ❌ Requires exact or similar words to match
- ❌ Sensitive to vocabulary mismatch
- ❌ No understanding of synonyms or context

## Implementation Details

### rank-bm25 Library
We use the `rank-bm25` Python library which provides:
- BM25Okapi implementation (most common variant)
- BM25L (with length normalization)
- BM25Plus (improved version)

### Installation
```bash
pip install rank-bm25>=0.2.2
```

### Basic Usage
```python
from rank_bm25 import BM25Okapi

# Tokenize documents
corpus = [
    ["machine", "learning", "algorithm"],
    ["neural", "network", "deep", "learning"],
    ["python", "programming", "language"]
]

# Create BM25 index
bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

# Query
query = ["machine", "learning"]
scores = bm25.get_scores(query)

# Get top documents
top_n = bm25.get_top_n(query, corpus, n=5)
```

## Comparison: Vector vs. BM25 vs. Hybrid

| Feature | Vector (Semantic) | BM25 (Keyword) | Hybrid |
|---------|------------------|----------------|---------|
| Semantic understanding | ✅ Excellent | ❌ None | ✅ Excellent |
| Exact keyword matching | ❌ Poor | ✅ Excellent | ✅ Excellent |
| Synonym handling | ✅ Good | ❌ None | ✅ Good |
| Speed | 🔶 Moderate | ✅ Fast | 🔶 Moderate |
| Setup complexity | 🔶 Moderate | ✅ Simple | 🔴 Complex |
| Works with paraphrasing | ✅ Yes | ❌ No | ✅ Yes |
| Technical term matching | 🔶 Moderate | ✅ Excellent | ✅ Excellent |
| Overall accuracy | 🔶 Good | 🔶 Good | ✅ Best |

## Best Practices

### Use Vector Search When:
- Queries are conceptual or paraphrased
- Dealing with synonyms and related concepts
- User queries use different vocabulary than documents
- Understanding context is important

### Use BM25 Search When:
- Exact keyword matching is critical
- Dealing with technical terms, IDs, or proper nouns
- Speed is paramount
- Documents have consistent vocabulary

### Use Hybrid Search When:
- You want the best of both worlds (recommended default)
- Diverse query types expected
- Critical that nothing is missed
- Accuracy is more important than speed

## Fusion Methods

### Weighted Fusion
Combines normalized scores using weights:
```
combined_score = w1 * semantic_score + w2 * bm25_score
```

### Reciprocal Rank Fusion (RRF)
Combines based on ranks:
```
RRF_score = Σ(1 / (k + rank))
```

## References

- ChromaDB Documentation: https://docs.trychroma.com/
- BM25 Wikipedia: https://en.wikipedia.org/wiki/Okapi_BM25
- rank-bm25 Library: https://github.com/dorianbrown/rank_bm25
- Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25 and Beyond"

## Conclusion

While ChromaDB is excellent for vector similarity search, it does not support BM25 or traditional keyword search. To get the benefits of both approaches, we've implemented a hybrid retrieval system that:

1. Uses ChromaDB for semantic (vector) search
2. Uses rank-bm25 library for keyword (BM25) search
3. Combines results using score fusion techniques

This gives users the flexibility to choose:
- **SEMANTIC mode** for understanding meaning
- **BM25 mode** for keyword matching
- **HYBRID mode** for best overall results
