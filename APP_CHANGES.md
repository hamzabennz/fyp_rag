# app.py Changes Summary

## Overview

The Flask REST API (`app.py`) has been updated to support **hybrid retrieval** combining vector embeddings and BM25 keyword search.

## Key Changes

### 1. Added RetrievalMode Import

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode
```

### 2. Updated get_pipeline() Function

**Before:**
```python
def get_pipeline(strategy="semantic", device="cuda"):
```

**After:**
```python
def get_pipeline(strategy="semantic", retrieval_mode="hybrid", device="cuda"):
```

Now supports:
- `retrieval_mode="semantic"` - Vector embeddings only
- `retrieval_mode="bm25"` - Keyword search only
- `retrieval_mode="hybrid"` - Combined (default)

### 3. Enhanced /query Endpoint

**New Parameters:**
- `retrieval_mode` (string, default: "hybrid"): Choose retrieval method
- `show_scores` (boolean, default: true): Show similarity scores

**Request Example:**
```json
{
  "payload": {
    "query": "machine learning",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "show_scores": true
  }
}
```

**Response Now Includes:**
```json
{
  "success": true,
  "query": "machine learning",
  "retrieval_mode": "hybrid",
  "results": [...]
}
```

### 4. New /query/hybrid Endpoint

**Purpose:** In-memory hybrid search with detailed score breakdown

**Features:**
- Shows individual semantic and BM25 scores
- Includes rank information
- More transparent results

**Request Example:**
```json
{
  "payload": {
    "query": "neural networks",
    "top_k": 3
  }
}
```

**Response Example:**
```json
{
  "results": [
    {
      "chunk_id": "doc1_chunk_3",
      "scores": {
        "semantic": 0.8734,
        "bm25": 0.7048,
        "combined": 0.7891,
        "semantic_rank": 1,
        "bm25_rank": 2
      }
    }
  ]
}
```

### 5. Enhanced /stats Endpoint

**New Fields:**
```json
{
  "stats": {
    "retrieval_mode": "hybrid",
    "chunking_strategy": "semantic"
  }
}
```

## Complete API Endpoints

### 1. GET /health
Health check endpoint (unchanged)

### 2. POST /query
Main query endpoint with hybrid retrieval support

**Parameters:**
- `query` (required): Query text
- `top_k` (default: 5): Number of results
- `strategy` (default: "semantic"): Chunking strategy
- `retrieval_mode` (default: "hybrid"): **NEW** - Retrieval method
- `device` (default: "cuda"): Compute device
- `filter_source`: Filter by document source
- `resources`: Filter by resource types
- `show_text` (default: false): Include full text
- `show_scores` (default: true): **NEW** - Include scores

### 3. POST /query/hybrid
**NEW** - In-memory hybrid query with detailed scores

**Parameters:**
- `query` (required): Query text
- `top_k` (default: 5): Number of results
- `show_text` (default: false): Include full text

**Returns:** Detailed score breakdown with semantic, BM25, and combined scores

### 4. GET /stats
System statistics (enhanced with retrieval mode info)

## Usage Examples

### Curl Examples

**Semantic Search:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"payload": {"query": "AI", "retrieval_mode": "semantic"}}'
```

**BM25 Search:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"payload": {"query": "Python", "retrieval_mode": "bm25"}}'
```

**Hybrid Search (Default):**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"payload": {"query": "machine learning", "retrieval_mode": "hybrid"}}'
```

**Hybrid with Details:**
```bash
curl -X POST http://localhost:5000/query/hybrid \
  -H "Content-Type: application/json" \
  -d '{"payload": {"query": "neural networks", "top_k": 3}}'
```

### Python Examples

**Basic Query:**
```python
import requests

response = requests.post(
    "http://localhost:5000/query",
    json={
        "payload": {
            "query": "What is deep learning?",
            "retrieval_mode": "hybrid",
            "top_k": 5
        }
    }
)

data = response.json()
print(f"Mode: {data['retrieval_mode']}")
print(f"Results: {data['results_count']}")
```

**Hybrid with Details:**
```python
response = requests.post(
    "http://localhost:5000/query/hybrid",
    json={
        "payload": {
            "query": "neural networks",
            "top_k": 3
        }
    }
)

data = response.json()
for result in data['results']:
    scores = result['scores']
    print(f"Semantic: {scores['semantic']:.4f}")
    print(f"BM25: {scores['bm25']:.4f}")
    print(f"Combined: {scores['combined']:.4f}")
```

## Backward Compatibility

✅ **Fully backward compatible**

Old requests without `retrieval_mode` will default to "hybrid":

```json
{
  "payload": {
    "query": "machine learning",
    "top_k": 5
  }
}
```

This will use hybrid retrieval automatically.

## Testing

### 1. Start Server
```bash
python app.py --port 5000
```

### 2. Test Health
```bash
curl http://localhost:5000/health
```

### 3. Test Query
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"payload": {"query": "test", "retrieval_mode": "hybrid"}}'
```

### 4. Test Stats
```bash
curl http://localhost:5000/stats
```

## Error Handling

**Invalid retrieval_mode:**
```json
{
  "error": "Invalid retrieval_mode: xyz. Must be 'semantic', 'bm25', or 'hybrid'"
}
```

**No documents (for /query/hybrid):**
```json
{
  "success": false,
  "error": "No documents loaded in memory..."
}
```

## Configuration

Server initializes with hybrid retrieval by default:

```python
pipeline = RAGPipeline(
    retrieval_mode=retrieval_map.get(retrieval_mode, RetrievalMode.HYBRID),
    # ... other parameters
)
```

## Performance Comparison

| Retrieval Mode | Speed | Accuracy | Use Case |
|---------------|-------|----------|----------|
| semantic | Medium | Good | Conceptual queries |
| bm25 | Fast | Good | Keyword matching |
| hybrid | Slower | Best | General use (default) |

## Related Files

- **app.py** - Main Flask application (modified)
- **src/rag_pipeline.py** - Pipeline with RetrievalMode (modified)
- **src/bm25_retriever.py** - BM25 implementation (new)
- **src/hybrid_retriever.py** - Hybrid retrieval (new)
- **API_DOCUMENTATION.md** - Complete API docs (new)
- **HYBRID_RETRIEVAL_GUIDE.md** - Usage guide (new)

## Next Steps

1. ✅ app.py updated with hybrid retrieval
2. ✅ New /query/hybrid endpoint added
3. ✅ Backward compatible
4. 🧪 Test with real queries
5. 📊 Monitor performance
6. 🔧 Tune fusion weights if needed

## Summary

The updated `app.py` now supports:
- ✅ Three retrieval modes (semantic, BM25, hybrid)
- ✅ New /query/hybrid endpoint with detailed scores
- ✅ Enhanced /query endpoint with retrieval_mode parameter
- ✅ Enhanced /stats endpoint with mode information
- ✅ Fully backward compatible
- ✅ Default to hybrid mode for best results

All existing code continues to work, with hybrid retrieval as the new recommended default.
