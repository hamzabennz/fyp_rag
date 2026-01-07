# RAG API Documentation - Hybrid Retrieval Support

## Overview

The Flask REST API now supports **hybrid retrieval** combining vector embeddings (semantic search) and BM25 keyword search.

## Base URL

```
http://localhost:5000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG Query Service",
  "version": "0.1.0"
}
```

---

### 2. Query (Persistent Storage)

**POST** `/query`

Query documents using persistent storage (ChromaDB). Supports multiple retrieval modes.

**Request Body:**
```json
{
  "payload": {
    "query": "What is machine learning?",
    "top_k": 5,
    "strategy": "semantic",
    "retrieval_mode": "hybrid",
    "device": "cuda",
    "filter_source": null,
    "resources": ["emails", "sms"],
    "show_text": false,
    "show_scores": true
  }
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Query text |
| `top_k` | integer | 5 | Number of results to return |
| `strategy` | string | "semantic" | Chunking strategy: "semantic", "layout", or "hybrid" |
| `retrieval_mode` | string | "hybrid" | **NEW**: Retrieval mode: "semantic", "bm25", or "hybrid" |
| `device` | string | "cuda" | Device: "cpu", "cuda", or "mps" |
| `filter_source` | string | null | Filter by specific document source |
| `resources` | array | null | Resource types to search (e.g., ["emails", "sms"]) |
| `show_text` | boolean | false | Include full text in results |
| `show_scores` | boolean | true | Include similarity scores |

**Retrieval Modes:**

- **`semantic`**: Vector embedding search only (semantic similarity)
- **`bm25`**: Keyword search only (BM25 algorithm)
- **`hybrid`**: Combined semantic + BM25 (recommended for best results)

**Response:**
```json
{
  "success": true,
  "query": "What is machine learning?",
  "retrieval_mode": "hybrid",
  "results_count": 5,
  "results": [
    {
      "chunk_id": "doc1_chunk_0",
      "source": "ml_guide.txt",
      "chunk_index": 0,
      "total_chunks": 10,
      "resource_type": "documents",
      "similarity_score": 0.8734,
      "preview": "Machine learning is a subset of artificial intelligence..."
    }
  ],
  "stats": {
    "total_documents": 3,
    "total_chunks": 25
  }
}
```

---

### 3. Hybrid Query (In-Memory)

**POST** `/query/hybrid`

Query documents using in-memory hybrid retrieval with **detailed score breakdown**.

This endpoint shows individual semantic and BM25 scores for transparency.

**Note:** Requires documents to be loaded in memory. For persistent storage, use `/query` endpoint.

**Request Body:**
```json
{
  "payload": {
    "query": "neural networks deep learning",
    "top_k": 5,
    "show_text": false
  }
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Query text |
| `top_k` | integer | 5 | Number of results to return |
| `show_text` | boolean | false | Include full text in results |

**Response:**
```json
{
  "success": true,
  "query": "neural networks deep learning",
  "retrieval_mode": "hybrid",
  "results_count": 5,
  "results": [
    {
      "chunk_id": "doc1_chunk_3",
      "source": "ai_overview.txt",
      "chunk_index": 3,
      "retrieval_method": "hybrid",
      "scores": {
        "semantic": 0.8734,
        "bm25": 0.7048,
        "combined": 0.7891,
        "semantic_rank": 1,
        "bm25_rank": 2
      },
      "preview": "Neural networks are the foundation of deep learning..."
    }
  ],
  "stats": {
    "total_documents": 3,
    "total_chunks": 25,
    "retrieval_mode": "hybrid"
  }
}
```

**Score Explanation:**
- `semantic`: Cosine similarity score from vector embeddings (0-1)
- `bm25`: BM25 keyword matching score (normalized 0-1)
- `combined`: Fused score combining both methods
- `semantic_rank`: Rank in semantic search results
- `bm25_rank`: Rank in BM25 search results

---

### 4. Statistics

**GET** `/stats`

Get statistics about the RAG system.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_documents": 3,
    "total_chunks": 25,
    "db_url": "sqlite:///./data/rag_docs.db",
    "chroma_collection": "rag_chunks",
    "retrieval_mode": "hybrid",
    "chunking_strategy": "semantic"
  },
  "documents": [
    {
      "source": "doc1.txt",
      "chunk_count": 10,
      "created_at": "2026-01-07T10:30:00"
    }
  ]
}
```

---

## Usage Examples

### Example 1: Semantic Search Only

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "What is artificial intelligence?",
      "retrieval_mode": "semantic",
      "top_k": 3
    }
  }'
```

### Example 2: BM25 Keyword Search Only

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "Python programming language",
      "retrieval_mode": "bm25",
      "top_k": 5
    }
  }'
```

### Example 3: Hybrid Search (Recommended)

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "machine learning algorithms",
      "retrieval_mode": "hybrid",
      "top_k": 5,
      "show_scores": true
    }
  }'
```

### Example 4: Hybrid with Full Score Breakdown

```bash
curl -X POST http://localhost:5000/query/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "neural networks",
      "top_k": 3,
      "show_text": true
    }
  }'
```

### Example 5: Filter by Source

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "machine learning",
      "retrieval_mode": "hybrid",
      "filter_source": "ml_guide.txt"
    }
  }'
```

---

## Python Client Example

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/query"

# Query payload
payload = {
    "payload": {
        "query": "What is deep learning?",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "show_text": False,
        "show_scores": True
    }
}

# Send request
response = requests.post(url, json=payload)

# Parse response
if response.status_code == 200:
    data = response.json()
    
    print(f"Query: {data['query']}")
    print(f"Retrieval Mode: {data['retrieval_mode']}")
    print(f"Results: {data['results_count']}\n")
    
    for i, result in enumerate(data['results'], 1):
        print(f"[{i}] Source: {result['source']}")
        print(f"    Score: {result['similarity_score']:.4f}")
        print(f"    Preview: {result['preview']}\n")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

---

## Starting the Server

```bash
# Default (port 5000)
python app.py

# Custom port
python app.py --port 8080

# Custom host and port
python app.py --host 0.0.0.0 --port 8080

# Debug mode
python app.py --debug
```

---

## Retrieval Mode Comparison

| Feature | Semantic | BM25 | Hybrid |
|---------|----------|------|--------|
| **Understanding Context** | ✅ Excellent | ❌ None | ✅ Excellent |
| **Exact Keywords** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Synonyms** | ✅ Good | ❌ None | ✅ Good |
| **Speed** | 🔶 Moderate | ✅ Fast | 🔶 Moderate |
| **Best For** | Conceptual queries | Technical terms | All queries |

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message here"
}
```

### Common Errors

**400 Bad Request**
- Missing required fields
- Invalid parameter values

**500 Internal Server Error**
- Database connection issues
- Model loading failures
- Processing errors

---

## Configuration

Edit `config.py` to change default settings:

```python
# config.py
DEVICE = "cuda"  # or "cpu", "mps"
CHROMA_PERSIST_DIR = "./data/chroma"
DOC_DB_URL = "sqlite:///./data/rag_docs.db"
```

---

## Performance Tips

1. **Use Hybrid Mode by Default**: Best overall accuracy
2. **Use BM25 for Speed**: When exact keywords matter more
3. **Use Semantic for Context**: When understanding meaning is critical
4. **Adjust top_k**: Higher values = more results but slower
5. **Set show_text=false**: Reduces response size
6. **Use GPU**: Set `device="cuda"` for faster processing

---

## Advanced Configuration

### Hybrid Retrieval Parameters

When creating the pipeline (server initialization), you can configure:

```python
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.6,  # 60% semantic
    hybrid_bm25_weight=0.4,      # 40% BM25
    hybrid_fusion_method="weighted",  # or "rrf"
    bm25_k1=1.5,  # Term frequency saturation
    bm25_b=0.75,  # Length normalization
)
```

---

## Troubleshooting

### No results returned
- Check if documents are ingested
- Try different retrieval modes
- Increase top_k value

### Slow performance
- Use BM25 mode for speed
- Ensure GPU is available (check device setting)
- Reduce top_k value

### Hybrid endpoint returns error
- Ensure documents are loaded in memory
- Use /query endpoint for persistent storage
- Check if pipeline is initialized

---

## Related Documentation

- **HYBRID_RETRIEVAL_GUIDE.md** - Comprehensive retrieval guide
- **CHROMADB_AND_BM25.md** - Technical details on ChromaDB and BM25
- **ARCHITECTURE.md** - System architecture documentation
