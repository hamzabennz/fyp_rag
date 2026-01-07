# Quick Reference: Hybrid Retrieval System & API

## ✅ ChromaDB Keyword Search Answer

**Question:** Does ChromaDB support keyword search?

**Answer:** **NO**. ChromaDB only supports vector similarity search. For keyword search (BM25), we've implemented a separate retriever using the `rank-bm25` library.

## 🚀 Quick Start

### Installation
```bash
pip install rank-bm25>=0.2.2
# or
pip install -r requirements.txt
```

### Basic Usage - Hybrid Mode (Recommended)
```python
from src.rag_pipeline import RAGPipeline, RetrievalMode

# Create pipeline with hybrid retrieval
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.HYBRID)

# Add documents
pipeline.add_document("Your text here", "doc1.txt")

# Search (uses both vector + BM25)
results = pipeline.retrieve("your query", top_k=5)

# Access results
for result in results:
    print(result['chunk']['text'])
    print(result['score_details'])  # Shows semantic, bm25, combined scores
```

## 🎯 Three Retrieval Modes

| Mode | Best For | Speed | Accuracy |
|------|----------|-------|----------|
| **SEMANTIC** | Conceptual queries, synonyms | Medium | Good |
| **BM25** | Exact keywords, technical terms | Fast | Good |
| **HYBRID** | Best overall (recommended) | Slower | Best |

## 📝 Mode Examples

### Semantic Mode (Vector Embeddings)
```python
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.SEMANTIC)
```
✅ Good for: "What is machine learning?"
✅ Handles: Paraphrasing, synonyms, context

### BM25 Mode (Keyword Search)
```python
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.BM25)
```
✅ Good for: "Python programming language"
✅ Handles: Exact matches, technical terms, IDs

### Hybrid Mode (Combined)
```python
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.6,  # 60% semantic
    hybrid_bm25_weight=0.4,      # 40% BM25
    hybrid_fusion_method="weighted"  # or "rrf"
)
```
✅ Good for: Everything (recommended default)
✅ Handles: Both semantic understanding + keyword matching

## ⚙️ Key Parameters

### BM25 Parameters
```python
bm25_k1=1.5      # Term frequency saturation (1.2-2.0)
bm25_b=0.75      # Length normalization (0-1)
```

### Hybrid Parameters
```python
hybrid_semantic_weight=0.5    # Weight for semantic (0-1)
hybrid_bm25_weight=0.5        # Weight for BM25 (0-1)
hybrid_fusion_method="weighted"  # "weighted" or "rrf"
```

## 🔍 Result Structure

### Semantic Mode
```python
{
    "chunk": {...},
    "similarity_score": 0.87,
    "retrieval_method": "semantic"
}
```

### BM25 Mode
```python
{
    "chunk": {...},
    "bm25_score": 12.45,
    "retrieval_method": "bm25"
}
```

### Hybrid Mode
```python
{
    "chunk": {...},
    "combined_score": 0.79,
    "score_details": {
        "semantic": 0.87,
        "bm25": 0.70,
        "combined": 0.79
    },
    "retrieval_method": "hybrid"
}
```

## 💡 When to Use Each Mode

### Use SEMANTIC when:
- Query is conceptual: "How does X work?"
- Dealing with synonyms or related concepts
- User uses different vocabulary than documents

### Use BM25 when:
- Need exact keyword matches
- Searching for IDs, names, technical terms
- Speed is critical

### Use HYBRID when:
- Want best results (recommended)
- Not sure which is better
- Critical not to miss any matches

## 🎛️ Tuning Fusion Weights

### Equal weights (default):
```python
hybrid_semantic_weight=0.5, hybrid_bm25_weight=0.5
```
Use when: Balanced approach needed

### Favor semantic:
```python
hybrid_semantic_weight=0.7, hybrid_bm25_weight=0.3
```
Use when: Conceptual queries, diverse vocabulary

### Favor BM25:
```python
hybrid_semantic_weight=0.3, hybrid_bm25_weight=0.7
```
Use when: Technical docs, exact terminology important

## 🔧 Fusion Methods

### Weighted Fusion (Default)
```python
hybrid_fusion_method="weighted"
```
- Combines normalized scores
- Direct control via weights
- Good for most cases

### Reciprocal Rank Fusion (RRF)
```python
hybrid_fusion_method="rrf"
```
- Combines based on ranks
- More stable across different score scales
- Good when score ranges differ greatly

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `HYBRID_RETRIEVAL_GUIDE.md` | Comprehensive usage guide |
| `CHROMADB_AND_BM25.md` | ChromaDB capabilities & BM25 details |
| `ARCHITECTURE.md` | System architecture & diagrams |
| `CHANGES_SUMMARY.md` | What's new summary |
| `demo_hybrid.py` | Live demonstrations |
| `test_hybrid_retrieval.py` | Test suite |

## 🧪 Testing

### Run Demo
```bash
python demo_hybrid.py
```

### Run Tests
```bash
python test_hybrid_retrieval.py
```

## 📊 Performance Comparison

```
Query: "machine learning algorithms"

SEMANTIC Results:  Documents with ML concepts (even if worded differently)
BM25 Results:      Documents containing exact words "machine", "learning", "algorithms"
HYBRID Results:    Best of both - catches semantic matches AND keyword matches
```

## 🔄 Migration from Old System

### Old Code (Still Works)
```python
pipeline = RAGPipeline()  # Defaults to semantic
results = pipeline.retrieve(query)
```

### New Code (Recommended)
```python
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.HYBRID)
results = pipeline.retrieve(query)
```

## 🎓 Advanced Example

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode

# Create pipeline with custom settings
pipeline = RAGPipeline(
    # Chunking
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    semantic_threshold=0.75,
    
    # Retrieval
    retrieval_mode=RetrievalMode.HYBRID,
    
    # BM25 settings
    bm25_k1=1.5,
    bm25_b=0.75,
    
    # Hybrid settings
    hybrid_semantic_weight=0.6,
    hybrid_bm25_weight=0.4,
    hybrid_fusion_method="weighted",
    
    # Device
    device="cuda"  # or "cpu", "mps"
)

# Add documents
pipeline.add_document(content1, "doc1.txt")
pipeline.add_document(content2, "doc2.txt")

# Search
results = pipeline.retrieve(
    query="machine learning",
    top_k=5
)

# Process results
for i, result in enumerate(results, 1):
    chunk = result["chunk"]
    scores = result.get("score_details", {})
    
    print(f"\n[{i}] {chunk['source']}")
    print(f"Text: {chunk['text'][:100]}...")
    
    if "score_details" in result:
        print(f"Semantic: {scores['semantic']:.4f}")
        print(f"BM25: {scores['bm25']:.4f}")
        print(f"Combined: {scores['combined']:.4f}")

# Get pipeline stats
stats = pipeline.get_pipeline_stats()
print(f"\nTotal documents: {stats['total_documents']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Retrieval mode: {stats['retrieval_mode']}")
```

## ❓ Troubleshooting

### Import Error: rank_bm25
```bash
pip install rank-bm25>=0.2.2
```

### BM25 returns unexpected results
- Check if query keywords actually appear in documents
- Try adjusting `bm25_k1` and `bm25_b` parameters
- Switch to HYBRID mode

### Semantic search misses obvious matches
- Verify embedding model is appropriate
- Increase `top_k` to see more results
- Switch to HYBRID mode

### Hybrid results seem biased
- Adjust `hybrid_semantic_weight` and `hybrid_bm25_weight`
- Try switching from "weighted" to "rrf" fusion
- Check individual scores in `score_details`

## 📞 Support

For detailed documentation, see:
- `HYBRID_RETRIEVAL_GUIDE.md` - Full usage guide
- `CHROMADB_AND_BM25.md` - Technical details
- `ARCHITECTURE.md` - System design

---

**Summary:** ChromaDB does NOT support BM25/keyword search. We've added a separate BM25 retriever that works alongside vector search for true hybrid retrieval. Use `RetrievalMode.HYBRID` for best results!
