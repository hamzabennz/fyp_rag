# Summary of Changes: Hybrid Retrieval System

## Overview
Added BM25 keyword search capability to the RAG system, creating a hybrid retrieval approach that combines vector embeddings and keyword matching.

## Key Finding: ChromaDB and Keyword Search
**ChromaDB does NOT support BM25 or keyword search natively.** ChromaDB is designed exclusively for vector similarity search. To add keyword search, we implemented a separate BM25 retriever using the `rank-bm25` library.

## New Files Created

1. **`src/bm25_retriever.py`**
   - BM25Retriever class for keyword-based retrieval
   - Uses rank-bm25 library (Okapi BM25 algorithm)
   - Configurable k1 and b parameters

2. **`src/hybrid_retriever.py`**
   - HybridRetriever class combining semantic + BM25
   - Two fusion methods: weighted and RRF (Reciprocal Rank Fusion)
   - Returns detailed score breakdowns

3. **`demo_hybrid.py`**
   - Comprehensive demonstration of all retrieval modes
   - 5 different demos showing various use cases
   - Comparison of SEMANTIC, BM25, and HYBRID modes

4. **`test_hybrid_retrieval.py`**
   - Unit tests for new functionality
   - Tests BM25Retriever, HybridRetriever, and RAGPipeline modes

5. **`HYBRID_RETRIEVAL_GUIDE.md`**
   - Complete documentation on hybrid retrieval
   - Usage examples and best practices
   - Configuration parameters guide

6. **`CHROMADB_AND_BM25.md`**
   - Research findings on ChromaDB capabilities
   - Detailed explanation of BM25 algorithm
   - Comparison table: Vector vs BM25 vs Hybrid

## Modified Files

1. **`src/rag_pipeline.py`**
   - Added `RetrievalMode` enum (SEMANTIC, BM25, HYBRID)
   - New initialization parameters for BM25 and hybrid settings
   - Updated `retrieve()` method to route to appropriate retriever
   - Added `_retrieve_semantic()`, `_retrieve_bm25()`, `_retrieve_hybrid()` methods
   - Updated all retriever-related methods to support multiple modes

2. **`requirements.txt`**
   - Added: `rank-bm25>=0.2.2`

3. **`README.md`**
   - Added section on hybrid retrieval capabilities
   - Updated feature list
   - Reference to new documentation

## Three Retrieval Modes

### 1. SEMANTIC (Vector Embeddings)
```python
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.SEMANTIC)
```
- Uses sentence transformers for embeddings
- Best for conceptual understanding
- Handles synonyms and paraphrasing

### 2. BM25 (Keyword Search)
```python
pipeline = RAGPipeline(retrieval_mode=RetrievalMode.BM25)
```
- Uses BM25 algorithm (rank-bm25 library)
- Best for exact keyword matches
- Fast and efficient

### 3. HYBRID (Combined)
```python
pipeline = RAGPipeline(
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.6,
    hybrid_bm25_weight=0.4,
    hybrid_fusion_method="weighted"  # or "rrf"
)
```
- Combines semantic and BM25 search
- Best overall accuracy
- Two fusion methods: weighted or RRF

## Configuration Parameters

### BM25 Parameters
- `bm25_k1`: Term frequency saturation (default: 1.5, range: 1.2-2.0)
- `bm25_b`: Length normalization (default: 0.75, range: 0-1)

### Hybrid Parameters
- `hybrid_semantic_weight`: Weight for semantic scores (default: 0.5)
- `hybrid_bm25_weight`: Weight for BM25 scores (default: 0.5)
- `hybrid_fusion_method`: "weighted" or "rrf"

## Usage Example

```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode

# Create pipeline with hybrid retrieval
pipeline = RAGPipeline(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    retrieval_mode=RetrievalMode.HYBRID,
    hybrid_semantic_weight=0.6,
    hybrid_bm25_weight=0.4,
    hybrid_fusion_method="weighted"
)

# Add document
pipeline.add_document(
    content="Your document text...",
    source="doc.txt"
)

# Search with hybrid retrieval
results = pipeline.retrieve("machine learning", top_k=5)

# Access results
for result in results:
    chunk = result["chunk"]
    score_details = result["score_details"]
    
    print(f"Text: {chunk['text']}")
    print(f"Semantic: {score_details['semantic']:.4f}")
    print(f"BM25: {score_details['bm25']:.4f}")
    print(f"Combined: {score_details['combined']:.4f}")
```

## Installation

```bash
pip install rank-bm25>=0.2.2
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Testing

Run the test script:

```bash
python test_hybrid_retrieval.py
```

Run the demo:

```bash
python demo_hybrid.py
```

## Benefits

1. **Improved Accuracy**: Hybrid approach catches both semantic and keyword matches
2. **Flexibility**: Choose the right mode for your use case
3. **Transparency**: Score breakdowns show how each method contributed
4. **No External Dependencies**: All processing happens locally
5. **Backward Compatible**: Existing code works with SEMANTIC mode (default)

## When to Use Each Mode

### Use SEMANTIC when:
- Queries are conceptual
- Dealing with synonyms
- Understanding context matters

### Use BM25 when:
- Exact keywords are critical
- Technical terms or IDs
- Speed is paramount

### Use HYBRID when:
- Want best overall results (recommended)
- Diverse query types
- Critical not to miss matches

## Technical Details

### BM25 Algorithm
- Statistical ranking function
- Based on term frequency and inverse document frequency
- Adjustable saturation and length normalization

### Fusion Methods

**Weighted Fusion**
```
score = w1 × semantic + w2 × bm25
```

**Reciprocal Rank Fusion (RRF)**
```
score = Σ(1 / (k + rank))
```

## Documentation

- **HYBRID_RETRIEVAL_GUIDE.md**: Comprehensive usage guide
- **CHROMADB_AND_BM25.md**: Research on ChromaDB and BM25
- **demo_hybrid.py**: Live demonstrations
- **test_hybrid_retrieval.py**: Test suite

## Next Steps

1. Install dependencies: `pip install rank-bm25`
2. Read HYBRID_RETRIEVAL_GUIDE.md
3. Run demo_hybrid.py to see examples
4. Test with your own data
5. Tune weights and parameters for your use case
