# Multi-Collection RAG Architecture Guide

## Overview

The RAG system now supports **multiple collections** (resource types), allowing you to organize and query different types of evidence separately:

- **emails**: Email communications
- **sms**: SMS/text messages  
- **transactions**: Bank transactions
- **logs**: System/application logs
- **documents**: General documents
- **custom**: Any custom resource type

Each resource type is stored in its own ChromaDB collection, enabling:
- ✅ **Isolated queries**: Search only emails, only transactions, etc.
- ✅ **Multi-resource queries**: Search across emails + SMS simultaneously
- ✅ **Better organization**: Clean separation of data types
- ✅ **Improved precision**: Avoid cross-contamination between resource types

---

## 1. Ingesting Data with Resource Types

### Single Document Ingestion

Use the `--resource-type` parameter to specify the collection:

```bash
# Ingest an email
python ingest.py \
  --source "email_001.txt" \
  --file path/to/email.txt \
  --resource-type emails

# Ingest a transaction record
python ingest.py \
  --source "transaction_001.json" \
  --text "Transaction from Account X to Y, amount: 5000 USD..." \
  --resource-type transactions

# Ingest SMS messages
python ingest.py \
  --source "sms_batch_1" \
  --file path/to/sms.txt \
  --resource-type sms
```

### Batch Ingestion

For batch ingestion (JSON files), modify `ingest_batches.py`:

```python
# Example: Ingest emails with resource_type
num_chunks = pipeline.add_document(
    content=formatted_text,
    source=f"{batch_type}_batch_{batch_num}_id_{record['id']}",
    metadata=metadata,
    resource_type="emails",  # Specify resource type here
)
```

**Updated batch types:**
- `--batch-types emails` → resource_type="emails"
- `--batch-types sms` → resource_type="sms"  
- `--batch-types transactions` → resource_type="transactions"

---

## 2. Querying by Resource Type

### Python Script (`query.py`)

#### Query Only Emails
```bash
python query.py \
  --q "Erebus Project security vulnerability" \
  --resources emails \
  --top-k 5
```

#### Query Only Transactions
```bash
python query.py \
  --q "Large transfers to offshore accounts" \
  --resources transactions \
  --top-k 10
```

#### Query Multiple Resources (Emails + SMS)
```bash
python query.py \
  --q "Meeting at 2pm location" \
  --resources emails sms \
  --top-k 5
```

#### Query All Resources (Default)
```bash
# Omit --resources to search all collections
python query.py \
  --q "Project X deadline" \
  --top-k 5
```

---

### REST API (`/query` endpoint)

#### Query Only Emails
```bash
curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "Erebus Project quantum processor security",
      "top_k": 5,
      "resources": ["emails"],
      "show_text": true
    }
  }'
```

#### Query Multiple Resources
```bash
curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "suspicious transfer 2025",
      "top_k": 10,
      "resources": ["emails", "sms", "transactions"],
      "show_text": false
    }
  }'
```

#### Query All Resources (Omit `resources` field)
```bash
curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "query": "data exfiltration",
      "top_k": 5,
      "show_text": true
    }
  }'
```

---

## 3. API Response Format

Results now include `resource_type` field:

```json
{
  "success": true,
  "query": "Erebus Project",
  "results_count": 3,
  "results": [
    {
      "chunk_id": "emails_batch_5_id_0_0",
      "source": "emails_batch_5_id_0",
      "chunk_index": 0,
      "total_chunks": 3,
      "similarity_score": 0.8245,
      "resource_type": "emails",
      "text": "EMAIL RECORD\nFrom: s.jenkins@qubitdynamics.io..."
    },
    {
      "chunk_id": "transactions_batch_1_id_42_0",
      "source": "transactions_batch_1_id_42",
      "chunk_index": 0,
      "total_chunks": 1,
      "similarity_score": 0.7234,
      "resource_type": "transactions",
      "text": "Transaction ID: TX-2025-07-15-001..."
    }
  ],
  "stats": {
    "total_documents": 375,
    "total_chunks": 697,
    "collections": {
      "emails": 450,
      "sms": 120,
      "transactions": 127
    }
  }
}
```

---

## 4. Storage Statistics

Get stats for all collections:

```bash
curl http://127.0.0.1:5000/stats
```

**Response:**
```json
{
  "success": true,
  "doc_store": {
    "total_documents": 375,
    "total_chunks": 697
  },
  "vector_store": {
    "total_chunks": 697,
    "collections": {
      "emails": 450,
      "sms": 120,
      "transactions": 127
    },
    "resource_types": ["emails", "sms", "transactions"]
  }
}
```

---

## 5. Planner Agent Integration

Update `planner_agent.py` to generate resource-specific queries:

```python
# Example subtask output
{
  "id": 1,
  "category": "Emails",
  "search_query": "Emails 2025-07-01..2025-12-31 from @qubitdynamics.com containing 'Erebus Project' OR 'quantum processor'",
  "resource_type": "emails",  # <-- Add this field
  "reasoning": "Find email communications about the Erebus Project..."
}
```

When executing subtasks, pass `resource_type` to the API:

```python
response = requests.post("http://127.0.0.1:5000/query", json={
    "payload": {
        "query": subtask["search_query"],
        "top_k": 10,
        "resources": [subtask["resource_type"]],  # Use planner's resource type
        "show_text": True
    }
})
```

---

## 6. Migration from Old System

### For Existing Data (No Resource Type)

All existing data remains in the `default` collection. To migrate:

1. **Re-ingest with resource types:**
   ```bash
   python ingest.py --source "old_doc.txt" --file old_doc.txt --resource-type emails
   ```

2. **Or query default collection explicitly:**
   ```bash
   python query.py --q "test" --resources default
   ```

### Clear Specific Collections

```python
from src.vector_store import ChromaVectorStore

vector_store = ChromaVectorStore(persist_directory="./data/chroma")

# Clear only emails
vector_store.clear(resource_type="emails")

# Clear all collections
vector_store.clear()
```

---

## 7. Best Practices

### Resource Type Naming
- Use lowercase, plural names: `emails`, `transactions`, `sms`
- Be consistent across ingestion and queries
- Avoid special characters (use underscores: `bank_transactions`)

### Query Optimization
- **Narrow scope**: Query specific resources when possible (`resources: ["emails"]`)
- **Broad scope**: Query multiple resources for cross-referencing (`resources: ["emails", "sms"]`)
- **Default**: Omit `resources` only for exploratory queries

### Planner Agent Guidelines
- Always specify `resource_type` in subtasks
- Match resource types to investigation needs:
  - Communication analysis → `emails`, `sms`
  - Financial forensics → `transactions`
  - System forensics → `logs`

---

## 8. Technical Details

### Collection Naming Convention
Collections are named: `{base_collection_name}_{resource_type}`

Examples:
- `rag_chunks_emails`
- `rag_chunks_sms`
- `rag_chunks_transactions`

### Metadata Stored
Each chunk includes:
- `source`: Document identifier
- `chunk_index`: Chunk position
- `total_chunks`: Total chunks in document
- `word_count`: Chunk word count
- `resource_type`: Collection/resource type

### Performance
- Each collection is independent (isolated indexes)
- Multi-resource queries merge results and re-rank by similarity
- Top-k is applied **per collection**, then merged and trimmed
