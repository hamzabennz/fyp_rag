#!/usr/bin/env python3
"""
Flask REST API for RAG Query Service

Exposes query endpoint for the RAG system with persistent storage.
"""

from flask import Flask, request, jsonify
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, ChunkingStrategy
from config import CHROMA_PERSIST_DIR, DOC_DB_URL, DEVICE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Add CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global pipeline instance (initialized on first request)
pipeline = None


def get_pipeline(strategy="semantic", device="cuda"):
    """Get or create pipeline instance."""
    global pipeline
    
    strategy_map = {
        "semantic": ChunkingStrategy.SEMANTIC,
        "layout": ChunkingStrategy.LAYOUT,
        "hybrid": ChunkingStrategy.HYBRID,
    }
    
    if pipeline is None:
        logger.info(f"Initializing RAG pipeline with {strategy} strategy on {device}")
        pipeline = RAGPipeline(
            chunking_strategy=strategy_map.get(strategy, ChunkingStrategy.SEMANTIC),
            device=device,
            use_persistent_storage=True,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
            sqlite_db_url=DOC_DB_URL,
        )
        logger.info("Pipeline initialized successfully")
    
    return pipeline


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "RAG Query Service",
        "version": "0.1.0"
    }), 200


@app.route('/query', methods=['POST'])
def query():
    """
    Query endpoint for RAG system.
    
    Expected JSON payload:
    {
        "payload": {
            "query": "Your question here",
            "top_k": 5,
            "strategy": "semantic",
            "device": "cuda",
            "filter_source": null,
            "resources": ["emails", "sms"],
            "show_text": false,
            "retrieval_mode": "embedding"
        }
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'payload' not in data:
            return jsonify({
                "error": "Missing 'payload' in request body"
            }), 400
        
        payload = data['payload']
        
        # Extract parameters with defaults
        query_text = payload.get('query')
        if not query_text:
            return jsonify({
                "error": "Missing required field: 'query' in payload"
            }), 400
        
        top_k = payload.get('top_k', 5)
        strategy = payload.get('strategy', 'semantic')
        device = payload.get('device', DEVICE)
        filter_source = payload.get('filter_source', None)
        resources = payload.get('resources', None)  # e.g., ['emails', 'sms']
        show_text = payload.get('show_text', False)
        retrieval_mode = payload.get('retrieval_mode', 'embedding')  # 'embedding', 'bm25', 'hybrid'
        
        # Validate parameters
        if strategy not in ['semantic', 'layout', 'hybrid']:
            return jsonify({
                "error": f"Invalid strategy: {strategy}. Must be 'semantic', 'layout', or 'hybrid'"
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1:
            return jsonify({
                "error": f"Invalid top_k: {top_k}. Must be a positive integer"
            }), 400
        
        if retrieval_mode not in ['embedding', 'bm25', 'hybrid']:
            return jsonify({
                "error": f"Invalid retrieval_mode: {retrieval_mode}. Must be 'embedding', 'bm25', or 'hybrid'"
            }), 400
        
        # Get pipeline
        rag_pipeline = get_pipeline(strategy=strategy, device=device)
        
        # Execute query
        logger.info(f"Processing query: {query_text[:50]}... (top_k={top_k}, resources={resources}, retrieval_mode={retrieval_mode})")
        results = rag_pipeline.retrieve_persistent(
            query=query_text,
            top_k=top_k,
            filter_source=filter_source,
            resource_types=resources,
            retrieval_mode=retrieval_mode,
        )
        
        # Format response
        formatted_results = []
        for result in results:
            result_data = {
                "chunk_id": result['chunk_id'],
                "source": result['metadata']['source'],
                "chunk_index": result['metadata']['chunk_index'],
                "total_chunks": result['metadata']['total_chunks'],
                "similarity_score": result['similarity_score'],
                "resource_type": result.get('resource_type', 'unknown'),
                "retrieval_method": result.get('retrieval_method', retrieval_mode),
            }
            
            # Add scores for hybrid mode
            if retrieval_mode == 'hybrid':
                result_data['embedding_score'] = result.get('embedding_score', 0.0)
                result_data['bm25_score'] = result.get('bm25_score', 0.0)
            
            if show_text:
                result_data['text'] = result['text']
            else:
                # Provide preview
                preview = result['text'][:150]
                if len(result['text']) > 150:
                    preview += "..."
                result_data['preview'] = preview
            
            formatted_results.append(result_data)
        
        # Get storage stats
        doc_stats = rag_pipeline.doc_store.get_stats()
        vector_stats = rag_pipeline.vector_store.get_stats()
        
        response = {
            "success": True,
            "query": query_text,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "stats": {
                "total_documents": doc_stats['total_documents'],
                "total_chunks": vector_stats['total_chunks'],
            }
        }
        
        logger.info(f"Query successful: returned {len(formatted_results)} results")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get storage statistics."""
    try:
        rag_pipeline = get_pipeline()
        
        doc_stats = rag_pipeline.doc_store.get_stats()
        vector_stats = rag_pipeline.vector_store.get_stats()
        
        # List documents
        documents = rag_pipeline.doc_store.list_documents()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_documents": doc_stats['total_documents'],
                "total_chunks": vector_stats['total_chunks'],
                "db_url": doc_stats['db_url'],
                "chroma_collection": vector_stats['collection_name'],
                "collections": vector_stats.get('collections', {}),  # Add this
                "resource_types": vector_stats.get('resource_types', []),  # Add this
            },
            "documents": [
                {
                    "source": doc['source'],
                    "chunk_count": doc['chunk_count'],
                    "created_at": doc['created_at'],
                }
                for doc in documents
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/documents', methods=['GET'])
def documents():
    """
    Get list of all documents with metadata.
    
    Optional query parameters:
    - resource_type: Filter by resource type (e.g., "emails")
    """
    try:
        rag_pipeline = get_pipeline()
        resource_type = request.args.get('resource_type', None)
        
        # Get all documents
        all_documents = rag_pipeline.doc_store.list_documents()
        
        # Filter by resource type if specified
        if resource_type:
            filtered_docs = []
            for doc in all_documents:
                doc_resource_type = doc.get('metadata', {}).get('resource_type', 'default')
                if doc_resource_type == resource_type:
                    filtered_docs.append(doc)
            documents_list = filtered_docs
        else:
            documents_list = all_documents
        
        # Format response
        formatted_docs = [
            {
                "id": doc['id'],
                "source": doc['source'],
                "chunk_count": doc['chunk_count'],
                "resource_type": doc.get('metadata', {}).get('resource_type', 'default'),
                "created_at": doc['created_at'],
                "updated_at": doc['updated_at'],
            }
            for doc in documents_list
        ]
        
        return jsonify({
            "success": True,
            "count": len(formatted_docs),
            "documents": formatted_docs
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/document/<path:source_id>', methods=['GET'])
def get_document(source_id):
    """
    Get full content of a document by source ID.
    
    Args:
        source_id: Document source identifier (from query results, can include slashes)
    """
    try:
        rag_pipeline = get_pipeline()
        
        # Get document metadata
        doc = rag_pipeline.doc_store.get_document(source_id)
        
        if not doc:
            return jsonify({
                "success": False,
                "error": f"Document not found: {source_id}"
            }), 404
        
        # Get all chunks for this source from vector store
        chunks = rag_pipeline.vector_store.get_chunks_by_source(source_id)
        
        # Sort by chunk index and reconstruct full content
        sorted_chunks = sorted(chunks, key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
        full_content = "\n\n---\n\n".join([chunk['text'] for chunk in sorted_chunks])
        
        return jsonify({
            "success": True,
            "source": source_id,
            "metadata": doc.get('metadata', {}),
            "chunk_count": doc['chunk_count'],
            "created_at": doc['created_at'],
            "chunks": [
                {
                    "index": chunk.get('metadata', {}).get('chunk_index', 0),
                    "total": chunk.get('metadata', {}).get('total_chunks', 0),
                    "text": chunk['text'],
                    "word_count": chunk.get('metadata', {}).get('word_count', 0),
                }
                for chunk in sorted_chunks
            ],
            "full_content": full_content
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/collection/<resource_type>', methods=['GET'])
def get_collection(resource_type):
    """
    Get all documents in a specific collection (resource type).
    
    Args:
        resource_type: Resource type (e.g., "emails", "sms")
    """
    try:
        rag_pipeline = get_pipeline()
        
        # Get all documents and filter by resource type
        all_documents = rag_pipeline.doc_store.list_documents()
        
        collection_docs = [
            doc for doc in all_documents
            if doc.get('metadata', {}).get('resource_type', 'default') == resource_type
        ]
        
        if not collection_docs:
            return jsonify({
                "success": True,
                "resource_type": resource_type,
                "count": 0,
                "documents": [],
                "message": f"No documents found in collection '{resource_type}'"
            }), 200
        
        # Format response
        formatted_docs = [
            {
                "id": doc['id'],
                "source": doc['source'],
                "chunk_count": doc['chunk_count'],
                "created_at": doc['created_at'],
                "updated_at": doc['updated_at'],
            }
            for doc in collection_docs
        ]
        
        # Get total chunks in this collection
        vector_stats = rag_pipeline.vector_store.get_stats()
        collection_chunk_count = vector_stats.get('collections', {}).get(resource_type, 0)
        
        return jsonify({
            "success": True,
            "resource_type": resource_type,
            "document_count": len(formatted_docs),
            "chunk_count": collection_chunk_count,
            "documents": formatted_docs
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving collection: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Query Service")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting RAG Query Service on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
