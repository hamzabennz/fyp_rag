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

from src.rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalMode
from config import CHROMA_PERSIST_DIR, DOC_DB_URL, DEVICE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global pipeline instance (initialized on first request)
pipeline = None


def get_pipeline(strategy="semantic", retrieval_mode="hybrid", device="cuda"):
    """
    Get or create pipeline instance.
    
    Args:
        strategy: Chunking strategy ('semantic', 'layout', 'hybrid')
        retrieval_mode: Retrieval mode ('semantic', 'bm25', 'hybrid')
        device: Device to use ('cpu', 'cuda', 'mps')
    """
    global pipeline
    
    strategy_map = {
        "semantic": ChunkingStrategy.SEMANTIC,
        "layout": ChunkingStrategy.LAYOUT,
        "hybrid": ChunkingStrategy.HYBRID,
    }
    
    retrieval_map = {
        "semantic": RetrievalMode.SEMANTIC,
        "bm25": RetrievalMode.BM25,
        "hybrid": RetrievalMode.HYBRID,
    }
    
    if pipeline is None:
        logger.info(
            f"Initializing RAG pipeline: "
            f"chunking={strategy}, retrieval={retrieval_mode}, device={device}"
        )
        pipeline = RAGPipeline(
            chunking_strategy=strategy_map.get(strategy, ChunkingStrategy.SEMANTIC),
            retrieval_mode=retrieval_map.get(retrieval_mode, RetrievalMode.HYBRID),
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
    Query endpoint for RAG system with hybrid retrieval support.
    
    Expected JSON payload:
    {
        "payload": {
            "query": "Your question here",
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
    
    Retrieval Modes:
    - "semantic": Vector embedding search only
    - "bm25": Keyword search only (BM25)
    - "hybrid": Combined semantic + BM25 (recommended)
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
        retrieval_mode = payload.get('retrieval_mode', 'hybrid')  # NEW: default to hybrid
        device = payload.get('device', DEVICE)
        filter_source = payload.get('filter_source', None)
        resources = payload.get('resources', None)  # e.g., ['emails', 'sms']
        show_text = payload.get('show_text', False)
        show_scores = payload.get('show_scores', True)  # NEW: show score breakdown
        
        # Validate parameters
        if strategy not in ['semantic', 'layout', 'hybrid']:
            return jsonify({
                "error": f"Invalid strategy: {strategy}. Must be 'semantic', 'layout', or 'hybrid'"
            }), 400
        
        if retrieval_mode not in ['semantic', 'bm25', 'hybrid']:
            return jsonify({
                "error": f"Invalid retrieval_mode: {retrieval_mode}. Must be 'semantic', 'bm25', or 'hybrid'"
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1:
            return jsonify({
                "error": f"Invalid top_k: {top_k}. Must be a positive integer"
            }), 400
        
        # Get pipeline
        rag_pipeline = get_pipeline(strategy=strategy, retrieval_mode=retrieval_mode, device=device)
        
        # Execute query
        logger.info(
            f"Processing query: '{query_text[:50]}...' "
            f"(top_k={top_k}, retrieval={retrieval_mode}, resources={resources})"
        )
        results = rag_pipeline.retrieve_persistent(
            query=query_text,
            top_k=top_k,
            filter_source=filter_source,
        )
        
        # Format response based on retrieval mode
        formatted_results = []
        for result in results:
            result_data = {
                "chunk_id": result['chunk_id'],
                "source": result['metadata']['source'],
                "chunk_index": result['metadata']['chunk_index'],
                "total_chunks": result['metadata']['total_chunks'],
                "resource_type": result.get('resource_type', 'unknown'),
            }
            
            # Add scores based on retrieval mode
            if show_scores:
                result_data['similarity_score'] = result['similarity_score']
                # Note: retrieve_persistent uses semantic search
                # For full hybrid with score breakdown, use in-memory retrieve()
            
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
            "retrieval_mode": retrieval_mode,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "stats": {
                "total_documents": doc_stats['total_documents'],
                "total_chunks": vector_stats['total_chunks'],
            }
        }
        
        logger.info(
            f"Query successful: returned {len(formatted_results)} results "
            f"using {retrieval_mode} retrieval"
        )
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/query/hybrid', methods=['POST'])
def query_hybrid():
    """
    Hybrid query endpoint with detailed score breakdown.
    Uses in-memory retrieval to show both semantic and BM25 scores.
    
    Note: This requires documents to be loaded in memory.
    For persistent storage queries, use /query endpoint.
    
    Expected JSON payload:
    {
        "payload": {
            "query": "Your question here",
            "top_k": 5,
            "semantic_weight": 0.6,
            "bm25_weight": 0.4,
            "fusion_method": "weighted",
            "show_text": false
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
        
        # Extract parameters
        query_text = payload.get('query')
        if not query_text:
            return jsonify({
                "error": "Missing required field: 'query' in payload"
            }), 400
        
        top_k = payload.get('top_k', 5)
        show_text = payload.get('show_text', False)
        
        # Get pipeline (always use hybrid mode for this endpoint)
        rag_pipeline = get_pipeline(retrieval_mode="hybrid")
        
        # Check if documents are loaded
        if not rag_pipeline.processed_chunks:
            return jsonify({
                "success": False,
                "error": "No documents loaded in memory. Please ingest documents first or use /query endpoint for persistent storage."
            }), 400
        
        # Execute hybrid query
        logger.info(f"Processing hybrid query: '{query_text[:50]}...' (top_k={top_k})")
        results = rag_pipeline.retrieve(
            query=query_text,
            top_k=top_k
        )
        
        # Format response with score breakdown
        formatted_results = []
        for result in results:
            chunk = result['chunk']
            
            result_data = {
                "chunk_id": chunk.get('chunk_id', 'unknown'),
                "source": chunk.get('source', 'unknown'),
                "chunk_index": chunk.get('chunk_index', 0),
                "retrieval_method": result.get('retrieval_method', 'hybrid'),
            }
            
            # Add score details
            if 'score_details' in result:
                score_details = result['score_details']
                result_data['scores'] = {
                    "semantic": score_details.get('semantic', 0.0),
                    "bm25": score_details.get('bm25', 0.0),
                    "combined": score_details.get('combined', 0.0),
                }
                if 'semantic_rank' in score_details:
                    result_data['scores']['semantic_rank'] = score_details['semantic_rank']
                if 'bm25_rank' in score_details:
                    result_data['scores']['bm25_rank'] = score_details['bm25_rank']
            elif 'combined_score' in result:
                result_data['combined_score'] = result['combined_score']
            elif 'similarity_score' in result:
                result_data['semantic_score'] = result['similarity_score']
            elif 'bm25_score' in result:
                result_data['bm25_score'] = result['bm25_score']
            
            if show_text:
                result_data['text'] = chunk.get('text', '')
            else:
                text = chunk.get('text', '')
                preview = text[:150]
                if len(text) > 150:
                    preview += "..."
                result_data['preview'] = preview
            
            formatted_results.append(result_data)
        
        # Get pipeline stats
        pipeline_stats = rag_pipeline.get_pipeline_stats()
        
        response = {
            "success": True,
            "query": query_text,
            "retrieval_mode": "hybrid",
            "results_count": len(formatted_results),
            "results": formatted_results,
            "stats": {
                "total_documents": pipeline_stats.get('total_documents', 0),
                "total_chunks": pipeline_stats.get('total_chunks', 0),
                "retrieval_mode": pipeline_stats.get('retrieval_mode', 'hybrid'),
            }
        }
        
        logger.info(f"Hybrid query successful: returned {len(formatted_results)} results")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing hybrid query: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get storage and pipeline statistics."""
    try:
        rag_pipeline = get_pipeline()
        
        doc_stats = rag_pipeline.doc_store.get_stats()
        vector_stats = rag_pipeline.vector_store.get_stats()
        pipeline_stats = rag_pipeline.get_pipeline_stats()
        
        # List documents
        documents = rag_pipeline.doc_store.list_documents()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_documents": doc_stats['total_documents'],
                "total_chunks": vector_stats['total_chunks'],
                "db_url": doc_stats['db_url'],
                "chroma_collection": vector_stats['collection_name'],
                "retrieval_mode": pipeline_stats.get('retrieval_mode', 'unknown'),
                "chunking_strategy": pipeline_stats.get('chunking_strategy', 'unknown'),
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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Query Service")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting RAG Query Service on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
