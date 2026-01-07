"""
Vector Store Module using ChromaDB

Handles persistent storage and retrieval of embeddings using ChromaDB.
Configured for Kubeflow/Docker environments with persistent volumes.
"""

from typing import List, Dict, Any, Optional
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Vector store using ChromaDB for persistent embedding storage.
    Supports multiple collections for different resource types (emails, SMS, transactions, etc.).
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_directory: str = "./data/chroma",
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Base name for collections (will be prefixed with resource type)
            persist_directory: Directory for persistent storage
            embedding_function: Custom embedding function (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Cache for collections (resource_type -> collection)
        self.collections = {}

        logger.info(
            f"ChromaVectorStore initialized: base_collection={collection_name}, "
            f"path={persist_directory}"
        )

    def _get_collection(self, resource_type: str = "default"):
        """
        Get or create a collection for a specific resource type.

        Args:
            resource_type: Type of resource (e.g., 'emails', 'sms', 'transactions')

        Returns:
            ChromaDB collection
        """
        # Check cache
        if resource_type in self.collections:
            return self.collections[resource_type]

        # Create collection name
        coll_name = f"{self.collection_name}_{resource_type}"

        # Get or create collection
        if self.embedding_function:
            collection = self.client.get_or_create_collection(
                name=coll_name,
                embedding_function=self.embedding_function,
            )
        else:
            collection = self.client.get_or_create_collection(
                name=coll_name,
            )

        # Cache it
        self.collections[resource_type] = collection
        logger.info(f"Collection created/loaded: {coll_name}")

        return collection

    def list_collections(self) -> List[str]:
        """
        List all available collections (resource types).

        Returns:
            List of resource types
        """
        all_collections = self.client.list_collections()
        prefix = f"{self.collection_name}_"
        resource_types = []
        
        for coll in all_collections:
            if coll.name.startswith(prefix):
                resource_type = coll.name[len(prefix):]
                resource_types.append(resource_type)
        
        return resource_types

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        resource_type: str = "default",
    ) -> None:
        """
        Add chunks with their embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors
            resource_type: Type of resource (e.g., 'emails', 'sms', 'transactions')
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Get collection for this resource type
        collection = self._get_collection(resource_type)

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 0),
                "word_count": chunk.get("word_count", 0),
                "resource_type": resource_type,
            }
            for chunk in chunks
        ]

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to collection '{resource_type}'")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        resource_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding similarity.
        Can search across multiple resource types (collections).

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return per collection
            filter_dict: Optional metadata filters (e.g., {"source": "doc.txt"})
            resource_types: List of resource types to search (e.g., ['emails', 'sms']).
                          If None, searches all available collections.

        Returns:
            List of result dictionaries with chunk data and scores
        """
        # Determine which collections to search
        if resource_types is None:
            # Search all collections
            resource_types = self.list_collections()
            if not resource_types:
                # Fallback to default collection if no collections exist
                resource_types = ["default"]
        
        where = filter_dict if filter_dict else None
        all_results = []

        # Query each collection
        for resource_type in resource_types:
            try:
                collection = self._get_collection(resource_type)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )

                # Format results from this collection
                for i in range(len(results["ids"][0])):
                    all_results.append({
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i],
                        "resource_type": resource_type,
                    })
            except Exception as e:
                logger.warning(f"Error querying collection '{resource_type}': {e}")
                continue

        # Sort all results by similarity score and return top_k
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        formatted_results = all_results[:top_k]

        logger.info(
            f"Retrieved {len(formatted_results)} results from {len(resource_types)} collection(s)"
        )
        return formatted_results

    def search_legacy(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Legacy search method for backward compatibility (searches default collection only).

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of result dictionaries with chunk data and scores
        """
        where = filter_dict if filter_dict else None
        collection = self._get_collection("default")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity_score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
            })

        logger.info(f"Retrieved {len(formatted_results)} results from vector store")
        return formatted_results

    def delete_by_source(self, source: str, resource_type: str = "default") -> None:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier
            resource_type: Type of resource
        """
        collection = self._get_collection(resource_type)
        collection.delete(where={"source": source})
        logger.info(f"Deleted chunks from source '{source}' in collection '{resource_type}'")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        resource_types = self.list_collections()
        total_chunks = 0
        collection_stats = {}
        
        for resource_type in resource_types:
            collection = self._get_collection(resource_type)
            count = collection.count()
            total_chunks += count
            collection_stats[resource_type] = count
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": total_chunks,
            "persist_directory": self.persist_directory,
            "collections": collection_stats,
            "resource_types": resource_types,
        }

    def clear(self, resource_type: Optional[str] = None) -> None:
        """
        Clear data from collections.
        
        Args:
            resource_type: If specified, clear only this resource type.
                         If None, clear all collections.
        """
        if resource_type:
            # Clear specific collection
            coll_name = f"{self.collection_name}_{resource_type}"
            self.client.delete_collection(coll_name)
            if resource_type in self.collections:
                del self.collections[resource_type]
            logger.info(f"Cleared collection: {resource_type}")
        else:
            # Clear all collections
            resource_types = self.list_collections()
            for rt in resource_types:
                coll_name = f"{self.collection_name}_{rt}"
                self.client.delete_collection(coll_name)
            self.collections.clear()
            logger.info("All collections cleared")

    def reset(self) -> None:
        """Reset the entire ChromaDB instance (use with caution)."""
        self.client.reset()
        self.collections.clear()
        logger.warning("ChromaDB instance reset")
