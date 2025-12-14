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
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            embedding_function: Custom embedding function (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Get or create collection
        if embedding_function:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
            )

        logger.info(
            f"ChromaVectorStore initialized: collection={collection_name}, "
            f"path={persist_directory}"
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add chunks with their embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 0),
                "word_count": chunk.get("word_count", 0),
            }
            for chunk in chunks
        ]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"source": "doc.txt"})

        Returns:
            List of result dictionaries with chunk data and scores
        """
        where = filter_dict if filter_dict else None

        results = self.collection.query(
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

    def delete_by_source(self, source: str) -> None:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier
        """
        self.collection.delete(where={"source": source})
        logger.info(f"Deleted chunks from source: {source}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
        }

    def clear(self) -> None:
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.info("Vector store cleared")

    def reset(self) -> None:
        """Reset the entire ChromaDB instance (use with caution)."""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.warning("ChromaDB instance reset")
