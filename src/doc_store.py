"""
Document Store Module using SQLite

Handles persistent storage of document metadata and tracking.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import hashlib
import json
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class Document(Base):
    """SQLAlchemy model for documents."""
    
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    source = Column(String, unique=True, nullable=False, index=True)
    checksum = Column(String, nullable=False)
    chunk_count = Column(Integer, nullable=False)
    doc_metadata = Column(Text)  # JSON string (renamed from metadata to avoid SQLAlchemy conflict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SQLiteDocStore:
    """
    Document store using SQLite for persistent metadata storage.
    """

    def __init__(self, db_url: str = "sqlite:///./data/rag_docs.db"):
        """
        Initialize SQLite document store.

        Args:
            db_url: SQLAlchemy database URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"SQLiteDocStore initialized: {db_url}")

    def _calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of content."""
        return hashlib.md5(content.encode()).hexdigest()

    def add_document(
        self,
        source: str,
        content: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add or update a document record.

        Args:
            source: Document source identifier
            content: Document content (for checksum)
            chunk_count: Number of chunks created
            metadata: Additional metadata

        Returns:
            Document ID
        """
        checksum = self._calculate_checksum(content)
        doc_id = f"doc_{checksum[:12]}"
        
        # Check if document exists
        existing = self.session.query(Document).filter_by(source=source).first()
        
        if existing:
            # Update existing
            existing.checksum = checksum
            existing.chunk_count = chunk_count
            existing.doc_metadata = json.dumps(metadata or {})
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated document: {source}")
            return existing.id
        else:
            # Create new
            doc = Document(
                id=doc_id,
                source=source,
                checksum=checksum,
                chunk_count=chunk_count,
                doc_metadata=json.dumps(metadata or {}),
            )
            self.session.add(doc)
            self.session.commit()
            logger.info(f"Added document: {source} (id={doc_id})")
            return doc_id

    def get_document(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Get document by source.

        Args:
            source: Document source identifier

        Returns:
            Document data or None
        """
        doc = self.session.query(Document).filter_by(source=source).first()
        if doc:
            return {
                "id": doc.id,
                "source": doc.source,
                "checksum": doc.checksum,
                "chunk_count": doc.chunk_count,
                "metadata": json.loads(doc.doc_metadata) if doc.doc_metadata else {},
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
            }
        return None

    def exists(self, source: str) -> bool:
        """
        Check if document exists.

        Args:
            source: Document source identifier

        Returns:
            True if exists, False otherwise
        """
        count = self.session.query(Document).filter_by(source=source).count()
        return count > 0

    def has_changed(self, source: str, content: str) -> bool:
        """
        Check if document content has changed since last ingestion.

        Args:
            source: Document source identifier
            content: Current document content

        Returns:
            True if changed or new, False if unchanged
        """
        doc = self.get_document(source)
        if not doc:
            return True
        
        current_checksum = self._calculate_checksum(content)
        return doc["checksum"] != current_checksum

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents.

        Returns:
            List of document data dictionaries
        """
        docs = self.session.query(Document).all()
        return [
            {
                "id": doc.id,
                "source": doc.source,
                "checksum": doc.checksum,
                "chunk_count": doc.chunk_count,
                "metadata": json.loads(doc.doc_metadata) if doc.doc_metadata else {},
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
            }
            for doc in docs
        ]

    def delete_document(self, source: str) -> bool:
        """
        Delete a document record.

        Args:
            source: Document source identifier

        Returns:
            True if deleted, False if not found
        """
        doc = self.session.query(Document).filter_by(source=source).first()
        if doc:
            self.session.delete(doc)
            self.session.commit()
            logger.info(f"Deleted document: {source}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document store.

        Returns:
            Dictionary with stats
        """
        total = self.session.query(Document).count()
        total_chunks = self.session.query(Document).with_entities(
            Document.chunk_count
        ).all()
        total_chunk_count = sum(c[0] for c in total_chunks)
        
        return {
            "total_documents": total,
            "total_chunks": total_chunk_count,
            "db_url": self.db_url,
        }

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.session.query(Document).delete()
        self.session.commit()
        logger.info("Document store cleared")

    def close(self) -> None:
        """Close the database session."""
        self.session.close()
        logger.info("Document store session closed")
