"""
Vector Database Service - ChromaDB abstraction
Provides unified interface for vector operations
"""
import chromadb
from chromadb import PersistentClient
from typing import List, Dict, Any, Optional
from pathlib import Path


class VectorDBService:
    """ChromaDB vector database service for REFRAG system"""
    
    def __init__(self, persist_directory: str, collection_name: str = "rag_embeddings"):
        """
        Initialize vector database service
        
        Args:
            persist_directory: Path to ChromaDB persistence directory
            collection_name: Name of the collection to use
        """
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"[VectorDBService] Collection '{collection_name}' ready")
        print(f"[VectorDBService] Current document count: {self.collection.count()}")
    
    def add_documents(self, ids: List[str], metadatas: List[Dict], documents: List[str],
                     embeddings: Optional[List[List[float]]] = None):
        """
        Add documents to vector database (with or without embeddings)
        
        Args:
            ids: List of unique IDs for chunks
            metadatas: List of metadata dicts
            documents: List of text content
            embeddings: Optional list of embedding vectors (if None, ChromaDB auto-embeds)
        """
        try:
            if embeddings is not None:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            else:
                # Let ChromaDB auto-embed the documents
                self.collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=documents
                )
            print(f"[VectorDB] Added {len(ids)} documents {'with' if embeddings else 'without'} embeddings")
        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            raise
    
    def insert_embeddings(self, ids: List[str], embeddings: List[List[float]],
                         metadatas: List[Dict], documents: List[str]):
        """
        Insert embeddings into vector database
        
        Args:
            ids: List of unique IDs for chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            documents: List of text content
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"[VectorDB] Inserted {len(ids)} embeddings")
        except Exception as e:
            print(f"[ERROR] Failed to insert embeddings: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 10,
              where_filter: Optional[Dict] = None,
              where_document: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where_filter: Metadata filter
            where_document: Document content filter
            
        Returns:
            Dictionary with ids, distances, metadatas, documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                where_document=where_document
            )
            return results
        except Exception as e:
            print(f"[ERROR] Failed to search vectors: {e}")
            return {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
    
    def search_by_text(self, query_text: str, top_k: int = 10,
                      embedding_fn = None,
                      where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search using text query (requires embedding function)
        
        Args:
            query_text: Query text
            top_k: Number of results
            embedding_fn: Function to generate embeddings
            where_filter: Metadata filter
            
        Returns:
            Search results
        """
        if embedding_fn is None:
            raise ValueError("embedding_fn required for text search")
        
        query_embedding = embedding_fn(query_text)
        return self.search(query_embedding, top_k, where_filter)
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Documents and metadata
        """
        try:
            return self.collection.get(ids=ids)
        except Exception as e:
            print(f"[ERROR] Failed to get documents: {e}")
            return {'ids': [], 'metadatas': [], 'documents': []}
    
    def delete_by_ids(self, ids: List[str]):
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            print(f"[VectorDB] Deleted {len(ids)} documents")
        except Exception as e:
            print(f"[ERROR] Failed to delete documents: {e}")
    
    def delete_by_document(self, doc_id: str):
        """
        Delete all chunks for a document
        
        Args:
            doc_id: Document ID
        """
        try:
            self.collection.delete(where={"document_id": doc_id})
            print(f"[VectorDB] Deleted all chunks for document: {doc_id}")
        except Exception as e:
            print(f"[ERROR] Failed to delete document chunks: {e}")
    
    def delete_by_filter(self, where_filter: Dict):
        """
        Delete documents matching filter
        
        Args:
            where_filter: Metadata filter
        """
        try:
            self.collection.delete(where=where_filter)
            print(f"[VectorDB] Deleted documents matching filter")
        except Exception as e:
            print(f"[ERROR] Failed to delete by filter: {e}")
    
    def update_metadata(self, ids: List[str], metadatas: List[Dict]):
        """
        Update metadata for existing documents
        
        Args:
            ids: List of document IDs
            metadatas: List of new metadata dicts
        """
        try:
            self.collection.update(ids=ids, metadatas=metadatas)
            print(f"[VectorDB] Updated metadata for {len(ids)} documents")
        except Exception as e:
            print(f"[ERROR] Failed to update metadata: {e}")
    
    def count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at first N documents
        
        Args:
            limit: Number of documents to return
            
        Returns:
            Sample documents
        """
        return self.collection.peek(limit=limit)
    
    def reset_collection(self):
        """Delete all documents in collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorDB] Collection '{self.collection_name}' reset")
        except Exception as e:
            print(f"[ERROR] Failed to reset collection: {e}")
