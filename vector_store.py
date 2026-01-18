"""Vector store management using ChromaDB."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self):
        """Initialize ChromaDB and embedding model."""
        logger.info("🗄️  Initializing Vector Store...")
        logger.info(f"   Database path: {CHROMA_DB_DIR}")
        logger.info(f"   Collection name: {COLLECTION_NAME}")
        
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        logger.info("   ✓ ChromaDB client initialized")
        
        logger.info(f"   Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("   ✓ Embedding model loaded")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"   ✓ Collection ready (documents: {self.collection.count()})")
        logger.info("✓ Vector Store initialized successfully!")

    def add_documents(self, chunks: List[Dict], car_model: str):
        """Add document chunks to the vector store."""
        logger.info(f"Adding documents for {car_model}...")
        logger.info(f"   Number of chunks: {len(chunks)}")
        
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"   Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts).tolist()
        logger.info("   ✓ Embeddings generated")

        ids = [f"{car_model}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "car_model": car_model,
                "page": chunk["page"],
                "start_char": chunk["start_char"],
            }
            for chunk in chunks
        ]

        logger.info(f"   Storing {len(chunks)} chunks in ChromaDB...")
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

        logger.info(f"✓ Added {len(chunks)} chunks for {car_model}")

    def search(self, query: str, car_model: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant chunks."""
        logger.info(f"Searching for: '{query[:100]}...'")
        logger.info(f"   Car model filter: {car_model}")
        logger.info(f"   Top K results: {top_k}")
        
        logger.info("   Generating query embedding...")
        query_embedding = self.embedding_model.encode([query]).tolist()

        logger.info("   Querying ChromaDB...")
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"car_model": car_model},
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            logger.info(f"   Found {len(results['documents'][0])} results")
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "page": results["metadatas"][0][i]["page"],
                        "distance": results["distances"][0][i],
                    }
                )
        else:
            logger.warning("   No results found")

        logger.info(f"✓ Search complete, returning {len(formatted_results)} results")
        return formatted_results

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return self.collection.count() == 0
