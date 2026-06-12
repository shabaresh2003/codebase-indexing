import os
import json
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL_NAME = "text-embedding-004"
CHROMA_DB_DIR = "./code_index"

class VectorStore:
    def __init__(self, project_id: str, location: str):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
    def get_collection(self, repo_name: str):
        """Get or create a collection for a specific repository."""
        return self.chroma_client.get_or_create_collection(name=repo_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Vertex AI."""
        if not texts:
            return []
            
        try:
            # Vertex AI has limits on batch size, usually 250
            batch_size = 250
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch_texts]
                embeddings = self.embedding_model.get_embeddings(inputs)
                all_embeddings.extend([emb.values for emb in embeddings])
                
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_chunks(self, repo_name: str, chunks: List[Dict[str, Any]], deleted_files: List[str]):
        """
        Embeds chunks, updates ChromaDB, and handles incremental indexing.
        """
        collection = self.get_collection(repo_name)
        
        # 1. Remove deleted files from ChromaDB
        if deleted_files:
            logger.info(f"Removing {len(deleted_files)} deleted files from index.")
            # ChromaDB supports deleting by metadata
            for filepath in deleted_files:
                collection.delete(where={"file": filepath})
                
        if not chunks:
            logger.info("No new chunks to process.")
            return
            
        # 2. For modified files, first remove old chunks before adding new ones
        # We find all unique files in the new chunks
        modified_files = set(c["file"] for c in chunks)
        for filepath in modified_files:
            collection.delete(where={"file": filepath})
            
        logger.info(f"Embedding and storing {len(chunks)} chunks...")
        
        # Format text for embedding: combine docstring, name, and code
        texts_to_embed = []
        ids = []
        metadatas = []
        documents = []
        
        for c in chunks:
            # Combining context for better embedding quality
            combined_text = f"Name: {c['name']}\nType: {c.get('type', 'Unknown')}\nDocstring: {c['docstring']}\nCode:\n{c['code']}"
            texts_to_embed.append(combined_text)
            
            # Create a unique ID for the chunk (file::name::lineno)
            chunk_id = f"{c['file']}::{c['name']}::{c['lineno']}"
            ids.append(chunk_id)
            
            documents.append(c["code"])
            
            metadatas.append({
                "file": c["file"],
                "name": c["name"],
                "lineno": c["lineno"],
                "type": c.get("type", "Unknown")
            })

        # Generate embeddings
        embeddings = self.embed_texts(texts_to_embed)
        
        # Add to Chroma
        # We need to batch adds if there are too many chunks
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                documents=documents[i:i + batch_size]
            )
        logger.info("Chunks successfully stored in ChromaDB.")

def get_project_id_from_service_account(service_account_path: str):
    """Extract project_id from service.json."""
    try:
        with open(service_account_path, 'r') as f:
            data = json.load(f)
            return data.get('project_id')
    except Exception as e:
        logger.error(f"Error reading service account file: {e}")
        return None
