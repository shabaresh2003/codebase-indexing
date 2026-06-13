import logging
import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from vertexai.language_models import TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from embeddings import VectorStore

logger = logging.getLogger(__name__)

# Constants
LLM_MODEL_NAME = "gemini-2.5-pro" # Excellent context window and coding ability

class SearchEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.generative_model = GenerativeModel(LLM_MODEL_NAME)
        self._bm25_cache = {}

    def get_or_create_bm25_index(self, repo_name: str) -> Dict[str, Any]:
        """Loads all documents from ChromaDB and builds a BM25 index on the fly."""
        if repo_name in self._bm25_cache:
            return self._bm25_cache[repo_name]
            
        logger.info(f"Building BM25 index for {repo_name}...")
        collection = self.vector_store.get_collection(repo_name)
        all_data = collection.get(include=['documents', 'metadatas'])
        
        documents = all_data.get('documents', [])
        metadatas = all_data.get('metadatas', [])
        ids = all_data.get('ids', [])
        
        if not documents:
            return None
            
        # Tokenize by splitting on non-alphanumeric characters
        tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        self._bm25_cache[repo_name] = {
            'bm25': bm25,
            'documents': documents,
            'metadatas': metadatas,
            'ids': ids,
            # Create a quick lookup for RRF
            'id_to_data': {doc_id: {'code': doc, 'metadata': meta} for doc_id, doc, meta in zip(ids, documents, metadatas)}
        }
        return self._bm25_cache[repo_name]

    def invalidate_bm25_cache(self, repo_name: str):
        """Clear the cache if the repository is re-indexed."""
        if repo_name in self._bm25_cache:
            del self._bm25_cache[repo_name]

    def semantic_search(self, repo_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Embeds the query and searches ChromaDB for the most relevant code chunks.
        """
        collection = self.vector_store.get_collection(repo_name)
        
        # Embed the query
        inputs = [TextEmbeddingInput(query, "RETRIEVAL_QUERY")]
        query_embeddings = self.vector_store.embedding_model.get_embeddings(inputs)
        query_emb_values = query_embeddings[0].values
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_emb_values],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for doc_id, doc, meta, dist in zip(ids, docs, metadatas, distances):
                formatted_results.append({
                    "id": doc_id,
                    "code": doc,
                    "metadata": meta,
                    "score": 1 - dist # Convert distance to similarity score
                })
                
        return formatted_results

    def hybrid_search(self, repo_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Combines Semantic Search and BM25 Search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Get Top 60 Semantic Results
        semantic_results = self.semantic_search(repo_name, query, top_k=60)
        
        # 2. Get Top 60 BM25 Results
        bm25_data = self.get_or_create_bm25_index(repo_name)
        if not bm25_data:
            return semantic_results[:top_k]
            
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = bm25_data['bm25'].get_scores(tokenized_query)
        
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:60]
        
        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_k = 60 # Standard constant for RRF
        
        # Add semantic ranks
        for rank, res in enumerate(semantic_results):
            doc_id = res['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # Add BM25 ranks
        for rank, idx in enumerate(top_bm25_indices):
            # Only consider non-zero BM25 scores
            if bm25_scores[idx] <= 0:
                break
                
            doc_id = bm25_data['ids'][idx]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # 4. Sort by RRF score and format top_k results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids[:top_k]:
            data = bm25_data['id_to_data'][doc_id]
            final_results.append({
                "id": doc_id,
                "code": data['code'],
                "metadata": data['metadata'],
                "score": rrf_scores[doc_id]
            })
            
        return final_results

    def generate_answer_with_citations(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Uses Gemini to generate an answer based on the retrieved code chunks, including citations.
        """
        if not search_results:
            return "I couldn't find any relevant code to answer your question."
            
        context = ""
        for i, result in enumerate(search_results):
            meta = result['metadata']
            context += f"\n--- Source [{i+1}]: {meta['file']} (Function/Class: {meta['name']}, Line: {meta['lineno']}) ---\n"
            context += result['code']
            context += "\n---------------------------------------------------\n"
            
        prompt = f"""
        You are an expert software engineer analyzing a codebase.
        Answer the user's query using ONLY the provided code snippets from the codebase.
        
        When referring to specific parts of the code, you MUST use inline citations in the format [Source N] where N is the source number.
        For example: "The user is fetched using the `get_user` function [Source 1], which handles caching [Source 2]."
        
        If the answer cannot be found in the provided context, say so clearly. Do not make up facts about the codebase.

        <query>
        {query}
        </query>

        <context>
        {context}
        </context>
        """
        
        response = self.generative_model.generate_content(prompt)
        return response.text
