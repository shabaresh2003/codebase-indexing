import logging
from typing import List, Dict, Any
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
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                formatted_results.append({
                    "code": doc,
                    "metadata": meta,
                    "score": 1 - dist # Convert distance to similarity score
                })
                
        return formatted_results

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
