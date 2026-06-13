import streamlit as st
import os
import json

from indexer import clone_or_pull_repo, index_repo
from embeddings import VectorStore, get_project_id_from_service_account
from search import SearchEngine

# Page config
st.set_page_config(page_title="Semantic Codebase Search", page_icon="🔍", layout="wide")

# Session state initialization
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "search_engine" not in st.session_state:
    st.session_state.search_engine = None

st.title("🔍 Semantic Codebase Search")
st.markdown("Index any Python GitHub repository and search it using natural language.")

# Setup Authentication
service_account_path = "service.json"
if not os.path.exists(service_account_path):
    st.error("Error: `service.json` not found in the root directory. Please provide your Google Cloud service account credentials.")
    st.stop()

# Set environment variable for Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

# Initialize clients once
if st.session_state.vector_store is None:
    project_id = get_project_id_from_service_account(service_account_path)
    if not project_id:
        st.error("Failed to read project_id from service.json")
        st.stop()
        
    try:
        # Assuming location is us-central1, can be made configurable
        st.session_state.vector_store = VectorStore(project_id, "us-central1")
        st.session_state.search_engine = SearchEngine(st.session_state.vector_store)
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI / ChromaDB: {e}")
        st.stop()

# Sidebar for indexing
with st.sidebar:
    st.header("1. Index Repository")
    github_url = st.text_input("GitHub URL", placeholder="https://github.com/pallets/flask")
    
    if st.button("Index / Update"):
        if github_url:
            with st.spinner("Cloning / Pulling repository..."):
                try:
                    repo_path, repo_name = clone_or_pull_repo(github_url)
                    st.session_state.repo_name = repo_name
                except Exception as e:
                    st.error(f"Failed to clone repository: {e}")
                    st.stop()
                    
            with st.spinner("Parsing and embedding Python files..."):
                try:
                    # Get previous hashes for this repo if it exists
                    # For simplicity in this demo, we store them in session state. 
                    # In a real app, save hashes to a local JSON file per repo.
                    hashes_file = f"./repos/{repo_name}_hashes.json"
                    previous_hashes = {}
                    if os.path.exists(hashes_file):
                        with open(hashes_file, 'r') as f:
                            previous_hashes = json.load(f)
                            
                    new_chunks, new_hashes, deleted_files = index_repo(repo_path, previous_hashes)
                    
                    if not new_chunks and not deleted_files:
                        st.success("Repository is already up to date!")
                    else:
                        st.session_state.vector_store.process_chunks(repo_name, new_chunks, deleted_files)
                        st.session_state.search_engine.invalidate_bm25_cache(repo_name)
                        
                        # Save new hashes
                        with open(hashes_file, 'w') as f:
                            json.dump(new_hashes, f)
                            
                        st.success(f"Successfully indexed! Processed {len(new_chunks)} new/modified chunks, deleted {len(deleted_files)} files.")
                except Exception as e:
                    st.error(f"Failed to index repository: {e}")
        else:
            st.warning("Please enter a GitHub URL.")

# Main area for search
st.header("2. Search")
if not st.session_state.repo_name:
    st.info("Please index a repository first using the sidebar.")
else:
    st.markdown(f"**Current Repository:** `{st.session_state.repo_name}`")
    query = st.text_input("Ask a question about the codebase:", placeholder="How is user authentication handled?")
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching and generating answer..."):
                try:
                    results = st.session_state.search_engine.hybrid_search(st.session_state.repo_name, query)
                    answer = st.session_state.search_engine.generate_answer_with_citations(query, results)
                    
                    st.markdown("### Answer")
                    st.markdown(answer)
                    
                    st.markdown("### Sources")
                    for i, res in enumerate(results):
                        with st.expander(f"[{i+1}] {res['metadata']['file']} (Score: {res['score']:.3f})"):
                            st.markdown(f"**Name:** `{res['metadata']['name']}` | **Type:** `{res['metadata']['type']}` | **Line:** `{res['metadata']['lineno']}`")
                            st.code(res['code'], language="python")
                            
                except Exception as e:
                    st.error(f"Search failed: {e}")
        else:
            st.warning("Please enter a search query.")
