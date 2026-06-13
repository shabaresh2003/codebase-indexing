# 🔍 Semantic Codebase Search

An intelligent, AI-powered search engine for Python codebases. This application clones any Python GitHub repository, parses the code into semantic chunks, and allows you to search and ask questions about your codebase using natural language. It leverages Google's Vertex AI and ChromaDB to synthesize precise answers backed by code citations.

---

## ✨ Features

- **🧠 Hybrid Search (Semantic + BM25)**: Combines Vertex AI embeddings (`text-embedding-004`) for conceptual understanding with BM25 keyword search for exact identifier matching. Uses Reciprocal Rank Fusion (RRF) to deliver the absolute best results.
- **⚡ Incremental Indexing**: Uses Python's `ast` module to intelligently parse code down to the function and class level. Caches file hashes to ensure that subsequent updates only re-index files that have been modified or added.
- **🤖 Retrieval-Augmented Generation (RAG)**: Employs `gemini-2.5-pro` to answer complex questions about your codebase.
- **📌 Inline Citations**: Generates answers with strict, inline citations linking directly back to the exact file, class, and line number where the context was found.
- **🗂️ Local Vector Storage**: Uses ChromaDB to efficiently store and query embeddings locally, completely avoiding the need for expensive cloud database hosting.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM & Embeddings**: [Google Vertex AI](https://cloud.google.com/vertex-ai) (`gemini-2.5-pro`, `text-embedding-004`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Keyword Search**: `rank_bm25`
- **AST Parsing**: Python built-in `ast` module
- **Git Integration**: `GitPython`

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** installed on your system.
2. A **Google Cloud Project** with Vertex AI API enabled.
3. A Google Cloud Service Account JSON key (`service.json`) with permissions to access Vertex AI.

### Installation

1. **Clone this repository** (or download the source):
   ```bash
   git clone https://github.com/shabaresh2003/codebase-indexing.git
   cd codebase-indexing
   ```

2. **Place your Google Cloud credentials** in the root directory and name the file `service.json`.

3. **Set up a virtual environment** and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit application:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Index a Repository**: 
   - Open the app and look at the sidebar.
   - Enter a valid Python GitHub repository URL (e.g., `https://github.com/pallets/flask`).
   - Click **Index / Update**. 
   - *Note: The first time you index a large repo, it will take a minute or two to generate embeddings. Subsequent updates will be instant!*

2. **Search and Ask Questions**:
   - In the main search bar, ask a question about the codebase. 
   - Example: *"How is user authentication handled?"* or *"Where is the calculate_revenue function defined?"*
   - The engine will run a Hybrid Search, retrieve the top context, and Gemini will generate a summarized answer with inline citations.
   - You can expand the **Sources** accordions below the answer to view the exact code chunks that were referenced.

---

## 🔮 Future Roadmap

- [ ] Support for Multi-Language parsing using `Tree-sitter` (JS, Go, Rust, etc.)
- [ ] Indexing non-code context files (Markdown, Configs)
- [ ] Conversational UI with persistent chat history
- [ ] Knowledge Graph extraction (GraphRAG) for mapping dependencies
