# Obsidian Hybrid Graph RAG

A powerful, local-first Retrieval-Augmented Generation (RAG) assistant specifically designed to chat with and explore your Obsidian Vault. 

This application combines multiple advanced retrieval strategies to give you highly accurate answers based purely on your personal notes:
- **Vector Search**: Semantic similarity using ChromaDB and `BAAI/bge-m3` embeddings.
- **Keyword Search (BM25)**: Fast, exact-match text search.
- **Graph Traversal**: Explores the logical links and connections between your Obsidian notes.
- **Reranking**: Refines and scores the retrieved context using `BAAI/bge-reranker-v2-m3`.
- **Local LLMs**: Entirely private generation using **Ollama**.

It features a robust **FastAPI backend** and a simple **Next.js frontend** with conversational history, source-tracking, and feedback controls.

---

## Project Structure
```text
obsidianRAG/
├── app/                  # Core RAG logic, config, and retrieval modules
├── data/                 # Stores your Obsidian Vault, graphs, and BM25 chunks
├── web/                  # Next.js web UI (TypeScript)
├── indexing/             # Pipeline for chunking notes, building vectors & graphs
├── scripts/              # Helper scripts (e.g., Docker entrypoints)
├── vectorstore/          # ChromaDB vector database files
├── api.py                # FastAPI server handling queries and DB history
├── main.py               # Single unified CLI entrypoint
└── Dockerfile            # Fully automated Docker setup
```

---

##  Setup & Usage (Local Python Environment)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your host machine.

### 2. Configuration
1. Place your Obsidian vault folder inside `data/obsidianVault/` (or update `VAULT_PATH` in `app/config.py`).
2. Make sure Ollama is running and pull your preferred model (e.g., `ollama pull qwen3:8b` or whatever is defined as `OLLAMA_MODEL` in `app/config.py`).

### 3. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Indexing Your Vault
Before you can chat, you must index your Obsidian vault. This processes your Markdown files into vector embeddings, keyword chunks, and a knowledge graph.
```bash
python main.py index
```
*Note: You need to re-run this command whenever you significantly update your Obsidian notes.*

### 5. Running the Application
You have two ways to interact with your assistant:

**Option A: Web Application (Recommended)**
Starts both the FastAPI backend and the Next.js UI.
```bash
python main.py web
```
- UI available at: `http://localhost:3000`
- API documentation at: `http://localhost:8000/docs`

**Option B: Command Line Interface**
Chat with your vault directly in the terminal.
```bash
python main.py cli
```

---

## Setup & Usage (Docker)

If you prefer to run the application entirely containerized (including Ollama), use Docker. 

> **Note:** The Docker container is configured to automatically start the Ollama service and pull your configured model on startup.

1. Ensure your vault is located at `data/obsidianVault/`.
2. Build the Docker image:
```bash
docker build -t obsidian-rag .
```
3. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 obsidian-rag
```

**What the container does automatically:**
- Installs Python dependencies.
- Installs the Ollama service.
- When started, runs `scripts/docker-entrypoint.sh` which boots Ollama, pulls your target model, and finally executes `python main.py web`.

Once the container says "Both services are running!", navigate to `http://localhost:3000` in your browser.

## Python to TypeScript API Contract (Recommended)

Best approach for this stack:
- Keep **FastAPI** as the source of truth for request/response models.
- Use FastAPI's OpenAPI schema at `http://localhost:8000/openapi.json`.
- Generate TypeScript types/client from OpenAPI (for example with `openapi-typescript` or `orval`) and consume from Next.js.

Current repo includes a typed API layer in `web/lib/types.ts` and `web/lib/api.ts` as a direct bridge.

---

## Configuration Details
All major settings are located in `app/config.py`:
- `OLLAMA_MODEL`: The local LLM to use (default setup implies a Qwen model, but Llama 3 or Mistral work perfectly).
- `EMBED_MODEL`: The HuggingFace embedding model (default: `BAAI/bge-m3`).
- `RERANK_MODEL`: The cross-encoder used for reranking (default: `BAAI/bge-reranker-v2-m3`).
- Chunking limits, vector store paths, and device settings (`cuda` vs `cpu`) can also be toggled here.
