# Benzyl RAG

**Agentic AI Assistant for Private Document Collections.**

Benzyl RAG is a RAG (Retrieval-Augmented Generation) pipeline that parses multi-column layouts, tables, and scanned PDFs, indexes them into a hybrid vector + lexical index, and synthesizes accurate answers using local LLMs—without sending sensitive documents to the cloud.

## Key Features

- **Structured Layout Parsing (`Docling` + `Apache Tika`)**: Preserves visual hierarchy, multi-column layouts, section headings, and complex markdown tables across `.pdf`, `.docx`, `.xlsx`, `.html`, `.txt`, and more.
- **OCR Fallback (`tesseract` + `poppler-utils`)**: Automatically rasterizes and OCRs scanned or image-only PDFs lacking a text layer.
- **Hybrid Retrieval (`Qdrant` + `BM25` + `RRF`)**: Combines 1024-dimensional dense semantic embeddings (`BAAI/bge-m3`) with sparse exact keyword matching (`Rank-BM25`), normalized via Reciprocal Rank Fusion ($k_{rrf}=60$) and rescored with a Cross-Encoder reranker (`BAAI/bge-reranker-v2-m3`).
- **Adversarial Guardrails**: Defends against OWASP LLM Top 10 threats including **Corpus Poisoning** (near-duplicate clustering & outlier quarantine) and **Indirect Prompt Injection** (Unicode normalization & untrusted data block framing).
- **Human-In-The-Loop (HITL) Safety Gate**: Gated filesystem operations (`SAVE`, `APPEND`, `DELETE`) block mutations until explicitly approved.
- **Local-First Privacy**: Runs against a local [Ollama](https://ollama.com/) daemon (`qwen3:8b`). Your documents and queries never leave your machine.

## Quickstart: All-in-One Self-Contained Appliance (Recommended)

Our official prebuilt Docker image is an **All-in-One Self-Contained Appliance**. It comes pre-packaged with:
- **Embedded Ollama Daemon (`ollama serve`)** running locally inside the container
- **Pre-pulled LLM Weights (`qwen2.5:3b`)** baked directly into the image layer
- **Pre-cached HuggingFace Models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`)** pre-downloaded into container storage
- **Embedded Qdrant Client** using local file mode

You do **not** need to install Ollama separately, wait for multi-gigabyte runtime model downloads, or configure separate database containers. Everything runs offline out of the box.

### 1. Pull & Index Your Documents

Place your files (`.pdf`, `.docx`, etc.) into a local directory named `./data/` on your host machine and run:

```bash
# Pull the self-contained appliance
docker pull tastytaco/benzyl-rag:latest

# Index your documents (supervisor entrypoint automatically starts Ollama and runs indexing)
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.data:/app/.data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.mission_state:/app/.mission_state \
  tastytaco/benzyl-rag:latest index
```

### 2. Query & Interact via CLI

Start the interactive multi-agent assistant:

```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.data:/app/.data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.mission_state:/app/.mission_state \
  tastytaco/benzyl-rag:latest cli
```

Or execute one-shot queries:

```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.data:/app/.data \
  tastytaco/benzyl-rag:latest query -q "What is the authentication flow?"
```

---

## Optional: Multi-Service Docker Compose Setup

If you prefer separating services across distinct containers, create a `docker-compose.yaml` file on your host machine:

```yaml
name: benzyl-rag

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: benzyl-rag-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    container_name: benzyl-rag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama_storage:/root/.ollama

  ollama-init:
    image: ollama/ollama:latest
    container_name: benzyl-rag-ollama-init
    depends_on:
      - ollama
    entrypoint: /bin/sh
    command:
      - "-c"
      - "sleep 5; ollama --host http://ollama:11434 pull qwen2.5:3b"

  rag-app:
    image: tastytaco/benzyl-rag:latest
    container_name: benzyl-rag-app
    depends_on:
      - qdrant
      - ollama
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - DATA_DIR=/app/data
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_MODEL=qwen2.5:3b
    volumes:
      - ./data:/app/data
      - ./.data:/app/.data
      - ./outputs:/app/outputs
      - ./.mission_state:/app/.mission_state
    stdin_open: true
    tty: true
```

---

## Volume Mounts & Persistence

| Host Path          | Container Path        | Purpose                                                             |
| :----------------- | :-------------------- | :------------------------------------------------------------------ |
| `./data`           | `/app/data`           | Source directory containing your raw documents to be indexed        |
| `./.data`          | `/app/.data`          | Cached BM25 indices, metadata summaries, and graph relationships    |
| `./outputs`        | `/app/outputs`        | Exported markdown summaries and files created by FileAgent          |
| `./.mission_state` | `/app/.mission_state` | Persistent state for Human-in-the-Loop (HITL) file action approvals |

---

## License & Source Code

- **GitHub Repository**: [Kaushikmak/benzyl-RAG](https://github.com/Kaushikmak/benzyl-RAG)
