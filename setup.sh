#!/usr/bin/env bash
set -e

echo "[Setup] Creating local database, model storage, and document directories..."
mkdir -p data/qdrant_storage data/ollama_storage .data .mission_state

echo "[Setup] Building and launching containers (Qdrant, Ollama, RAG App)..."
docker compose up -d --build

echo ""
echo "===================================================="
echo " Ready to go! "
echo "===================================================="
echo "Note: Ollama is running inside Docker and will automatically"
echo "pull default model 'qwen3:8b' in the background."
echo ""
echo "1. Drop your files into the local './data' folder."
echo "2. Index them: docker compose exec -it rag-app python main.py index"
echo "3. Run the CLI: docker compose exec -it rag-app python main.py cli"
echo "===================================================="