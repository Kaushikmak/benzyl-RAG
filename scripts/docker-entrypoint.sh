#!/bin/bash
set -e

# Start Ollama service in the background
echo "Starting Ollama service..."
ollama serve > /app/ollama.log 2>&1 &

# Wait for Ollama to be available
echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is running."

# Determine the model name from config using python
MODEL_NAME=$(python -c "import sys; import os; sys.path.insert(0, os.path.abspath('.')); import app.config; print(app.config.OLLAMA_MODEL)")
if [ -z "$MODEL_NAME" ]; then
    echo "Warning: Could not determine OLLAMA_MODEL from app.config. Defaulting to qwen2.5:7b."
    MODEL_NAME="qwen2.5:7b"
fi

echo "Pulling Ollama model: $MODEL_NAME..."
ollama pull "$MODEL_NAME"

echo "Model pulled successfully. Ready to start the application."

# Execute the command passed to the docker container (e.g., python main.py web)
exec "$@"
