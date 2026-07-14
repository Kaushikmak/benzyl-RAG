#!/usr/bin/env bash
set -e

# ── 1. Create essential directories & ensure safe permissions ────────────────
mkdir -p /app/data /app/.data /app/outputs /app/.mission_state
chmod -R 777 /app/data /app/.data /app/outputs /app/.mission_state 2>/dev/null || true

# ── 2. Initialize embedded Ollama daemon if running locally ──────────────────
OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:3b}

if [[ "$OLLAMA_URL" =~ localhost|127\.0\.0\.1|0\.0\.0\.0 ]]; then
    echo "[Entrypoint] Starting embedded Ollama server in background ($OLLAMA_URL)..."
    ollama serve >/dev/null 2>&1 &
    
    # Wait until Ollama API responds
    MAX_RETRIES=30
    RETRY_COUNT=0
    until curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT+1))
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "[Entrypoint] Warning: Ollama daemon did not become ready within ${MAX_RETRIES}s. Proceeding anyway..."
    else
        echo "[Entrypoint] Ollama daemon ready."
        
        # Verify if required model is present in storage
        if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$OLLAMA_MODEL\"" && ! ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL%%:*}"; then
            echo "[Entrypoint] Model '$OLLAMA_MODEL' not found in volume storage. Pulling model..."
            ollama pull "$OLLAMA_MODEL"
            echo "[Entrypoint] Model '$OLLAMA_MODEL' ready."
        else
            echo "[Entrypoint] Model '$OLLAMA_MODEL' already present."
        fi
    fi
fi

# ── 3. Route shorthand CLI commands (`index`, `cli`, `query`, `test`) ────────
if [ $# -eq 0 ]; then
    set -- python main.py cli
elif [ "$1" = "index" ] || [ "$1" = "cli" ] || [ "$1" = "test" ]; then
    set -- python main.py "$@"
elif [ "$1" = "query" ]; then
    shift
    set -- python main.py query "$@"
fi

echo "[Entrypoint] Executing: $*"
exec "$@"
