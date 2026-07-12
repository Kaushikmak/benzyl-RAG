#!/usr/bin/env bash
# Unified execution runner for benzyl-RAG & Guardrails
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source ".venv/bin/activate"
fi

show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  cli                 Start the interactive RAG CLI"
    echo "  index               Run or rebuild the index pipeline (Qdrant, BM25, Graph)"
    echo "  query -q \"TEXT\"     Run a one-shot query through Guardrails defenses"
    echo "  eval                Run Guardrails defense benchmarks and output ASR summary table"
    echo "  test                Run unit test suite"
    echo ""
    echo "Options:"
    echo "  --reindex           Force reindexing before running CLI or Query command"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 cli"
    echo "  $0 index"
    echo "  $0 query -q \"How does the authentication middleware work?\""
    echo "  $0 cli --reindex"
    echo "  $0 test"
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

REINDEX=false
COMMAND=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reindex)
            REINDEX=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        index|cli|query|eval|test)
            COMMAND="$1"
            shift
            # Collect remaining arguments for the command
            while [[ $# -gt 0 ]]; do
                if [ "$1" = "--reindex" ]; then
                    REINDEX=true
                else
                    PASSTHROUGH_ARGS+=("$1")
                fi
                shift
            done
            break
            ;;
        *)
            echo "Unknown command or option: $1"
            show_help
            exit 1
            ;;
    esac
done

PYTHON="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
fi

# 1. Handle explicit --reindex flag
if [ "$REINDEX" = true ]; then
    echo "[Guardrails Pipeline] Force reindexing requested..."
    echo "[Guardrails Pipeline] Wiping .data/ directory for a clean slate..."
    rm -rf .data/
    $PYTHON main.py index
fi

# 2. Auto-index if indices are completely missing and user wants to query/cli
if [[ "$COMMAND" =~ ^(cli|query)$ ]] && [ ! -d ".data/qdrant_db" ]; then
    echo "[Guardrails Pipeline] Indices not found in .data/qdrant_db. Indexing first..."
    echo "[Guardrails Pipeline] Wiping .data/ directory for a clean slate..."
    rm -rf .data/
    $PYTHON main.py index
fi

# 3. Handle explicit "index" command to ensure data is wiped before execution
# (We skip this if REINDEX=true to avoid double-deleting and double-indexing)
if [ "$COMMAND" = "index" ] && [ "$REINDEX" = false ]; then
    echo "[Guardrails Pipeline] Wiping .data/ directory for a clean slate before indexing..."
    rm -rf .data/
fi

echo "[Guardrails Pipeline] Running command: $COMMAND ${PASSTHROUGH_ARGS[*]}"
$PYTHON main.py "$COMMAND" "${PASSTHROUGH_ARGS[@]}"