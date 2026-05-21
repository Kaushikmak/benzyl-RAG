#!/bin/bash

set -e

source .venv/bin/activate

gnome-terminal -- bash -c "
watch -n 1 nvidia-smi;
exec bash
"

if [ ! -d "vectorstore" ]; then
    echo "Vectorstore missing..."
    echo "Building indexes..."

    python -m indexing.build_indexes
fi

python main.py