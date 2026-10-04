#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Edit these values for your dataset and model.
PYTHON="python"
DATASET="$REPO_ROOT/data/fiqa"
DIRECTIONS=(vi-en en-en csw-en) # query-document languages; fixed English corpus.
MODEL="MODEL_ID"               # Embedding model name or local path.
RERANKER=""                    # Optional cross-encoder; empty uses cosine.
TOP_K=100
BATCH_SIZE=32
DEVICE="auto"                 # auto, cpu, cuda.
OUTPUT="$REPO_ROOT/results/analysis_fiqa_en"

args=(
    --dataset "$DATASET"
    --direction "${DIRECTIONS[@]}"
    --model "$MODEL"
    --top-k "$TOP_K"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --output "$OUTPUT"
)

if [[ -n "$RERANKER" ]]; then
    args+=(--reranker "$RERANKER")
fi

# Additional CLI arguments override the configuration above.
exec "$PYTHON" "$SCRIPT_DIR/run_analysis.py" "${args[@]}" "$@"
