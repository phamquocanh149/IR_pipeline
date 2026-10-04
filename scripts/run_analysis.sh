#!/usr/bin/env bash
# Script: Analysis example
# ========================
# Requirements:
# - Configure dataset, query-document directions and model below.
# - Use canonical loader files: queries.vi/en/csw.jsonl, documents.vi/en.jsonl.
# - Use one shared qrels.tsv; keep DatasetLoader unchanged.
# - Call the existing Python CLI and allow extra arguments to override defaults.
# Output: analysis.json and report.html under OUTPUT.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Edit these values for your dataset and model.
PYTHON="python"
DATASET="$REPO_ROOT/data/fiqa"
QUERIES=(vi en csw)
DOCUMENTS=(en)                 # Fixed English corpus.
MODEL="intfloat/multilingual-e5-small"
RERANKER=""                    # Optional cross-encoder; empty uses cosine.
TOP_K=10
METRICS=(ndcg@10 mrr@10 recall@10)
BATCH_SIZE=32
DEVICE="auto"                 # auto, cpu, cuda.
OUTPUT="$REPO_ROOT/results/analysis_fiqa_en"

args=(
    --dataset "$DATASET"
    --queries "${QUERIES[@]}"
    --documents "${DOCUMENTS[@]}"
    --model "$MODEL"
    --model-type single-vector
    --retriever dense
    --top-k "$TOP_K"
    --metrics "${METRICS[@]}"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --output "$OUTPUT"
)

if [[ -n "$RERANKER" ]]; then
    args+=(--reranker "$RERANKER")
fi

# Additional CLI arguments override the configuration above.
exec "$PYTHON" "$SCRIPT_DIR/run_analysis.py" "${args[@]}" "$@"
