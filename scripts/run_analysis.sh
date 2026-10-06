#!/usr/bin/env bash
# Script: Analysis example
# ========================
# Requirements:
# - Configure dataset, query-document directions and model below.
# - Use canonical loader files: queries.vi/en/csw.jsonl, documents.vi/en.jsonl.
# - Use one shared qrels.tsv; keep DatasetLoader unchanged.
# - Call the existing Python CLI and allow extra arguments to override defaults.
# - Dense runs reuse matching indexes/ caches; preserve them between runs.
# Output: analysis.json and report.html under OUTPUT.
# Notebook example: !bash scripts/run_analysis.sh --dataset data/exports/fiqa --output results/exports/fiqa
# CUDA example: bash scripts/run_analysis.sh --dataset data/exports/fiqa --device cuda --batch-size 64
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Edit these values for your dataset and model.
PYTHON="${PYTHON:-python}"
DATASET="${ANALYSIS_DATASET:-$REPO_ROOT/data/fiqa}"
QUERIES=(vi en csw)
DOCUMENTS=(vi en)              # Separate Vietnamese and English indexes.
MODEL="${ANALYSIS_MODEL:-intfloat/multilingual-e5-small}"
RERANKER=""                    # Optional full-corpus cross-encoder; much more work.
TOP_K=10
METRICS=(ndcg@10 mrr@10 recall@10)
BATCH_SIZE=32
DEVICE="auto"                 # auto, cpu, cuda.
OUTPUT="${ANALYSIS_OUTPUT:-$REPO_ROOT/results/analysis_fiqa}"

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
