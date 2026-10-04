#!/usr/bin/env bash
# Script: Analyze four benchmarks and render the combined HTML report
# ==================================================================
# Requirements:
# - Supply four dataset directories with distinct folder names.
# - Reuse run_analysis.sh and render_analysis_four_benchmarks.sh.
# - Keep one model/scorer, VI/EN/CSW queries, and fixed document indexes.
# - Stop on failure; do not render partially completed results.
# - Extra analysis options apply to all benchmarks; dataset/output are managed here.
# Output: results/<benchmark>/{analysis.json,report.html} and root report.html.
# Usage: bash scripts/run_analysis_four_benchmarks.sh data/b1 data/b2 data/b3 data/b4 --documents vi --device cpu
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_analysis_four_benchmarks.sh data/b1 data/b2 data/b3 data/b4 [analysis options]"
    echo "Defaults: queries vi en csw; documents vi en; multilingual-e5-small; top-k 10."
    echo "Options apply to all datasets, e.g. --model MODEL --documents vi --device cpu."
    echo "Environment: PYTHON, ANALYSIS_OUTPUT_ROOT, ANALYSIS_REPORT_PATH."
    exit 0
fi
if [[ $# -lt 4 ]]; then
    echo "Expected four dataset directories. Use --help for an example." >&2
    exit 2
fi
datasets=("${@:1:4}")
shift 4
for option in "$@"; do
    case "$option" in
        --dataset|--dataset=*|--output|--output=*)
            echo "Dataset/output are managed by this script; use ANALYSIS_OUTPUT_ROOT for results." >&2
            exit 2 ;;
    esac
done

output_root="${ANALYSIS_OUTPUT_ROOT:-results}"
report_path="${ANALYSIS_REPORT_PATH:-report.html}"
names=()
for dataset in "${datasets[@]}"; do
    if [[ ! -d "$dataset" ]]; then
        echo "Dataset directory missing: $dataset" >&2
        exit 2
    fi
    name="$(basename -- "${dataset%/}")"
    for previous in "${names[@]}"; do
        if [[ "$previous" == "$name" ]]; then
            echo "Dataset folder names must differ to avoid overwriting results: $name" >&2
            exit 2
        fi
    done
    names+=("$name")
done

inputs=()
for i in "${!datasets[@]}"; do
    output="$output_root/${names[$i]}"
    bash scripts/run_analysis.sh --dataset "${datasets[$i]}" --output "$output" "$@"
    inputs+=("$output/analysis.json")
done
bash scripts/render_analysis_four_benchmarks.sh "${inputs[@]}" --output "$report_path"
