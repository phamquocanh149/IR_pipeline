#!/usr/bin/env bash
# Render four saved benchmark reports from the same model and scorer.
# Requirements: reuse render_analysis.py; never reload models or change source JSON.
# Output: report.html at repository root; extra arguments can override --output.
# bash scripts/render_analysis_four_benchmarks.sh results/b1/analysis.json results/b2/analysis.json results/b3/analysis.json results/b4/analysis.json
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/render_analysis_four_benchmarks.sh b1/analysis.json b2/analysis.json b3/analysis.json b4/analysis.json [--output report.html]"
    exit 0
fi
if [[ $# -lt 4 ]]; then
    echo "Expected four analysis.json paths. Use --help for an example." >&2
    exit 2
fi
inputs=("${@:1:4}")
shift 4
exec "${PYTHON:-python}" scripts/render_analysis.py "${inputs[0]}" \
    --benchmark-inputs "${inputs[@]}" --output report.html "$@"
