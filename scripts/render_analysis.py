"""
Script: Refresh an analysis report
=================================
Render saved analysis.json with the current HTML layout.

Requirements
------------
- Reuse write_report; do not reload models, encode texts or repeat retrieval.
- Keep the input JSON unchanged and write report.html beside it by default.
- Optional --embedding-inputs creates a two-column model/benchmark grid from
  saved PCA projections without combining benchmarks or fitting PCA again.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_system.analysis.report import write_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh HTML from saved analysis results.")
    parser.add_argument("input", type=Path, help="Path to analysis.json.")
    parser.add_argument("--output", type=Path, help="Default: report.html beside the input JSON.")
    parser.add_argument("--embedding-inputs", nargs="+", type=Path,
                        help="Saved reports to show as a two-column model/benchmark PCA grid.")
    parser.add_argument("--benchmark-inputs", nargs=4, type=Path,
                        help="Four benchmark reports from the same model/scorer; one benchmark per panel.")
    args = parser.parse_args()
    output = args.output or args.input.with_name("report.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if args.benchmark_inputs:
        benchmarks = [json.loads(p.read_text(encoding="utf-8")) for p in args.benchmark_inputs]
        if len({p["config"]["dataset"] for p in benchmarks}) != 4:
            parser.error("--benchmark-inputs requires four distinct benchmarks.")
        if len({(p["config"]["model"], p["config"]["scorer"]) for p in benchmarks}) != 1:
            parser.error("Use the same model and scorer for the four benchmark panels.")
        report = dict(benchmarks[0])
        report["benchmark_reports"] = benchmarks
        report["embedding_panels"] = [{"config": p["config"], "query_projection": p["query_projection"]}
                                      for p in benchmarks]
    if args.embedding_inputs:
        panels = [json.loads(p.read_text(encoding="utf-8")) for p in args.embedding_inputs]
        report["embedding_panels"] = [{"config": p["config"], "query_projection": p["query_projection"]}
                                      for p in panels]
    write_report(report, output)
    print(f"Saved {output}")
