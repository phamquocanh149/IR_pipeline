"""CLI for paired Vietnamese, English and code-switched retrieval analysis."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ir_system.analysis.language_views import (
    LANGUAGES, fixed_index_analysis, normalize, paired_gaps, project_3d,
)
from ir_system.io.dataset_loader import DatasetLoader
from ir_system.domain.run import Run
from ir_system.models.factory import create_model
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.pipeline.search_pipeline import SearchPipeline
from ir_system.retrievers.cross_encoder import CrossEncoderRetriever
from ir_system.retrievers.dense import DenseRetriever
from ir_system.cli.main import _setup_logging


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="One directory with queries/documents.<vi|vn|en|csw>.jsonl and qrels.tsv.")
    parser.add_argument("--direction", required=True, nargs="+",
                        choices=[f"{query}-{document}" for query in LANGUAGES for document in LANGUAGES],
                        help="Query-document languages, e.g. vi-en csw-en. Vietnamese files may use .vi or .vn.")
    parser.add_argument("--model", required=True, help="Single-vector encoder for representation analysis.")
    parser.add_argument("--reranker", help="Optional cross-encoder for alignment/margins; otherwise cosine scores.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100, help="Hard-negative pool cutoff.")
    parser.add_argument("--output", type=Path, default=Path("results/language_analysis"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if args.top_k <= 0 or args.batch_size <= 0:
        parser.error("--top-k and --batch-size must be positive.")
    try:
        loader = DatasetLoader(strict_qrels=True)
        directions = list(dict.fromkeys(args.direction))
        selected_queries = {direction.split("-")[0] for direction in directions}
        query_paths = {lang: loader.query_path(args.dataset, language=lang) for lang in LANGUAGES}
        queries = {
            lang: {q.qid: q for q in loader.load_queries(args.dataset, language=lang)}
            for lang in LANGUAGES if lang in selected_queries or query_paths[lang].is_file()
        }
        # Other available query views are used only for cosine/PCA diagnostics.
        corpora = {lang: loader.load_corpus(args.dataset, language=lang)
                   for lang in LANGUAGES if any(direction.endswith(f"-{lang}") for direction in directions)}
        model = create_model(args.model, model_type="single-vector", device=args.device,
                             batch_size=args.batch_size)
        if not isinstance(model, SingleVectorEmbeddingModel):
            raise ValueError("Representation analysis requires a single-vector encoder.")
        reranker = (create_model(args.reranker, model_type="cross-encoder", device=args.device,
                                batch_size=args.batch_size) if args.reranker else None)
        query_vectors = {}
        for lang in queries:
            logging.info("Encoding %s query views", lang)
            qids = list(queries[lang])
            qv = normalize(model.encode([queries[lang][qid].text for qid in qids],
                                        batch_size=args.batch_size, show_progress=True))
            if len(qv) != len(qids):
                raise ValueError("Encoder returned an incorrect number of embeddings.")
            query_vectors[lang] = dict(zip(qids, qv))
        points = [{"qid": qid, "language": lang, "text": queries[lang][qid].text}
                  for lang in queries for qid in sorted(queries[lang])]
        coordinates, variance = project_3d(np.stack([
            query_vectors[p["language"]][p["qid"]] for p in points
        ]))
        for point, xyz in zip(points, coordinates):
            point["xyz"] = xyz.tolist()
        indexes, document_vectors = {}, {}
        for index_lang, (documents, qrels) in corpora.items():
            logging.info("Analyzing fixed %s index", index_lang)
            query_languages = [lang for lang in LANGUAGES if f"{lang}-{index_lang}" in directions]
            dense = DenseRetriever(model=model, batch_size=args.batch_size)
            retriever = (CrossEncoderRetriever(
                model=reranker, candidate_retriever=dense,
                candidate_top_k=len(documents), batch_size=args.batch_size,
            ) if reranker else dense)
            retriever.build(documents)
            if len(corpora) > 1:
                document_vectors[index_lang] = dense.document_embeddings()
            pipeline = SearchPipeline(retriever)
            # Full rankings preserve scores for positives outside the negative cutoff.
            def run_batches():
                qids = sorted(set.union(*(set(queries[lang]) for lang in query_languages)))
                for qid in qids:
                    yield {lang: pipeline.search([queries[lang][qid]], len(documents))
                           if qid in queries[lang] else Run()
                           for lang in query_languages}
            indexes[index_lang] = fixed_index_analysis(run_batches(), qrels, args.top_k)
            for row in indexes[index_lang]["margins"]:
                row["direction"] = f"{row['language']}-{index_lang}"
        report = {
            "config": {"model": args.model, "scorer": args.reranker or "cosine",
                       "top_k": args.top_k, "batch_size": args.batch_size,
                       "dataset": str(args.dataset),
                       "query_files": {lang: str(query_paths[lang]) for lang in queries},
                       "directions": directions},
            "policies": {
                "pairing": "Exact shared qid and doc_id denote information needs and document groups.",
                "projection": "One joint PCA on available query-language views; cosine gaps use original dimensions.",
                "positives": "All relevance > 0 judgments in each fixed index, including positives outside top-k.",
                "negatives": "Non-positive documents in each query view's own top-k; unjudged documents treated as non-relevant.",
                "missing": "Undefined margins/deltas are null. Deltas require a matching vi query direction on the same index. Unpaired IDs are reported; their margins are still evaluated.",
                "aggregation": "Alignment is averaged over query-positive-document pairs; margins over each requested query direction. Only requested document indexes are built.",
            },
            "query_projection": {"points": points, "explained_variance_ratio": variance},
            "query_gaps": paired_gaps(query_vectors),
            "document_gaps": paired_gaps(document_vectors),
            "indexes": indexes,
        }
        from ir_system.analysis.report import write_report
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_report(report, args.output / "report.html")
        logging.info("Saved analysis to %s", args.output)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        logging.error("Analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
