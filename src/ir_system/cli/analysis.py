"""
CLI: Language-view analysis
==========================
Analyze representation drift and positive/negative separation on fixed indexes.

Responsibilities
----------------
1. Parse --queries/--documents or query-document directions, e.g. vi-en and csw-en.
2. Load separate language views through DatasetLoader's public multilingual API.
3. Build only requested document indexes using existing retrievers.
4. Encode available query views and invoke cosine, PCA and margin analysis.
5. Evaluate optional retrieval metrics with the existing Evaluator.
6. Save detailed JSON and a self-contained interactive HTML report.

Requirements (mandatory)
------------------------
- Query languages come from QUERY_LANGS; document languages from DOC_LANGS.
- load_queries(..., lang="all") returns language -> queries; keep views separate.
- load_documents(..., lang=[...]) returns language -> documents; do not merge IDs.
- Requested query/document languages must exist; fail before loading models.
- Use the shared, language-agnostic qrels.tsv through load_qrels(...).
- Never modify DatasetLoader or duplicate its JSONL/TSV parsing rules.
- Score all corpus documents so positives outside the negative top-k retain scores.
- Delegate scoring/ranking to DenseRetriever or CrossEncoderRetriever.

Output
------
<output>/analysis.json : configuration, cosine/gap pairs, PCA and margins.
<output>/report.html   : interactive query 3D plot and analysis distributions.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ir_system.analysis.language_views import (
    LANGUAGES, fixed_index_analysis, normalize, paired_gaps, project_3d,
)
from ir_system.io.dataset_loader import DOC_LANGS, DatasetLoader
from ir_system.domain.run import Run
from ir_system.models.factory import create_model
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.pipeline.search_pipeline import SearchPipeline
from ir_system.retrievers.cross_encoder import CrossEncoderRetriever
from ir_system.retrievers.dense import DenseRetriever
from ir_system.cli.main import _parse_metric_spec, _setup_logging
from ir_system.evaluation.evaluator import Evaluator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze query representation drift and positive/negative margins by query-document direction.",
    )
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Multilingual dataset directory with split language files and shared qrels.tsv.")
    parser.add_argument("--direction", nargs="+",
                        choices=[f"{query}-{document}" for query in LANGUAGES for document in DOC_LANGS],
                        help="Query-document languages, e.g. vi-en csw-en; queries vi/en/csw, documents vi/en.")
    parser.add_argument("--queries", nargs="+", choices=[*LANGUAGES, "all"],
                        help="Query languages, e.g. csw or vi en csw.")
    parser.add_argument("--documents", nargs="+", choices=[*DOC_LANGS, "all"],
                        help="Document languages, e.g. vi or en.")
    parser.add_argument("--model", required=True, help="Single-vector encoder for representation analysis.")
    parser.add_argument("--model-type", choices=["single-vector"], default="single-vector")
    parser.add_argument("--retriever", choices=["dense"], default="dense",
                        help="Dense retrieval; use --reranker for optional cross-encoder scoring.")
    parser.add_argument("--metrics", nargs="+", help="Optional retrieval metrics, e.g. ndcg@10 mrr@10 recall@10.")
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
    if args.direction and (args.queries or args.documents):
        parser.error("Use either --direction or --queries with --documents.")
    if not args.direction and not (args.queries and args.documents):
        parser.error("Provide --queries and --documents, or --direction.")
    try:
        loader = DatasetLoader(strict_qrels=True)
        if args.direction:
            directions = list(dict.fromkeys(args.direction))
        else:
            query_tags = LANGUAGES if "all" in args.queries else args.queries
            document_tags = DOC_LANGS if "all" in args.documents else args.documents
            directions = list(dict.fromkeys(f"{q}-{d}" for q in query_tags for d in document_tags))
        evaluator = Evaluator([_parse_metric_spec(spec) for spec in args.metrics]) if args.metrics else None
        selected_queries = {direction.split("-")[0] for direction in directions}
        selected_documents = [lang for lang in DOC_LANGS
                              if any(direction.endswith(f"-{lang}") for direction in directions)]
        query_views = loader.load_queries(args.dataset, lang="all")
        queries = {lang: {q.qid: q for q in view} for lang, view in query_views.items()}
        missing_queries = selected_queries - queries.keys()
        if missing_queries:
            raise FileNotFoundError(f"Requested query languages missing from dataset: {sorted(missing_queries)}")
        # Other available query views are used only for cosine/PCA diagnostics.
        corpora = loader.load_documents(args.dataset, lang=selected_documents)
        missing_documents = set(selected_documents) - corpora.keys()
        if missing_documents:
            raise FileNotFoundError(f"Requested document languages missing from dataset: {sorted(missing_documents)}")
        qrels = loader.load_qrels(args.dataset, known_doc_ids={doc.doc_id
                                 for documents in corpora.values() for doc in documents})
        model = create_model(args.model, model_type=args.model_type, device=args.device,
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
        for index_lang, documents in corpora.items():
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
            metric_runs = {lang: Run() for lang in query_languages} if evaluator else {}
            # Full rankings preserve scores for positives outside the negative cutoff.
            def run_batches():
                qids = sorted(set.union(*(set(queries[lang]) for lang in query_languages)))
                for qid in qids:
                    runs = {lang: pipeline.search([queries[lang][qid]], len(documents))
                            if qid in queries[lang] else Run() for lang in query_languages}
                    if evaluator:
                        for lang, run in runs.items():
                            if qid in run.qids:
                                metric_runs[lang].add(qid, run.get(qid)[:args.top_k])
                    yield runs
            indexes[index_lang] = fixed_index_analysis(
                run_batches(), qrels, args.top_k, document_ids={doc.doc_id for doc in documents},
            )
            for row in indexes[index_lang]["margins"]:
                row["direction"] = f"{row['language']}-{index_lang}"
            if evaluator:
                indexes[index_lang]["retrieval_metrics"] = {
                    f"{lang}-{index_lang}": evaluator.evaluate(run, qrels)
                    for lang, run in metric_runs.items()
                }
        report = {
            "config": {"model": args.model, "scorer": args.reranker or "cosine",
                       "top_k": args.top_k, "batch_size": args.batch_size,
                       "dataset": str(args.dataset),
                       "query_languages": list(queries), "document_languages": list(corpora),
                       "directions": directions, "model_type": args.model_type,
                       "retriever": args.retriever, "metrics": args.metrics},
            "policies": {
                "pairing": "Exact shared qid and doc_id denote information needs and document groups.",
                "projection": "One joint PCA on available query-language views; cosine gaps use original dimensions.",
                "positives": "Shared qrels.tsv, restricted to relevance > 0 documents present in each fixed index, including positives outside top-k.",
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
