"""
CLI: main.py
=============
Entry point for the IR pipeline.

Responsibilities (ONLY):
1. Parse CLI arguments.
2. Setup logging.
3. Load dataset (with multilingual split-file support).
4. Instantiate model (via ModelFactory).
5. Instantiate retriever (mapping --retriever string to concrete class).
6. Instantiate metrics (parsing --metrics string list).
7. Instantiate Evaluator and SearchPipeline.
8. Build retriever index.
9. Execute evaluation over every (queries_lang x docs_lang) pair.
10. Print results as per-metric tables.
11. Save results to results/<run_id>/.

No BM25/NDCG/RRF formula logic here.
No business logic here -- only wiring.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ensure stdout can handle Unicode box-drawing characters on Windows.
# ---------------------------------------------------------------------------
def _ensure_utf8_stdout() -> None:
    """Re-configure stdout/stderr to UTF-8 when the shell encoding differs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        elif hasattr(stream, "buffer"):
            new_stream = io.TextIOWrapper(
                stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            setattr(sys, stream_name, new_stream)


_ensure_utf8_stdout()



def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_metric_spec(spec: str):
    """
    Parse a metric specification like 'ndcg@10', 'mrr@10', 'recall@100'.

    Returns the appropriate Metric instance.
    """
    from ir_system.evaluation.mrr import MRRAtK
    from ir_system.evaluation.ndcg import NDCGAtK
    from ir_system.evaluation.recall import RecallAtK
    from ir_system.evaluation.precision import PrecisionAtK
    from ir_system.evaluation.map import MAPAtK

    spec = spec.strip().lower()
    if "@" not in spec:
        raise ValueError(
            f"Invalid metric spec {spec!r}. Expected format: <name>@<k>, e.g. ndcg@10"
        )
    name, k_str = spec.rsplit("@", 1)
    try:
        k = int(k_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid metric spec {spec!r}: cutoff {k_str!r} is not an integer."
        ) from exc

    if name == "ndcg":
        return NDCGAtK(k)
    elif name == "mrr":
        return MRRAtK(k)
    elif name == "recall":
        return RecallAtK(k)
    elif name in ("precision", "p"):
        return PrecisionAtK(k)
    elif name == "map":
        return MAPAtK(k)
    else:
        raise ValueError(
            f"Unknown metric {name!r} in spec {spec!r}. "
            "Supported: ndcg@K, mrr@K, recall@K, precision@K, map@K"
        )


def _build_candidate_retriever(
    candidate_type: str,
    candidate_model,
    batch_size: int,
):
    """
    Build the first-stage (candidate) retriever for cross-encoder reranking.
    """
    from ir_system.retrievers.bm25 import BM25Retriever
    from ir_system.retrievers.dense import DenseRetriever
    from ir_system.models.single_vector import SingleVectorEmbeddingModel

    ctype = candidate_type.lower().strip()

    if ctype == "bm25":
        return BM25Retriever()
    elif ctype == "dense":
        if candidate_model is None:
            raise ValueError(
                "--candidate-retriever dense requires --candidate-model "
                "to specify the embedding model for Stage 1."
            )
        if not isinstance(candidate_model, SingleVectorEmbeddingModel):
            raise ValueError(
                f"--candidate-retriever dense requires a SingleVectorEmbeddingModel. "
                f"Got: {type(candidate_model).__name__}."
            )
        return DenseRetriever(model=candidate_model, batch_size=batch_size)
    else:
        raise ValueError(
            f"Unknown candidate retriever type: {candidate_type!r}. "
            "Supported: bm25, dense"
        )


def _build_retriever(
    retriever_type: str,
    model,
    candidate_top_k: int,
    batch_size: int,
    candidate_retriever_type: str = "bm25",
    candidate_model=None,
    candidate_batch_size: int = 32,
):
    """
    Map --retriever string to a concrete Retriever instance.
    """
    from ir_system.retrievers.bm25 import BM25Retriever
    from ir_system.retrievers.dense import DenseRetriever
    from ir_system.retrievers.late_interaction import LateInteractionRetriever
    from ir_system.retrievers.cross_encoder import CrossEncoderRetriever
    from ir_system.retrievers.hybrid import HybridRetriever
    from ir_system.fusion.rrf import RRFusion
    from ir_system.models.single_vector import SingleVectorEmbeddingModel
    from ir_system.models.multi_vector import MultiVectorEmbeddingModel
    from ir_system.models.cross_encoder import CrossEncoderModel

    rtype = retriever_type.lower().strip()

    if rtype == "bm25":
        return BM25Retriever()

    elif rtype == "dense":
        if not isinstance(model, SingleVectorEmbeddingModel):
            raise ValueError(
                f"--retriever dense requires a SingleVectorEmbeddingModel. "
                f"Got: {type(model).__name__}. "
                "Use --model-type single-vector or a different model."
            )
        return DenseRetriever(model=model, batch_size=batch_size)

    elif rtype in ("late-interaction", "late_interaction", "colbert"):
        if not isinstance(model, MultiVectorEmbeddingModel):
            raise ValueError(
                f"--retriever late-interaction requires a MultiVectorEmbeddingModel. "
                f"Got: {type(model).__name__}. "
                "Use --model-type multi-vector."
            )
        return LateInteractionRetriever(model=model, batch_size=batch_size)

    elif rtype in ("cross-encoder", "cross_encoder", "reranker"):
        if not isinstance(model, CrossEncoderModel):
            raise ValueError(
                f"--retriever cross-encoder requires a CrossEncoderModel. "
                f"Got: {type(model).__name__}. "
                "Use --model-type cross-encoder."
            )
        stage1 = _build_candidate_retriever(
            candidate_type=candidate_retriever_type,
            candidate_model=candidate_model,
            batch_size=candidate_batch_size,
        )
        return CrossEncoderRetriever(
            model=model,
            candidate_retriever=stage1,
            candidate_top_k=candidate_top_k,
            batch_size=batch_size,
        )

    elif rtype == "hybrid":
        if not isinstance(model, SingleVectorEmbeddingModel):
            raise ValueError(
                f"--retriever hybrid requires a SingleVectorEmbeddingModel. "
                f"Got: {type(model).__name__}. "
                "Use --model-type single-vector."
            )
        bm25 = BM25Retriever()
        dense = DenseRetriever(model=model, batch_size=batch_size)
        rrf = RRFusion(k=60)
        return HybridRetriever(
            retrievers=[bm25, dense],
            fusion_strategy=rrf,
            candidate_top_k=candidate_top_k,
        )

    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type!r}. "
            "Supported: bm25, dense, late-interaction, cross-encoder, hybrid"
        )


# ---------------------------------------------------------------------------
# Table printing helpers
# ---------------------------------------------------------------------------

def _col_width(values: Sequence[str], header: str, min_w: int = 8) -> int:
    """Return the column width needed to fit the header and all values."""
    return max(min_w, len(header), *(len(v) for v in values))


def _print_metric_table(
    metric_name: str,
    pair_labels: List[str],
    scores: List[float],
) -> None:
    """
    Print a single metric table.

    Uses Unicode box-drawing characters when the terminal supports them,
    falls back to ASCII otherwise.

    Example output (Unicode)
    ------------------------
    ╔══════════════════════════════════╦══════════╗
    ║  METRIC: ndcg@10                 ║   score  ║
    ╠══════════════════════════════════╬══════════╣
    ║  queries=vi  x  docs=vi          ║   0.7500 ║
    ║  queries=en  x  docs=vi          ║   0.6230 ║
    ╚══════════════════════════════════╩══════════╝
    """
    # Detect if stdout can render Unicode box chars
    _enc = getattr(sys.stdout, "encoding", "ascii") or "ascii"
    _use_unicode = _enc.lower().replace("-", "") in (
        "utf8", "utf16", "utf32", "cp65001"
    )

    if _use_unicode:
        C = dict(tl="\u2554", tr="\u2557", bl="\u255a", br="\u255d",
                 h="\u2550", v="\u2551", tc="\u2566", bc="\u2569",
                 ml="\u2560", mr="\u2563", mc="\u256c")
    else:
        C = dict(tl="+", tr="+", bl="+", br="+",
                 h="-", v="|", tc="+", bc="+",
                 ml="+", mr="+", mc="+")

    SCORE_HDR = "  score  "
    SCORE_W = max(len(SCORE_HDR), 10)
    pair_w = _col_width(pair_labels, f"  METRIC: {metric_name}  ", min_w=30)

    top    = C["tl"] + C["h"] * pair_w + C["tc"] + C["h"] * SCORE_W + C["tr"]
    header = C["v"] + f"  METRIC: {metric_name:<{pair_w - 2}}" + C["v"] + f"{SCORE_HDR:^{SCORE_W}}" + C["v"]
    sep    = C["ml"] + C["h"] * pair_w + C["mc"] + C["h"] * SCORE_W + C["mr"]
    bottom = C["bl"] + C["h"] * pair_w + C["bc"] + C["h"] * SCORE_W + C["br"]

    print(top)
    print(header)
    print(sep)
    for label, score in zip(pair_labels, scores):
        score_str = f"{score:.4f}"
        row = C["v"] + f"  {label:<{pair_w - 2}}" + C["v"] + f"{score_str:^{SCORE_W}}" + C["v"]
        print(row)
    print(bottom)
    print()


def _print_all_tables(
    all_results: Dict[str, Dict[str, float]],
    metric_names: List[str],
) -> None:
    """
    Print one table per metric, with rows = retrieval pairs.

    Parameters
    ----------
    all_results : dict  pair_label -> {metric_name: score}
    metric_names : list[str]  ordered list of metric names to display
    """
    pair_labels = list(all_results.keys())

    print()
    print("=" * 60)
    print("  CROSS-LINGUAL RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print()

    for metric in metric_names:
        scores = [all_results[label].get(metric, float("nan")) for label in pair_labels]
        _print_metric_table(metric, pair_labels, scores)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_results(
    output_dir: Path,
    run,
    metrics_dict: Dict,
    config_dict: Dict,
) -> None:
    """Save config.json, metrics.json, and run.jsonl to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.json"
    run_path = output_dir / "run.jsonl"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    with run_path.open("w", encoding="utf-8") as f:
        for qid in sorted(run.qids):
            hits = run.get(qid)
            record = {
                "qid": qid,
                "hits": [
                    {"doc_id": h.doc_id, "score": float(h.score)}
                    for h in hits
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.getLogger(__name__).info("[RESULT] Saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ir_system.cli.main",
        description="IR Pipeline -- end-to-end information retrieval evaluation.",
    )
    # ---- Required ----
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Path to dataset directory. For multilingual datasets may contain "
            "queries.{vi,en,csw}.jsonl and documents.{vi,en}.jsonl. "
            "For legacy datasets: queries.jsonl, documents.jsonl, qrels.tsv."
        ),
    )
    parser.add_argument(
        "--retriever",
        required=True,
        choices=["bm25", "dense", "late-interaction", "cross-encoder", "hybrid"],
        help="Retriever type.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=True,
        help="Number of documents to retrieve per query.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metrics to compute, e.g. ndcg@10 mrr@10 recall@100",
    )

    # ---- Multilingual selection ----
    parser.add_argument(
        "--queries",
        "--queries-lang",
        dest="queries_lang",
        nargs="+",
        default=["all"],
        metavar="LANG",
        help=(
            "Query language variant(s) to use. "
            "Choices: vi en csw all (default: all). "
            "Example: --queries vi en  -- uses vi and en query files."
        ),
    )
    parser.add_argument(
        "--documents",
        "--docs-lang",
        "--docs",
        dest="docs_lang",
        nargs="+",
        default=["all"],
        metavar="LANG",
        help=(
            "Document language variant(s) to use. "
            "Choices: vi en all (default: all). "
            "Example: --documents vi  -- uses only vi document file."
        ),
    )

    # ---- Model options ----
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (HuggingFace name or path). Required for non-BM25 retrievers.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help=(
            "Prefix added to every query for single-vector models. Default: taken from "
            "the model (Qwen3, ...) or inferred from its name (e5, bge). Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--doc-prefix",
        default=None,
        help=(
            "Prefix added to every document for single-vector models. Default: taken from "
            "the model or inferred from its name (e5). Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["single-vector", "multi-vector", "cross-encoder"],
        default=None,
        help="Override model type detection.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to run model on (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding/scoring batch size (default: 32).",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=100,
        help="Candidate retrieval count for hybrid/reranking (default: 100).",
    )
    parser.add_argument(
        "--candidate-retriever",
        choices=["bm25", "dense"],
        default="bm25",
        help="First-stage retriever for cross-encoder reranking (default: bm25).",
    )
    parser.add_argument(
        "--candidate-model",
        default=None,
        help="Model for first-stage candidate retriever when --candidate-retriever=dense.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Batch size for Stage 1 candidate retriever (default: same as --batch-size).",
    )

    # ---- Output / reproducibility ----
    parser.add_argument(
        "--run-name",
        "--name",
        "-n",
        default=None,
        help="Custom run name (saved under results/<run_name>/). Replaces auto-generated timestamp.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results (default: results/<run_name> or results/<timestamp>_<retriever>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    # ---- Index persistence ----
    parser.add_argument(
        "--save-index",
        default=None,
        help="Save an extra copy of the FAISS index to this directory after building (e.g. indexes/fiqa_bge).",
    )
    parser.add_argument(
        "--load-index",
        default=None,
        help="Load a previously saved FAISS index from this directory (skip encoding).",
    )
    return parser


# ---------------------------------------------------------------------------
# Retriever index helpers (DenseRetriever aware)
# ---------------------------------------------------------------------------

def _find_dense_retrievers(ret):
    """Discover all DenseRetriever instances inside a (possibly composite) retriever."""
    from ir_system.retrievers.dense import DenseRetriever

    found = []
    if isinstance(ret, DenseRetriever):
        found.append(ret)
    if hasattr(ret, "_candidate_retriever") and ret._candidate_retriever is not None:
        found.extend(_find_dense_retrievers(ret._candidate_retriever))
    if hasattr(ret, "_retrievers"):
        for sub in ret._retrievers:
            found.extend(_find_dense_retrievers(sub))
    return found


def _build_index(retriever, documents, load_dir, logger):
    """Build (or load) the retrieval index for *documents*."""
    dense_retrievers = _find_dense_retrievers(retriever)
    if load_dir:
        if not dense_retrievers:
            raise ValueError("--load-index requires a dense retriever component.")
        logger.info("[INDEX] Loading FAISS index from %s...", load_dir)
        for dr in dense_retrievers:
            dr.load_index(load_dir)
    logger.info("[RETRIEVER] Building index...")
    retriever.build(documents)
    logger.info("[RETRIEVER] Index ready.")



# ---------------------------------------------------------------------------
# Resolve lang spec helper (wraps DatasetLoader utility)
# ---------------------------------------------------------------------------

def _parse_lang_spec(raw: List[str], kind: str) -> str | List[str]:
    """
    Normalise --queries-lang / --docs-lang into "all" or a list of tags.

    ``raw`` comes straight from argparse (always a list because nargs="+").
    """
    if len(raw) == 1 and raw[0].lower() == "all":
        return "all"
    return [t.lower() for t in raw]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # --- Seed ---
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
        except ImportError:
            pass
        logger.info("[SETUP] Seed set to %d", args.seed)

    # --- Validate top-k ---
    if args.top_k <= 0:
        logger.error("--top-k must be > 0, got %d", args.top_k)
        return 1

    # --- Model requirement check ---
    needs_model = args.retriever != "bm25"
    if needs_model and not args.model:
        logger.error("--model is required for --retriever=%s", args.retriever)
        return 1

    # --- Validate candidate-retriever usage ---
    is_cross_encoder = args.retriever in ("cross-encoder", "cross_encoder", "reranker")
    if not is_cross_encoder and args.candidate_model:
        logger.warning("--candidate-model is ignored when --retriever is not cross-encoder.")
    if is_cross_encoder and args.candidate_retriever == "dense" and not args.candidate_model:
        logger.error("--candidate-model is required when --candidate-retriever=dense.")
        return 1

    # --- Resolve lang specs ---
    from ir_system.io.dataset_loader import DatasetLoader, QUERY_LANGS, DOC_LANGS, _resolve_lang_spec

    queries_lang_spec = _parse_lang_spec(args.queries_lang, "queries")
    docs_lang_spec = _parse_lang_spec(args.docs_lang, "docs")

    # Validate lang tags early (before loading models)
    try:
        q_tags = _resolve_lang_spec(queries_lang_spec, QUERY_LANGS)
        d_tags = _resolve_lang_spec(docs_lang_spec, DOC_LANGS)
    except ValueError as exc:
        logger.error("Language spec error: %s", exc)
        return 1

    # --- Load dataset ---
    try:
        loader = DatasetLoader(strict_qrels=False)

        # Check dataset dir exists
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"DatasetLoader: dataset directory not found: {dataset_path}"
            )

        # Detect mode: multilingual split files vs legacy
        has_legacy_queries = (dataset_path / "queries.jsonl").exists()
        has_legacy_docs = (dataset_path / "documents.jsonl").exists()
        is_legacy = has_legacy_queries and has_legacy_docs

        if is_legacy:
            # Legacy: load once, ignore --queries-lang / --docs-lang
            logger.info(
                "[DATASET] Legacy single-file layout detected. "
                "--queries-lang / --docs-lang are ignored."
            )
            queries, documents, qrels = loader.load(args.dataset)
            queries_by_lang: Dict = {"all": list(queries)}
            docs_by_lang: Dict = {"all": list(documents)}
        else:
            # Multilingual: load requested lang variants
            queries_by_lang = loader.load_queries(args.dataset, lang=queries_lang_spec)
            docs_by_lang = loader.load_documents(args.dataset, lang=docs_lang_spec)

            # Collect all doc ids across all doc langs for qrels validation
            all_doc_ids: set = set()
            for docs in docs_by_lang.values():
                for doc in docs:
                    all_doc_ids.add(doc.doc_id)

            qrels = loader.load_qrels(args.dataset, known_doc_ids=all_doc_ids)

        # Filter to tags actually found on disk
        q_tags = list(queries_by_lang.keys())
        d_tags = list(docs_by_lang.keys())

    except (FileNotFoundError, ValueError, NotADirectoryError) as exc:
        logger.error("Dataset loading failed:\n%s", exc)
        return 1

    logger.info(
        "[DATASET] Query langs: %s | Doc langs: %s", q_tags, d_tags
    )
    logger.info(
        "[DATASET] Cross-retrieval pairs to run: %d",
        len(q_tags) * len(d_tags),
    )

    # --- Load model ---
    from ir_system.models.factory import create_model

    model = None
    if args.model:
        try:
            model = create_model(
                model_id=args.model,
                model_type=args.model_type,
                device=args.device,
                batch_size=args.batch_size,
                query_prompt=args.query_prefix,
                document_prompt=args.doc_prefix,
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.error("Model loading failed: %s", exc)
            return 1

    # --- Load candidate model (for cross-encoder Stage 1) ---
    candidate_model = None
    if is_cross_encoder and args.candidate_model:
        try:
            candidate_model = create_model(
                model_id=args.candidate_model,
                model_type="single-vector",
                device=args.device,
                batch_size=args.batch_size,
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.error("Candidate model loading failed: %s", exc)
            return 1

    # --- Parse metrics ---
    try:
        metrics = [_parse_metric_spec(spec) for spec in args.metrics]
    except ValueError as exc:
        logger.error("Metric parsing failed: %s", exc)
        return 1

    # --- Evaluator + Pipeline ---
    from ir_system.evaluation.evaluator import Evaluator
    from ir_system.pipeline.search_pipeline import SearchPipeline

    evaluator = Evaluator(metrics)

    # --- Cross-lingual retrieval loop ---
    candidate_batch_size = args.candidate_batch_size or args.batch_size

    # all_results: pair_label -> {metric_name: score}
    all_results: Dict[str, Dict[str, float]] = {}
    all_runs: Dict[str, object] = {}

    # The index depends only on the document set, so build it once per d_lang
    # and evaluate every query language against it.
    results_by_pair: Dict[Tuple[str, str], Dict[str, float]] = {}
    runs_by_pair: Dict[Tuple[str, str], object] = {}

    for d_lang in d_tags:
        cur_documents = docs_by_lang[d_lang]
        logger.info("[RUN] === Documents: %s ===", d_lang)

        # With several doc sets, each one needs its own index directory.
        def _index_dir(base: Optional[str]) -> Optional[str]:
            if base and len(d_tags) > 1:
                return str(Path(base) / d_lang)
            return base

        try:
            retriever = _build_retriever(
                retriever_type=args.retriever,
                model=model,
                candidate_top_k=args.candidate_top_k,
                batch_size=args.batch_size,
                candidate_retriever_type=args.candidate_retriever,
                candidate_model=candidate_model,
                candidate_batch_size=candidate_batch_size,
            )
        except (ValueError, TypeError) as exc:
            logger.error("Retriever construction failed: %s", exc)
            return 1

        try:
            _build_index(
                retriever,
                cur_documents,
                _index_dir(args.load_index),
                logger,
            )
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            logger.error("Index building/loading failed: %s", exc)
            return 1

        pipeline = SearchPipeline(retriever=retriever, evaluator=evaluator)

        for q_lang in q_tags:
            pair_label = f"queries={q_lang}  x  docs={d_lang}"
            logger.info("[RUN] === Pair: %s ===", pair_label)

            try:
                run, metrics_result = pipeline.evaluate(
                    queries_by_lang[q_lang], qrels, top_k=args.top_k
                )
            except (RuntimeError, ValueError) as exc:
                logger.error("Evaluation failed for pair %s: %s", pair_label, exc)
                return 1

            results_by_pair[(q_lang, d_lang)] = metrics_result
            runs_by_pair[(q_lang, d_lang)] = run
            logger.info("[RUN] Pair %s done.", pair_label)

        del pipeline, retriever  # Release this corpus index before building the next.

    # Report in the original order: grouped by query language, then doc language.
    for q_lang in q_tags:
        for d_lang in d_tags:
            pair_label = f"queries={q_lang}  x  docs={d_lang}"
            all_results[pair_label] = results_by_pair[(q_lang, d_lang)]
            all_runs[pair_label] = runs_by_pair[(q_lang, d_lang)]

    # --- Print results as tables ---
    # Flush logging (stderr) first so table is not interleaved with log lines.
    logging.shutdown()
    sys.stderr.flush()
    metric_names = list(next(iter(all_results.values())).keys()) if all_results else []
    _print_all_tables(all_results, metric_names)
    sys.stdout.flush()
    # Re-open basic logging for remaining INFO messages (save results).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Resolve output directory ---
    timestamp = time.strftime("%Y%m%dT%H%M%S")

    if args.output:
        raw_output = Path(args.output)
        if len(raw_output.parts) == 1 and not args.output.startswith((".", "/", "\\")):
            output_path = Path("results") / raw_output
        else:
            output_path = raw_output
        run_name = raw_output.name
    elif args.run_name:
        run_name = args.run_name
        output_path = Path("results") / run_name
    else:
        run_name = f"{timestamp}_{args.retriever}"
        output_path = Path("results") / run_name

    # --- Build config dict ---
    config = {
        "run_id": run_name,
        "model": args.model,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "metrics": args.metrics,
        "queries_lang": q_tags,
        "docs_lang": d_tags,
        "device": args.device,
        "batch_size": args.batch_size,
        "candidate_top_k": args.candidate_top_k,
        "candidate_retriever": args.candidate_retriever,
        "candidate_model": args.candidate_model,
        "candidate_batch_size": candidate_batch_size,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "timestamp": timestamp,
    }

    # --- Save combined results ---
    # Save aggregate metrics summary + one sub-dir per pair
    combined_metrics = {
        label: scores for label, scores in all_results.items()
    }
    try:
        # Use the last run for the top-level run.jsonl (single-pair compat)
        last_pair = list(all_runs.keys())[-1]
        _save_results(output_path, all_runs[last_pair], combined_metrics, config)

        # Per-pair subdirectories for multi-pair runs
        if len(all_runs) > 1:
            for label, run in all_runs.items():
                safe_label = label.replace(" ", "").replace("=", "_").replace("x", "X")
                pair_dir = output_path / safe_label
                _save_results(pair_dir, run, all_results[label], config)

    except OSError as exc:
        logger.error("Failed to save results: %s", exc)
        return 1

    logger.info("[RESULT] Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())