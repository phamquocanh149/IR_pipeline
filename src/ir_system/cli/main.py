"""
CLI: main.py
=============
Entry point for the IR pipeline.

Responsibilities (ONLY):
1. Parse CLI arguments.
2. Setup logging.
3. Load dataset.
4. Instantiate model (via ModelFactory).
5. Instantiate retriever (mapping --retriever string to concrete class).
6. Instantiate metrics (parsing --metrics string list).
7. Instantiate Evaluator and SearchPipeline.
8. Build retriever index.
9. Execute evaluation.
10. Print summary.
11. Save results to results/<run_id>/.

No BM25/NDCG/RRF formula logic here.
No business logic here — only wiring.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


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
    else:
        raise ValueError(
            f"Unknown metric {name!r} in spec {spec!r}. "
            "Supported: ndcg@K, mrr@K, recall@K"
        )


def _build_retriever(
    retriever_type: str,
    model,
    candidate_top_k: int,
    batch_size: int,
):
    """
    Map --retriever string to a concrete Retriever instance.

    This is the composition root — the only place that knows string→class mapping.
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
        # Cross-encoder with BM25 candidate retriever for large corpora.
        candidate_bm25 = BM25Retriever()
        return CrossEncoderRetriever(
            model=model,
            candidate_retriever=candidate_bm25,
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


def _save_results(
    output_dir: Path,
    run,
    metrics_dict,
    config_dict,
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

    logging.getLogger(__name__).info(
        "[RESULT] Saved to %s", output_dir
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ir_system.cli.main",
        description="IR Pipeline — end-to-end information retrieval evaluation.",
    )
    # Required
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory (must contain queries.jsonl, documents.jsonl, qrels.tsv).",
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
    # Model options
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (HuggingFace name or path). Required for non-BM25 retrievers.",
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
    # Output / reproducibility
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: results/<auto_run_id>).",
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
    return parser


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
        logger.error(
            "--model is required for --retriever=%s", args.retriever
        )
        return 1

    # --- Load dataset ---
    from ir_system.io.dataset_loader import DatasetLoader

    try:
        loader = DatasetLoader(strict_qrels=False)
        queries, documents, qrels = loader.load(args.dataset)
    except (FileNotFoundError, ValueError, NotADirectoryError) as exc:
        logger.error("Dataset loading failed:\n%s", exc)
        return 1

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
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.error("Model loading failed: %s", exc)
            return 1

    # --- Build retriever ---
    try:
        retriever = _build_retriever(
            retriever_type=args.retriever,
            model=model,
            candidate_top_k=args.candidate_top_k,
            batch_size=args.batch_size,
        )
    except (ValueError, TypeError) as exc:
        logger.error("Retriever construction failed: %s", exc)
        return 1

    # --- Build index ---
    try:
        logger.info("[RETRIEVER] Building index...")
        retriever.build(documents)
        logger.info("[RETRIEVER] Index ready.")
    except (ValueError, RuntimeError, ImportError) as exc:
        logger.error("Index building failed: %s", exc)
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
    pipeline = SearchPipeline(retriever=retriever, evaluator=evaluator)

    # --- Run evaluation ---
    try:
        run, metrics_result = pipeline.evaluate(queries, qrels, top_k=args.top_k)
    except (RuntimeError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1

    # --- Print summary ---
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric_name, score in metrics_result.items():
        print(f"  {metric_name:<20}  {score:.4f}")
    print("=" * 50 + "\n")

    # --- Build config dict ---
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_id = args.output or f"results/{timestamp}_{args.retriever}"

    config = {
        "model": args.model,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "metrics": args.metrics,
        "device": args.device,
        "batch_size": args.batch_size,
        "candidate_top_k": args.candidate_top_k,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "timestamp": timestamp,
    }

    # --- Save results ---
    output_path = Path(run_id) if not Path(run_id).is_absolute() else Path(run_id)
    try:
        _save_results(output_path, run, metrics_result, config)
    except OSError as exc:
        logger.error("Failed to save results: %s", exc)
        return 1

    logger.info("[RESULT] Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
