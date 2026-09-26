"""
Evaluation: Evaluator
======================
Orchestrates a list of Metric instances over a Run and Qrels.

Evaluator does NOT know:
- BM25, Dense, HuggingFace, FAISS
- Any specific metric formula

Contract
--------
    Evaluator(metrics: Sequence[Metric])
    evaluator.evaluate(run, qrels) → dict[metric_name, score]
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Computes a set of metrics over a retrieval Run.

    Example
    -------
    evaluator = Evaluator([NDCGAtK(10), MRRAtK(10), RecallAtK(100)])
    scores = evaluator.evaluate(run, qrels)
    # → {"ndcg@10": 0.42, "mrr@10": 0.38, "recall@100": 0.91}
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        if not metrics:
            raise ValueError("Evaluator requires at least one Metric.")
        self._metrics = list(metrics)

    def evaluate(self, run: Run, qrels: Qrels) -> Dict[str, float]:
        """
        Compute all metrics.

        Parameters
        ----------
        run   : Run
        qrels : Qrels

        Returns
        -------
        dict[metric_name, score]
        """
        results: Dict[str, float] = {}
        for metric in self._metrics:
            logger.info("[EVALUATION] Computing %s", metric.name)
            score = metric.compute(run, qrels)
            results[metric.name] = score
            logger.info("[EVALUATION] %s = %.4f", metric.name, score)
        return results

    @property
    def metric_names(self) -> list[str]:
        """Return names of all metrics."""
        return [m.name for m in self._metrics]
