"""
Evaluation: Metric (abstract base)
====================================
All metrics compute a scalar aggregate over a Run and Qrels.

Metric knows nothing about:
- Which retriever produced the Run
- BM25, Dense, HuggingFace, FAISS

Contract
--------
    compute(run, qrels) → float  (macro average over queries)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run


class Metric(ABC):
    """Abstract base for all IR evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical metric name, e.g. 'ndcg@10', 'mrr@10', 'recall@100'."""
        ...

    @abstractmethod
    def compute(self, run: Run, qrels: Qrels) -> float:
        """
        Compute the metric as a macro-average over all queries in `run`.

        Parameters
        ----------
        run   : Run    — retrieved hits per query (ordered)
        qrels : Qrels  — relevance judgments

        Returns
        -------
        float  in [0.0, 1.0] (or other appropriate range)
        """
        ...
