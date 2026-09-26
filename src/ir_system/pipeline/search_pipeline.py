"""
Pipeline: SearchPipeline
=========================
Orchestrates retrieval and (optionally) evaluation for a full query set.

Contract
--------
- Accepts any Retriever and any Evaluator (dependency injection).
- Does NOT hard-code BM25/Dense/NDCG/model logic.
- Calls retriever.retrieve() per query, accumulates into Run.
- Passes Run + Qrels to Evaluator for metric computation.

API
---
    pipeline.search(queries, top_k) → Run
    pipeline.evaluate(queries, qrels, top_k) → dict[metric_name, score]
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

from ir_system.domain.hit import Hit
from ir_system.domain.qrels import Qrels
from ir_system.domain.query import Query
from ir_system.domain.run import Run
from ir_system.evaluation.evaluator import Evaluator
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    End-to-end search pipeline: retrieval → (optional) evaluation.

    Parameters
    ----------
    retriever : Retriever
        Any concrete Retriever (BM25, Dense, Hybrid, etc.).
        Must have been built before calling search/evaluate.
    evaluator : Evaluator, optional
        If provided, evaluate() will compute metrics.
        If None, calling evaluate() raises RuntimeError.
    """

    def __init__(
        self,
        retriever: Retriever,
        evaluator: Optional[Evaluator] = None,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator

    def search(
        self,
        queries: Sequence[Query],
        top_k: int,
    ) -> Run:
        """
        Run retrieval for all queries.

        Parameters
        ----------
        queries : Sequence[Query]   Must not be empty.
        top_k   : int               > 0

        Returns
        -------
        Run   with hits for every query.

        Raises
        ------
        ValueError    If queries empty or top_k <= 0.
        """
        if not queries:
            raise ValueError("SearchPipeline.search: queries must not be empty.")
        if top_k <= 0:
            raise ValueError(
                f"SearchPipeline.search: top_k must be > 0, got {top_k}"
            )

        logger.info("[SEARCH] Processing %d queries (top_k=%d)...", len(queries), top_k)

        run = Run()
        for i, query in enumerate(queries):
            hits: Sequence[Hit] = self._retriever.retrieve(query, top_k)
            run.add(query.qid, hits)
            if (i + 1) % 100 == 0 or (i + 1) == len(queries):
                logger.info("[SEARCH] %d / %d queries processed.", i + 1, len(queries))

        logger.info("[SEARCH] Done. Run contains %d queries.", len(run))
        return run

    def evaluate(
        self,
        queries: Sequence[Query],
        qrels: Qrels,
        top_k: int,
    ) -> tuple:
        """
        Run retrieval then compute all metrics.

        Parameters
        ----------
        queries : Sequence[Query]
        qrels   : Qrels
        top_k   : int

        Returns
        -------
        tuple (run: Run, metrics: dict[metric_name, float])

        Raises
        ------
        RuntimeError  If no evaluator was provided.
        """
        if self._evaluator is None:
            raise RuntimeError(
                "SearchPipeline.evaluate: no Evaluator provided. "
                "Pass evaluator= to SearchPipeline constructor."
            )

        run = self.search(queries, top_k)
        metrics = self._evaluator.evaluate(run, qrels)
        return run, metrics

