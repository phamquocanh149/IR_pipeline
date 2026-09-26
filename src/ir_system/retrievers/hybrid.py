"""
Retrievers: HybridRetriever
============================
Combines multiple retrievers via a pluggable FusionStrategy.

Architecture (Composition + Strategy Pattern)
---------------------------------------------
    Retriever A  →  ranking A  ──┐
                                  ├──→ FusionStrategy → final ranking
    Retriever B  →  ranking B  ──┘

HybridRetriever does NOT contain any RRF formula directly.
The formula lives in the FusionStrategy (e.g. RRFusion).

Constructor
-----------
    HybridRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        fusion_strategy=RRFusion(k=60),
        candidate_top_k=100,
    )
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.fusion.base import FusionStrategy
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """
    Hybrid retriever that fuses results from multiple sub-retrievers.

    Parameters
    ----------
    retrievers       : Sequence[Retriever]   At least two.
    fusion_strategy  : FusionStrategy
    candidate_top_k  : int
        How many candidates each sub-retriever retrieves before fusion.
        Should be >= final top_k. Default 100.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fusion_strategy: FusionStrategy,
        candidate_top_k: int = 100,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError(
                f"HybridRetriever requires at least 2 retrievers, got {len(retrievers)}."
            )
        self._retrievers = list(retrievers)
        self._fusion = fusion_strategy
        self._candidate_top_k = candidate_top_k
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Build all sub-retriever indexes.

        Raises
        ------
        ValueError  If corpus is empty.
        """
        if not documents:
            raise ValueError("HybridRetriever.build: corpus must not be empty.")

        logger.info(
            "[RETRIEVER] Building HybridRetriever (%d sub-retrievers) "
            "over %d documents...",
            len(self._retrievers),
            len(documents),
        )

        for retriever in self._retrievers:
            retriever.build(documents)

        self._built = True
        logger.info("[RETRIEVER] HybridRetriever index ready.")

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Retrieve from all sub-retrievers and fuse results.

        Parameters
        ----------
        query : Query
        top_k : int   > 0

        Returns
        -------
        List[Hit]  length ≤ top_k, fused ranking.
        """
        if not self._built:
            raise RuntimeError(
                "HybridRetriever.retrieve: build() must be called before retrieve()."
            )
        if top_k <= 0:
            raise ValueError(
                f"HybridRetriever.retrieve: top_k must be > 0, got {top_k}"
            )

        candidate_k = max(self._candidate_top_k, top_k)
        runs: List[Sequence[Hit]] = []
        for retriever in self._retrievers:
            hits = retriever.retrieve(query, candidate_k)
            runs.append(hits)

        return self._fusion.fuse(runs, top_k)
