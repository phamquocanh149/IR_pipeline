"""
Fusion: FusionStrategy (abstract base)
=======================================
Strategy interface for combining multiple ranked lists into one.

FusionStrategy knows nothing about BM25, Dense, or any retriever type.
It operates purely on Sequence[Hit] lists (one per retriever).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ir_system.domain.hit import Hit


class FusionStrategy(ABC):
    """Abstract base for ranking fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        runs: Sequence[Sequence[Hit]],
        top_k: int,
    ) -> Sequence[Hit]:
        """
        Fuse multiple ranked hit lists into a single ranking.

        Parameters
        ----------
        runs  : Sequence[Sequence[Hit]]
            One hit list per retriever, each in rank order (rank 1 = index 0).
        top_k : int   > 0   Maximum number of results to return.

        Returns
        -------
        Sequence[Hit]
            Fused ranking, at most top_k results, score desc / doc_id asc tie-break.

        Raises
        ------
        ValueError  If runs is empty or top_k <= 0.
        """
        ...
