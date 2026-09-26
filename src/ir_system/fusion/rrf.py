"""
Fusion: RRFusion (Reciprocal Rank Fusion)
==========================================
RRF formula:

    RRF(d) = Σ_r  1 / (k + rank_r(d))

where:
- rank is 1-based (rank_r(d) = 1 for position 0 in list r)
- k is the smoothing constant (default 60)
- documents not appearing in ranking r contribute 0 from that ranking

Correctness requirements
------------------------
- rank is 1-based, NOT 0-based.
- Duplicate doc_ids within a single ranking are de-duplicated before fusion
  (only the best rank is used per list).
- Tie-break: RRF score desc, doc_id asc.
- RRFusion knows nothing about BM25 or Dense — operates on abstract Hit lists.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from ir_system.domain.hit import Hit
from ir_system.fusion.base import FusionStrategy


class RRFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion strategy.

    Parameters
    ----------
    k : int   RRF smoothing constant (default 60, as in the original paper).
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError(f"RRFusion: k must be > 0, got {k}")
        self._k = k

    def fuse(
        self,
        runs: Sequence[Sequence[Hit]],
        top_k: int,
    ) -> Sequence[Hit]:
        """
        Fuse multiple ranked lists via RRF.

        Parameters
        ----------
        runs  : Sequence[Sequence[Hit]]
            Hit lists in rank order (index 0 = rank 1).
        top_k : int

        Returns
        -------
        List[Hit]  length ≤ top_k.
        """
        if not runs:
            raise ValueError("RRFusion.fuse: runs must not be empty.")
        if top_k <= 0:
            raise ValueError(f"RRFusion.fuse: top_k must be > 0, got {top_k}")

        # Accumulate RRF scores per doc_id
        rrf_scores: Dict[str, float] = defaultdict(float)

        for ranked_list in runs:
            # De-duplicate doc_ids within this list: keep only first (best rank) occurrence
            seen_in_list: set[str] = set()
            for position, hit in enumerate(ranked_list):
                if hit.doc_id in seen_in_list:
                    continue
                seen_in_list.add(hit.doc_id)
                rank = position + 1  # 1-based rank
                rrf_scores[hit.doc_id] += 1.0 / (self._k + rank)

        # Sort by RRF score desc, doc_id asc for deterministic tie-break
        sorted_pairs: List[tuple[str, float]] = sorted(
            rrf_scores.items(),
            key=lambda x: (-x[1], x[0]),
        )

        return [
            Hit(doc_id=doc_id, score=score)
            for doc_id, score in sorted_pairs[:top_k]
        ]
