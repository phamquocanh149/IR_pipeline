"""
Evaluation: MRR@K
==================
Mean Reciprocal Rank at cutoff K.

Per-query reciprocal rank:
    RR(q) = 1 / rank_of_first_relevant_in_top_K

    rank is 1-based. If no relevant document in top K, RR(q) = 0.

Macro MRR:
    MRR@K = (1 / |Q|) * Σ_q RR(q)

Policy
------
- Relevant condition: qrels relevance > 0 (graded relevance supported).
- Queries with no relevant document in top K contribute RR = 0.
- Denominator is |Q| (all queries), not just queries that found relevant docs.
- Queries not in qrels are treated as having no relevant documents.
"""
from __future__ import annotations

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric


class MRRAtK(Metric):
    """Mean Reciprocal Rank at cutoff K."""

    def __init__(self, k: int, relevance_threshold: float = 0.0) -> None:
        """
        Parameters
        ----------
        k                   : int    Cutoff (top-K hits to consider)
        relevance_threshold : float  Minimum relevance to count as relevant (exclusive).
                                     Default 0.0 means relevance > 0 is relevant.
        """
        if k <= 0:
            raise ValueError(f"MRRAtK: k must be > 0, got {k}")
        self._k = k
        self._threshold = relevance_threshold

    @property
    def name(self) -> str:
        return f"mrr@{self._k}"

    def compute(self, run: Run, qrels: Qrels) -> float:
        """Compute MRR@K as macro average over all queries in run."""
        qids = list(run.qids)
        if not qids:
            return 0.0

        total_rr = 0.0
        for qid in qids:
            hits = run.get(qid)
            top_k_hits = hits[: self._k]
            rr = 0.0
            for rank_zero, hit in enumerate(top_k_hits):
                rel = qrels.relevance(qid, hit.doc_id)
                if rel > self._threshold:
                    rr = 1.0 / (rank_zero + 1)  # rank is 1-based
                    break
            total_rr += rr

        return total_rr / len(qids)
