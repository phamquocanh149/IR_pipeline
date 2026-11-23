"""
Evaluation: Precision@K
========================
Precision at cutoff K.

    Precision@K(q) = |relevant docs retrieved in top K| / K

Macro Precision@K:
    Precision@K = (1 / |Q|) * Σ_q Precision@K(q)

Policy
------
- Consider only ranks 1..K.
- Denominator is K, NOT the number of retrieved hits.
- If fewer than K documents are returned, missing positions are treated as nonrelevant.
- Relevant condition: relevance > 0 by default (relevance >= relevance_threshold for threshold > 0).
- Duplicate retrieved docs are counted at most ONCE.
- Queries with no positive judgments in qrels contribute Precision@K = 0.0.
- If run contains no queries (|Q| == 0), returns 0.0.
- Validate k > 0.
"""
from __future__ import annotations

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric


class PrecisionAtK(Metric):
    """
    Precision at cutoff K.

    Parameters
    ----------
    k                   : int    Cutoff rank (must be > 0).
    relevance_threshold : int    Relevance threshold (default: 1).
                                 Documents with relevance >= threshold (for threshold > 0)
                                 or relevance > threshold (for threshold <= 0)
                                 are considered relevant.
    """

    def __init__(self, k: int, relevance_threshold: int = 1) -> None:
        if k <= 0:
            raise ValueError(f"PrecisionAtK: k must be > 0, got {k}")
        self.k = k
        self._k = k
        self._threshold = relevance_threshold

    @property
    def name(self) -> str:
        return f"precision@{self.k}"

    def _is_relevant(self, rel: float) -> bool:
        if self._threshold > 0:
            return rel >= self._threshold
        return rel > self._threshold

    def compute(self, run: Run, qrels: Qrels) -> float:
        """
        Compute Precision@K as macro average over all queries in run.

        Parameters
        ----------
        run   : Run    — retrieved hits per query
        qrels : Qrels  — relevance judgments

        Returns
        -------
        float in [0.0, 1.0]
        """
        qids = list(run.qids)
        if not qids:
            return 0.0

        total_precision = 0.0
        for qid in qids:
            hits = run.get(qid)
            top_k_hits = hits[: self._k]

            relevant_retrieved = 0
            seen_docs = set()

            for hit in top_k_hits:
                if hit.doc_id in seen_docs:
                    continue
                seen_docs.add(hit.doc_id)

                rel = qrels.relevance(qid, hit.doc_id)
                if self._is_relevant(rel):
                    relevant_retrieved += 1

            # Denominator is strictly K; missing positions count as non-relevant
            total_precision += relevant_retrieved / self._k

        return total_precision / len(qids)
