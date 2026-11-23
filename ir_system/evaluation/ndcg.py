"""
Evaluation: NDCG@K
===================
Normalized Discounted Cumulative Gain at cutoff K.

DCG@K = Σ_{i=1..K}  (2^rel_i - 1) / log2(i + 1)

NDCG@K = DCG@K / IDCG@K

where IDCG@K is the DCG of the ideal (perfect) ranking.

Policy
------
- Graded relevance is preserved — qrels are NOT binarized.
- Cutoff K is applied exactly.
- If IDCG@K == 0 (query has no positive judgments), NDCG@K = 0.0
  and the query is EXCLUDED from the macro average denominator.
  (Rationale: a query with no positive judgments cannot be evaluated.)
- If a query has positive judgments but the system retrieved nothing,
  the query contributes NDCG = 0.0 and IS included in the denominator.
- No divide-by-zero: all edge cases handled explicitly.
"""
from __future__ import annotations

import math

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric


def _dcg(relevances: list[float], k: int) -> float:
    """Compute DCG@K for an ordered list of relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)  # i+2 because log2(rank+1), rank=i+1
    return dcg


class NDCGAtK(Metric):
    """Normalized Discounted Cumulative Gain at cutoff K."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"NDCGAtK: k must be > 0, got {k}")
        self._k = k

    @property
    def name(self) -> str:
        return f"ndcg@{self._k}"

    def compute(self, run: Run, qrels: Qrels) -> float:
        """
        Compute NDCG@K as macro average over evaluable queries.

        Queries with no positive judgment are excluded from the denominator.
        Queries with positive judgments but zero DCG contribute 0.0.
        """
        qids = list(run.qids)
        if not qids:
            return 0.0

        total_ndcg = 0.0
        num_evaluable = 0

        for qid in qids:
            judgments = qrels.judgments_for(qid)
            positive_rels = [r for r in judgments.values() if r > 0]

            if not positive_rels:
                # No positive judgments → query cannot be evaluated → skip
                continue

            num_evaluable += 1

            # Ideal ranking: sort all relevant docs by relevance desc
            ideal_rels = sorted(positive_rels, reverse=True)
            idcg = _dcg(ideal_rels, self._k)

            if idcg == 0.0:
                # Defensive: should not happen since we checked positive_rels
                total_ndcg += 0.0
                continue

            # Actual ranking
            hits = run.get(qid)
            actual_rels = [qrels.relevance(qid, hit.doc_id) for hit in hits[: self._k]]
            dcg = _dcg(actual_rels, self._k)

            total_ndcg += dcg / idcg

        if num_evaluable == 0:
            return 0.0

        return total_ndcg / num_evaluable
