"""
Evaluation: Recall@K
=====================
Recall at cutoff K.

    Recall@K = |relevant docs retrieved in top K| / |relevant docs in qrels|

Policy
------
- Relevant condition: qrels relevance > 0 (consistent with MRR/NDCG policy).
- Duplicate retrieved docs count only ONCE (deduplicated by doc_id).
- Denominator = total relevant docs for query in qrels (NOT len(top_k_hits)).
- Numerator never exceeds denominator.
- Query with no relevant docs in qrels: excluded from macro average (denominator=0 guard).
- Query with relevant docs but none retrieved: contributes 0.0 to the average.
"""
from __future__ import annotations

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric


class RecallAtK(Metric):
    """Recall at cutoff K."""

    def __init__(self, k: int, relevance_threshold: float = 0.0) -> None:
        """
        Parameters
        ----------
        k                   : int    Cutoff
        relevance_threshold : float  Exclusive lower bound to count as relevant.
        """
        if k <= 0:
            raise ValueError(f"RecallAtK: k must be > 0, got {k}")
        self._k = k
        self._threshold = relevance_threshold

    @property
    def name(self) -> str:
        return f"recall@{self._k}"

    def compute(self, run: Run, qrels: Qrels) -> float:
        """
        Compute Recall@K as macro average over evaluable queries.

        Queries with no relevant documents in qrels are excluded from denominator.
        """
        qids = list(run.qids)
        if not qids:
            return 0.0

        total_recall = 0.0
        num_evaluable = 0

        for qid in qids:
            judgments = qrels.judgments_for(qid)
            relevant_set = {
                doc_id
                for doc_id, rel in judgments.items()
                if rel > self._threshold
            }

            if not relevant_set:
                # No relevant documents → cannot compute recall → skip
                continue

            num_evaluable += 1
            denominator = len(relevant_set)

            # De-duplicate retrieved doc_ids (Run already prevents duplicates,
            # but we guard defensively here).
            hits = run.get(qid)
            retrieved_relevant = set()
            for hit in hits[: self._k]:
                if hit.doc_id in relevant_set:
                    retrieved_relevant.add(hit.doc_id)

            numerator = len(retrieved_relevant)
            # Invariant: numerator <= denominator
            assert numerator <= denominator, (
                f"RecallAtK: numerator {numerator} > denominator {denominator} "
                f"for qid={qid!r}. This is a bug."
            )

            total_recall += numerator / denominator

        if num_evaluable == 0:
            return 0.0

        return total_recall / num_evaluable
