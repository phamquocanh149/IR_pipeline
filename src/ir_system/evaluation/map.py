"""
Evaluation: MAP@K
=================
Mean Average Precision at cutoff K.

For one query:
    AP@K(q) = (1 / R_q) * Σ_{i=1..K} [rel_i * Precision@i]

where:
    rel_i = 1 if rank i contains a relevant document, else 0
    Precision@i = (number of relevant documents retrieved up to rank i) / i
    R_q = total number of relevant documents for query q in Qrels

Macro MAP@K:
    MAP@K = (1 / |Q'|) * Σ_{q in Q'} AP@K(q)
where Q' is the set of queries with at least one relevant document in Qrels (R_q > 0).

Policy
------
- Consider only ranks 1..K.
- Denominator for a query is R_q (total relevant documents in Qrels for q).
- Relevant condition: relevance > 0 by default (relevance >= relevance_threshold for threshold > 0).
- Relevant documents outside top K contribute 0.
- Duplicate retrieved docs are counted at most ONCE.
- Zero-relevant queries (R_q == 0) are excluded from the macro aggregation to prevent division by zero.
- If no queries have relevant documents (|Q'| == 0), returns 0.0.
- Validate k > 0.
"""
from __future__ import annotations

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run
from ir_system.evaluation.metric import Metric


class MAPAtK(Metric):
    """
    Mean Average Precision at cutoff K.

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
            raise ValueError(f"MAPAtK: k must be > 0, got {k}")
        self.k = k
        self._k = k
        self._threshold = relevance_threshold

    @property
    def name(self) -> str:
        return f"map@{self.k}"

    def _is_relevant(self, rel: float) -> bool:
        if self._threshold > 0:
            return rel >= self._threshold
        return rel > self._threshold

    def compute(self, run: Run, qrels: Qrels) -> float:
        """
        Compute MAP@K as macro average over evaluable queries.

        Queries with no relevant documents in qrels are excluded from aggregation.
        """
        qids = list(run.qids)
        if not qids:
            return 0.0

        total_ap = 0.0
        num_evaluable = 0

        for qid in qids:
            judgments = qrels.judgments_for(qid)
            relevant_docs = {
                doc_id
                for doc_id, rel in judgments.items()
                if self._is_relevant(rel)
            }
            num_relevant = len(relevant_docs)

            if num_relevant == 0:
                # Exclude zero-relevant queries from MAP aggregation
                continue

            num_evaluable += 1

            hits = run.get(qid)
            top_k_hits = hits[: self._k]

            ap_numerator = 0.0
            relevant_seen = 0
            seen_docs = set()

            for rank_zero, hit in enumerate(top_k_hits):
                rank = rank_zero + 1  # 1-based rank
                doc_id = hit.doc_id

                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)

                if doc_id in relevant_docs:
                    relevant_seen += 1
                    precision_at_rank = relevant_seen / rank
                    ap_numerator += precision_at_rank

            ap = ap_numerator / num_relevant
            total_ap += ap

        if num_evaluable == 0:
            return 0.0

        return total_ap / num_evaluable
