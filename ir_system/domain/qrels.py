"""
Domain: Qrels
=============
Ground-truth relevance judgments: (qid, doc_id) → relevance.

Supports graded relevance — relevance values are NOT binarized here.
Qrels does not know which metrics will consume it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping


class Qrels:
    """
    Container for relevance judgments.

    Relevance is graded (float); callers choose their own threshold.
    Missing (qid, doc_id) pairs implicitly have relevance = 0.
    """

    def __init__(self) -> None:
        # _data[qid][doc_id] = relevance
        self._data: dict[str, dict[str, float]] = defaultdict(dict)

    def add(self, qid: str, doc_id: str, relevance: float) -> None:
        """Add a single judgment."""
        if not qid or not qid.strip():
            raise ValueError(f"Qrels.add: qid must be non-empty, got {qid!r}")
        if not doc_id or not doc_id.strip():
            raise ValueError(
                f"Qrels.add: doc_id must be non-empty, got {doc_id!r}"
            )
        if not isinstance(relevance, (int, float)):
            raise ValueError(
                f"Qrels.add: relevance must be numeric, got {type(relevance)} "
                f"for ({qid!r}, {doc_id!r})"
            )
        self._data[qid][doc_id] = float(relevance)

    def relevance(self, qid: str, doc_id: str) -> float:
        """
        Return relevance for (qid, doc_id).
        Returns 0.0 for any pair not in the judgment set.
        """
        return self._data.get(qid, {}).get(doc_id, 0.0)

    def judgments_for(self, qid: str) -> Mapping[str, float]:
        """
        Return all judgments for a query as {doc_id: relevance}.
        Returns an empty mapping if qid has no judgments.
        """
        return dict(self._data.get(qid, {}))

    @property
    def qids(self) -> frozenset[str]:
        """Return all query IDs that have at least one judgment."""
        return frozenset(self._data.keys())

    def __len__(self) -> int:
        """Total number of (qid, doc_id) judgment pairs."""
        return sum(len(docs) for docs in self._data.values())

    def __repr__(self) -> str:
        return f"Qrels(num_queries={len(self._data)}, num_pairs={len(self)})"
