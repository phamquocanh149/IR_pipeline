"""
Domain: Run
===========
Output of retrieval for an entire query set.

    qid → ordered list[Hit]  (ranking order preserved)

Invariants
----------
- Ranking order is preserved (index 0 = rank 1).
- Duplicate doc_id within the same query are rejected.
- qid must be non-empty.
"""
from __future__ import annotations

from typing import Sequence


class Run:
    """
    Container for retrieval results across all queries.

    Maintains per-query ordered Hit lists; insertion order determines rank.
    Duplicate doc_id within a single query is disallowed.
    """

    def __init__(self) -> None:
        # _data[qid] = list of Hit in rank order
        self._data: dict[str, list] = {}

    def add(self, qid: str, hits: Sequence) -> None:
        """
        Add ranked hits for a query.

        Parameters
        ----------
        qid  : str
            Query identifier.
        hits : Sequence[Hit]
            Hits in descending score order (rank 1 = hits[0]).

        Raises
        ------
        ValueError
            If qid already has results, or if duplicate doc_ids are present.
        """
        if not qid or not qid.strip():
            raise ValueError(f"Run.add: qid must be non-empty, got {qid!r}")
        if qid in self._data:
            raise ValueError(
                f"Run.add: qid {qid!r} already has results. "
                "Call add() once per query."
            )
        seen: set[str] = set()
        for hit in hits:
            if hit.doc_id in seen:
                raise ValueError(
                    f"Run.add: duplicate doc_id {hit.doc_id!r} in hits for qid {qid!r}"
                )
            seen.add(hit.doc_id)
        self._data[qid] = list(hits)

    def get(self, qid: str) -> Sequence:
        """
        Return hits for a query in rank order.

        Returns an empty list if qid not present.
        """
        return self._data.get(qid, [])

    @property
    def qids(self) -> frozenset[str]:
        """Return all query IDs in this run."""
        return frozenset(self._data.keys())

    def __len__(self) -> int:
        """Number of queries in this run."""
        return len(self._data)

    def __repr__(self) -> str:
        return f"Run(num_queries={len(self._data)})"
