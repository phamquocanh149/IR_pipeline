"""
Retrievers: Base
=================
Abstract base for all retriever types.

Contract
--------
    build(documents)             → index/encode corpus (once)
    retrieve(query, top_k) → Sequence[Hit]

Invariants
----------
- Must call build() before retrieve().
- top_k > 0.
- doc_id in results must belong to indexed corpus.
- Ranking is deterministic.
- Corpus is NOT re-encoded for each query.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query


class Retriever(ABC):
    """Abstract base for all retrieval strategies."""

    @abstractmethod
    def build(self, documents: Sequence[Document]) -> None:
        """
        Build the retrieval index from a document corpus.

        Must be called exactly once before retrieve().
        Implementations must NOT rebuild for each query.

        Parameters
        ----------
        documents : Sequence[Document]   The full corpus.

        Raises
        ------
        ValueError  If corpus is empty or documents are invalid.
        """
        ...

    @abstractmethod
    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Retrieve top-K documents for a single query.

        Parameters
        ----------
        query : Query
        top_k : int   Must be > 0.

        Returns
        -------
        Sequence[Hit]
            At most top_k hits, sorted by score descending (rank 1 = index 0).
            All doc_ids belong to the indexed corpus.
            Ties broken deterministically (score desc, doc_id asc).

        Raises
        ------
        RuntimeError  If called before build().
        ValueError    If top_k <= 0.
        """
        ...
