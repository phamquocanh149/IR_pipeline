"""
Domain: Query
=============
Immutable value object representing a single retrieval query.

Invariants
----------
- qid must be non-empty.
- text must be non-empty after strip().
- Unicode is preserved as-is; no silent normalization.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    """Immutable value object for a search query."""

    qid: str
    text: str

    def __post_init__(self) -> None:
        if not self.qid or not self.qid.strip():
            raise ValueError(f"Query.qid must be non-empty, got: {self.qid!r}")
        if not self.text or not self.text.strip():
            raise ValueError(
                f"Query.text must be non-empty after strip (qid={self.qid!r})"
            )
