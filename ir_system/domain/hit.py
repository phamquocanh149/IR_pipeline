"""
Domain: Hit
===========
Immutable value object representing a single retrieved document with its score.

Invariants
----------
- doc_id must be non-empty.
- score must be a finite float (no NaN, +inf, -inf).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """Immutable value object representing a retrieved document and its retrieval score."""

    doc_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.doc_id or not self.doc_id.strip():
            raise ValueError(f"Hit.doc_id must be non-empty, got: {self.doc_id!r}")
        if not isinstance(self.score, (int, float)):
            raise ValueError(
                f"Hit.score must be a numeric type, got: {type(self.score)} (doc_id={self.doc_id!r})"
            )
        if not math.isfinite(float(self.score)):
            raise ValueError(
                f"Hit.score must be finite (no NaN/inf), got: {self.score!r} "
                f"(doc_id={self.doc_id!r})"
            )
