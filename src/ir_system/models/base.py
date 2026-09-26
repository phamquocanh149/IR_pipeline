"""
Models: Base
============
Root abstraction for all model types.

Contract
--------
- Only exposes `name` property.
- encode() and score() are NOT here — they belong to subtypes with different
  computational contracts (EmbeddingModel vs CrossEncoderModel).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Model(ABC):
    """Root abstraction for all models in the IR system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable / canonical identifier for this model."""
        ...
