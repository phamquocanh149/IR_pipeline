"""
Models: EmbeddingModel
======================
Abstract base for models that encode text into vector representations.

Contract
--------
- encode(texts) must:
  - support batch processing;
  - preserve input order;
  - return exactly len(texts) representations;
  - not silently drop empty items.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

from ir_system.models.base import Model


class EmbeddingModel(Model):
    """
    Abstract base for text embedding models.

    Subclasses must implement `encode()`.
    The return type is intentionally flexible (numpy array, torch tensor, list of arrays)
    because SingleVector and MultiVector models return different shapes.
    Concrete subclasses document their exact return types.
    """

    @abstractmethod
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        """
        Encode a sequence of texts into vector representations.

        Parameters
        ----------
        texts    : Sequence[str]
            Input texts. May be empty.
        **kwargs : Any
            Implementation-specific options (e.g. batch_size, show_progress).

        Returns
        -------
        Any
            Shape and type determined by the concrete subclass.
            Guaranteed: len(return_value) == len(texts) when texts is non-empty.
        """
        ...
