"""
Models: SingleVectorEmbeddingModel
====================================
Abstract base for models that produce exactly one dense vector per text.

Contract
--------
    N input texts → N vectors  (shape [N, dim])

Used by DenseRetriever.
DenseRetriever must NOT assume any concrete backend (SentenceTransformer, HF, etc.).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import numpy as np

from ir_system.models.embedding import EmbeddingModel


class SingleVectorEmbeddingModel(EmbeddingModel):
    """
    Abstract base for single-vector text embeddings.

    encode() must return a 2-D array-like with shape [N, dim] where N = len(texts).
    """

    @abstractmethod
    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        """
        Encode texts into a matrix of dense vectors.

        Parameters
        ----------
        texts    : Sequence[str]
        **kwargs : e.g. batch_size=32, show_progress=False

        Returns
        -------
        np.ndarray
            Shape [N, dim], dtype float32.
            Row i corresponds to texts[i].
        """
        ...
