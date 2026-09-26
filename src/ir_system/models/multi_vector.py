"""
Models: MultiVectorEmbeddingModel
====================================
Abstract base for models that produce a sequence of vectors per text (late-interaction).

Contract
--------
    N input texts → N multi-vector representations (list of variable-length arrays)

Example: ColBERT-style — each token produces one vector.

DO NOT flatten multi-vector into a single vector — that breaks late-interaction scoring.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Sequence

import numpy as np

from ir_system.models.embedding import EmbeddingModel


class MultiVectorEmbeddingModel(EmbeddingModel):
    """
    Abstract base for multi-vector text embeddings (late-interaction).

    encode() must return a list of arrays, one per input text.
    Each array has shape [T_i, dim] where T_i is the number of vectors for text i.
    """

    @abstractmethod
    def encode(
        self, texts: Sequence[str], **kwargs: Any
    ) -> List[np.ndarray]:
        """
        Encode texts into per-text multi-vector representations.

        Parameters
        ----------
        texts    : Sequence[str]
        **kwargs : e.g. batch_size=32

        Returns
        -------
        List[np.ndarray]
            Length N (= len(texts)).
            Element i has shape [T_i, dim] where T_i is the token count for text i.
        """
        ...
