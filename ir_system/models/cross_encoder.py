"""
Models: CrossEncoderModel
==========================
Abstract base for cross-encoder models (query+document joint scoring).

Cross-encoder is a SIBLING of EmbeddingModel — NOT a subtype.
Do NOT inherit EmbeddingModel; do NOT add a fake encode() method.

Contract
--------
    N (query, document) pairs → N scores

Used by CrossEncoderRetriever (direct scoring or reranking).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

from ir_system.models.base import Model


class CrossEncoderModel(Model):
    """
    Abstract base for cross-encoder scoring models.

    score() takes parallel sequences of query texts and document texts
    and returns a relevance score for each pair.
    """

    @abstractmethod
    def score(
        self,
        query_texts: Sequence[str],
        document_texts: Sequence[str],
        **kwargs: Any,
    ) -> Sequence[float]:
        """
        Score query-document pairs jointly.

        Parameters
        ----------
        query_texts    : Sequence[str]  length N
        document_texts : Sequence[str]  length N
        **kwargs       : e.g. batch_size=32

        Returns
        -------
        Sequence[float]  length N
            Relevance score for each pair (query_texts[i], document_texts[i]).

        Raises
        ------
        ValueError
            If len(query_texts) != len(document_texts).
        """
        ...
