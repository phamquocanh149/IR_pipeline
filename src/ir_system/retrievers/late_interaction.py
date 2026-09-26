"""
Retrievers: LateInteractionRetriever
======================================
Late-interaction retrieval using ColBERT-style MaxSim scoring.

Architecture
------------
- Accepts a MultiVectorEmbeddingModel.
- Corpus encoded ONCE during build(); stored per-document token matrices.
- Scoring:  score(Q, D) = Σ_i max_j similarity(q_i, d_j)
  where Q = {q_i} are query token vectors, D = {d_j} are doc token vectors.
- Token/vector boundaries preserved per document (NOT flattened).
- Row-to-doc_id mapping maintained via _doc_ids list.
- Tie-break: score desc, doc_id asc.

NOTE: Full corpus MaxSim is O(N * T_q * T_d) which can be slow for large corpora.
For production, use approximate indexing (PLAID, etc.). This is the reference
implementation for correctness.
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence

import numpy as np

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.multi_vector import MultiVectorEmbeddingModel
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class LateInteractionRetriever(Retriever):
    """
    ColBERT-style late-interaction retriever.

    Parameters
    ----------
    model      : MultiVectorEmbeddingModel
    batch_size : int   Encoding batch size (default 32)
    """

    def __init__(
        self,
        model: MultiVectorEmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self._batch_size = batch_size

        # State populated by build()
        self._doc_ids: List[str] = []
        # _doc_vecs[i] = np.ndarray shape [T_i, dim] for document i
        self._doc_vecs: List[np.ndarray] = []
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Encode all documents and store per-document token matrices.

        Each self._doc_vecs[i] corresponds to self._doc_ids[i].
        Multi-vector structure is preserved (NOT flattened).

        Raises
        ------
        ValueError    If corpus is empty.
        RuntimeError  If encoding fails.
        """
        if not documents:
            raise ValueError(
                "LateInteractionRetriever.build: corpus must not be empty."
            )

        logger.info(
            "[RETRIEVER] Building late-interaction index with model=%s "
            "over %d documents...",
            self._model.name,
            len(documents),
        )

        self._doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.text for doc in documents]

        all_vecs = self._model.encode(texts, batch_size=self._batch_size)

        if len(all_vecs) != len(documents):
            raise RuntimeError(
                f"LateInteractionRetriever.build: expected {len(documents)} "
                f"multi-vector representations, got {len(all_vecs)}."
            )

        self._doc_vecs = [v.astype(np.float32) for v in all_vecs]
        self._built = True

        logger.info(
            "[RETRIEVER] Late-interaction index ready: %d documents.",
            len(self._doc_ids),
        )

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Score all documents via MaxSim and return top_k.

        score(Q, D) = Σ_i max_j cos(q_i, d_j)

        Parameters
        ----------
        query : Query
        top_k : int   > 0

        Returns
        -------
        List[Hit]  length ≤ top_k, score desc / doc_id asc tie-break.
        """
        if not self._built:
            raise RuntimeError(
                "LateInteractionRetriever.retrieve: "
                "build() must be called before retrieve()."
            )
        if top_k <= 0:
            raise ValueError(
                f"LateInteractionRetriever.retrieve: top_k must be > 0, got {top_k}"
            )

        # Encode query: List[np.ndarray] of length 1
        q_vecs_list = self._model.encode([query.text], batch_size=1)
        if len(q_vecs_list) != 1:
            raise RuntimeError(
                f"LateInteractionRetriever.retrieve: expected 1 query representation, "
                f"got {len(q_vecs_list)}."
            )
        q_vecs = q_vecs_list[0].astype(np.float32)  # [T_q, dim]

        if q_vecs.ndim != 2:
            raise RuntimeError(
                f"LateInteractionRetriever.retrieve: query vectors must be 2-D, "
                f"got shape {q_vecs.shape}"
            )

        # Score each document
        scores: List[tuple[str, float]] = []
        for i, d_vecs in enumerate(self._doc_vecs):
            if d_vecs.ndim != 2 or d_vecs.shape[0] == 0:
                scores.append((self._doc_ids[i], 0.0))
                continue

            # MaxSim: for each query token q_i, find max similarity over doc tokens
            # sim[t_q, t_d] = q_i · d_j  (vectors already L2-normalized in adapter)
            sim_matrix = q_vecs @ d_vecs.T  # [T_q, T_d]
            max_per_query_token = sim_matrix.max(axis=1)  # [T_q]
            doc_score = float(max_per_query_token.sum())

            if not math.isfinite(doc_score):
                doc_score = 0.0

            scores.append((self._doc_ids[i], doc_score))

        # Sort: score desc, doc_id asc
        scores.sort(key=lambda x: (-x[1], x[0]))

        return [
            Hit(doc_id=doc_id, score=score)
            for doc_id, score in scores[:top_k]
        ]
