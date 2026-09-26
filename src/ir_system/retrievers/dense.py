"""
Retrievers: DenseRetriever
===========================
Dense retrieval using a SingleVectorEmbeddingModel + cosine similarity search.

Architecture
------------
- Accepts any SingleVectorEmbeddingModel (no concrete model assumed).
- Encodes corpus ONCE during build(); never re-encodes per query.
- Maintains strict row-index ↔ doc_id mapping.
- Similarity: cosine (via normalized inner product if model normalizes,
  or explicit normalization otherwise).
- Tie-break: score descending, doc_id ascending.

Key invariant
-------------
    vector row index ↔ Document.doc_id

This mapping is maintained via self._doc_ids[i] ↔ self._embeddings[i].
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence

import numpy as np

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """
    Dense retriever using single-vector embeddings and cosine similarity.

    Parameters
    ----------
    model      : SingleVectorEmbeddingModel
    batch_size : int   Encoding batch size (default 32)
    """

    def __init__(
        self,
        model: SingleVectorEmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self._batch_size = batch_size

        # State populated by build()
        self._doc_ids: List[str] = []
        self._embeddings: np.ndarray | None = None  # shape [N, dim]
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Encode all documents and build the dense index.

        Row i of self._embeddings corresponds to self._doc_ids[i].

        Raises
        ------
        ValueError  If corpus is empty.
        RuntimeError  If embedding shape is unexpected.
        """
        if not documents:
            raise ValueError("DenseRetriever.build: corpus must not be empty.")

        logger.info(
            "[RETRIEVER] Building dense index with model=%s over %d documents...",
            self._model.name,
            len(documents),
        )

        self._doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.text for doc in documents]

        embeddings = self._model.encode(texts, batch_size=self._batch_size)

        # Validate output shape
        if not hasattr(embeddings, "shape") or embeddings.ndim != 2:
            raise RuntimeError(
                f"DenseRetriever.build: model.encode must return a 2-D array, "
                f"got shape {getattr(embeddings, 'shape', type(embeddings))}"
            )
        if embeddings.shape[0] != len(documents):
            raise RuntimeError(
                f"DenseRetriever.build: expected {len(documents)} embeddings, "
                f"got {embeddings.shape[0]}. Row-to-doc_id mapping is broken."
            )

        # L2-normalize for cosine similarity via inner product.
        # If the model already normalizes (e.g. SentenceTransformerAdapter with
        # normalize_embeddings=True), this is a no-op (norms ≈ 1).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)  # avoid divide-by-zero
        self._embeddings = (embeddings / norms).astype(np.float32)

        self._built = True
        logger.info(
            "[RETRIEVER] Dense index ready: %d vectors, dim=%d.",
            self._embeddings.shape[0],
            self._embeddings.shape[1],
        )

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Encode query and return top-K hits by cosine similarity.

        Parameters
        ----------
        query : Query
        top_k : int   > 0

        Returns
        -------
        List[Hit]  length ≤ top_k, score desc / doc_id asc tie-break.

        Raises
        ------
        RuntimeError  If called before build().
        ValueError    If top_k <= 0.
        """
        if not self._built:
            raise RuntimeError(
                "DenseRetriever.retrieve: build() must be called before retrieve()."
            )
        if top_k <= 0:
            raise ValueError(
                f"DenseRetriever.retrieve: top_k must be > 0, got {top_k}"
            )

        # Encode query — single text, returns [1, dim]
        q_emb = self._model.encode([query.text], batch_size=1)
        if q_emb.ndim != 2 or q_emb.shape[0] != 1:
            raise RuntimeError(
                f"DenseRetriever.retrieve: unexpected query embedding shape {q_emb.shape}"
            )

        # Normalize query vector
        q_vec = q_emb[0].astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-10:
            q_vec = q_vec / q_norm

        # Cosine similarity = dot product (both sides normalized)
        scores = self._embeddings @ q_vec  # [N]

        # Sanitize non-finite scores (defensive)
        scores = np.where(np.isfinite(scores), scores, -np.inf)

        # Partial sort for efficiency: argpartition then sort top_k
        n = len(self._doc_ids)
        k_actual = min(top_k, n)

        if k_actual == n:
            indices = np.arange(n)
        else:
            # Get top_k indices (unsorted)
            partition_idx = np.argpartition(scores, -k_actual)[-k_actual:]
            indices = partition_idx

        # Sort selected indices by score desc, doc_id asc for tie-break
        top_pairs = sorted(
            ((self._doc_ids[i], float(scores[i])) for i in indices),
            key=lambda x: (-x[1], x[0]),
        )[:top_k]

        return [
            Hit(doc_id=doc_id, score=score)
            for doc_id, score in top_pairs
            if math.isfinite(score)
        ]
