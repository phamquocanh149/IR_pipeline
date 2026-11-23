"""
Retrievers: CrossEncoderRetriever
===================================
Two-stage cross-encoder retriever:

  Stage 1 (Candidate Retrieval):
    A first-stage Retriever (e.g. BM25, Dense) retrieves candidate_top_k
    documents from the corpus.

  Stage 2 (Cross-Encoder Reranking):
    CrossEncoderModel scores each (query, candidate_doc) pair and reranks
    to produce the final top-k results.

The first-stage retriever is injected via constructor (Dependency Injection).
CrossEncoderRetriever does NOT know which concrete retriever is used —
that decision belongs to the CLI composition root.
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.cross_encoder import CrossEncoderModel
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class CrossEncoderRetriever(Retriever):
    """
    Two-stage cross-encoder retriever.

    Stage 1: candidate_retriever retrieves candidate_top_k documents.
    Stage 2: CrossEncoderModel scores and reranks candidates.

    Parameters
    ----------
    model               : CrossEncoderModel
        The cross-encoder model used for scoring (query, document) pairs.
    candidate_retriever : Retriever
        First-stage retriever for candidate selection (e.g. BM25, Dense).
    candidate_top_k     : int
        Number of candidates to retrieve in Stage 1 (default 100).
    batch_size          : int
        Scoring batch size for Stage 2 (default 32).
    """

    def __init__(
        self,
        model: CrossEncoderModel,
        candidate_retriever: Retriever,
        candidate_top_k: int = 100,
        batch_size: int = 32,
    ) -> None:
        if candidate_retriever is None:
            raise ValueError(
                "CrossEncoderRetriever requires a candidate_retriever "
                "(Stage 1). Pass a concrete Retriever instance "
                "(e.g. BM25Retriever, DenseRetriever)."
            )
        self._model = model
        self._candidate_retriever = candidate_retriever
        self._candidate_top_k = candidate_top_k
        self._batch_size = batch_size

        # State populated by build()
        self._documents: dict[str, str] = {}  # doc_id → text
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Build indexes for both stages.

        - Stores doc_id → text mapping for cross-encoder input (Stage 2).
        - Delegates to candidate_retriever.build() for Stage 1 index.

        Raises
        ------
        ValueError  If corpus is empty.
        """
        if not documents:
            raise ValueError(
                "CrossEncoderRetriever.build: corpus must not be empty."
            )

        logger.info(
            "[RETRIEVER] Building CrossEncoder 2-stage index "
            "(stage1=%s, stage2=%s) over %d documents...",
            type(self._candidate_retriever).__name__,
            self._model.name,
            len(documents),
        )

        self._documents = {doc.doc_id: doc.text for doc in documents}
        self._candidate_retriever.build(documents)

        self._built = True
        logger.info("[RETRIEVER] CrossEncoder 2-stage index ready.")

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Two-stage retrieval using cross-encoder scoring.

        Stage 1: candidate_retriever retrieves candidate_top_k documents.
        Stage 2: CrossEncoderModel scores each (query, doc) pair and reranks.

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
                "CrossEncoderRetriever.retrieve: "
                "build() must be called before retrieve()."
            )
        if top_k <= 0:
            raise ValueError(
                f"CrossEncoderRetriever.retrieve: top_k must be > 0, got {top_k}"
            )

        # --- Stage 1: Candidate retrieval ---
        candidates = self._candidate_retriever.retrieve(
            query, self._candidate_top_k
        )
        doc_ids = [h.doc_id for h in candidates]

        if not doc_ids:
            return []

        # --- Stage 2: Cross-encoder scoring ---
        query_texts = [query.text] * len(doc_ids)
        doc_texts = [self._documents[did] for did in doc_ids]

        raw_scores = self._model.score(
            query_texts, doc_texts, batch_size=self._batch_size
        )

        pairs: List[tuple[str, float]] = []
        for doc_id, s in zip(doc_ids, raw_scores):
            s_float = float(s)
            if not math.isfinite(s_float):
                logger.warning(
                    "[RETRIEVER] CrossEncoder: non-finite score for doc_id=%r, "
                    "replacing with -inf.",
                    doc_id,
                )
                s_float = float("-inf")
            pairs.append((doc_id, s_float))

        pairs.sort(key=lambda x: (-x[1], x[0]))

        return [
            Hit(doc_id=doc_id, score=score)
            for doc_id, score in pairs[:top_k]
            if math.isfinite(score)
        ]

