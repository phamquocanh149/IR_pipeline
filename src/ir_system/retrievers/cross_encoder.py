"""
Retrievers: CrossEncoderRetriever
===================================
Cross-encoder based retriever supporting two modes:

1. Direct scoring (small corpus):
   query + all documents → CrossEncoderModel → ranking

2. Reranking (large corpus):
   candidate_retriever → Top-N candidates → CrossEncoderModel → reranked Top-K

The mode is determined by whether a candidate_retriever is provided.

Dependency injection: candidate_retriever is optional.
CrossEncoderRetriever does NOT scan the entire corpus in reranking mode.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.cross_encoder import CrossEncoderModel
from ir_system.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class CrossEncoderRetriever(Retriever):
    """
    Cross-encoder retriever supporting direct scoring and reranking.

    Parameters
    ----------
    model               : CrossEncoderModel
    candidate_retriever : Retriever, optional
        If provided, operates in reranking mode:
        first retrieves candidate_top_k docs, then reranks with cross-encoder.
    candidate_top_k     : int
        Number of initial candidates for reranking (default 100).
    batch_size          : int
        Scoring batch size (default 32).
    """

    def __init__(
        self,
        model: CrossEncoderModel,
        candidate_retriever: Optional[Retriever] = None,
        candidate_top_k: int = 100,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self._candidate_retriever = candidate_retriever
        self._candidate_top_k = candidate_top_k
        self._batch_size = batch_size

        # State populated by build()
        self._documents: dict[str, str] = {}  # doc_id → text
        self._built = False

    @property
    def _reranking_mode(self) -> bool:
        return self._candidate_retriever is not None

    def build(self, documents: Sequence[Document]) -> None:
        """
        Prepare the retriever for cross-encoder scoring.

        - Always stores doc_id → text mapping for cross-encoder input.
        - Also calls candidate_retriever.build() if in reranking mode.

        Raises
        ------
        ValueError  If corpus is empty.
        """
        if not documents:
            raise ValueError(
                "CrossEncoderRetriever.build: corpus must not be empty."
            )

        logger.info(
            "[RETRIEVER] Building CrossEncoder index (mode=%s) with model=%s "
            "over %d documents...",
            "reranking" if self._reranking_mode else "direct",
            self._model.name,
            len(documents),
        )

        self._documents = {doc.doc_id: doc.text for doc in documents}

        if self._reranking_mode:
            self._candidate_retriever.build(documents)  # type: ignore[union-attr]

        self._built = True
        logger.info("[RETRIEVER] CrossEncoder index ready.")

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Retrieve using cross-encoder scoring.

        In direct mode:  scores all documents in corpus.
        In reranking mode: scores top candidate_top_k from candidate_retriever.

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

        if self._reranking_mode:
            candidates = self._candidate_retriever.retrieve(  # type: ignore[union-attr]
                query, self._candidate_top_k
            )
            doc_ids = [h.doc_id for h in candidates]
        else:
            doc_ids = list(self._documents.keys())

        if not doc_ids:
            return []

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
