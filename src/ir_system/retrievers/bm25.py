"""
Retrievers: BM25Retriever
==========================
Concrete sparse retriever using the BM25 algorithm via rank_bm25.

Implementation guarantees
-------------------------
- Index built once via build(); NOT rebuilt for each query.
- Document IDs stored alongside the index (no array-index-as-identity).
- BM25 scores mapped to correct doc_id via stored mapping.
- Tie-break: score descending, doc_id ascending (deterministic).
- Empty queries and documents handled with clear policy.
- No dependency on any neural model or Hugging Face library.

Tokenization
------------
Default: whitespace split + lowercase. Configurable via tokenizer callable.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.retrievers.sparse import SparseRetriever

logger = logging.getLogger(__name__)

_DEFAULT_TOKENIZER: Callable[[str], List[str]] = lambda text: text.lower().split()


class BM25Retriever(SparseRetriever):
    """
    BM25 retriever backed by rank_bm25.

    Parameters
    ----------
    tokenizer : Callable[[str], List[str]], optional
        Tokenization function. Default: lowercase whitespace split.
    k1        : float   BM25 term-saturation parameter (default 1.5)
    b         : float   BM25 length-normalization parameter (default 0.75)
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        try:
            import rank_bm25  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install it with: pip install rank-bm25"
            ) from exc

        self._tokenizer = tokenizer or _DEFAULT_TOKENIZER
        self._k1 = k1
        self._b = b

        # State populated by build()
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Tokenize documents and build the BM25 index.

        Raises
        ------
        ValueError  If corpus is empty or any document has empty text.
        """
        if not documents:
            raise ValueError("BM25Retriever.build: corpus must not be empty.")

        logger.info("[RETRIEVER] Building BM25 index over %d documents...", len(documents))

        self._doc_ids = []
        tokenized_corpus: List[List[str]] = []

        for i, doc in enumerate(documents):
            if not doc.text.strip():
                raise ValueError(
                    f"BM25Retriever.build: document at index {i} "
                    f"(doc_id={doc.doc_id!r}) has empty text. "
                    "BM25 requires non-empty document text."
                )
            self._doc_ids.append(doc.doc_id)
            tokenized_corpus.append(self._tokenizer(doc.text))

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(
            tokenized_corpus, k1=self._k1, b=self._b
        )
        self._built = True
        logger.info("[RETRIEVER] BM25 index ready (%d documents).", len(self._doc_ids))

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        """
        Retrieve top_k documents via BM25 scoring.

        Parameters
        ----------
        query : Query
        top_k : int   > 0

        Returns
        -------
        List[Hit]  length ≤ top_k, sorted by BM25 score desc / doc_id asc.

        Raises
        ------
        RuntimeError  If called before build().
        ValueError    If top_k <= 0 or query text is empty after tokenization.
        """
        if not self._built:
            raise RuntimeError(
                "BM25Retriever.retrieve: build() must be called before retrieve()."
            )
        if top_k <= 0:
            raise ValueError(f"BM25Retriever.retrieve: top_k must be > 0, got {top_k}")

        tokens = self._tokenizer(query.text)
        if not tokens:
            # Empty query after tokenization: return no results (explicit policy).
            logger.warning(
                "[RETRIEVER] BM25: query qid=%r produced no tokens after tokenization; "
                "returning empty result.",
                query.qid,
            )
            return []

        scores = self._bm25.get_scores(tokens)  # shape: [N_docs]

        # Build (doc_id, score) pairs and sort: score desc, doc_id asc for tie-break.
        paired = [
            (self._doc_ids[i], float(scores[i]))
            for i in range(len(self._doc_ids))
        ]
        paired.sort(key=lambda x: (-x[1], x[0]))

        top = paired[:top_k]
        return [Hit(doc_id=doc_id, score=score) for doc_id, score in top]
