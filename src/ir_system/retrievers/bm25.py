"""
Retrievers: BM25Retriever
==========================
Concrete sparse retriever using eager sparse scoring via BM25S (Lucene variant).

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

import numpy as np
from tqdm import tqdm

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.retrievers.sparse import SparseRetriever

logger = logging.getLogger(__name__)

_DEFAULT_TOKENIZER: Callable[[str], List[str]] = lambda text: text.lower().split()


class BM25Retriever(SparseRetriever):
    """
    BM25 retriever backed by BM25S's precomputed sparse score index.

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
            import bm25s  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "bm25s is required for BM25 retrieval. "
                "Install it with: pip install bm25s"
            ) from exc

        self._tokenizer = tokenizer or _DEFAULT_TOKENIZER
        self._k1 = k1
        self._b = b

        # State populated by build()
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_id_order = np.empty(0, dtype=np.intp)
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        """
        Tokenize documents and build the BM25 index.

        Raises
        ------
        ValueError  If corpus is empty or contains no tokens after tokenization.
        """
        if not documents:
            raise ValueError("BM25Retriever.build: corpus must not be empty.")

        self._built = False
        logger.info("[RETRIEVER] Building BM25 index over %d documents...", len(documents))

        doc_ids: List[str] = []
        tokenized_corpus: List[List[str]] = []

        for doc in tqdm(documents, desc="[BM25] Tokenizing corpus", unit="doc"):
            doc_ids.append(doc.doc_id)
            if not doc.text or not doc.text.strip():
                tokenized_corpus.append([])
            else:
                tokenized_corpus.append(self._tokenizer(doc.text))

        if not any(tokenized_corpus):
            raise ValueError("BM25Retriever.build: corpus contains no tokens after tokenization.")

        import bm25s

        index = bm25s.BM25(k1=self._k1, b=self._b, method="lucene", backend="numpy")
        # Preserve the caller's tokenizer: BM25S's default tokenizer would
        # otherwise introduce different punctuation/stopword handling.
        index.index(tokenized_corpus, show_progress=False)
        self._bm25 = index
        self._doc_ids = doc_ids
        self._doc_id_order = np.asarray(
            sorted(range(len(doc_ids)), key=doc_ids.__getitem__), dtype=np.intp
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
        ValueError    If top_k <= 0.
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

        scores = self._bm25.get_scores(tokens)  # sparse postings -> [N_docs]
        n = len(self._doc_ids)
        k = min(top_k, n)
        if k == n:
            indices = np.arange(n)
        else:
            cutoff = np.partition(scores, n - k)[n - k]
            above = np.flatnonzero(scores > cutoff)
            # Include only the lexicographically first IDs tied at the cutoff.
            # Plain argpartition could arbitrarily discard a tied top-k hit.
            tied = self._doc_id_order[scores[self._doc_id_order] == cutoff]
            indices = np.concatenate((above, tied[:k - len(above)]))

        # Sort only selected hits; scoring is not repeated for cutoff ties.
        paired = [
            (self._doc_ids[i], float(scores[i]))
            for i in indices
        ]
        paired.sort(key=lambda x: (-x[1], x[0]))

        return [Hit(doc_id=doc_id, score=score) for doc_id, score in paired]
