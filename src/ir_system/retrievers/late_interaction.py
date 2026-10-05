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
implementation with optional blocked CUDA scoring.
"""
from __future__ import annotations

import heapq
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
    batch_size : int   Document encoding batch size (default 32)
    score_batch_size : int   Documents per GPU block (default 128)
    query_batch_size : int   Queries sharing document transfers (default 8)
    scoring_device : str | None   Follow model.device when omitted
    verify_scores : bool   Check GPU scores and return NumPy reference scores
    """

    def __init__(
        self,
        model: MultiVectorEmbeddingModel,
        batch_size: int = 32,
        score_batch_size: int = 128,
        query_batch_size: int = 8,
        scoring_device: str | None = None,
        verify_scores: bool = False,
    ) -> None:
        for name, value in (("batch_size", batch_size), ("score_batch_size", score_batch_size),
                            ("query_batch_size", query_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._score_batch_size = score_batch_size
        self._query_batch_size = query_batch_size
        self._scoring_device = str(scoring_device or getattr(model, "device", "cpu"))
        self._verify_scores = verify_scores
        if self._scoring_device != "cpu" and not self._scoring_device.startswith("cuda"):
            raise ValueError("scoring_device must be cpu or cuda")
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

        try:
            if hasattr(self._model, "encode_documents"):
                all_vecs = self._model.encode_documents(
                    texts, batch_size=self._batch_size, show_progress=True
                )
            else:
                all_vecs = self._model.encode(
                    texts, batch_size=self._batch_size, show_progress=True
                )
        except TypeError:
            if hasattr(self._model, "encode_documents"):
                all_vecs = self._model.encode_documents(texts, batch_size=self._batch_size)
            else:
                all_vecs = self._model.encode(texts, batch_size=self._batch_size)


        if len(all_vecs) != len(documents):
            raise RuntimeError(
                f"LateInteractionRetriever.build: expected {len(documents)} "
                f"multi-vector representations, got {len(all_vecs)}."
            )

        self._doc_vecs = [np.asarray(v, dtype=np.float32) for v in all_vecs]
        self._built = True
        logger.info("[MaxSim] device=%s document_block=%d query_group=%d verify=%s",
                    self._scoring_device, self._score_batch_size,
                    self._query_batch_size, self._verify_scores)

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

        if self._scoring_device.startswith("cuda"):
            return next(self.retrieve_many([query], top_k))

        # Encode query: List[np.ndarray] of length 1.
        # Pass is_query=True so ColBERTAdapter uses query encoding (MASK augmentation).
        # Fallback to plain encode() for other MultiVectorEmbeddingModel implementations.
        if hasattr(self._model, "encode_queries"):
            q_vecs_list = self._model.encode_queries([query.text], batch_size=1)
        else:
            q_vecs_list = self._model.encode([query.text], batch_size=1)
        if len(q_vecs_list) != 1:
            raise RuntimeError(
                f"LateInteractionRetriever.retrieve: expected 1 query representation, "
                f"got {len(q_vecs_list)}."
            )
        q_vecs = np.asarray(q_vecs_list[0], dtype=np.float32)  # [T_q, dim]

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

        # Select only top-k while preserving score/doc_id ordering.
        top_scores = heapq.nsmallest(top_k, scores, key=lambda x: (-x[1], x[0]))

        return [
            Hit(doc_id=doc_id, score=score)
            for doc_id, score in top_scores
        ]

    def retrieve_many(self, queries: Sequence[Query], top_k: int):
        """Yield results in query order; CUDA reuses each document transfer.

        Exhaustive FP32 MaxSim, not approximate candidate retrieval. GPU floating
        point results can differ from NumPy near ties. Select on CPU with the
        original score/doc_id ordering, including ties across block boundaries.
        """
        if not self._built:
            raise RuntimeError("build() must be called before retrieve_many()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._scoring_device.startswith("cuda"):
            for query in queries:
                yield self.retrieve(query, top_k)
            return
        import torch
        for start in range(0, len(queries), self._query_batch_size):
            group = queries[start:start + self._query_batch_size]
            texts = [query.text for query in group]
            encoder = getattr(self._model, "encode_queries", self._model.encode)
            # Preserve the original per-query encoder numerics. Queries share
            # scoring transfers, but encoding batch shape remains unchanged.
            vectors = encoder(texts, batch_size=1)
            if len(vectors) != len(group):
                raise RuntimeError("Unexpected number of query embeddings")
            queries_np = [np.asarray(v, dtype=np.float32) for v in vectors]
            best = [[] for _ in group]
            # Restore global precision settings even if scoring fails.
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                    for offset in range(0, len(self._doc_vecs), self._score_batch_size):
                        docs = self._doc_vecs[offset:offset + self._score_batch_size]
                        block = self._score_block(queries_np, docs, self._scoring_device)
                        if self._verify_scores:
                            reference = np.asarray([
                                [self._reference_score(q, d) for d in docs]
                                for q in queries_np], dtype=np.float32)
                            np.testing.assert_allclose(block, reference, rtol=1e-4, atol=1e-5)
                            # Exact old scores and ordering in verification mode.
                            block = reference
                        ids = self._doc_ids[offset:offset + len(docs)]
                        for row, scores in enumerate(block):
                            candidates = best[row] + list(zip(ids, map(float, scores)))
                            best[row] = heapq.nsmallest(
                                top_k, candidates, key=lambda item: (-item[1], item[0]))
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            for hits in best:
                yield [Hit(doc_id=did, score=score) for did, score in hits]

    @staticmethod
    def _score_block(queries, documents, device):
        """One document transfer, then bounded [documents, Q tokens, D tokens] work."""
        import torch
        if not queries:
            return np.empty((0, len(documents)), dtype=np.float32)
        dimension = queries[0].shape[1]
        lengths = [len(d) if d.ndim == 2 else 0 for d in documents]
        width = max(max(lengths, default=0), 1)
        padded = np.zeros((len(documents), width, dimension), dtype=np.float32)
        for row, (document, length) in enumerate(zip(documents, lengths)):
            if length:
                padded[row, :length] = document
        docs = torch.as_tensor(padded, device=device)
        sizes = torch.tensor(lengths, device=device)
        mask = torch.arange(width, device=device)[None, :] < sizes[:, None]
        result = []
        for query in queries:
            if query.ndim != 2 or query.shape[1] != dimension:
                raise ValueError("Query embeddings must have shape [tokens, dimension]")
            q = torch.as_tensor(query, device=device)
            similarities = torch.matmul(q, docs.transpose(1, 2))
            similarities.masked_fill_(~mask[:, None, :], -torch.inf)
            scores = similarities.max(dim=-1).values.sum(dim=-1)
            scores = torch.where((sizes > 0) & torch.isfinite(scores), scores, 0.0)
            result.append(scores)
        return torch.stack(result).cpu().numpy()

    @staticmethod
    def _reference_score(query, document):
        if document.ndim != 2 or document.shape[0] == 0:
            return 0.0
        score = float((query @ document.T).max(axis=1).sum())
        return score if math.isfinite(score) else 0.0
