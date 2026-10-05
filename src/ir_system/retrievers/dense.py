"""Dense retrieval with mandatory FAISS persistence and cosine search."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.retrievers.base import Retriever
from ir_system.indexing.faiss_store import load_faiss_index, save_faiss_index

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Build and persist normalized IndexFlatIP vectors; search directly in FAISS.

    Default storage is indexes/<SHA256 of model and ordered corpus>.
    index_dir can override this location for Python callers.
    """

    def __init__(self, model: SingleVectorEmbeddingModel, batch_size: int = 32,
                 index_dir: str | Path | None = None) -> None:
        self._model = model
        self._batch_size = batch_size
        self._index_dir = Path(index_dir) if index_dir is not None else None
        self._index = None
        self._doc_ids: list[str] = []
        self._built = False
        self._loaded = False
        self._corpus_hash = None

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("DenseRetriever.build: corpus must not be empty.")
        doc_ids = [doc.doc_id for doc in documents]
        digest = hashlib.sha256()
        digest.update(json.dumps(self._model.name).encode("utf-8"))
        document_prompt = self._model.document_prompt
        if document_prompt:
            # Changes the document vectors, so indexes built without it are stale.
            digest.update(json.dumps(document_prompt, ensure_ascii=False).encode("utf-8"))
        for doc in documents:
            digest.update(json.dumps([doc.doc_id, doc.text], ensure_ascii=False).encode("utf-8"))
        corpus_hash = digest.hexdigest()
        if self._loaded:
            if doc_ids != self._doc_ids or (
                self._corpus_hash is not None and self._corpus_hash != corpus_hash
            ):
                raise ValueError("Loaded FAISS index does not match the current corpus.")
            self._loaded = False
            return

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS is required for dense retrieval. Install faiss-cpu.") from exc
        self._built = False
        texts = [doc.text for doc in documents]
        try:
            embeddings = self._model.encode_documents(
                texts, batch_size=self._batch_size, show_progress=True
            )
        except TypeError:
            embeddings = self._model.encode_documents(texts, batch_size=self._batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents) or embeddings.shape[1] == 0:
            raise RuntimeError("DenseRetriever.build: invalid embedding shape or document count.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Dense embeddings must contain only finite values.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.ascontiguousarray(embeddings / np.where(norms < 1e-10, 1.0, norms))
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        index_dir = self._index_dir or Path("indexes") / corpus_hash
        save_faiss_index(index_dir, index, doc_ids, self._model.name,
                         extra_config={"corpus_hash": corpus_hash})
        self._index, self._doc_ids = index, doc_ids
        self._corpus_hash = corpus_hash
        self._built = True
        logger.info("[RETRIEVER] FAISS index ready and saved to %s", index_dir)

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        if not self._built:
            raise RuntimeError("DenseRetriever.retrieve: build() or load_index() must be called first.")
        if top_k <= 0:
            raise ValueError("DenseRetriever.retrieve: top_k must be > 0.")
        vector = np.asarray(self._model.encode_queries([query.text], batch_size=1), dtype=np.float32)
        if vector.shape != (1, self._index.d):
            raise RuntimeError(f"Unexpected query embedding shape {vector.shape}.")
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding must contain only finite values.")
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        vector = np.ascontiguousarray(vector)
        k = min(top_k, self._index.ntotal)
        # Fetch one extra result to detect ties at the cutoff. Expand only when
        # needed so doc_id tie breaking is deterministic across the full corpus.
        limit = min(k + 1, self._index.ntotal)
        while True:
            scores, rows = self._index.search(vector, limit)
            if limit == self._index.ntotal or scores[0, k - 1] > scores[0, -1]:
                break
            limit = min(limit * 2, self._index.ntotal)
        pairs = sorted(((self._doc_ids[int(row)], float(score))
                        for row, score in zip(rows[0], scores[0])
                        if row >= 0 and np.isfinite(score)), key=lambda pair: (-pair[1], pair[0]))
        return [Hit(doc_id=doc_id, score=score) for doc_id, score in pairs[:k]]

    def document_embeddings(self) -> dict[str, np.ndarray]:
        """Return normalized vectors on demand for representation diagnostics."""
        if not self._built:
            raise RuntimeError("DenseRetriever.document_embeddings: build or load the index first.")
        return {doc_id: self._index.reconstruct(row).copy()
                for row, doc_id in enumerate(self._doc_ids)}

    def save_index(self, index_dir: str | Path) -> Path:
        """Save an additional copy of the current index."""
        if not self._built:
            raise RuntimeError("Build or load an index before saving.")
        return save_faiss_index(index_dir, self._index, self._doc_ids, self._model.name,
                                extra_config={"corpus_hash": self._corpus_hash})

    def load_index(self, index_dir: str | Path) -> None:
        """Load FAISS directly without reconstructing a NumPy embedding matrix."""
        index, doc_ids, config = load_faiss_index(index_dir, expected_model=self._model.name)
        self._index, self._doc_ids = index, doc_ids
        self._corpus_hash = config.get("corpus_hash")
        self._built = True
        self._loaded = True
