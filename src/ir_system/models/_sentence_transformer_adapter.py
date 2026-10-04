"""
Models: SentenceTransformer Adapter
=====================================
Concrete SingleVectorEmbeddingModel backed by sentence-transformers.

This adapter is the ONLY place that knows about sentence_transformers.
Retrievers only depend on SingleVectorEmbeddingModel (the abstraction).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ir_system.models.single_vector import SingleVectorEmbeddingModel

logger = logging.getLogger(__name__)


class SentenceTransformerAdapter(SingleVectorEmbeddingModel):
    """
    Wraps sentence_transformers.SentenceTransformer as SingleVectorEmbeddingModel.

    Invariants
    ----------
    - encode(N texts) → float32 array of shape [N, dim]
    - Input order preserved.
    - Runs in torch.no_grad() / inference_mode during encode.
    - Normalizes vectors if normalize_embeddings=True (default).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        default_batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_id = model_id
        self._device = device
        self._default_batch_size = default_batch_size
        self._normalize = normalize_embeddings
        logger.info("[MODEL] Loading SentenceTransformer: %s on %s", model_id, device)
        self._model = SentenceTransformer(model_id, device=device)
        self._model.max_seq_length = 512

    @property
    def name(self) -> str:
        return self._model_id

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode texts into dense vectors.

        Parameters
        ----------
        texts        : Sequence[str]
        batch_size   : int, optional   (uses default_batch_size if None)
        show_progress: bool

        Returns
        -------
        np.ndarray  shape [N, dim], dtype float32
        """
        texts = list(texts)
        if len(texts) == 0:
            return np.empty((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)

        bs = batch_size if batch_size is not None else self._default_batch_size

        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)

        if embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"SentenceTransformerAdapter.encode: expected {len(texts)} vectors, "
                f"got {embeddings.shape[0]}. Input order may be corrupted."
            )

        return embeddings
