"""
Models: ColBERT Adapter
========================
Concrete MultiVectorEmbeddingModel for ColBERT-style late-interaction.

Uses sentence-transformers with per-token output.
Each text produces a variable-length array of token vectors.

NOTE: This adapter provides a practical implementation using a standard
SentenceTransformer model in token-output mode. For true ColBERT checkpoints,
replace this with pylate or colbert-ai integration.
"""
from __future__ import annotations

import logging
from typing import Any, List, Sequence

import numpy as np

from ir_system.models.multi_vector import MultiVectorEmbeddingModel

logger = logging.getLogger(__name__)


class ColBERTAdapter(MultiVectorEmbeddingModel):
    """
    ColBERT-style multi-vector adapter using HuggingFace transformers.

    Produces one vector per token (last-hidden-state), excluding padding.
    Vectors are L2-normalized per token.

    Invariants
    ----------
    - encode(N texts) → List[np.ndarray] of length N
    - Element i has shape [T_i, dim]
    - Token boundaries are preserved (no flattening).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        default_batch_size: int = 32,
        max_length: int = 128,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for late-interaction (ColBERT) retrieval. "
                "Install it with: pip install transformers torch"
            ) from exc

        import torch

        self._model_id = model_id
        self._device = torch.device(device)
        self._default_batch_size = default_batch_size
        self._max_length = max_length

        logger.info("[MODEL] Loading ColBERT tokenizer/model: %s on %s", model_id, device)
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(self._device)
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_id

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """
        Encode texts into lists of token vectors.

        Returns
        -------
        List[np.ndarray]
            Length N. Element i: shape [T_i, dim], dtype float32.
            Token count T_i excludes padding tokens.
        """
        import torch

        texts = list(texts)
        if not texts:
            return []

        bs = batch_size if batch_size is not None else self._default_batch_size
        all_vectors: List[np.ndarray] = []

        batch_starts = range(0, len(texts), bs)
        if show_progress:
            from tqdm import tqdm
            batch_starts = tqdm(batch_starts, desc="[ColBERT] Encoding batches", unit="batch")

        with torch.inference_mode():
            for start in batch_starts:

                batch = texts[start: start + bs]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state  # [B, T, dim]
                attention_mask = encoded["attention_mask"]  # [B, T]

                for i in range(hidden.shape[0]):
                    mask_i = attention_mask[i].bool()
                    vecs = hidden[i][mask_i]  # [T_i, dim]
                    # L2-normalize each token vector
                    norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    vecs = (vecs / norms).cpu().numpy().astype(np.float32)
                    all_vectors.append(vecs)

        if len(all_vectors) != len(texts):
            raise RuntimeError(
                f"ColBERTAdapter.encode: expected {len(texts)} representations, "
                f"got {len(all_vectors)}."
            )
        return all_vectors
