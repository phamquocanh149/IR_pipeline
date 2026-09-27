"""
Models: CrossEncoder Adapter
=============================
Concrete CrossEncoderModel backed by sentence-transformers CrossEncoder.

This adapter is the ONLY file that knows about sentence_transformers.CrossEncoder.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Sequence

from ir_system.models.cross_encoder import CrossEncoderModel

logger = logging.getLogger(__name__)


class CrossEncoderAdapter(CrossEncoderModel):
    """
    Wraps sentence_transformers.CrossEncoder as CrossEncoderModel.

    Invariants
    ----------
    - score(N queries, N docs) → N float scores
    - len(query_texts) must equal len(document_texts)
    - All returned scores are finite floats
    - Input order preserved
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        default_batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_id = model_id
        self._device = device
        self._default_batch_size = default_batch_size

        logger.info("[MODEL] Loading CrossEncoder: %s on %s", model_id, device)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_id, device=device)

    @property
    def name(self) -> str:
        return self._model_id

    def score(
        self,
        query_texts: Sequence[str],
        document_texts: Sequence[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Score query-document pairs.

        Parameters
        ----------
        query_texts    : Sequence[str]  length N
        document_texts : Sequence[str]  length N

        Returns
        -------
        List[float]  length N, all finite.

        Raises
        ------
        ValueError if lengths mismatch.
        """
        if len(query_texts) != len(document_texts):
            raise ValueError(
                f"CrossEncoderAdapter.score: len(query_texts)={len(query_texts)} "
                f"!= len(document_texts)={len(document_texts)}"
            )

        if not query_texts:
            return []

        bs = batch_size if batch_size is not None else self._default_batch_size
        pairs = list(zip(query_texts, document_texts))

        show_progress = kwargs.get("show_progress", False)
        raw_scores = self._model.predict(pairs, batch_size=bs, show_progress_bar=show_progress)


        result: List[float] = []
        for i, s in enumerate(raw_scores):
            s_float = float(s)
            if not math.isfinite(s_float):
                raise RuntimeError(
                    f"CrossEncoderAdapter.score: non-finite score {s_float!r} "
                    f"at pair index {i}"
                )
            result.append(s_float)

        if len(result) != len(query_texts):
            raise RuntimeError(
                f"CrossEncoderAdapter.score: expected {len(query_texts)} scores, "
                f"got {len(result)}."
            )

        return result
