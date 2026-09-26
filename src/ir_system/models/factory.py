"""
Models: Factory
===============
Maps a model identifier string (as supplied by CLI) to a concrete model instance.

Rules
-----
- Validates the model identifier.
- Determines the model type (single-vector, multi-vector, cross-encoder).
- Instantiates the correct concrete class.
- Raises an explicit error if type cannot be determined.
- Retrievers must NOT know Hugging Face identifiers.
- SearchPipeline must NOT hard-code any model.

Model type detection heuristics (can be overridden by --model-type):
  - Known cross-encoder prefixes → CrossEncoderModel
  - "bm25" → no neural model (return None)
  - Otherwise → SingleVectorEmbeddingModel (dense embedding default)

The factory also provides a special sentinel "bm25" that returns None,
signalling that no neural model is needed.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Union

from ir_system.models.cross_encoder import CrossEncoderModel
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.models.multi_vector import MultiVectorEmbeddingModel

logger = logging.getLogger(__name__)

ModelType = Literal["single-vector", "multi-vector", "cross-encoder"]

# Prefix heuristics — can be extended without changing retrievers or pipeline.
_CROSS_ENCODER_PREFIXES = (
    "cross-encoder/",
    "cross_encoder/",
    "reranker/",
)

_MULTI_VECTOR_PREFIXES = (
    "colbert",
    "col-bert",
)

AnyModel = Union[
    SingleVectorEmbeddingModel,
    MultiVectorEmbeddingModel,
    CrossEncoderModel,
    None,
]


def create_model(
    model_id: str,
    model_type: Optional[ModelType] = None,
    device: str = "auto",
    batch_size: int = 32,
) -> AnyModel:
    """
    Instantiate a model from its identifier.

    Parameters
    ----------
    model_id   : str
        Hugging Face model name, path, or "bm25".
    model_type : Optional[ModelType]
        Explicit override. If None, type is inferred from model_id.
    device     : str
        "cpu", "cuda", or "auto".
    batch_size : int
        Default batch size for encoding/scoring.

    Returns
    -------
    AnyModel
        Concrete model instance or None for "bm25".

    Raises
    ------
    ValueError
        If model_id is empty, or model type is unsupported.
    RuntimeError
        If requested device is unavailable.
    """
    if not model_id or not model_id.strip():
        raise ValueError("ModelFactory: model_id must be non-empty.")

    normalized = model_id.strip()

    # BM25 needs no neural model.
    if normalized.lower() == "bm25":
        logger.info("[MODEL] No neural model required for BM25.")
        return None

    # Resolve device.
    resolved_device = _resolve_device(device)

    # Determine model type.
    resolved_type: ModelType = model_type or _infer_model_type(normalized)

    logger.info("[MODEL] Loading %s", normalized)
    logger.info("[MODEL] Type: %s", resolved_type)
    logger.info("[MODEL] Device: %s", resolved_device)

    if resolved_type == "single-vector":
        return _build_single_vector(normalized, resolved_device, batch_size)
    elif resolved_type == "multi-vector":
        return _build_multi_vector(normalized, resolved_device, batch_size)
    elif resolved_type == "cross-encoder":
        return _build_cross_encoder(normalized, resolved_device, batch_size)
    else:
        raise ValueError(
            f"ModelFactory: unsupported model type {resolved_type!r}. "
            "Expected one of: single-vector, multi-vector, cross-encoder."
        )


def _resolve_device(device: str) -> str:
    """Resolve "auto" to "cuda" or "cpu"; validate explicit choices."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ModelFactory: device='cuda' requested but CUDA is not available. "
                "Use device='cpu' or device='auto'."
            )
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(
            f"ModelFactory: unknown device {device!r}. Use 'cpu', 'cuda', or 'auto'."
        )


def _infer_model_type(model_id: str) -> ModelType:
    """Heuristically infer model type from model identifier."""
    lower = model_id.lower()

    for prefix in _CROSS_ENCODER_PREFIXES:
        if lower.startswith(prefix) or prefix.rstrip("/") in lower:
            return "cross-encoder"

    for prefix in _MULTI_VECTOR_PREFIXES:
        if lower.startswith(prefix) or prefix in lower:
            return "multi-vector"

    # Default: single-vector dense embedding.
    return "single-vector"


def _build_single_vector(
    model_id: str, device: str, batch_size: int
) -> SingleVectorEmbeddingModel:
    """Build a concrete single-vector embedding model adapter."""
    # Import deferred to avoid top-level dependency on torch/transformers.
    from ir_system.models._sentence_transformer_adapter import (
        SentenceTransformerAdapter,
    )

    return SentenceTransformerAdapter(
        model_id=model_id, device=device, default_batch_size=batch_size
    )


def _build_multi_vector(
    model_id: str, device: str, batch_size: int
) -> MultiVectorEmbeddingModel:
    """Build a concrete multi-vector embedding model adapter."""
    from ir_system.models._colbert_adapter import ColBERTAdapter

    return ColBERTAdapter(
        model_id=model_id, device=device, default_batch_size=batch_size
    )


def _build_cross_encoder(
    model_id: str, device: str, batch_size: int
) -> CrossEncoderModel:
    """Build a concrete cross-encoder model adapter."""
    from ir_system.models._cross_encoder_adapter import CrossEncoderAdapter

    return CrossEncoderAdapter(
        model_id=model_id, device=device, default_batch_size=batch_size
    )
