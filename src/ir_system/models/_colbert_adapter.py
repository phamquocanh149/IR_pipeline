"""ColBERTv2 multi-vector adapter backed by the official colbert-ai Checkpoint.

Dependencies: torch, colbert-ai==0.2.22, transformers<5, numpy, tqdm,
and the platform-appropriate FAISS package. CPU ColBERT initialization may
compile its C++ scoring extension (a C++ compiler and ninja are required).

Integration:
    model = ColBERTAdapter("colbert-ir/colbertv2.0")
    documents = model.encode_documents(document_texts)
    queries = model.encode_queries(query_texts)
    # Existing document callers can retain model.encode(document_texts).
    # Query callers MUST use encode_queries or encode(..., is_query=True).

Each result is a list of independent float32 [tokens, 128] arrays.
Queries retain augmentation tokens; documents omit padding and punctuation
according to ColBERT's own mask. Special tokens are handled by ColBERT.
Rebuild previously stored embeddings/indexes after replacing the old adapter.
"""
from __future__ import annotations

import logging
from typing import Any, List, Sequence

import numpy as np

from ir_system.models.multi_vector import MultiVectorEmbeddingModel

logger = logging.getLogger(__name__)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


class ColBERTAdapter(MultiVectorEmbeddingModel):
    """Preserve the old public interface, with explicit query/document routing.

    max_length is the DOCUMENT limit (default changed from 128 to 180).
    query_max_length is independent and defaults to 32.
    Plain encode() defaults to documents, with a one-time warning. There is
    no reliable way to infer a text's retrieval role from its content/length.
    Supported devices: CPU and CUDA. No pooling or vector flattening is used.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        default_batch_size: int = 32,
        max_length: int = 180,
        *,
        query_max_length: int = 32,
    ) -> None:
        _positive_int(default_batch_size, "default_batch_size")
        for key, value in (("max_length", max_length),
                           ("query_max_length", query_max_length)):
            _positive_int(value, key)
            if not 3 <= value <= 512:
                raise ValueError(f"{key} must be between 3 and 512")

        try:
            import torch
            from colbert.infra import ColBERTConfig
            from colbert.modeling.checkpoint import Checkpoint
            from colbert.utils.amp import MixedPrecisionManager
        except ImportError as exc:
            raise ImportError(
                "ColBERTAdapter requires torch and colbert-ai (including its "
                "dependencies). See the module docstring for installation. "
                f"Original error: {exc}"
            ) from exc

        self._device = torch.device(device)
        if self._device.type not in {"cpu", "cuda"}:
            raise ValueError("This adapter supports only CPU and CUDA")
        if self._device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            index = self._device.index
            if index is None:
                index = torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError(f"Unavailable CUDA device: {device}")
            self._device = torch.device("cuda", index)

        self._model_id = model_id
        self._default_batch_size = default_batch_size
        self._max_length = max_length
        self._warned_default_role = False

        config = ColBERTConfig(
            checkpoint=model_id,
            dim=128,
            query_maxlen=query_max_length,
            doc_maxlen=max_length,
            mask_punctuation=True,
            attend_to_mask_tokens=False,
            interaction="colbert",
            similarity="cosine",
        )
        logger.info("[MODEL] Loading ColBERT checkpoint %s on %s", model_id, device)
        # Checkpoint loads the trained projection as well as the BERT encoder.
        self._model = Checkpoint(model_id, colbert_config=config, verbose=0)
        self._model.to(self._device)
        self._model.eval()
        # Upstream initially derives use_gpu from visible hardware, not the
        # requested device. Keep its document dtype/AMP behavior consistent.
        self._model.use_gpu = self._device.type == "cuda"
        self._model.amp_manager = MixedPrecisionManager(self._model.use_gpu)
        if self._model.linear.weight.shape[0] != 128:
            raise RuntimeError("Expected the ColBERTv2 128-dimensional projection")

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
        """Encode in input order; use is_query=True or input_type='query'.

        input_type accepts 'query' or 'document'. Unknown options and
        conflicting role flags raise errors instead of being ignored.
        """
        import torch

        role = kwargs.pop("input_type", None)
        is_query = kwargs.pop("is_query", None)
        if kwargs:
            raise TypeError(f"Unsupported encode options: {', '.join(sorted(kwargs))}")
        if role is not None and role not in ("query", "document"):
            raise ValueError("input_type must be 'query' or 'document'")
        if is_query is not None:
            if not isinstance(is_query, bool):
                raise TypeError("is_query must be a bool")
            flag_role = "query" if is_query else "document"
            if role is not None and role != flag_role:
                raise ValueError("input_type and is_query disagree")
            role = flag_role

        bs = _positive_int(
            self._default_batch_size if batch_size is None else batch_size,
            "batch_size",
        )
        if isinstance(texts, (str, bytes)):
            raise TypeError("Pass a sequence of strings, e.g. [query], not a string")
        texts = list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every input text must be a string")
        if not texts:
            return []
        if role is None:
            role = "document"
            if not self._warned_default_role:
                logger.warning(
                    "encode() defaults to DOCUMENT encoding. For queries use "
                    "encode_queries() or encode(..., is_query=True)."
                )
                self._warned_default_role = True

        starts = range(0, len(texts), bs)
        if show_progress:
            from tqdm.auto import tqdm
            starts = tqdm(starts, desc=f"[ColBERT] {role}", unit="batch")

        result: List[np.ndarray] = []
        with torch.inference_mode():
            for start in starts:
                batch = texts[start:start + bs]
                if role == "query":
                    vectors = self._model.queryFromText(batch)
                    # Do NOT filter with the tokenizer attention mask: MASK
                    # augmentation vectors are part of ColBERT query scoring.
                else:
                    # Batch here, not inside docFromText: bsize=None preserves
                    # order and returns a plain list, not a wrapper tuple.
                    # to_cpu=False avoids calling .cpu() on a Python list.
                    vectors = self._model.docFromText(
                        batch, keep_dims=False, to_cpu=False
                    )
                if len(vectors) != len(batch):
                    raise RuntimeError("ColBERT returned an unexpected batch size")
                for vector in vectors:
                    array = vector.detach().float().cpu().numpy().copy()
                    if array.ndim != 2 or array.shape[1] != 128:
                        raise RuntimeError(f"Unexpected embedding shape: {array.shape}")
                    result.append(array)
        return result

    def encode_queries(
        self, texts: Sequence[str], batch_size: int | None = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        return self.encode(texts, batch_size, show_progress, input_type="query")

    def encode_documents(
        self, texts: Sequence[str], batch_size: int | None = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        return self.encode(texts, batch_size, show_progress, input_type="document")