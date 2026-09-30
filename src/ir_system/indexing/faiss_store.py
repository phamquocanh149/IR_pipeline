"""
Indexing: FaissIndexStore
==========================
Persistence layer for FAISS indexes used by dense retrieval.

Responsibilities
----------------
- Save a FAISS index + doc_id mapping + config metadata to disk.
- Load them back and validate compatibility (model, dimension, etc.).

Directory layout per saved index:

    indexes/<index_name>/
    ├── index.faiss      # FAISS binary index
    ├── doc_ids.json     # Ordered list of doc_ids (position ↔ FAISS row)
    └── config.json      # Model name, embedding dim, similarity, create time

This module knows nothing about Retrievers, Models or Pipeline —
it is a standalone I/O utility that DenseRetriever calls when needed.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Filenames (constants — not configurable, keeps layout deterministic)
_INDEX_FILE = "index.faiss"
_DOC_IDS_FILE = "doc_ids.json"
_CONFIG_FILE = "config.json"


def save_faiss_index(
    index_dir: str | Path,
    embeddings: np.ndarray,
    doc_ids: List[str],
    model_name: str,
    *,
    similarity: str = "cosine",
    extra_config: Optional[Dict] = None,
) -> Path:
    """
    Save a FAISS index, doc_id mapping, and config to disk.

    Parameters
    ----------
    index_dir   : Path to output directory (created if needed).
    embeddings  : np.ndarray shape [N, dim], L2-normalized float32 vectors.
    doc_ids     : List[str] length N — position i maps to embeddings row i.
    model_name  : Canonical model identifier for compatibility checking.
    similarity  : Similarity metric used (default "cosine").
    extra_config: Optional extra metadata to store in config.json.

    Returns
    -------
    Path  The index directory.

    Raises
    ------
    ImportError   If faiss is not installed.
    ValueError    If embeddings/doc_ids length mismatch.
    OSError       If writing to disk fails.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss is required for index persistence. "
            "Install with: pip install faiss-cpu  (or faiss-gpu)"
        )

    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            f"save_faiss_index: embeddings rows ({embeddings.shape[0]}) "
            f"!= doc_ids length ({len(doc_ids)})"
        )

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    n_vectors, dim = embeddings.shape

    # --- Build FAISS index (FlatIP for cosine with pre-normalized vectors) ---
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # --- Write files ---
    faiss.write_index(index, str(index_dir / _INDEX_FILE))

    with open(index_dir / _DOC_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)

    config = {
        "model": model_name,
        "embedding_dim": dim,
        "n_vectors": n_vectors,
        "similarity": similarity,
        "normalized": True,
        "faiss_index_type": "IndexFlatIP",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_config:
        config.update(extra_config)

    with open(index_dir / _CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(
        "[INDEX] Saved FAISS index to %s  (%d vectors, dim=%d, model=%s)",
        index_dir,
        n_vectors,
        dim,
        model_name,
    )
    return index_dir


def load_faiss_index(
    index_dir: str | Path,
    expected_model: Optional[str] = None,
) -> tuple[np.ndarray, List[str], Dict]:
    """
    Load a previously saved FAISS index from disk.

    Parameters
    ----------
    index_dir      : Path to the index directory.
    expected_model : If provided, validate that the saved index was built with
                     this model. Raises ValueError on mismatch.

    Returns
    -------
    (embeddings, doc_ids, config)
        embeddings : np.ndarray [N, dim] float32, reconstructed from FAISS.
        doc_ids    : List[str] length N.
        config     : dict from config.json.

    Raises
    ------
    ImportError    If faiss is not installed.
    FileNotFoundError  If index_dir or required files are missing.
    ValueError    If model mismatch or data integrity check fails.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss is required for index persistence. "
            "Install with: pip install faiss-cpu  (or faiss-gpu)"
        )

    index_dir = Path(index_dir)

    if not index_dir.is_dir():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    index_path = index_dir / _INDEX_FILE
    doc_ids_path = index_dir / _DOC_IDS_FILE
    config_path = index_dir / _CONFIG_FILE

    for path in (index_path, doc_ids_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required index file missing: {path}")

    # --- Load config ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # --- Model compatibility check ---
    if expected_model and config.get("model") != expected_model:
        raise ValueError(
            f"Index model mismatch: index was built with "
            f"'{config.get('model')}', but current model is '{expected_model}'. "
            f"Rebuild the index with --save-index or use the matching model."
        )

    # --- Load FAISS index ---
    index = faiss.read_index(str(index_path))
    n_vectors = index.ntotal
    dim = index.d

    # Reconstruct embeddings from FAISS (FlatIP stores raw vectors)
    embeddings = np.zeros((n_vectors, dim), dtype=np.float32)
    for i in range(n_vectors):
        embeddings[i] = index.reconstruct(i)

    # --- Load doc_ids ---
    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    # --- Integrity checks ---
    if len(doc_ids) != n_vectors:
        raise ValueError(
            f"Data integrity error: doc_ids has {len(doc_ids)} entries "
            f"but FAISS index has {n_vectors} vectors."
        )

    expected_dim = config.get("embedding_dim")
    if expected_dim and dim != expected_dim:
        raise ValueError(
            f"Dimension mismatch: config says {expected_dim}, "
            f"FAISS index has {dim}."
        )

    logger.info(
        "[INDEX] Loaded FAISS index from %s  (%d vectors, dim=%d, model=%s)",
        index_dir,
        n_vectors,
        dim,
        config.get("model", "unknown"),
    )

    return embeddings, doc_ids, config
