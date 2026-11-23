"""
Retrievers: SparseRetriever (abstract)
=======================================
Intermediate abstraction for lexical/sparse retrieval methods.
"""
from __future__ import annotations

from ir_system.retrievers.base import Retriever


class SparseRetriever(Retriever):
    """Abstract base for sparse (lexical) retrieval strategies."""
    pass
