"""
Domain: Document
================
Immutable value object representing a corpus document.

Invariants
----------
- doc_id must be non-empty.
- text must be non-empty if retriever requires text content.
- doc_id is the sole identity — never use array index as identity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Immutable value object for a corpus document."""

    doc_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.doc_id or not self.doc_id.strip():
            raise ValueError(
                f"Document.doc_id must be non-empty, got: {self.doc_id!r}"
            )
        # text validation is deferred to retriever that requires it,
        # but we still reject completely None text.
        if self.text is None:
            raise ValueError(
                f"Document.text must not be None (doc_id={self.doc_id!r})"
            )
