"""
Models: query/document prompt resolution
=========================================
Many embedding models were trained with a text prefix that differs between
queries and documents (e5: "query: " / "passage: ", bge: an instruction on the
query side, Qwen3-Embedding: an "Instruct: ... Query:" prompt on the query side).
Encoding without it silently lowers retrieval quality.

Resolution order for each side:
    1. explicit value passed by the caller (an empty string disables it)
    2. prompts published by the model itself (sentence-transformers ``prompts``)
    3. name heuristics below
    4. no prompt
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Tuple

_BGE_EN_QUERY = "Represent this sentence for searching relevant passages: "
_BGE_ZH_QUERY = "为这个句子生成表示以用于检索相关文章："

# Models whose prefix is not a plain query/passage pair; leave them to the
# model's own published prompts.
_NO_HEURISTIC = ("mistral", "bge-m3", "bge-reranker", "gemma")


def _heuristic_prompts(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    name = model_id.lower().rstrip("/").split("/")[-1]
    if any(token in name for token in _NO_HEURISTIC):
        return None, None
    if re.search(r"(^|[-_])e5($|[-_])", name):
        return "query: ", "passage: "
    if name.startswith("bge-"):
        if re.search(r"-en($|[-_])", name):
            return _BGE_EN_QUERY, None
        if re.search(r"-zh($|[-_])", name):
            return _BGE_ZH_QUERY, None
    return None, None


def resolve_prompts(
    model_id: str,
    query_prompt: Optional[str] = None,
    document_prompt: Optional[str] = None,
    model_prompts: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(query_prompt, document_prompt)``; ``None`` means no prompt."""
    model_prompts = model_prompts or {}
    heuristic_query, heuristic_doc = _heuristic_prompts(model_id)

    def pick(explicit: Optional[str], published: Optional[str],
             heuristic: Optional[str]) -> Optional[str]:
        if explicit is not None:
            return explicit or None
        return published or heuristic or None

    query = pick(
        query_prompt, model_prompts.get("query"), heuristic_query
    )
    document = pick(
        document_prompt,
        model_prompts.get("document") or model_prompts.get("passage"),
        heuristic_doc,
    )
    return query, document
