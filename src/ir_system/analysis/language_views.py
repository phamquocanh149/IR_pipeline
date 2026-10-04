"""Representation gaps and fixed-index alignment for single-vector encoders."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import numpy as np

from ir_system.domain.qrels import Qrels
from ir_system.domain.run import Run


LANGUAGES = ("vi", "en", "csw")


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite two-dimensional array.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cosine analysis is undefined for zero embeddings.")
    return vectors / norms


def summary(values) -> dict:
    values = [float(v) for v in values if v is not None]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def project_3d(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Fit ONE PCA to all views; pad missing axes for small inputs."""
    vectors = np.asarray(vectors, dtype=np.float64)
    centered = vectors - vectors.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    n_axes = min(3, len(axes))
    coordinates = np.zeros((len(vectors), 3))
    coordinates[:, :n_axes] = centered @ axes[:n_axes].T
    variance = singular ** 2
    ratios = np.zeros(3)
    if variance.sum() > 0:
        ratios[:n_axes] = variance[:n_axes] / variance.sum()
    return coordinates, ratios.tolist()


def paired_gaps(embeddings: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    result = {}
    for first, other in combinations(embeddings, 2):
        shared = sorted(set(embeddings[first]) & set(embeddings[other]))
        rows = []
        for key in shared:
            cosine = float(np.clip(np.dot(embeddings[first][key], embeddings[other][key]), -1, 1))
            rows.append({"group_id": key, "cosine": cosine, "gap": 1 - cosine})
        result[f"{first}_{other}"] = {
            "summary": summary(row["gap"] for row in rows), "pairs": rows,
            "cosine_summary": summary(row["cosine"] for row in rows),
            "missing_first": sorted(set(embeddings[other]) - set(embeddings[first])),
            "missing_other": sorted(set(embeddings[first]) - set(embeddings[other])),
        }
    return result


def fixed_index_analysis(
    run_batches: Iterable[Mapping[str, Run]],
    qrels: Qrels,
    top_k: int,
) -> dict:
    """Consume batches of existing full-corpus rankings on one fixed index.

    Batches may contain a single query to keep full rankings out of memory.

    All positive judgments in the fixed index are scored, even outside top-k.
    Hard negatives are non-positive documents in each query's own top-k.
    Missing positives/negatives produce null margins, never fabricated zeros.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    rows, alignment = [], []
    paired = set()
    query_ids = {}
    for runs in run_batches:
        languages = list(runs)
        shared = set.intersection(*(set(run.qids) for run in runs.values()))
        paired.update(shared)
        for lang in languages:
            query_ids.setdefault(lang, set()).update(runs[lang].qids)
        for qid in sorted(set.union(*(set(run.qids) for run in runs.values()))):
            judgments = qrels.judgments_for(qid)
            relevant = {did for did, rel in judgments.items() if rel > 0}
            scores, margins = {}, {}
            query_rows = []
            for lang in languages:
                if qid not in runs[lang].qids:
                    continue
                hits = runs[lang].get(qid)
                scores[lang] = {hit.doc_id: hit.score for hit in hits}
                if not relevant.issubset(scores[lang]):
                    raise ValueError("Analysis needs scores for every positive, including those outside top-k.")
                negatives = [hit.doc_id for hit in hits[:top_k] if hit.doc_id not in relevant]
                positive = next((hit.doc_id for hit in hits if hit.doc_id in relevant), None)
                negative = negatives[0] if negatives else None
                margin = (scores[lang][positive] - scores[lang][negative]
                          if positive is not None and negative is not None else None)
                margins[lang] = margin
                query_rows.append({"qid": qid, "language": lang, "margin": margin,
                             "best_positive": positive, "hardest_negative": negative,
                             "positive_score": scores[lang][positive] if positive else None,
                             "negative_score": scores[lang][negative] if negative else None,
                             "hard_negative_count": len(negatives),
                             "unjudged_negative_count": sum(did not in judgments
                                                            for did in negatives)})
            for lang in scores:
                if lang == "vi" or "vi" not in scores:
                    continue
                for did in sorted(relevant):
                    alignment.append({"qid": qid, "doc_group_id": did, "language": lang,
                                      "score_vi": scores["vi"][did], "score_variant": scores[lang][did],
                                      "delta_alignment": scores[lang][did] - scores["vi"][did]})
            for row in query_rows:
                row["delta_margin"] = (row["margin"] - margins["vi"]
                                       if row["margin"] is not None and margins.get("vi") is not None
                                       else None)
            rows.extend(query_rows)
    return {
        "paired_query_count": len(paired),
        "excluded_query_ids": {lang: sorted(query_ids[lang] - paired) for lang in query_ids},
        "alignment": alignment, "margins": rows,
        "summary": {lang: {
            "delta_alignment": summary(r["delta_alignment"] for r in alignment if r["language"] == lang),
            "margin": summary(r["margin"] for r in rows if r["language"] == lang),
            "delta_margin": summary(r["delta_margin"] for r in rows if r["language"] == lang),
            "undefined_margin_count": sum(r["margin"] is None for r in rows if r["language"] == lang),
        } for lang in query_ids},
    }
