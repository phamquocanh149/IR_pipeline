import math
from collections import Counter

import numpy as np
import pytest

from ir_system.domain.document import Document
from ir_system.domain.query import Query
from ir_system.retrievers.bm25 import BM25Retriever
from ir_system.retrievers.cross_encoder import CrossEncoderRetriever


def reference_ranking(documents, query, k1=1.5, b=0.75):
    """Independent Lucene BM25 calculation, including repeated query terms."""
    corpus = [document.text.lower().split() for document in documents]
    average_length = sum(map(len, corpus)) / len(corpus)
    frequencies = Counter(token for tokens in corpus for token in set(tokens))
    ranking = []
    for document, tokens in zip(documents, corpus):
        counts = Counter(tokens)
        score = 0.0
        for token in query.lower().split():
            if token not in counts:
                continue
            frequency = counts[token]
            idf = math.log(1 + (len(corpus) - frequencies[token] + 0.5)
                           / (frequencies[token] + 0.5))
            score += idf * frequency / (
                frequency + k1 * (1 - b + b * len(tokens) / average_length)
            )
        ranking.append((document.doc_id, score))
    return sorted(ranking, key=lambda hit: (-hit[1], hit[0]))


@pytest.mark.parametrize("top_k", [1, 2, 4, 20])
@pytest.mark.parametrize("text", ["cat", "cat cat dog", "unknown", "CAT dog"])
@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (0.9, 0.4)])
def test_scores_and_top_k_match_reference(top_k, text, k1, b):
    documents = [Document("z", "cat cat dog"), Document("b", "dog"),
                 Document("a", "cat cat dog"), Document("empty", "")]
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.build(documents)
    actual = retriever.retrieve(Query("q", text), top_k)
    expected = reference_ranking(documents, text, k1, b)[:top_k]
    assert [hit.doc_id for hit in actual] == [doc_id for doc_id, _ in expected]
    assert [hit.score for hit in actual] == pytest.approx(
        [score for _, score in expected], rel=1e-5, abs=1e-7
    )


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_cutoff_ties_use_doc_id_without_repeating_scoring(monkeypatch, score):
    documents = [Document(doc_id, "same") for doc_id in ["z", "m", "b", "a"]]
    retriever = BM25Retriever()
    retriever.build(documents)
    calls = []

    def scores(tokens):
        calls.append(tokens)
        return np.full(len(documents), score)

    monkeypatch.setattr(retriever._bm25, "get_scores", scores)
    assert [hit.doc_id for hit in retriever.retrieve(Query("q", "same"), 2)] == ["a", "b"]
    assert len(calls) == 1


def test_custom_tokenizer_is_preserved():
    retriever = BM25Retriever(tokenizer=lambda text: text.split("|"))
    retriever.build([Document("wrong", "A"), Document("right", "a|b")])
    assert retriever.retrieve(Query("q", "a"), 1)[0].doc_id == "right"


def test_empty_tokenized_query_returns_no_hits():
    retriever = BM25Retriever(tokenizer=lambda text: [] if text == "skip" else text.split())
    retriever.build([Document("d", "document")])
    assert retriever.retrieve(Query("q", "skip"), 1) == []


def test_failed_rebuild_cannot_search_with_mismatched_state():
    retriever = BM25Retriever()
    retriever.build([Document("old", "document")])
    with pytest.raises(ValueError, match="no tokens"):
        retriever.build([Document("new", " ")])
    with pytest.raises(RuntimeError, match="build"):
        retriever.retrieve(Query("q", "document"), 1)


def test_retrieve_requires_build_and_positive_top_k():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.retrieve(Query("q", "query"), 1)
    with pytest.raises(ValueError):
        retriever.build([])
    retriever.build([Document("d", "document")])
    with pytest.raises(ValueError):
        retriever.retrieve(Query("q", "document"), 0)


def test_bm25_candidates_feed_cross_encoder():
    class Scorer:
        name = "fixture-reranker"

        def score(self, query_texts, document_texts, **kwargs):
            assert query_texts == ["cat", "cat"]
            assert set(document_texts) == {"cat", "cat dog"}
            return [2.0 if text == "cat dog" else 1.0 for text in document_texts]

    retriever = CrossEncoderRetriever(Scorer(), BM25Retriever(), candidate_top_k=2)
    retriever.build([Document("a", "cat"), Document("b", "cat dog"),
                     Document("c", "bird")])
    hits = retriever.retrieve(Query("q", "cat"), 1)
    assert [(hit.doc_id, hit.score) for hit in hits] == [("b", 2.0)]
