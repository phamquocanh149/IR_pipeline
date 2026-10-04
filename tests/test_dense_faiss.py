import logging
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ir_system.cli.main import _build_index, _make_parser
from ir_system.domain.document import Document
from ir_system.domain.query import Query
from ir_system.fusion.rrf import RRFusion
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.retrievers.dense import DenseRetriever
from ir_system.retrievers.hybrid import HybridRetriever
from ir_system.retrievers.bm25 import BM25Retriever


class Model(SingleVectorEmbeddingModel):
    name = 'test-model'

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([[float(x) for x in text.split()] for text in texts], dtype=np.float32)


class DenseFaissTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'index'
        self.model = Model()
        self.docs = [Document('z', '2 0'), Document('b', '1 0'),
                     Document('a', '1 0'), Document('c', '0 1'), Document('d', '0 0')]
        self.query = Query('q', '1 0')
        self.ret = DenseRetriever(self.model, index_dir=self.path)
        self.ret.build(self.docs)

    def test_persistence_search_and_cutoff_ties(self):
        self.assertTrue((self.path / 'index.faiss').is_file())
        self.assertEqual([h.doc_id for h in self.ret.retrieve(self.query, 2)], ['a', 'b'])
        hits = self.ret.retrieve(self.query, 99)
        self.assertEqual([h.doc_id for h in hits], ['a', 'b', 'z', 'c', 'd'])
        np.testing.assert_allclose([h.score for h in hits], [1, 1, 1, 0, 0])
        loaded = DenseRetriever(self.model)
        loaded.load_index(self.path)
        self.assertEqual(loaded.retrieve(self.query, 3), self.ret.retrieve(self.query, 3))

    def test_search_matches_numpy(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(200, 16)).astype(np.float32)
        query = rng.normal(size=16).astype(np.float32)
        docs = [Document(str(i), ' '.join(map(str, v))) for i, v in enumerate(vectors)]
        self.ret.build(docs)
        hits = self.ret.retrieve(Query('q', ' '.join(map(str, query))), 10)
        scores = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
        expected = np.argsort(-scores)[:10]
        self.assertEqual([h.doc_id for h in hits], list(map(str, expected)))
        np.testing.assert_allclose([h.score for h in hits], scores[expected], atol=1e-6)

    def test_hybrid_load_skips_corpus_encoding(self):
        dense = DenseRetriever(self.model)
        hybrid = HybridRetriever([BM25Retriever(), dense], RRFusion())
        self.model.calls.clear()
        _build_index(hybrid, self.docs, Namespace(load_index=str(self.path)), logging.getLogger())
        self.assertEqual(self.model.calls, [])
        self.assertTrue(hybrid.retrieve(self.query, 2))

    def test_loaded_corpus_mismatch(self):
        self.ret.load_index(self.path)
        with self.assertRaises(ValueError):
            self.ret.build([Document(d.doc_id, '0 1') for d in self.docs])

    def test_invalid_query_and_top_k(self):
        with self.assertRaises(ValueError):
            self.ret.retrieve(self.query, 0)
        with self.assertRaises(RuntimeError):
            self.ret.retrieve(Query('q', '1 0 0'), 1)

    def test_write_failure_is_not_searchable(self):
        with patch('ir_system.retrievers.dense.save_faiss_index', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.ret.build(self.docs)
        with self.assertRaises(RuntimeError):
            self.ret.retrieve(self.query, 1)

    def test_cli_has_no_save_switch(self):
        self.assertNotIn('--save-index', _make_parser().format_help())
        self.assertIn('--load-index', _make_parser().format_help())


if __name__ == '__main__':
    unittest.main()
