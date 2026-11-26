"""Offline retrieval regressions using real FAISS and deterministic model outputs."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ir_system.cli.main import main
from ir_system.domain.document import Document
from ir_system.domain.query import Query
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.models.cross_encoder import CrossEncoderModel
from ir_system.retrievers.dense import DenseRetriever


class Embeddings(SingleVectorEmbeddingModel):
    name = "test-embeddings"

    def encode(self, texts, **kwargs):
        return np.array([[1, 0] if text == "apple" else [0, 1]
                         for text in texts], dtype=np.float32)


class Reranker(CrossEncoderModel):
    name = "test-reranker"

    def score(self, query_texts, document_texts, **kwargs):
        return [float(q == d) for q, d in zip(query_texts, document_texts)]


class RetrievalTests(unittest.TestCase):
    def test_dense_ties_cache_and_explicit_reload(self):
        documents = [Document("b", "apple"), Document("a", "apple"),
                     Document("c", "orange")]
        with tempfile.TemporaryDirectory() as folder:
            old = Path.cwd()
            os.chdir(folder)
            try:
                model = Embeddings()
                retriever = DenseRetriever(model)
                retriever.build(documents)
                expected = [("a", 1.0), ("b", 1.0), ("c", 0.0)]
                for k in (1, 2, 3, 10):
                    self.assertEqual([(h.doc_id, h.score) for h in
                                      retriever.retrieve(Query("q", "apple"), k)], expected[:k])
                retriever.save_index("saved")
                with patch.object(model, "encode_documents", side_effect=AssertionError("re-encoded")):
                    cached = DenseRetriever(model)
                    cached.build(documents)
                    loaded = DenseRetriever(model)
                    loaded.load_index("saved")
                    loaded.build(documents)
                    for instance in (cached, loaded):
                        self.assertEqual([(h.doc_id, h.score) for h in
                                          instance.retrieve(Query("q", "apple"), 3)], expected)
            finally:
                os.chdir(old)

    def test_cli_dense_and_reranking_language_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = Path.cwd()
            os.chdir(root)
            try:
                dataset = root / "dataset"
                dataset.mkdir()
                for lang in ("vi", "en", "csw"):
                    (dataset / f"queries.{lang}.jsonl").write_text(
                        json.dumps({"qid": "q", "text": "apple"}) + "\n", encoding="utf-8")
                for lang in ("vi", "en"):
                    (dataset / f"documents.{lang}.jsonl").write_text(
                        "\n".join(json.dumps({"doc_id": did, "text": text})
                                  for did, text in [("a", "apple"), ("b", "orange")]) + "\n",
                        encoding="utf-8")
                (dataset / "qrels.tsv").write_text("q\ta\t1\n", encoding="utf-8")
                for kind in ("dense", "cross-encoder"):
                    with self.subTest(retriever=kind):
                        output = root / kind
                        def factory(model_id, **kwargs):
                            return Reranker() if model_id == "reranker" else Embeddings()
                        args = ["--dataset", str(dataset), "--retriever", kind,
                                "--model", "reranker" if kind == "cross-encoder" else "embedding",
                                "--top-k", "1", "--metrics", "ndcg@1", "mrr@1", "recall@1",
                                "precision@1", "map@1", "--output", str(output)]
                        if kind == "cross-encoder":
                            args += ["--candidate-retriever", "dense", "--candidate-model", "embedding",
                                     "--candidate-top-k", "2"]
                        with patch("ir_system.models.factory.create_model", side_effect=factory):
                            self.assertEqual(main(args), 0)
                        metrics = json.loads((output / "metrics.json").read_text())
                        self.assertEqual(len(metrics), 6)
                        self.assertTrue(all(value == 1.0 for pair in metrics.values() for value in pair.values()))
                        runs = list(output.glob("*/run.jsonl"))
                        self.assertEqual(len(runs), 6)
                        for run in runs:
                            self.assertEqual(json.loads(run.read_text())["hits"], [{"doc_id": "a", "score": 1.0}])
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
