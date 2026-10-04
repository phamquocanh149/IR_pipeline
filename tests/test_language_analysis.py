import json

import numpy as np
import pytest

from ir_system.analysis.language_views import fixed_index_analysis, normalize, paired_gaps, project_3d
from ir_system.analysis.report import write_report
from ir_system.domain.qrels import Qrels
from ir_system.domain.hit import Hit
from ir_system.domain.run import Run


def make_runs(qid, rankings):
    runs = {}
    for lang, hits in rankings.items():
        runs[lang] = Run()
        runs[lang].add(qid, [Hit(doc_id, score) for doc_id, score in hits])
    return runs


def test_alignment_can_improve_while_margin_decreases():
    qrels = Qrels()
    qrels.add("q1", "positive", 1)
    runs = make_runs("q1", {"vi": [("positive", .6), ("negative", .3)],
                           "en": [("positive", .6), ("negative", .2)],
                           "csw": [("positive", .7), ("negative", .5)]})
    result = fixed_index_analysis([runs], qrels, 2)
    csw = next(row for row in result["margins"] if row["language"] == "csw")
    assert csw["margin"] == pytest.approx(.2)
    assert csw["delta_margin"] == pytest.approx(-.1)
    assert result["alignment"][1]["delta_alignment"] == pytest.approx(.1)
    assert csw["unjudged_negative_count"] == 1


def test_positive_outside_top_k_and_missing_negatives():
    qrels = Qrels()
    qrels.add("q", "positive", 1)
    runs = make_runs("q", {lang: [("negative", .9), ("positive", .2)]
                           for lang in ("vi", "en", "csw")})
    result = fixed_index_analysis([runs], qrels, 1)
    assert result["margins"][0]["margin"] == pytest.approx(-.7)
    assert len(result["alignment"]) == 2
    no_negative = fixed_index_analysis([make_runs("q", {lang: [("positive", .2)]
                                                      for lang in ("vi", "en", "csw")})], qrels, 1)
    assert no_negative["margins"][0]["margin"] is None
    assert no_negative["summary"]["csw"]["delta_margin"]["count"] == 0


def test_no_positive_and_unpaired_queries():
    runs = make_runs("q", {lang: [("d", .1)] for lang in ("vi", "en", "csw")})
    runs["vi"].add("extra", [])
    result = fixed_index_analysis([runs], Qrels(), 1)
    assert result["paired_query_count"] == 1
    assert result["excluded_query_ids"]["vi"] == ["extra"]
    assert result["margins"][0]["margin"] is None
    assert not result["alignment"]


def test_gap_and_joint_projection():
    vectors = normalize(np.array([[1, 0], [0, 1], [-1, 0]]))
    gaps = paired_gaps({lang: {"q": v} for lang, v in zip(("vi", "en", "csw"), vectors)})
    assert gaps["vi_en"]["summary"]["mean"] == pytest.approx(1)
    assert gaps["vi_csw"]["summary"]["median"] == pytest.approx(2)
    assert gaps["vi_csw"]["pairs"][0]["cosine"] == pytest.approx(-1)
    assert gaps["en_csw"]["cosine_summary"]["mean"] == pytest.approx(0)
    xyz, variance = project_3d(vectors)
    assert xyz.shape == (3, 3)
    assert sum(variance) == pytest.approx(1)
    assert np.linalg.norm(xyz[0] - xyz[1]) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="zero"):
        normalize(np.zeros((1, 2)))


def test_report_escapes_query_text(tmp_path):
    path = tmp_path / "report.html"
    write_report({"text": "</script><script>alert(1)</script>"}, path)
    html = path.read_text(encoding="utf-8")
    payload = html.split('id="data">', 1)[1].split('</script>', 1)[0]
    assert "<script>" not in payload
    assert json.loads(payload)["text"] == "</script><script>alert(1)</script>"


@pytest.mark.parametrize("use_reranker", [False, True])
@pytest.mark.parametrize("directions", [["vi-en"], ["csw-vi"], ["vi-en", "en-en", "csw-en"]])
def test_cli_end_to_end(tmp_path, monkeypatch, use_reranker, directions):
    from ir_system.cli import analysis as cli
    from ir_system.models.single_vector import SingleVectorEmbeddingModel
    corpus_encodes = []

    class Encoder(SingleVectorEmbeddingModel):
        @property
        def name(self):
            return "fixture-encoder"

        def encode(self, texts, **kwargs):
            if len(texts) == 2:
                corpus_encodes.append(list(texts))
            return np.array([[1., 0.] if text.startswith("positive") else [0., 1.]
                             for text in texts])

    class Reranker:
        name = "fixture-reranker"

        def score(self, query_texts, document_texts, **kwargs):
            return [.8 if text.startswith("positive") else .2 for text in document_texts]

    monkeypatch.setattr(cli, "create_model", lambda model_id, **kw:
                        Reranker() if model_id == "fixture-reranker" else Encoder())
    directory = tmp_path / "fiqa"
    directory.mkdir()
    document_languages = {direction.split("-")[1] for direction in directions}
    suffixes = {"vi": "vi" if directions == ["csw-vi"] else "vn", "en": "en", "csw": "csw"}
    for language, lang in suffixes.items():
        if directions == ["vi-en"] and language != "vi":
            continue  # A direction does not require query files for its document language.
        (directory / f"queries.{lang}.jsonl").write_text(json.dumps({"qid": "q", "text": f"positive {lang}"}))
    for language in document_languages:
        suffix = suffixes[language]
        (directory / f"documents.{suffix}.jsonl").write_text('\n'.join(json.dumps(row) for row in [
            {"doc_id": "d+", "text": f"positive {suffix}"},
            {"doc_id": "d-", "text": f"negative {suffix}"}]))
    (directory / "qrels.tsv").write_text("q\td+\t1\n")
    output = tmp_path / "output"
    args = ["--dataset", str(directory), "--model", "fixture-encoder", "--top-k", "2", "--output", str(output)]
    args.extend(["--direction", *directions])
    if use_reranker:
        args.extend(["--reranker", "fixture-reranker"])
    assert cli.main(args) == 0
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    if directions == ["vi-en"]:
        assert len(report["query_projection"]["points"]) == 1
        assert report["query_gaps"] == {}
    else:
        assert len(report["query_projection"]["points"]) == 3
        assert report["query_gaps"]["vi_csw"]["summary"]["mean"] == pytest.approx(0)
    assert set(report["indexes"]) == document_languages
    actual_directions = []
    for index_language, index in report["indexes"].items():
        actual_directions.extend(row["direction"] for row in index["margins"])
        for lang, stats in index["summary"].items():
            assert stats["margin"]["mean"] == pytest.approx(.6 if use_reranker else 1.)
            if f"vi-{index_language}" not in directions:
                assert stats["delta_margin"]["count"] == 0
                assert not index["alignment"]
    assert set(actual_directions) == set(directions)
    assert (output / "report.html").is_file()
    assert len(corpus_encodes) == len(document_languages)  # Only requested indexes, encoded once.


def test_loader_language_qrels_and_original_layout(tmp_path):
    from ir_system.io.dataset_loader import DatasetLoader

    for suffix in ("", ".vn"):
        (tmp_path / f"queries{suffix}.jsonl").write_text('{"qid":"q","text":"query"}')
        (tmp_path / f"documents{suffix}.jsonl").write_text('{"doc_id":"d","text":"document"}')
    (tmp_path / "qrels.tsv").write_text("q\td\t1\n")
    loader = DatasetLoader(strict_qrels=True)
    assert loader.load(tmp_path)[2].relevance("q", "d") == 1
    assert loader.load(tmp_path, language="vn")[2].relevance("q", "d") == 1
    (tmp_path / "qrels.vn.tsv").write_text("q\td\t2\n")
    assert loader.load(tmp_path, language="vn")[2].relevance("q", "d") == 2
    with pytest.raises(FileNotFoundError, match="queries.csw.jsonl"):
        loader.load(tmp_path, language="csw")


def test_loader_query_and_document_languages_are_independent(tmp_path):
    from ir_system.io.dataset_loader import DatasetLoader

    (tmp_path / "queries.vn.jsonl").write_text('{"qid":"q","text":"Vietnamese query"}')
    (tmp_path / "documents.en.jsonl").write_text('{"doc_id":"d","text":"English document"}')
    (tmp_path / "qrels.en.tsv").write_text("q\td\t1\n")
    loader = DatasetLoader(strict_qrels=True)
    assert loader.load_queries(tmp_path, language="vn")[0].text == "Vietnamese query"
    documents, qrels = loader.load_corpus(tmp_path, language="en")
    assert documents[0].text == "English document"
    assert qrels.relevance("q", "d") == 1


def test_loader_vietnamese_suffix_alias_and_precedence(tmp_path):
    from ir_system.io.dataset_loader import DatasetLoader

    (tmp_path / "queries.vn.jsonl").write_text('{"qid":"q","text":"vn query"}')
    (tmp_path / "documents.vn.jsonl").write_text('{"doc_id":"d","text":"vn document"}')
    (tmp_path / "qrels.vn.tsv").write_text("q\td\t1\n")
    loader = DatasetLoader(strict_qrels=True)
    assert loader.query_path(tmp_path, language="vi").name == "queries.vn.jsonl"
    queries, documents, qrels = loader.load(tmp_path, language="vi")
    assert queries[0].text == "vn query"
    assert documents[0].text == "vn document"
    assert qrels.relevance("q", "d") == 1
    for stem, record in [
        ("queries", {"qid": "q", "text": "vi query"}),
        ("documents", {"doc_id": "d", "text": "vi document"}),
    ]:
        (tmp_path / f"{stem}.vi.jsonl").write_text(json.dumps(record))
    (tmp_path / "qrels.vi.tsv").write_text("q\td\t2\n")
    queries, documents, qrels = loader.load(tmp_path, language="vi")
    assert queries[0].text == "vi query"
    assert documents[0].text == "vi document"
    assert qrels.relevance("q", "d") == 2
    assert loader.load(tmp_path, language="vn")[0][0].text == "vn query"


def test_report_javascript_syntax():
    import shutil
    import subprocess
    from ir_system.analysis.report import _HTML

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = _HTML.split("</script><script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run([node, "--check"], input=script.encode("utf-8"), capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
