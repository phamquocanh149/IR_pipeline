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


def test_shared_negative_pool_uses_scores_outside_each_views_top_k():
    qrels = Qrels()
    qrels.add("q", "p", 1)
    result = fixed_index_analysis([make_runs("q", {
        "vi": [("n1", .9), ("n2", .8), ("p", .6)],
        "csw": [("n2", .95), ("p", .7), ("n1", .1)],
    })], qrels, 1)
    assert all(r["negative_pool_ids"] == ["n1", "n2"] for r in result["margins"])
    assert result["margins"][0]["margin"] == pytest.approx(-.3)
    assert result["margins"][1]["margin"] == pytest.approx(-.25)


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


def test_sparse_dense_scores_match_full_corpus_analysis_and_reuse_index(tmp_path, monkeypatch):
    from ir_system.domain.document import Document
    from ir_system.domain.query import Query
    from ir_system.models.single_vector import SingleVectorEmbeddingModel
    from ir_system.retrievers.dense import DenseRetriever

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(18)
    vectors = {f'd{i}': v for i, v in enumerate(normalize(rng.normal(size=(100, 6))))}
    vectors.update({lang: v for lang, v in zip(('vi', 'en', 'csw'), normalize(rng.normal(size=(3, 6))))})

    class Encoder(SingleVectorEmbeddingModel):
        calls = 0
        name = 'sparse-analysis-fixture'

        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.stack([vectors[t] for t in texts])

    model = Encoder()
    dense = DenseRetriever(model)
    documents = [Document(doc_id=f'd{i}', text=f'd{i}') for i in range(100)]
    dense.build(documents)
    calls = model.calls
    reloaded = DenseRetriever(model)
    reloaded.build(documents)
    assert model.calls == calls  # Matching default index avoids corpus re-encoding.
    queries = {lang: Query(qid='q', text=lang) for lang in ('vi', 'en', 'csw')}
    qrels = Qrels()
    qrels.add('q', 'd99', 1)
    full = {l: list(dense.retrieve(q, 100)) for l, q in queries.items()}
    top = {l: list(dense.retrieve(q, 2)) for l, q in queries.items()}
    pool = {h.doc_id for hits in top.values() for h in hits if h.doc_id != 'd99'}
    sparse = {}
    for lang, hits in top.items():
        missing = sorted((pool | {'d99'}) - {h.doc_id for h in hits})
        hits += list(reloaded.score_documents(queries[lang], missing))
        assert len(hits) <= 7  # Three top-2 rankings plus one positive, not 100 hits.
        sparse[lang] = sorted(hits, key=lambda h: (-h.score, h.doc_id))
    def analyze(hits):
        return fixed_index_analysis([make_runs('q', {l: [(h.doc_id, h.score) for h in hs]
                                                   for l, hs in hits.items()})], qrels, 2)
    expected, actual = analyze(full), analyze(sparse)
    for before, after in zip(expected['margins'], actual['margins']):
        assert before['negative_pool_ids'] == after['negative_pool_ids']
        for key in ('positive_score', 'negative_score', 'margin', 'delta_margin'):
            assert after[key] == pytest.approx(before[key], abs=1e-6)


def test_query_cache_preserves_model_specific_encoding():
    from ir_system.cli.analysis import _QueryEmbeddingCache
    from ir_system.models.single_vector import SingleVectorEmbeddingModel

    class Encoder(SingleVectorEmbeddingModel):
        name = 'prompt-fixture'
        document_prompt = 'passage: '
        def encode(self, texts, **kwargs):
            pytest.fail('Side-specific encoding must be delegated.')
        def encode_queries(self, texts, **kwargs):
            return np.ones((len(texts), 2))
        def encode_documents(self, texts, **kwargs):
            return np.zeros((len(texts), 2))

    cache = _QueryEmbeddingCache(Encoder())
    cache.remember(['same text'], np.ones((1, 2)))
    cache.query_mode = True
    assert cache.document_prompt == 'passage: '
    assert np.all(cache.encode_queries(['same text']) == 1)
    assert np.all(cache.encode_documents(['same text']) == 0)


def test_report_escapes_query_text(tmp_path):
    path = tmp_path / "report.html"
    write_report({"text": "</script><script>alert(1)</script>"}, path)
    html = path.read_text(encoding="utf-8")
    payload = html.split('id="data">', 1)[1].split('</script>', 1)[0]
    assert "<script>" not in payload
    assert json.loads(payload)["text"] == "</script><script>alert(1)</script>"


@pytest.mark.parametrize("use_reranker", [False, True])
@pytest.mark.parametrize("directions", [["vi-en"], ["csw-vi"], ["vi-en", "en-en", "csw-en"], ["vi-vi", "csw-en"]])
@pytest.mark.parametrize("id_format", ["canonical", "_id", "id"])
def test_cli_end_to_end(tmp_path, monkeypatch, use_reranker, directions, id_format):
    from ir_system.cli import analysis as cli
    from ir_system.models.single_vector import SingleVectorEmbeddingModel
    corpus_encodes = []
    monkeypatch.chdir(tmp_path)  # Isolate persistent FAISS indexes across cases.

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
    query_key = "qid" if id_format == "canonical" else id_format
    document_key = "doc_id" if id_format == "canonical" else id_format
    # Numeric JSON IDs must match the string IDs in qrels and saved diagnostics.
    query_id, positive_id, negative_id = ("q", "d+", "d-") if id_format == "canonical" else (7, 11, 12)
    suffixes = {"vi": "vi", "en": "en", "csw": "csw"}
    for language, lang in suffixes.items():
        if directions == ["vi-en"] and language != "vi":
            continue  # A direction does not require query files for its document language.
        (directory / f"queries.{lang}.jsonl").write_text(json.dumps({query_key: query_id, "text": f"positive {lang}"}))
    for language in document_languages:
        suffix = suffixes[language]
        (directory / f"documents.{suffix}.jsonl").write_text('\n'.join(json.dumps(row) for row in [
            {document_key: positive_id, "text": f"positive {suffix}"},
            {document_key: negative_id, "text": f"negative {suffix}"}]))
    (directory / "qrels.tsv").write_text(f"{query_id}\t{positive_id}\t1\n")
    output = tmp_path / "output"
    args = ["--dataset", str(directory), "--model", "fixture-encoder", "--top-k", "2", "--output", str(output)]
    if directions == ["csw-vi"]:
        args.extend(["--queries", "csw", "--documents", "vi", "--model-type", "single-vector",
                     "--retriever", "dense", "--metrics", "ndcg@2", "mrr@2", "recall@2"])
    else:
        args.extend(["--direction", *directions])
    if use_reranker:
        args.extend(["--reranker", "fixture-reranker"])
    assert cli.main(args) == 0
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert {p["qid"] for p in report["query_projection"]["points"]} == {str(query_id)}
    if directions == ["vi-en"]:
        assert len(report["query_projection"]["points"]) == 1
        assert report["query_gaps"] == {}
    else:
        assert len(report["query_projection"]["points"]) == 3
        assert report["query_gaps"]["vi_csw"]["summary"]["mean"] == pytest.approx(0)
    assert set(report["indexes"]) == document_languages
    actual_directions = []
    for index_language, index in report["indexes"].items():
        assert all(r["best_positive"] == str(positive_id) and r["hardest_negative"] == str(negative_id)
                   for r in index["margins"])
        actual_directions.extend(row["direction"] for row in index["margins"])
        for lang, stats in index["summary"].items():
            assert stats["margin"]["mean"] == pytest.approx(.6 if use_reranker else 1.)
            if f"vi-{index_language}" not in directions:
                assert stats["delta_margin"]["count"] == 0
                assert not index["alignment"]
    assert set(actual_directions) == set(directions)
    if directions == ["csw-vi"]:
        assert report["indexes"]["vi"]["retrieval_metrics"]["csw-vi"] == {
            "ndcg@2": 1., "mrr@2": 1., "recall@2": 1.,
        }
    assert (output / "report.html").is_file()
    assert len(corpus_encodes) == len(document_languages)  # Only requested indexes, encoded once.


@pytest.mark.parametrize("missing", ["query", "document"])
def test_analysis_checks_missing_requested_languages(tmp_path, monkeypatch, caplog, missing):
    from ir_system.cli import analysis as cli

    (tmp_path / "queries.vi.jsonl").write_text('{"qid":"q","text":"query"}')
    if missing == "document":
        (tmp_path / "queries.en.jsonl").write_text('{"qid":"q","text":"English query"}')
    (tmp_path / "documents.vi.jsonl").write_text('{"doc_id":"d","text":"document"}')
    (tmp_path / "qrels.tsv").write_text("q\td\t1\n")
    monkeypatch.setattr(cli, "create_model", lambda *args, **kwargs:
                        pytest.fail("Missing requested languages must fail before model loading"))
    assert cli.main(["--dataset", str(tmp_path), "--direction", "vi-vi", "en-en",
                     "--model", "fixture-encoder"]) == 1
    assert f"Requested {missing} languages missing" in caplog.text


def test_shared_qrels_restricts_positives_to_fixed_index():
    qrels = Qrels()
    qrels.add("q", "positive", 1)
    qrels.add("q", "other-language-only", 1)
    runs = make_runs("q", {"vi": [("positive", .8), ("negative", .2)]})
    result = fixed_index_analysis([runs], qrels, 2, document_ids={"positive", "negative"})
    assert result["summary"]["vi"]["margin"]["mean"] == pytest.approx(.6)


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


def test_report_horizontal_panels_and_shared_references():
    """Execute report JS against a minimal DOM to check labels, scales and paging."""
    import shutil
    import subprocess
    from ir_system.analysis.language_views import summary
    from ir_system.analysis.report import _HTML

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    gaps = {}
    for name, value in [("vi_en", .12), ("vi_csw", .08), ("en_csw", .06)]:
        rows = [{"group_id": f"q{i}", "gap": value+(i-1)*.0001,
                 "cosine": 1-value-(i-1)*.0001} for i in range(1, 23)]
        gaps[name] = {"pairs": rows, "summary": summary(r["gap"] for r in rows),
                      "cosine_summary": summary(r["cosine"] for r in rows),
                      "missing_first": [], "missing_other": []}
    report = {
        "config": {"directions": ["vi-en"], "model": "fixture", "scorer": "cosine", "top_k": 10},
        "query_projection": {"points": [{"qid": f"q{i}", "language": lang, "text": f"query {i}",
                                         "xyz": [0, 0, 0]} for i in range(1, 23) for lang in ("vi", "csw", "en")],
                             "explained_variance_ratio": [1, 0, 0]},
        "query_gaps": gaps, "document_gaps": {}, "policies": {},
        "indexes": {"en": {"summary": {}, "margins": [
            {"qid": "q1", "language": "vi", "direction": "vi-en", "positive_score": .6,
             "negative_score": .3, "margin": .3, "delta_margin": 0, "negative_pool_ids": ["d-"]},
            {"qid": "q1", "language": "csw", "direction": "csw-en", "positive_score": .7,
             "negative_score": .5, "margin": .2, "delta_margin": -.1, "negative_pool_ids": ["d-"]},
            {"qid": "q1", "language": "en", "direction": "en-en", "positive_score": .5,
             "negative_score": .3, "margin": .2, "delta_margin": -.1, "negative_pool_ids": ["d-"]},
            {"qid": "q2", "language": "vi", "direction": "vi-en", "positive_score": None,
             "negative_score": .5, "margin": None, "delta_margin": None},
        ], "alignment": [
            {"qid": "q1", "language": "csw", "doc_group_id": "d1", "delta_alignment": .1},
            {"qid": "q1", "language": "csw", "doc_group_id": "d2", "delta_alignment": .3},
        ],
                           "paired_query_count": 22, "excluded_query_ids": {}}},
    }
    report["embedding_panels"] = [
        {"config": {"model": model, "dataset": benchmark}, "query_projection": report["query_projection"]}
        for model in ("Model A", "Model B") for benchmark in ("Benchmark 1", "Benchmark 2")
    ]
    report["benchmark_reports"] = [dict(report, config=dict(report["config"], dataset=f"Benchmark {i}"))
                                   for i in range(1, 5)]
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.style={};this.value='';this.textContent='';this.classList={add(){}};this.clientWidth=1000;this.clientHeight=520;}
 append(...items){this.children.push(...items);if(this.tag==='select'&&!this.value&&items[0])this.value=items[0].value;}
 replaceChildren(...items){this.children=[];this.append(...items);}
 setAttribute(key,value){this.attrs[key]=value;}
 querySelector(){return new Element('option');}
 addEventListener(){}
 getContext(){return new Proxy({}, {get(){return ()=>{}}});}
}
const nodes=new Map();
global.document={getElementById(id){if(!nodes.has(id))nodes.set(id,new Element(id==='index'?'select':'div'));return nodes.get(id);},querySelectorAll(){return [];},createElement(tag){return new Element(tag);},createElementNS(ns,tag){return new Element(tag);}};
global.window={devicePixelRatio:1,addEventListener(){}};
document.getElementById('drift-kind').value='query';document.getElementById('drift-metric').value='gap';
document.getElementById('data').textContent=JSON.stringify(REPORT_FIXTURE);
'''.replace("REPORT_FIXTURE", json.dumps(report, ensure_ascii=False))
    script = _HTML.split("</script><script>", 1)[1].split("</script>", 1)[0]
    checks = r'''
const panels=document.getElementById('score-panels');
const embeddingPanels=document.getElementById('embedding-panels').children;
assert.equal(embeddingPanels.length,4);
assert.equal(embeddingPanels[0].children[0].textContent,'Model A | Benchmark 1');
assert.equal(embeddingPanels[3].children[0].textContent,'Model B | Benchmark 2');
assert.match(embeddingPanels[0].children[1].textContent,/100.0%.*22 matched triplets/);
assert.equal(embeddingPanels[0].children[2].children[0].tag,'canvas');
assert.match(panels.children[0].textContent,/alignment change vs margin change/);
const grid=panels.children[1].children[0];
assert.equal(grid.className,'benchmark-row');assert.equal(grid.children.length,4);
const scatter=grid.children[0].children[2];
const dots=scatter.children.filter(e=>e.tag==='circle');
assert.equal(dots.length,1);assert.equal(dots[0].attrs.fill,'#eea048');
assert.match(dots[0].children[0].textContent,/Delta A 0.2000.*Delta M -0.1000/);
assert(scatter.children.some(e=>e.tag==='text'&&e.textContent==='100.0%'));
const references=grid.children.map(panel=>panel.children[2].children.filter(e=>e.tag==='line').map(e=>e.attrs));
assert.deepEqual(references[0],references[3]);
const gapGrid=document.getElementById('representation-violins').children[0].children[0];
assert.equal(gapGrid.children.length,4);
const violin=gapGrid.children[0].children[2];
assert.equal(violin.children.filter(e=>e.tag==='path').length,2);
assert.equal(violin.children.filter(e=>e.tag==='polygon').length,2);
const zeros=deltaPanel({config:{dataset:'benchmark'}},[{qid:'q',da:0,dm:0},{qid:'q2',da:.1,dm:.1}],.2);
assert(zeros.children[2].children.some(e=>e.tag==='text'&&e.textContent==='50.0%'));
assert.match(zeros.children[3].textContent,/zero-axis=1/);
assert(!nodes.has('distributions'));assert(!nodes.has('drift-chart'));
'''
    result = subprocess.run([node], input=(harness+script+checks).encode("utf-8"), capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
