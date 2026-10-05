import numpy as np
import pytest

from ir_system.domain.document import Document
from ir_system.domain.query import Query
from ir_system.models.multi_vector import MultiVectorEmbeddingModel
from ir_system.pipeline.search_pipeline import SearchPipeline
from ir_system.retrievers.late_interaction import LateInteractionRetriever


class Model(MultiVectorEmbeddingModel):
    name = 'fixture-colbert'
    device = 'cpu'

    def __init__(self, vectors):
        self.vectors = vectors
        self.roles = []

    def encode(self, texts, **kwargs):
        raise AssertionError('Must use explicit query/document routing')

    def encode_documents(self, texts, **kwargs):
        self.roles.append('document')
        return [self.vectors[t] for t in texts]

    def encode_queries(self, texts, **kwargs):
        self.roles.append(('query', kwargs['batch_size']))
        return [self.vectors[t] for t in texts]


def fixture():
    rng = np.random.default_rng(18)
    vectors = {str(i): rng.normal(size=(i % 5 + 1, 128)).astype('float32')
               for i in range(19)}
    vectors['empty'] = np.empty((0, 128), dtype='float32')
    vectors['tie'] = vectors['0'].copy()
    queries = [Query('q1', '1'), Query('q2', '2'), Query('q3', '3')]
    documents = [Document(key, key) for key in vectors]
    return vectors, queries, documents


def test_cpu_pipeline_matches_original_and_roles():
    vectors, queries, documents = fixture()
    model = Model(vectors)
    retriever = LateInteractionRetriever(model)
    retriever.build(documents)
    actual = list(retriever.retrieve_many(queries, 50))
    for query, hits in zip(queries, actual):
        expected = sorted(((d.doc_id, retriever._reference_score(vectors[query.text], vectors[d.text]))
                           for d in documents), key=lambda pair: (-pair[1], pair[0]))
        assert [(h.doc_id, h.score) for h in hits] == expected
    run = SearchPipeline(retriever).search(queries, 3)
    assert len(run) == 3
    assert model.roles[0] == 'document'
    assert all(role == ('query', 1) for role in model.roles[1:])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_block_matches_numpy(device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    vectors, queries, documents = fixture()
    q = [vectors[item.text] for item in queries]
    d = [vectors[item.text] for item in documents]
    # Include non-finite and all-empty blocks.
    d.append(np.full((1, 128), np.nan, dtype='float32'))
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for docs in (d, [vectors['empty']]):
            actual = LateInteractionRetriever._score_block(q, docs, device)
            expected = [[LateInteractionRetriever._reference_score(a, b) for b in docs] for a in q]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_negative_scores_padding_and_ties():
    pytest.importorskip('torch')
    q = np.ones((2, 128), dtype='float32')
    docs = [-np.ones((1, 128), dtype='float32'),
            -np.ones((3, 128), dtype='float32'), np.empty((0, 128), dtype='float32')]
    actual = LateInteractionRetriever._score_block([q], docs, 'cpu')
    np.testing.assert_array_equal(actual, [[-256, -256, 0]])


@pytest.mark.parametrize('block_size', [1, 4, 128])
@pytest.mark.parametrize('top_k', [1, 5, 50])
def test_cuda_retrieval_exact_verification(block_size, top_k):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    vectors, queries, documents = fixture()
    gpu = LateInteractionRetriever(Model(vectors), score_batch_size=block_size,
                                   query_batch_size=2, scoring_device='cuda', verify_scores=True)
    cpu = LateInteractionRetriever(Model(vectors))
    gpu.build(documents)
    cpu.build(documents)
    old = torch.backends.cuda.matmul.allow_tf32
    assert list(gpu.retrieve_many(queries, top_k)) == list(cpu.retrieve_many(queries, top_k))
    assert torch.backends.cuda.matmul.allow_tf32 == old
    assert gpu.retrieve(queries[0], top_k) == cpu.retrieve(queries[0], top_k)


@pytest.mark.parametrize('field', ['batch_size', 'score_batch_size', 'query_batch_size'])
def test_invalid_batch(field):
    with pytest.raises(ValueError):
        LateInteractionRetriever(Model({}), **{field: 0})

def test_batched_orchestration_on_cpu(monkeypatch):
    # Exercise query grouping, block merging and the final short batch even
    # on a CPU-only test machine. Tensor kernel is independently tested above.
    pytest.importorskip('torch')
    vectors, queries, documents = fixture()
    scorer = LateInteractionRetriever._score_block
    for top_k in (1, 5, 50):
        batched = LateInteractionRetriever(Model(vectors), score_batch_size=4,
                                           query_batch_size=2, scoring_device='cuda',
                                           verify_scores=True)
        monkeypatch.setattr(batched, '_score_block', lambda q, d, device: scorer(q, d, 'cpu'))
        reference = LateInteractionRetriever(Model(vectors))
        batched.build(documents)
        reference.build(documents)
        assert list(batched.retrieve_many(queries, top_k)) == list(reference.retrieve_many(queries, top_k))
        assert len(SearchPipeline(batched).search(queries, top_k)) == len(queries)

def test_colbert_cannot_be_used_as_cross_encoder():
    from ir_system.cli.main import _build_retriever
    with pytest.raises(ValueError, match='CrossEncoderModel'):
        _build_retriever('cross-encoder', Model({}), 100, 32)
    retriever = _build_retriever('late-interaction', Model({}), 100, 32,
                                colbert_score_batch_size=7, colbert_query_batch_size=3,
                                colbert_scoring_device='cpu')
    assert isinstance(retriever, LateInteractionRetriever)
    assert retriever._score_batch_size == 7
    assert retriever._query_batch_size == 3


def test_tie_break_across_blocks_and_precision_restored(monkeypatch):
    torch = pytest.importorskip('torch')
    vectors = {key: np.ones((1, 128), dtype='float32') for key in ('z', 'b', 'a', 'q')}
    retriever = LateInteractionRetriever(Model(vectors), scoring_device='cuda', score_batch_size=1)
    retriever.build([Document(key, key) for key in ('z', 'b', 'a')])
    scorer = LateInteractionRetriever._score_block
    monkeypatch.setattr(retriever, '_score_block', lambda q, d, device: scorer(q, d, 'cpu'))
    original = torch.backends.cuda.matmul.allow_tf32
    hits = retriever.retrieve(Query('q', 'q'), 2)
    assert [hit.doc_id for hit in hits] == ['a', 'b']
    assert torch.backends.cuda.matmul.allow_tf32 == original
    def fail(*args):
        raise RuntimeError('scoring failed')
    monkeypatch.setattr(retriever, '_score_block', fail)
    with pytest.raises(RuntimeError, match='scoring failed'):
        retriever.retrieve(Query('q', 'q'), 2)
    assert torch.backends.cuda.matmul.allow_tf32 == original

def test_adapter_keeps_query_augmentation_and_document_routing():
    torch = pytest.importorskip('torch')
    from ir_system.models._colbert_adapter import ColBERTAdapter
    class Checkpoint:
        def queryFromText(self, batch):
            # All augmentation vectors must survive, including zero vectors.
            return torch.zeros((len(batch), 32, 128))
        def docFromText(self, batch, keep_dims, to_cpu):
            assert keep_dims is False and to_cpu is False
            return [torch.ones((3, 128)) for _ in batch]
    adapter = ColBERTAdapter.__new__(ColBERTAdapter)
    adapter._model = Checkpoint()
    adapter._default_batch_size = 2
    adapter._warned_default_role = False
    queries = adapter.encode_queries(['a', 'b', 'c'])
    documents = adapter.encode_documents(['a', 'b', 'c'])
    assert [v.shape for v in queries] == [(32, 128)] * 3
    assert [v.shape for v in documents] == [(3, 128)] * 3
    assert all(v.dtype == np.float32 for v in queries + documents)
