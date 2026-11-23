# IR Pipeline

A modular information retrieval framework for evaluating lexical retrieval, dense
retrieval, and cross-encoder reranking across Vietnamese (`vi`), English (`en`),
and code-switched (`csw`) queries.

## Methods

| Method | Implementation |
|---|---|
| BM25 | BM25S, Lucene scoring; lowercase whitespace tokenization |
| Dense | Single-vector embeddings; exact cosine search with FAISS `IndexFlatIP` |
| Reranking | BM25 or dense candidate retrieval followed by cross-encoder scoring |

Evaluation supports NDCG@k, MRR@k, Recall@k, Precision@k, and MAP@k.

## Installation

From the repository root:

```bash
pip install -e ".[all]"
```

The inference stack is pinned in `pyproject.toml`: PyTorch **2.14.1**,
Transformers **4.57.6**, Sentence Transformers **5.7.0**, and
FAISS CPU **1.15.1**. For BM25 alone, use `pip install -e .`.

## Data

Datasets must be supplied separately. A single-language dataset contains:

```text
dataset/
├── queries.jsonl       # {"qid": "q1", "text": "query text"}
├── documents.jsonl     # {"doc_id": "d1", "text": "document text"}
└── qrels.tsv           # qid<TAB>doc_id<TAB>relevance
```

For multilingual evaluation, use `queries.{vi,en,csw}.jsonl` and
`documents.{vi,en}.jsonl` with a shared `qrels.tsv`. Matching IDs identify the
same query or document group across language views. Each JSONL record requires
`text`; IDs also accept `_id` or `id` and are normalized to strings.

## Retrieval experiments

Examples below use Bash syntax. Replace `path/to/dataset` with your dataset.

```bash
# Lexical baseline
python -m ir_system.cli.main \
  --dataset path/to/dataset --retriever bm25 \
  --top-k 10 --metrics ndcg@10 mrr@10 recall@10 --name bm25

# Dense retrieval
python -m ir_system.cli.main \
  --dataset path/to/dataset --retriever dense \
  --model Qwen/Qwen3-Embedding-0.6B \
  --top-k 10 --metrics ndcg@10 mrr@10 recall@10 --name dense

# Dense candidates followed by cross-encoder reranking
python -m ir_system.cli.main \
  --dataset path/to/dataset --retriever cross-encoder \
  --model jinaai/jina-reranker-v2-base-multilingual \
  --candidate-retriever dense --candidate-model Qwen/Qwen3-Embedding-0.6B \
  --candidate-top-k 100 --top-k 10 \
  --metrics ndcg@10 mrr@10 recall@10 --name reranking
```

Use `--candidate-retriever bm25` without `--candidate-model` for lexical
candidates. Add `--queries vi en csw --documents vi en` to evaluate all six
language pairs. Runtime options include `--device`, `--batch-size`, and
`--candidate-batch-size`.

## Language analysis

```bash
python -m ir_system.cli.analysis \
  --dataset path/to/dataset \
  --model Qwen/Qwen3-Embedding-0.6B \
  --queries vi en csw --documents vi en \
  --top-k 10 --output results/language_analysis
```

Produces `analysis.json` and an offline `report.html` with embedding gaps,
joint PCA projections, retrieval metrics, and relevant/non-relevant score
margins. Analysis compares language views against a fixed document index.

## Outputs and reproducibility

Retrieval runs save `config.json`, `metrics.json`, and `run.jsonl` under
`results/<name>/`, with separate results for each language pair. Dense indexes
are cached under `indexes/`; `--load-index PATH` explicitly loads a saved index.

For comparable experiments, keep dataset IDs, relevance judgments, candidate
selection, and metric cutoffs fixed. Set `--top-k` at least as large as the
largest metric cutoff. Record model revisions, prompts, seed, and hardware
alongside the saved configuration. Model-provided or inferred prompts are
applied automatically; `--query-prefix` and `--doc-prefix` override prompts for
the main embedding model.

Full CLI options:

```bash
python -m ir_system.cli.main --help
python -m ir_system.cli.analysis --help
```
