# IR System — Information Retrieval Pipeline

End-to-end IR pipeline supporting:
- **BM25** (sparse lexical retrieval)
- **Dense** (bi-encoder with single-vector embeddings)
- **Late Interaction** (ColBERT-style MaxSim)
- **Cross Encoder** (joint query-document scoring / reranking)
- **Hybrid** (BM25 + Dense fused via RRF)

Metrics: NDCG@K, MRR@K, Recall@K

## Installation

```bash
# Core (BM25 only)
pip install -e .

# With dense/hybrid retrieval
pip install -e ".[dense]"

# Full (all retrievers)
pip install -e ".[all]"
```

## Usage

### BM25

```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --retriever bm25 \
    --top-k 3 \
    --metrics ndcg@3 mrr@3 recall@3
```

### Dense

```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model BAAI/bge-base-en-v1.5 \
    --retriever dense \
    --top-k 10 \
    --metrics ndcg@10 mrr@10 recall@10
```

### Hybrid (BM25 + Dense via RRF)

```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model BAAI/bge-base-en-v1.5 \
    --retriever hybrid \
    --top-k 10 \
    --candidate-top-k 100 \
    --metrics ndcg@10 mrr@10 recall@100
```

### Cross Encoder (reranking)

```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --retriever cross-encoder \
    --top-k 10 \
    --candidate-top-k 100 \
    --metrics ndcg@10 mrr@10
```

## Dataset Format

```
data/<dataset_name>/
├── queries.jsonl      {"qid": "q1", "text": "what is bm25"}
├── documents.jsonl    {"doc_id": "d1", "text": "BM25 is ..."}
└── qrels.tsv          q1\td1\t2
```

## Results

Each run saves to `results/<run_id>/`:
- `config.json` — run configuration
- `metrics.json` — computed metric scores
- `run.jsonl` — per-query ranked hits

### Customizing Run Name / Output Directory

By default, `<run_id>` is an auto-generated timestamp string (e.g. `20260927T094744_bm25`). You can customize it using:

- `--name <name>` / `--run-name <name>` / `-n <name>`: Saves directly to `results/<name>/`.
  ```bash
  python -m ir_system.cli.main --dataset data/toy --retriever bm25 --top-k 10 --metrics ndcg@10 --name my_bm25_test
  ```
- `--output <dir>` / `-o <dir>`: Saves to a custom directory. If a single folder name is passed, it is placed under `results/<dir>/`.

