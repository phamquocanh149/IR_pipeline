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

### Cross Encoder (2-Stage Reranking)

```bash
# Stage 1: BM25 (default)
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --retriever cross-encoder \
    --candidate-retriever bm25 \
    --top-k 10 \
    --candidate-top-k 100 \
    --metrics ndcg@10 mrr@10

# Stage 1: Dense
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --retriever cross-encoder \
    --candidate-retriever dense \
    --candidate-model BAAI/bge-base-en-v1.5 \
    --top-k 10 \
    --candidate-top-k 100 \
    --metrics ndcg@10 mrr@10
```

### FAISS Index Persistence (Save / Load Offline)

Save FAISS index after build:
```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model BAAI/bge-base-en-v1.5 \
    --retriever dense \
    --save-index indexes/toy_bge \
    --top-k 10 \
    --metrics ndcg@10
```

Load previously saved FAISS index (skips corpus encoding):
```bash
python -m ir_system.cli.main \
    --dataset ./data/toy \
    --model BAAI/bge-base-en-v1.5 \
    --retriever dense \
    --load-index indexes/toy_bge \
    --top-k 10 \
    --metrics ndcg@10
```

Also works for the Dense branch in **Hybrid** and the Dense candidate retriever in **Cross-Encoder**.

## Multilingual & Cross-Lingual Evaluation

IR Pipeline supports evaluating cross-lingual retrieval across combinations of query and document languages (e.g. `queries.vi` vs `documents.en`).

```bash
# Evaluate all 6 cross-lingual pairs (3 query langs x 2 doc langs)
python -m ir_system.cli.main \
    --dataset ./data/toy-ml \
    --retriever bm25 \
    --queries all \
    --documents all \
    --top-k 10 \
    --metrics ndcg@10 mrr@10

# Or evaluate specific language combinations
python -m ir_system.cli.main \
    --dataset ./data/toy-ml \
    --retriever bm25 \
    --queries vi en \
    --documents vi \
    --top-k 10 \
    --metrics ndcg@10
```

Results are printed as a comprehensive summary table and saved under `results/<run_id>/queries_<lang>Xdocs_<lang>/`.

## Dataset Format

### Legacy (Single-file)
```
data/<dataset_name>/
├── queries.jsonl      {"qid": "q1", "text": "what is bm25"}
├── documents.jsonl    {"doc_id": "d1", "text": "BM25 is ..."}
└── qrels.tsv          q1\td1\t2
```

### Multilingual (Split-file)
```
data/<dataset_name>/
├── queries.vi.jsonl       {"qid": "q1", "text": "bm25 là gì"}
├── queries.en.jsonl       {"qid": "q1", "text": "what is bm25"}
├── queries.csw.jsonl      {"qid": "q1", "text": "bm25 la gi va how it works"}
├── documents.vi.jsonl     {"doc_id": "d1", "text": "BM25 là thuật toán..."}
├── documents.en.jsonl     {"doc_id": "d1", "text": "BM25 is a ranking function..."}
└── qrels.tsv              q1\td1\t2
```
*Note: `qid` and `doc_id` are shared across languages for consistent relevance judgments.*

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

