# ViEn-CSIR

Repository for **ViEn-CSIR: A Benchmark for Vietnamese–English Code-Switched Information Retrieval**.

ViEn-CSIR supports the evaluation of information retrieval with Vietnamese–English
code-switched queries. The benchmark provides Vietnamese, English, and
code-switched query variants and Vietnamese and English document collections.
These language variants enable evaluation across six
query–document language configurations.

This repository contains the benchmark data and a retrieval pipeline supporting
BM25, dense retrieval, and cross-encoder reranking. Evaluation metrics include
NDCG@k, MRR@k, Recall@k, Precision@k, and MAP@k.

## Dataset

Four subsets are provided under `data/`: CQA, FiQA, Gaming, and MLQA.
Each includes Vietnamese (`vi`), English (`en`), and code-switched (`csw`)
queries and documents in Vietnamese and English.

| Subset | Vietnamese queries | English queries | Code-switched queries | Documents per language (VI / EN) |
|---|---:|---:|---:|---:|
| CQA | 1,014 | 1,014 | 1,014 | 121,184 / 121,184 |
| FiQA | 1,025 | 1,025 | 1,025 | 57,638 / 57,638 |
| Gaming | 888 | 888 | 888 | 45,301 / 45,301 |
| MLQA | 981 | 981 | 981 | 5,279 / 5,279 |
| **Total** | **3,908** | **3,908** | **3,908** | **229,402 / 229,402** |

Queries and documents are stored in `queries.{vi,en,csw}.jsonl` and
`documents.{vi,en}.jsonl`. Before using CQA, assemble and verify its document files:

```bash
python data/cqa/prepare.py
```

## Installation

Python 3.10 or later is required. From the repository root, install the BM25
retrieval dependencies:

```bash
pip install -e .
```

For dense retrieval and cross-encoder reranking, install the optional dependencies
with `pip install -e ".[all]"`. Dependency specifications are provided in
`pyproject.toml`.

## Example retrieval run

Evaluate BM25 on MLQA across all six language configurations:

```bash
python -m ir_system.cli.main \
  --dataset data/mlqa \
  --retriever bm25 \
  --queries vi en csw \
  --documents vi en \
  --top-k 10 \
  --metrics ndcg@10 mrr@10 recall@10 \
  --name mlqa-bm25
```

Retrieve 100 candidates with Qwen3-Embedding-0.6B, then rerank with a
cross-encoder and return the top ten:

```bash
python -m ir_system.cli.main \
  --dataset data/mlqa \
  --retriever cross-encoder \
  --model jinaai/jina-reranker-v2-base-multilingual \
  --model-type cross-encoder \
  --candidate-retriever dense \
  --candidate-model Qwen/Qwen3-Embedding-0.6B \
  --candidate-top-k 100 \
  --queries vi en csw \
  --documents vi en \
  --top-k 10 \
  --metrics ndcg@10 mrr@10 recall@10 \
  --name mlqa-reranker
```

Configuration, metrics, and per-configuration rankings are saved under
`results/<name>/`. See `python -m ir_system.cli.main --help` for all options.
