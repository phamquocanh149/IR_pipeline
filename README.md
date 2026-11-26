# ViEn-CSIR

Repository for **ViEn-CSIR: A Benchmark for Vietnamese–English Code-Switched Information Retrieval**.

ViEn-CSIR supports the evaluation of information retrieval with Vietnamese–English
code-switched queries. The benchmark provides Vietnamese, English, and
code-switched query variants, Vietnamese and English document collections, and
shared relevance judgments. These language variants enable evaluation across six
query–document language configurations.

This repository contains the benchmark data and a retrieval pipeline supporting
BM25, dense retrieval, and cross-encoder reranking. Evaluation metrics include
NDCG@k, MRR@k, Recall@k, Precision@k, and MAP@k.

## Dataset

Four subsets are provided under `data/`: CQA, FiQA, Gaming, and MLQA.
Each includes Vietnamese (`vi`), English (`en`), and code-switched (`csw`)
queries, documents in Vietnamese and English, and shared relevance judgments.

| Subset | Vietnamese queries | English queries | Code-switched queries | Documents per language (VI / EN) | Relevance judgments |
|---|---:|---:|---:|---:|---:|
| CQA | 1,014 | 1,014 | 1,014 | 121,184 / 121,184 | 5,510 |
| FiQA | 1,025 | 1,025 | 1,025 | 57,638 / 57,638 | 17,110 |
| Gaming | 888 | 898 | 888 | 45,301 / 45,301 | 1,447 |
| MLQA | 981 | 981 | 981 | 5,279 / 5,279 | 983 |
| **Total** | **3,908** | **3,918** | **3,908** | **229,402 / 229,402** | **25,050** |

Each subset contains `queries.{vi,en,csw}.jsonl`, `documents.{vi,en}.jsonl`,
and `qrels.tsv`. Before using CQA, assemble and verify its document files:

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
python -m ir_system.cli.main --dataset data/mlqa --retriever bm25 --queries vi en csw --documents vi en --top-k 10 --metrics ndcg@10 mrr@10 recall@10 --name mlqa-bm25
```

Configuration, metrics, and per-configuration rankings are saved under
`results/mlqa-bm25/`. See `python -m ir_system.cli.main --help` for all options.
