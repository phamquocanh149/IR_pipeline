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

## Dataset Format

```
data/<dataset_name>/
├── queries.jsonl      {"qid": "q1", "text": "what is bm25"}
├── documents.jsonl    {"doc_id": "d1", "text": "BM25 is ..."}
└── qrels.tsv          q1\td1\t2
```

Query IDs can use `_id`, `qid`, or `id`; document IDs can use `_id`, `doc_id`,
or `id` (in that priority order). The loader converts IDs to strings, so a numeric
JSON ID `7` matches `7` in qrels. The `text` field remains required.

## Results

### Language analysis: run four benchmarks and create report.html

Run these commands from the repository root in Bash (Git Bash on Windows).
Install model dependencies once:

```bash
pip install -e ".[dense]"
```

Each benchmark has its own directory with language-suffixed files:

```text
data/benchmark_1/
  queries.vi.jsonl       {"qid":"q1","text":"Vietnamese query"}
  queries.en.jsonl       {"qid":"q1","text":"English query"}
  queries.csw.jsonl      {"qid":"q1","text":"Code-switched query"}
  documents.vi.jsonl     {"doc_id":"d1","text":"Vietnamese document"}
  documents.en.jsonl     {"doc_id":"d1","text":"English document"}
  qrels.tsv             q1<TAB>d1<TAB>2
```

Use `.vi.jsonl`, not `.vn.jsonl`: the current loader recognizes `vi`, `en`, `csw`
for queries and `vi`, `en` for documents. Matching `qid` values represent the same
information need; matching `doc_id` values represent the same document group.
Use one shared `qrels.tsv` per benchmark. Only requested document files are needed.
The analysis calls the existing `DatasetLoader` APIs without modifying the loader.
The same ID aliases work in split-language files, for example
`{"_id":7,"text":"query"}` and `{"_id":11,"text":"document"}` with
the qrels row `7<TAB>11<TAB>1`. Keep these IDs consistent across language views.

Replace the four dataset paths below with your actual benchmark folders:

```bash
bash scripts/run_analysis_four_benchmarks.sh \
    data/benchmark_1 data/benchmark_2 data/benchmark_3 data/benchmark_4 \
    --model intfloat/multilingual-e5-small \
    --documents vi en \
    --device auto
```

Defaults: `--queries vi en csw`, `--model-type single-vector`, `--retriever dense`,
`--top-k 10`, `--metrics ndcg@10 mrr@10 recall@10`, `--batch-size 32`.
Use `--documents vi` if you have only Vietnamese documents, or `--device cpu`
to run on CPU. Extra analysis options apply to all four benchmarks. Folder names
must be distinct, and one model/scorer is used across all panels.

The script stops on any failed analysis and creates:

```text
results/benchmark_1/analysis.json   results/benchmark_1/report.html
results/benchmark_2/analysis.json   results/benchmark_2/report.html
results/benchmark_3/analysis.json   results/benchmark_3/report.html
results/benchmark_4/analysis.json   results/benchmark_4/report.html
report.html                       Combined report: open this file in a browser
```

To change output locations or the Python executable:

```bash
PYTHON=python ANALYSIS_OUTPUT_ROOT=results/my_model ANALYSIS_REPORT_PATH=my_report.html \
    bash scripts/run_analysis_four_benchmarks.sh \
    data/benchmark_1 data/benchmark_2 data/benchmark_3 data/benchmark_4 \
    --documents vi --device cpu
```

### Run one benchmark or the toy example

```bash
bash scripts/run_analysis.sh \
    --dataset data/toy_multilingual \
    --queries vi en csw \
    --documents vi \
    --model intfloat/multilingual-e5-small \
    --top-k 10 \
    --metrics ndcg@10 mrr@10 recall@10 \
    --output results/toy_multilingual_analysis

# Source-checkout CLI with all options visible:
python scripts/run_analysis.py --help
```

`run_analysis.sh` defaults to `data/fiqa`, queries VI/EN/CSW, document indexes
VI/EN, and `results/analysis_fiqa`; the dataset must contain the split-language files
above. CLI options override these defaults. You can also set `PYTHON`,
`ANALYSIS_DATASET`, `ANALYSIS_MODEL`, and `ANALYSIS_OUTPUT` in the environment.

The underlying CLI also accepts `--direction vi-en en-en csw-en` instead of
`--queries vi en csw --documents en`. A direction means **query-document**:
`vi-en` uses Vietnamese queries against English documents. Do not combine
`--direction` with `--queries`/`--documents`. Keep VI as a query baseline for deltas.

### Refresh the combined report without running models again

```bash
bash scripts/render_analysis_four_benchmarks.sh \
    results/benchmark_1/analysis.json results/benchmark_2/analysis.json \
    results/benchmark_3/analysis.json results/benchmark_4/analysis.json

# Optional alternate HTML path:
bash scripts/render_analysis_four_benchmarks.sh \
    results/benchmark_1/analysis.json results/benchmark_2/analysis.json \
    results/benchmark_3/analysis.json results/benchmark_4/analysis.json \
    --output my_report.html

# Refresh a single report:
python scripts/render_analysis.py results/toy_multilingual_analysis/analysis.json
```

This step reads saved JSON only. It does not load models or repeat retrieval.
The four input reports must have distinct dataset paths and the same model/scorer.
Benchmarks and document indexes remain separate; the HTML contains:

- One horizontal row of four VI-CSW/EN-CSW gap violin/box panels.
- One horizontal row of four scatter panels using **delta A versus delta M**,
  with both reference lines at zero. Select CSW or EN versus VI and a fixed index.
  Delta A averages score changes over the same relevant document groups per query;
  delta M is `M(variant) - M(VI)`. Zero-axis points are counted separately.
- A/M boxplot rows across the four benchmarks, each showing VI, CSW and EN.
- A two-column embedding-space grid with one joint PCA per benchmark, matched
  query triplets, language colors, explained variance, rotation and zoom.
- Retrieval metrics and data policies under a collapsible details section.

HTML works offline without plotting dependencies. Small samples show observations
and boxes; violin density is estimated only for at least 20 nonconstant values.
Cosine gaps use original embeddings, not the PCA coordinates.

`A = max positive score`, `B = max shared hard-negative score`, `M = A - B`.
The shared negative pool is the union of non-positive documents from the selected
query views' top-k on the same fixed index. Positives outside top-k retain scores;
unjudged negatives are counted separately. Missing values are null and excluded.
Old reports without shared `negative_pool_ids` need a fresh analysis run.
Optional `--reranker MODEL` changes alignment/margin scoring while gaps/PCA still
use the single-vector encoder. Full-corpus scoring reuses existing retrievers.

### Retrieval pipeline outputs

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

