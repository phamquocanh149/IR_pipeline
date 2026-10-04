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

## Results

### Paired Vietnamese / English / code-switched analysis

Generate a self-contained interactive HTML report and detailed JSON:

A ready-to-edit Bash example is available in `scripts/run_analysis.sh`.
Set `DATASET`, `QUERIES`, `DOCUMENTS`, `MODEL`, `TOP_K`, `BATCH_SIZE`, `DEVICE`, and `OUTPUT`
at the top of the file, then run:

```bash
bash scripts/run_analysis.sh
```

The example uses `--queries vi en csw --documents en` (three query languages,
English documents) and `intfloat/multilingual-e5-small`. Set `PYTHON=python3` if needed.
Optional `RERANKER` selects
cross-encoder scores. CLI arguments can override the example configuration:

```bash
bash scripts/run_analysis.sh --model "YOUR_MODEL" --queries vi csw --documents en --device cpu
```

Run the script from the repository root (it sets up the local `src` import path
and forwards all arguments to the existing analysis CLI):

```powershell
python scripts/run_analysis.py --dataset data/fiqa --direction vi-en csw-en --model "MODEL_ID" --top-k 100 --output results/analysis
```

View all CLI options with `python scripts/run_analysis.py --help`.
Replace `MODEL_ID` with your embedding model identifier or local model path.
The model dependencies still require `pip install -e ".[dense]"`.

```bash
python -m ir_system.cli.analysis \
    --dataset data/fiqa \
    --direction vi-en csw-en \
    --model YOUR_MULTILINGUAL_EMBEDDING_MODEL \
    --top-k 100 \
    --output results/language_analysis
```

Requires the `dense` extra. `--direction` accepts one or more **query-document**
language pairs: `vi-en` means Vietnamese queries against English documents;
`csw-vi` means code-switched queries against Vietnamese documents. Query languages
are `vi`, `en`, `csw`; document languages are `vi`, `en`, following the loader's
`QUERY_LANGS` and `DOC_LANGS` (six directions).

Alternatively use separate `--queries` and `--documents` selectors:

```bash
python -m ir_system.cli.analysis \
    --dataset data/fiqa \
    --queries csw \
    --documents vi \
    --model intfloat/multilingual-e5-small \
    --model-type single-vector \
    --retriever dense \
    --top-k 10 \
    --metrics ndcg@10 mrr@10 recall@10
```

Each selector accepts one or more languages, or `all`. All selected query/document
combinations are analyzed. Use these selectors or `--direction`, not both.
Optional `--metrics` reuses the existing evaluator on top-k retrieval results
and includes scores in JSON and HTML without repeating retrieval.

For example, `--direction vi-en csw-en` builds only the English corpus and
computes Vietnamese and code-switched query margins on that fixed index, plus
their alignment/margin changes. To compare all three query views on English
documents, use `--direction vi-en en-en csw-en`.

All language views live in one dataset directory:

```text
data/fiqa/
├── queries.vi.jsonl
├── queries.en.jsonl
├── queries.csw.jsonl
├── documents.vi.jsonl
├── documents.en.jsonl
└── qrels.tsv
```

Use the same layout for any dataset name. JSONL record fields remain `qid`/`text`
and `doc_id`/`text`. Analysis uses the existing multilingual loader API unchanged:
`load_queries(..., lang="all")`, `load_documents(..., lang=[...])`, and
`load_qrels(...)`. Files use canonical `.vi`, `.en`, `.csw` query suffixes and
`.vi`, `.en` document suffixes. A single language-agnostic `qrels.tsv` applies to
all directions. Query/document views remain separate rather than using the
legacy loader's merged result. Requested views must be present even though the
loader itself skips missing optional files.
Only document files for the requested pairs are required. `vi-en` requires
`queries.vi.jsonl`, `documents.en.jsonl`, and shared `qrels.tsv`.
Other available query files are included in the cosine/3D diagnostics but are
not searched unless their query-document direction is requested.

Shared `qid` must denote the same
information need and shared `doc_id` the same document group; matching is by ID,
never row order. If your views use different IDs, remap them to shared group IDs
before running. Cosine pairs use shared query IDs; margins also cover query IDs
that have no counterpart in the other requested views.

Open `results/language_analysis/report.html` in a browser. No network or plotting
dependency is required. It contains:

- A draggable, zoomable 3D query scatter plot with language filters, query text
  tooltips, and lines linking the three versions of each information need.
  One PCA is fit jointly to available query views; explained variance is shown.
- Query cosine similarity and representation gaps `1 - cosine(E(view1), E(view2))`
  for available pairs `vi/en`, `vi/csw`, `en/csw`, with mean, median and
  distributions. Document gaps are computed only between requested corpus views.
  Gaps are computed in the original embedding space, not the PCA projection.
- Positive alignment `delta_A = s(variant, positive) - s(vi, positive)` for every
  relevant document group, separately on each requested fixed document index.
- Relevance margins `M = max_positive_score - max_hard_negative_score` and
  `delta_M = M(variant) - M(vi)`, with distributions and per-query details.

By default `s` is cosine from the specified single-vector encoder. Optionally
pass `--reranker YOUR_CROSS_ENCODER` to compute alignment/margins using a
cross-encoder, while representation gaps and PCA still use `--model`.
This analysis does not represent BM25, Hybrid RRF, or late-interaction scores.
Cross-encoder scoring covers the entire corpus and can be expensive.
The analysis CLI reuses `DatasetLoader`, `create_model`, `DenseRetriever`,
`CrossEncoderRetriever`, `SearchPipeline`, `Run`, and `Qrels`. Corpus embeddings
are built once per index and reused for document gaps; scoring and ranking stay
in the existing retrievers. `ir-analyze` is the installed command equivalent.

Positives are all judgments with relevance greater than zero in the fixed index,
including positives outside top-k. Hard negatives are non-positive documents in
**each query variant's own top-k**; unjudged documents are treated as non-relevant
and their counts are reported. No positive or no hard negative gives a null margin
excluded from means/medians. Alignment summaries average query-positive pairs;
margin summaries average queries per requested direction. Scores are never pooled
across indexes. Delta alignment/margin requires a requested `vi-<document>`
baseline and matching qid on the same corpus; otherwise deltas are null.
Gaps use pairwise shared IDs; the scatter plot shows all available query views.
`analysis.json` includes unpaired query IDs, coverage, requested directions,
individual gaps, positive scores, margins, projection coordinates and configuration.

Large representation gaps alone do not establish retrieval degradation. Negative
alignment changes indicate positive score loss; negative margin changes indicate
weaker separation from the strongest retrieved negative. These are distinct effects.

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

