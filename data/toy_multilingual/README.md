# Toy multilingual dataset

Small, manually authored finance dataset for testing language-view analysis.
The questions and documents are illustrative test fixtures, not financial advice.

- 5 shared query IDs, each with Vietnamese, English and code-switched text.
- 12 shared document IDs, each with Vietnamese and English text.
- Shared `qrels.tsv`: two positive documents and one explicit negative per query.
- Relevance 2 is directly relevant, 1 is supporting context, 0 is non-relevant.
- Other unjudged documents are treated as negatives by the analysis pipeline.

`queries.vn.jsonl` and `documents.vn.jsonl` are the requested Vietnamese files.
Identical copies `queries.vi.jsonl` and `documents.vi.jsonl` provide compatibility
with the current unmodified loader, which recognizes `vi`, `en`, `csw` queries
and `vi`, `en` documents. CLI language tags remain `vi`, not `vn`.
Records align by `qid` / `doc_id`, not by language-specific IDs.

Run from the repository root:

```bash
python scripts/run_analysis.py \
    --dataset data/toy_multilingual \
    --queries vi en csw \
    --documents vi \
    --model intfloat/multilingual-e5-small \
    --model-type single-vector \
    --retriever dense \
    --top-k 10 \
    --metrics ndcg@10 mrr@10 recall@10 \
    --output results/toy_multilingual_analysis
```

Or use the Bash example with parameter overrides:

```bash
bash scripts/run_analysis.sh --dataset data/toy_multilingual --queries vi en csw --documents vi --output results/toy_multilingual_analysis
```

This dataset tests loading, pairing, visualization and metric calculations. Its
size and hand-written judgments are not suitable for evaluating model quality.
