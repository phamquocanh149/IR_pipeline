# CQA dataset (no Git LFS required)

After cloning, run this command from the repository root **before running the pipeline**:

```sh
python data/exports/cqa/prepare.py
```

Requires Python 3.10+ only; no additional packages or downloads are needed.
Then run the pipeline as usual with `data/exports/cqa` as the dataset directory.
Cloning alone does not assemble the document files automatically.

The two original document files exceed GitHub's per-file limit. They are stored
as ordered JSONL parts smaller than 40 MiB, containing every original document.
The script restores `documents.en.jsonl` and `documents.vi.jsonl` and verifies
their exact original bytes using SHA-256, byte sizes, and line counts recorded in
`documents.manifest.json`. It handles both LF and CRLF Git checkouts and can be
run again safely. A failed integrity check does not replace an existing file.
Each language contains 121,184 documents. Queries and qrels remain unchanged.

The assembled files are ignored by Git. Commit the parts and manifest when
updating the corpus; do not add the assembled files or re-enable LFS.

Older commits still reference LFS; use the latest branch for this layout.
