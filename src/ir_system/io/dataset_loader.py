"""
IO: DatasetLoader
==================
Loads queries, documents, and qrels from a dataset directory.

Supported layouts
-----------------
**Legacy (single-file)**::

    <dataset_dir>/
        queries.jsonl      {"qid": "...", "text": "..."}
        documents.jsonl    {"doc_id": "...", "text": "..."}
        qrels.tsv          qid<TAB>doc_id<TAB>relevance

**Multilingual (split files)**::

    <dataset_dir>/
        queries.vi.jsonl    Vietnamese queries     (optional)
        queries.en.jsonl    English queries        (optional)
        queries.csw.jsonl   Code-switched queries  (optional)
        documents.vi.jsonl  Vietnamese documents   (optional)
        documents.en.jsonl  English documents      (optional)
        qrels.tsv           language-agnostic relevance judgements

Language variants
-----------------
- Queries  : ``vi``, ``en``, ``csw``
- Documents: ``vi``, ``en``

The loader detects which files are present and returns the requested subset.
The ``lang`` argument to :meth:`load_queries` / :meth:`load_documents` may be:

* ``"all"``       - load every available language file (default)
* A list of tags  - e.g. ``["vi", "en"]``

qrels are language-agnostic: qid / doc_id values are shared across all
language files, so a single ``qrels.tsv`` covers all combinations.

Validation (mandatory)
----------------------
1. Files must exist.
2. Each JSON line must have required fields.
3. qid must be unique per language file.
4. doc_id must be unique per language file.
5. Malformed rows raise ValueError with file and line number.
6. qrels references are checked; unknown doc_ids are logged as warnings
   (configurable: raise error or warn).

Query and Document objects do NOT read files themselves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ir_system.domain.document import Document
from ir_system.domain.qrels import Qrels
from ir_system.domain.query import Query

logger = logging.getLogger(__name__)

# Canonical language tags
QUERY_LANGS: Tuple[str, ...] = ("vi", "en", "csw")
DOC_LANGS: Tuple[str, ...] = ("vi", "en")

LangSpec = Union[str, Sequence[str]]  # "all" | ["vi", "en"] | "vi" ...


def _resolve_lang_spec(
    spec: LangSpec,
    supported: Tuple[str, ...],
) -> List[str]:
    """Return an ordered list of language tags from a spec string or list."""
    if isinstance(spec, str):
        if spec.lower() == "all":
            return list(supported)
        spec = [spec]
    unknown = [s for s in spec if s not in supported]
    if unknown:
        raise ValueError(
            f"Unsupported language tag(s): {unknown!r}. "
            f"Supported: {list(supported)!r}"
        )
    return [s for s in supported if s in spec]  # keep canonical order


class DatasetLoader:
    """
    Loads a (potentially multilingual) IR dataset from a directory.

    Parameters
    ----------
    strict_qrels : bool
        If True, raise an error when qrels reference unknown doc_ids.
        If False (default), log a warning and skip unknown references.
    """

    def __init__(self, strict_qrels: bool = False) -> None:
        self._strict_qrels = strict_qrels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        dataset_dir: str | Path,
    ) -> Tuple[Sequence[Query], Sequence[Document], Qrels]:
        """
        Legacy compatibility - load a single-language dataset.

        Falls back to split-language files if ``queries.jsonl`` /
        ``documents.jsonl`` are not found (loads *all* available langs and
        merges them).

        Returns
        -------
        (queries, documents, qrels)
        """
        dataset_dir = Path(dataset_dir)
        self._validate_dir(dataset_dir)

        # Try legacy single-file first
        q_path = dataset_dir / "queries.jsonl"
        d_path = dataset_dir / "documents.jsonl"

        if q_path.exists() and d_path.exists():
            return self._load_legacy(dataset_dir, q_path, d_path)

        # Fall back to split multilingual files
        logger.info(
            "[DATASET] No single-file queries/documents found -- "
            "loading all available language variants."
        )
        queries_by_lang = self.load_queries(dataset_dir, lang="all")
        docs_by_lang = self.load_documents(dataset_dir, lang="all")
        queries = self._merge_queries(queries_by_lang)
        documents = self._merge_documents(docs_by_lang)
        qrels = self._load_qrels_file(
            dataset_dir / "qrels.tsv",
            known_doc_ids={doc.doc_id for doc in documents},
        )
        return queries, documents, qrels

    def load_queries(
        self,
        dataset_dir: str | Path,
        lang: LangSpec = "all",
    ) -> Dict[str, List[Query]]:
        """
        Load queries for the requested language(s).

        Parameters
        ----------
        dataset_dir : str | Path
        lang        : "all" | str | list[str]
            Which language variants to load.  ``"all"`` loads every file
            that is present on disk.

        Returns
        -------
        dict mapping lang_tag -> list[Query]
        Only tags whose files are present are included.
        """
        dataset_dir = Path(dataset_dir)
        self._validate_dir(dataset_dir)
        tags = _resolve_lang_spec(lang, QUERY_LANGS)

        result: Dict[str, List[Query]] = {}
        for tag in tags:
            path = dataset_dir / f"queries.{tag}.jsonl"
            if not path.exists():
                logger.debug(
                    "[DATASET] queries.%s.jsonl not found -- skipping.", tag
                )
                continue
            logger.info("[DATASET] Loading %s queries from %s", tag, path)
            qs = self._load_queries_file(path)
            logger.info("[DATASET] Loaded %d %s queries.", len(qs), tag)
            result[tag] = qs

        if not result:
            raise FileNotFoundError(
                f"DatasetLoader: no query files found in {dataset_dir} "
                f"for language(s) {tags!r}.\n"
                f"Expected one or more of: "
                + ", ".join(f"queries.{t}.jsonl" for t in tags)
            )
        return result

    def load_documents(
        self,
        dataset_dir: str | Path,
        lang: LangSpec = "all",
    ) -> Dict[str, List[Document]]:
        """
        Load documents for the requested language(s).

        Parameters
        ----------
        dataset_dir : str | Path
        lang        : "all" | str | list[str]

        Returns
        -------
        dict mapping lang_tag -> list[Document]
        Only tags whose files are present are included.
        """
        dataset_dir = Path(dataset_dir)
        self._validate_dir(dataset_dir)
        tags = _resolve_lang_spec(lang, DOC_LANGS)

        result: Dict[str, List[Document]] = {}
        for tag in tags:
            path = dataset_dir / f"documents.{tag}.jsonl"
            if not path.exists():
                logger.debug(
                    "[DATASET] documents.%s.jsonl not found -- skipping.", tag
                )
                continue
            logger.info("[DATASET] Loading %s documents from %s", tag, path)
            docs = self._load_documents_file(path)
            logger.info("[DATASET] Loaded %d %s documents.", len(docs), tag)
            result[tag] = docs

        if not result:
            raise FileNotFoundError(
                f"DatasetLoader: no document files found in {dataset_dir} "
                f"for language(s) {tags!r}.\n"
                f"Expected one or more of: "
                + ", ".join(f"documents.{t}.jsonl" for t in tags)
            )
        return result

    def load_qrels(
        self,
        dataset_dir: str | Path,
        known_doc_ids: Optional[set] = None,
    ) -> Qrels:
        """Load qrels.tsv from the dataset directory."""
        dataset_dir = Path(dataset_dir)
        self._validate_dir(dataset_dir)
        qrels_path = dataset_dir / "qrels.tsv"
        self._check_file_exists(qrels_path)
        return self._load_qrels_file(qrels_path, known_doc_ids or set())

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_queries(
        queries_by_lang: Dict[str, List[Query]],
    ) -> List[Query]:
        """Merge multiple language lists into a single deduplicated list."""
        seen: set[str] = set()
        merged: List[Query] = []
        for lang, qs in queries_by_lang.items():
            for q in qs:
                if q.qid not in seen:
                    seen.add(q.qid)
                    merged.append(q)
                else:
                    logger.debug(
                        "[DATASET] Duplicate qid %r across langs (kept first).", q.qid
                    )
        return merged

    @staticmethod
    def _merge_documents(
        docs_by_lang: Dict[str, List[Document]],
    ) -> List[Document]:
        """Merge multiple language lists into a single deduplicated list."""
        seen: set[str] = set()
        merged: List[Document] = []
        for lang, docs in docs_by_lang.items():
            for doc in docs:
                if doc.doc_id not in seen:
                    seen.add(doc.doc_id)
                    merged.append(doc)
                else:
                    logger.debug(
                        "[DATASET] Duplicate doc_id %r across langs (kept first).",
                        doc.doc_id,
                    )
        return merged

    # ------------------------------------------------------------------
    # Legacy loader
    # ------------------------------------------------------------------

    def _load_legacy(
        self,
        dataset_dir: Path,
        queries_path: Path,
        documents_path: Path,
    ) -> Tuple[List[Query], List[Document], Qrels]:
        qrels_path = dataset_dir / "qrels.tsv"
        self._check_file_exists(qrels_path)

        logger.info("[DATASET] Loading queries from %s", queries_path)
        queries = self._load_queries_file(queries_path)
        logger.info("[DATASET] Loaded %d queries.", len(queries))

        logger.info("[DATASET] Loading documents from %s", documents_path)
        documents = self._load_documents_file(documents_path)
        logger.info("[DATASET] Loaded %d documents.", len(documents))

        logger.info("[DATASET] Loading qrels from %s", qrels_path)
        doc_id_set = {doc.doc_id for doc in documents}
        qrels = self._load_qrels_file(qrels_path, doc_id_set)
        logger.info("[DATASET] Loaded %d qrel judgments.", len(qrels))

        return queries, documents, qrels

    # ------------------------------------------------------------------
    # Private file-level loaders
    # ------------------------------------------------------------------

    def _validate_dir(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"DatasetLoader: dataset directory not found: {path}"
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"DatasetLoader: path is not a directory: {path}"
            )

    def _check_file_exists(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"DatasetLoader: required file not found: {path}"
            )

    def _load_queries_file(self, path: Path) -> List[Query]:
        queries: List[Query] = []
        seen_qids: set[str] = set()

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- invalid JSON: {exc}"
                    ) from exc

                for field in ("qid", "text"):
                    if field not in obj:
                        raise ValueError(
                            f"Dataset validation failed:\n"
                            f"  {path}:{lineno} -- missing required field {field!r}"
                        )

                qid = str(obj["qid"])
                text = str(obj["text"])

                if qid in seen_qids:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- duplicate qid={qid!r}"
                    )
                seen_qids.add(qid)

                try:
                    queries.append(Query(qid=qid, text=text))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- {exc}"
                    ) from exc

        if not queries:
            raise ValueError(
                f"Dataset validation failed:\n"
                f"  {path} -- file is empty or contains no valid queries."
            )
        return queries

    def _load_documents_file(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        seen_doc_ids: set[str] = set()

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- invalid JSON: {exc}"
                    ) from exc

                for field in ("doc_id", "text"):
                    if field not in obj:
                        raise ValueError(
                            f"Dataset validation failed:\n"
                            f"  {path}:{lineno} -- missing required field {field!r}"
                        )

                doc_id = str(obj["doc_id"])
                text = str(obj["text"])

                if doc_id in seen_doc_ids:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- duplicate doc_id={doc_id!r}"
                    )
                seen_doc_ids.add(doc_id)

                try:
                    documents.append(Document(doc_id=doc_id, text=text))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- {exc}"
                    ) from exc

        if not documents:
            raise ValueError(
                f"Dataset validation failed:\n"
                f"  {path} -- file is empty or contains no valid documents."
            )
        return documents

    def _load_qrels_file(
        self,
        path: Path,
        known_doc_ids: set[str],
    ) -> Qrels:
        qrels = Qrels()
        unknown_doc_refs: List[str] = []

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- expected 3 tab-separated fields "
                        f"(qid, doc_id, relevance), got {len(parts)}: {line!r}"
                    )

                qid, doc_id = parts[0].strip(), parts[1].strip()
                rel_str = parts[2].strip()

                if not qid:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- qid is empty"
                    )
                if not doc_id:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- doc_id is empty"
                    )

                try:
                    relevance = float(rel_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} -- relevance must be numeric, "
                        f"got {rel_str!r}: {exc}"
                    ) from exc

                if known_doc_ids and doc_id not in known_doc_ids:
                    msg = (
                        f"Dataset validation:\n"
                        f"  {path}:{lineno} references unknown doc_id={doc_id!r}"
                    )
                    unknown_doc_refs.append(msg)
                    if self._strict_qrels:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                        continue

                qrels.add(qid=qid, doc_id=doc_id, relevance=relevance)

        if unknown_doc_refs and not self._strict_qrels:
            logger.warning(
                "[DATASET] %d qrels entries referenced unknown doc_ids (skipped).",
                len(unknown_doc_refs),
            )

        return qrels
