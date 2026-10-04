"""
IO: DatasetLoader
==================
Loads queries, documents, and qrels from a dataset directory.

Expected layout
---------------
    <dataset_dir>/
        queries.jsonl      {"qid": "...", "text": "..."}
        documents.jsonl    {"doc_id": "...", "text": "..."}
        qrels.tsv          qid<TAB>doc_id<TAB>relevance

Validation (mandatory)
----------------------
1. Files must exist.
2. Each JSON line must have required fields.
3. qid must be unique across queries.
4. doc_id must be unique across documents.
5. Malformed rows raise ValueError with file and line number.
6. qrels references are checked; unknown doc_ids logged as warnings
   (configurable: raise error or warn).

Query and Document objects do NOT read files themselves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

from ir_system.domain.document import Document
from ir_system.domain.qrels import Qrels
from ir_system.domain.query import Query

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads a standard IR dataset from a directory.

    Parameters
    ----------
    strict_qrels : bool
        If True, raise an error when qrels reference unknown doc_ids.
        If False (default), log a warning and skip unknown references.
    """

    def __init__(self, strict_qrels: bool = False) -> None:
        self._strict_qrels = strict_qrels

    def load(
        self,
        dataset_dir: str | Path,
        *,
        language: str | None = None,
    ) -> Tuple[Sequence[Query], Sequence[Document], Qrels]:
        """
        Load full dataset from directory.

        Parameters
        ----------
        dataset_dir : str | Path
        language : str, optional
            Read queries.<language>.jsonl and documents.<language>.jsonl.
            Use qrels.<language>.tsv if present, otherwise shared qrels.tsv.

        Returns
        -------
        (queries, documents, qrels)

        Raises
        ------
        FileNotFoundError  If required file is missing.
        ValueError         If data is malformed (includes file + line number).
        """
        queries = self.load_queries(dataset_dir, language=language)
        documents, qrels = self.load_corpus(dataset_dir, language=language)
        return queries, documents, qrels

    def load_queries(self, dataset_dir: str | Path, *, language: str | None = None) -> Sequence[Query]:
        """Load query views independently of the corpus language."""
        path = self.query_path(dataset_dir, language=language)
        self._check_file_exists(path)
        logger.info("[DATASET] Loading queries from %s", path)
        return self._load_queries(path)

    def query_path(self, dataset_dir: str | Path, *, language: str | None = None) -> Path:
        """Resolve a query filename, also usable to discover optional views.

        Vietnamese vi/vn suffixes are aliases. Prefer the explicitly requested
        spelling when both exist. A missing view returns its expected path.
        """
        return self._language_file(dataset_dir, "queries", "jsonl", language)

    def load_corpus(
        self, dataset_dir: str | Path, *, language: str | None = None,
    ) -> tuple[Sequence[Document], Qrels]:
        """Load a fixed document-language index and its relevance judgments."""
        documents_path = self._language_file(dataset_dir, "documents", "jsonl", language)
        qrels_path = self._language_file(dataset_dir, "qrels", "tsv", language)
        if language and not qrels_path.is_file():
            qrels_path = documents_path.parent / "qrels.tsv"
        self._check_file_exists(documents_path)
        self._check_file_exists(qrels_path)
        logger.info("[DATASET] Loading documents from %s", documents_path)
        documents = self._load_documents(documents_path)
        logger.info("[DATASET] Loading qrels from %s", qrels_path)
        qrels = self._load_qrels(qrels_path, {doc.doc_id for doc in documents})
        return documents, qrels

    def _language_file(self, dataset_dir, stem, extension, language) -> Path:
        directory, suffix = self._language_path(dataset_dir, language)
        path = directory / f"{stem}{suffix}.{extension}"
        if not path.is_file() and language in {"vi", "vn"}:
            alias = "vn" if language == "vi" else "vi"
            alias_path = directory / f"{stem}.{alias}.{extension}"
            if alias_path.is_file():
                return alias_path
        return path

    def _language_path(self, dataset_dir, language):
        directory = Path(dataset_dir)
        self._validate_dir(directory)
        if language is not None and language not in {"vn", "vi", "en", "csw"}:
            raise ValueError(f"Unsupported dataset language: {language!r}")
        return directory, f".{language}" if language else ""

    # ------------------------------------------------------------------
    # Private helpers
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

    def _load_queries(self, path: Path) -> list[Query]:
        queries: list[Query] = []
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
                        f"  {path}:{lineno} — invalid JSON: {exc}"
                    ) from exc

                if "qid" not in obj:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — missing required field 'qid'"
                    )
                if "text" not in obj:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — missing required field 'text'"
                    )

                qid = str(obj["qid"])
                text = str(obj["text"])

                if qid in seen_qids:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — duplicate qid={qid!r}"
                    )
                seen_qids.add(qid)

                try:
                    queries.append(Query(qid=qid, text=text))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — {exc}"
                    ) from exc

        if not queries:
            raise ValueError(
                f"Dataset validation failed:\n"
                f"  {path} — file is empty or contains no valid queries."
            )

        return queries

    def _load_documents(self, path: Path) -> list[Document]:
        documents: list[Document] = []
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
                        f"  {path}:{lineno} — invalid JSON: {exc}"
                    ) from exc

                if "doc_id" not in obj:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — missing required field 'doc_id'"
                    )
                if "text" not in obj:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — missing required field 'text'"
                    )

                doc_id = str(obj["doc_id"])
                text = str(obj["text"])

                if doc_id in seen_doc_ids:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — duplicate doc_id={doc_id!r}"
                    )
                seen_doc_ids.add(doc_id)

                try:
                    documents.append(Document(doc_id=doc_id, text=text))
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — {exc}"
                    ) from exc

        if not documents:
            raise ValueError(
                f"Dataset validation failed:\n"
                f"  {path} — file is empty or contains no valid documents."
            )

        return documents

    def _load_qrels(self, path: Path, known_doc_ids: set[str]) -> Qrels:
        qrels = Qrels()
        unknown_doc_refs: list[str] = []

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — expected 3 tab-separated fields "
                        f"(qid, doc_id, relevance), got {len(parts)}: {line!r}"
                    )

                qid, doc_id = parts[0].strip(), parts[1].strip()
                rel_str = parts[2].strip()

                if not qid:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — qid is empty"
                    )
                if not doc_id:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — doc_id is empty"
                    )

                try:
                    relevance = float(rel_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset validation failed:\n"
                        f"  {path}:{lineno} — relevance must be numeric, "
                        f"got {rel_str!r}: {exc}"
                    ) from exc

                if doc_id not in known_doc_ids:
                    msg = (
                        f"Dataset validation failed:\n"
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
