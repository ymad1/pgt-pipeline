# pgt/io.py
"""Safe JSONL I/O for the PGT pipeline.

This module is the file-boundary companion to :mod:`pgt.schema`.  It keeps the
legacy ``read_jsonl(path)`` and ``write_jsonl(path, rows)`` interfaces while
adding:

* line-aware JSON and record errors;
* optional or automatically inferred schema validation;
* duplicate ``input_id`` detection for pipeline-stage files;
* atomic writes, so a failed validation never leaves a half-written result;
* deterministic JSON serialization for reproducible hashes; and
* per-record validation context for cross-file checks (e.g. MES vs. graph).

Validation modes
----------------
``record_kind="auto"`` (the default) validates records only when their structure
unambiguously identifies one of the six pipeline contracts.  Raw dataset rows,
labels, ATT&CK indexes, and other generic JSONL files remain readable.

For production stage boundaries, pass an explicit kind, for example::

    rows = read_jsonl("extraction.jsonl", record_kind="extraction")
    write_jsonl("mes.jsonl", mes_rows, record_kind="mes")

The explicit form is preferred because malformed records that have lost their
stage-identifying fields cannot evade validation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableSet,
    Optional,
    TypeAlias,
    Union,
)

from .schema import RecordKind, RecordValidationError, assert_valid_record


PathLike: TypeAlias = Union[str, os.PathLike[str]]
Record: TypeAlias = Dict[str, Any]
SchemaMode: TypeAlias = Union[RecordKind, Literal["auto"], None]
ValidationKwargs: TypeAlias = Union[
    Mapping[str, Any],
    Callable[[Mapping[str, Any], int], Mapping[str, Any]],
]


class JsonlError(ValueError):
    """Base class for JSONL boundary errors."""


class JsonlDecodeError(JsonlError):
    """Raised when one physical JSONL line is not valid JSON."""

    def __init__(self, path: Path, line_number: int, error: json.JSONDecodeError):
        self.path = path
        self.line_number = line_number
        self.original = error
        super().__init__(
            f"Invalid JSON in {path} at line {line_number}, column {error.colno}: "
            f"{error.msg}"
        )


class JsonlRecordError(JsonlError):
    """Raised when a parsed line is not an acceptable JSONL record."""

    def __init__(
        self,
        path: Path,
        line_number: int,
        message: str,
        *,
        input_id: Optional[str] = None,
    ):
        self.path = path
        self.line_number = line_number
        self.input_id = input_id
        suffix = f", input_id={input_id!r}" if input_id else ""
        super().__init__(f"Invalid record in {path} at line {line_number}{suffix}: {message}")


class DuplicateInputIdError(JsonlRecordError):
    """Raised when a stage file contains the same ``input_id`` twice."""


# ---------------------------------------------------------------------------
# Schema inference and validation helpers
# ---------------------------------------------------------------------------


def infer_record_kind(record: Mapping[str, Any]) -> Optional[RecordKind]:
    """Infer a pipeline record kind only when the structure is unambiguous.

    Deliberately *not* inferred as ``sentences`` are early/raw dataset rows that
    contain only ``input_id``, ``raw_text`` and ``sentences``.  The enriched
    evidence-segmentation contract is inferred only when ``evidence_spans`` and
    ``segmentation`` are also present.
    """

    keys = set(record)

    # MES and local graph both contain nodes/edges; inspect MES first.
    if {
        "status",
        "complete_core_chain",
        "nodes",
        "edges",
        "structural_node_ids",
    }.issubset(keys):
        return "mes"

    if {"nodes", "edges", "stats"}.issubset(keys):
        return "local_graph"

    if {
        "preconditions",
        "entry",
        "vuln_type",
        "behaviors",
        "impacts",
        "relations",
    }.issubset(keys):
        return "extraction"

    if {
        "raw_text",
        "sentences",
        "evidence_spans",
        "segmentation",
    }.issubset(keys):
        return "sentences"

    candidates = record.get("candidates")
    if isinstance(candidates, list) and (
        "rerank_metadata" in record
        or any(
            isinstance(item, Mapping)
            and any(key in item for key in ("final_score", "llm_score", "rerank_rank"))
            for item in candidates
        )
    ):
        return "reranking"

    if isinstance(candidates, list) and (
        "retrieval_metadata" in record
        or any(
            isinstance(item, Mapping)
            and any(key in item for key in ("score_fused", "score_text", "score_structure"))
            for item in candidates
        )
    ):
        return "candidates"

    return None


def _validation_kwargs_for(
    provider: Optional[ValidationKwargs],
    record: Mapping[str, Any],
    line_number: int,
) -> Dict[str, Any]:
    if provider is None:
        return {}
    value = provider(record, line_number) if callable(provider) else provider
    if not isinstance(value, Mapping):
        raise TypeError("validation_kwargs must be a mapping or return a mapping")
    return dict(value)


def _validate_record(
    *,
    path: Path,
    line_number: int,
    record: Mapping[str, Any],
    record_kind: SchemaMode,
    validate: bool,
    validation_kwargs: Optional[ValidationKwargs],
) -> Optional[RecordKind]:
    if not validate or record_kind is None:
        return None

    resolved: Optional[RecordKind]
    if record_kind == "auto":
        resolved = infer_record_kind(record)
    else:
        resolved = record_kind

    if resolved is None:
        return None

    kwargs = _validation_kwargs_for(validation_kwargs, record, line_number)
    try:
        assert_valid_record(resolved, record, **kwargs)
    except RecordValidationError as exc:
        input_id = record.get("input_id") if isinstance(record.get("input_id"), str) else None
        raise JsonlRecordError(
            path,
            line_number,
            f"{resolved} schema validation failed: {exc}",
            input_id=input_id,
        ) from exc
    return resolved


def _should_enforce_unique_ids(
    explicit: Optional[bool],
    resolved_kind: Optional[RecordKind],
) -> bool:
    if explicit is not None:
        return explicit
    # Stage artifacts are one-record-per-CVE. Generic JSONL files are not.
    return resolved_kind is not None


def _check_duplicate_input_id(
    *,
    path: Path,
    line_number: int,
    record: Mapping[str, Any],
    seen_ids: MutableSet[str],
    enforce: bool,
) -> None:
    if not enforce:
        return
    input_id = record.get("input_id")
    if not isinstance(input_id, str) or not input_id.strip():
        # The stage schema reports a clearer missing-ID error when applicable.
        return
    if input_id in seen_ids:
        raise DuplicateInputIdError(
            path,
            line_number,
            "duplicate input_id in the same JSONL file",
            input_id=input_id,
        )
    seen_ids.add(input_id)


# ---------------------------------------------------------------------------
# Public reading API
# ---------------------------------------------------------------------------


def read_jsonl(
    path: PathLike,
    *,
    record_kind: SchemaMode = "auto",
    validate: bool = True,
    validation_kwargs: Optional[ValidationKwargs] = None,
    require_object: bool = True,
    enforce_unique_input_ids: Optional[bool] = None,
) -> Iterator[Record]:
    """Stream JSON objects from a JSONL file.

    Parameters are keyword-only after ``path`` to preserve the original API.
    Blank physical lines are ignored. UTF-8 BOMs are accepted.

    ``validation_kwargs`` may be a mapping applied to every record or a callable
    ``(record, line_number) -> mapping``.  The callable form supports cross-file
    validation, for example supplying each MES record's source graph.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Expected a JSONL file, got directory: {p}")

    seen_ids: set[str] = set()
    detected_kind: Optional[RecordKind] = None

    with p.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, physical_line in enumerate(handle, start=1):
            stripped = physical_line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlDecodeError(p, line_number, exc) from exc

            if require_object and not isinstance(value, dict):
                raise JsonlRecordError(
                    p,
                    line_number,
                    f"top-level JSON value must be an object, got {type(value).__name__}",
                )
            if not isinstance(value, dict):
                # Type narrowing for callers that deliberately allow non-objects.
                yield value  # type: ignore[misc]
                continue

            resolved_kind = _validate_record(
                path=p,
                line_number=line_number,
                record=value,
                record_kind=record_kind,
                validate=validate,
                validation_kwargs=validation_kwargs,
            )

            if resolved_kind is not None:
                if detected_kind is None:
                    detected_kind = resolved_kind
                elif record_kind == "auto" and resolved_kind != detected_kind:
                    raise JsonlRecordError(
                        p,
                        line_number,
                        f"mixed pipeline record kinds: first {detected_kind}, now {resolved_kind}",
                        input_id=value.get("input_id") if isinstance(value.get("input_id"), str) else None,
                    )

            enforce = _should_enforce_unique_ids(
                enforce_unique_input_ids,
                resolved_kind if resolved_kind is not None else detected_kind,
            )
            _check_duplicate_input_id(
                path=p,
                line_number=line_number,
                record=value,
                seen_ids=seen_ids,
                enforce=enforce,
            )
            yield value


# ---------------------------------------------------------------------------
# Public writing API
# ---------------------------------------------------------------------------


def _json_line(record: Mapping[str, Any], *, canonical: bool) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=canonical,
        separators=(",", ":") if canonical else None,
    ) + "\n"


def write_jsonl(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    *,
    record_kind: SchemaMode = "auto",
    validate: bool = True,
    validation_kwargs: Optional[ValidationKwargs] = None,
    enforce_unique_input_ids: Optional[bool] = None,
    atomic: bool = True,
    canonical: bool = True,
) -> None:
    """Write JSON objects to JSONL, validating before each line is committed.

    By default the destination is replaced atomically only after every record has
    serialized and validated successfully.  Thus a schema error, duplicate ID,
    non-finite float, or iterator exception leaves an existing destination file
    untouched.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    if atomic:
        fd, raw_temp = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            text=True,
        )
        os.close(fd)
        target = Path(raw_temp)
        temp_path = target
    else:
        target = destination

    seen_ids: set[str] = set()
    detected_kind: Optional[RecordKind] = None

    try:
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for line_number, row in enumerate(rows, start=1):
                if not isinstance(row, Mapping):
                    raise JsonlRecordError(
                        destination,
                        line_number,
                        f"row must be a mapping, got {type(row).__name__}",
                    )
                record = dict(row)

                resolved_kind = _validate_record(
                    path=destination,
                    line_number=line_number,
                    record=record,
                    record_kind=record_kind,
                    validate=validate,
                    validation_kwargs=validation_kwargs,
                )
                if resolved_kind is not None:
                    if detected_kind is None:
                        detected_kind = resolved_kind
                    elif record_kind == "auto" and resolved_kind != detected_kind:
                        raise JsonlRecordError(
                            destination,
                            line_number,
                            f"mixed pipeline record kinds: first {detected_kind}, now {resolved_kind}",
                            input_id=record.get("input_id")
                            if isinstance(record.get("input_id"), str)
                            else None,
                        )

                enforce = _should_enforce_unique_ids(
                    enforce_unique_input_ids,
                    resolved_kind if resolved_kind is not None else detected_kind,
                )
                _check_duplicate_input_id(
                    path=destination,
                    line_number=line_number,
                    record=record,
                    seen_ids=seen_ids,
                    enforce=enforce,
                )

                try:
                    handle.write(_json_line(record, canonical=canonical))
                except (TypeError, ValueError) as exc:
                    input_id = record.get("input_id") if isinstance(record.get("input_id"), str) else None
                    raise JsonlRecordError(
                        destination,
                        line_number,
                        f"record is not JSON-serializable: {exc}",
                        input_id=input_id,
                    ) from exc

            handle.flush()
            os.fsync(handle.fileno())

        if atomic:
            os.replace(target, destination)
            temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def append_jsonl(
    path: PathLike,
    row: Mapping[str, Any],
    *,
    record_kind: SchemaMode = "auto",
    validate: bool = True,
    validation_kwargs: Optional[ValidationKwargs] = None,
    canonical: bool = True,
    fsync: bool = True,
) -> None:
    """Append one validated record.

    This helper is intended for resumable API jobs.  It validates the new row but
    does not rescan the existing file for duplicate IDs; resumable callers should
    maintain their own completed-ID set, as ``pgt.rerank`` already does.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(row, Mapping):
        raise JsonlRecordError(p, 1, f"row must be a mapping, got {type(row).__name__}")
    record = dict(row)
    _validate_record(
        path=p,
        line_number=1,
        record=record,
        record_kind=record_kind,
        validate=validate,
        validation_kwargs=validation_kwargs,
    )
    try:
        line = _json_line(record, canonical=canonical)
    except (TypeError, ValueError) as exc:
        input_id = record.get("input_id") if isinstance(record.get("input_id"), str) else None
        raise JsonlRecordError(
            p,
            1,
            f"record is not JSON-serializable: {exc}",
            input_id=input_id,
        ) from exc

    with p.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


__all__ = [
    "DuplicateInputIdError",
    "JsonlDecodeError",
    "JsonlError",
    "JsonlRecordError",
    "append_jsonl",
    "infer_record_kind",
    "read_jsonl",
    "write_jsonl",
]
