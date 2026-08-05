# pgt/extract.py
"""Reproducible extraction stage with optional local-graph and MES construction.

This module is the controlled bridge between sentence-level evidence and the
structure-aware stages of the CVE-to-ATT&CK pipeline.  It provides:

* strict validation of ``sentences.jsonl``;
* incremental, resumable extraction output;
* explicit accounting of LLM versus rule-based fallback records;
* optional construction of local attack graphs and MES records using the same
  extraction records and sentence evidence;
* per-stage summaries, input/output hashes, and a run manifest;
* fail-fast defaults suitable for paper experiments.

The default policy is intentionally strict: rule-based fallback, validation
errors, and downstream construction failures stop the run.  They can be
allowed explicitly for smoke tests with ``--allow_fallback``,
``--allow_validation_errors``, or ``--continue_on_error``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tqdm import tqdm

from .build_local_graph import GRAPH_VERSION, build_local_graph_for_one
from .build_mes import ALGORITHM_VERSION as MES_ALGORITHM_VERSION
from .build_mes import _build_mes_record
from .io import read_jsonl
from .llm import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    EXTRACTION_PIPELINE_VERSION,
    PROMPT_VERSION,
    call_llm_extract,
)
from .schema import validate_evidence_ids


ORCHESTRATOR_VERSION = "extract-orchestrator-v2.0.0"
_EVIDENCE_ID_RE = re.compile(r"^E([1-9][0-9]*)$")
_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _safe_filename(input_id: str) -> str:
    cleaned = _SAFE_FILE_RE.sub("_", input_id).strip("._")
    if not cleaned:
        raise ValueError(f"input_id cannot be converted to a safe filename: {input_id!r}")
    return cleaned


def _read_ids(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"ID file not found: {path}")
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not ids:
        raise ValueError(f"ID file is empty: {path}")
    return ids


def _evidence_sort_key(eid: str) -> Tuple[int, str]:
    match = _EVIDENCE_ID_RE.fullmatch(eid)
    if match:
        return int(match.group(1)), eid
    return sys.maxsize, eid


# ---------------------------------------------------------------------------
# Sentence input validation
# ---------------------------------------------------------------------------


def _normalise_sentence_record(row: Mapping[str, Any], line_number: int) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    input_id = str(row.get("input_id", "")).strip()
    if not input_id:
        errors.append(f"line {line_number}: missing input_id")

    raw_sentences = row.get("sentences")
    if not isinstance(raw_sentences, Mapping) or not raw_sentences:
        errors.append(f"line {line_number} ({input_id or '<missing>'}): sentences must be a non-empty object")
        raw_sentences = {}

    sentences: Dict[str, str] = {}
    for raw_eid, raw_text in raw_sentences.items():
        eid = str(raw_eid).strip().upper()
        text = str(raw_text).strip() if raw_text is not None else ""
        if not _EVIDENCE_ID_RE.fullmatch(eid):
            errors.append(f"line {line_number} ({input_id}): invalid evidence id {raw_eid!r}")
            continue
        if not text:
            errors.append(f"line {line_number} ({input_id}): empty evidence text for {eid}")
            continue
        if eid in sentences:
            errors.append(f"line {line_number} ({input_id}): duplicate evidence id {eid}")
            continue
        sentences[eid] = text

    sentences = dict(sorted(sentences.items(), key=lambda item: _evidence_sort_key(item[0])))
    expected = [f"E{i}" for i in range(1, len(sentences) + 1)]
    actual = list(sentences)
    warnings: List[str] = []
    if sentences and actual != expected:
        warnings.append("non_contiguous_evidence_ids")

    normalised: Dict[str, Any] = {
        "input_id": input_id,
        "sentences": sentences,
    }
    raw_text = row.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        normalised["raw_text"] = raw_text.strip()
    if warnings:
        normalised["_input_warnings"] = warnings
    return normalised, errors


def _load_sentence_records(
    path: Path,
    selected_ids: Optional[set[str]],
    max_records: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"sentences.jsonl not found: {path}")

    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    errors: List[str] = []
    total_rows = 0
    skipped_by_id = 0
    warning_records = 0

    for line_number, row in enumerate(read_jsonl(str(path)), start=1):
        total_rows += 1
        if not isinstance(row, Mapping):
            errors.append(f"line {line_number}: JSON value must be an object")
            continue
        normalised, row_errors = _normalise_sentence_record(row, line_number)
        errors.extend(row_errors)
        input_id = normalised["input_id"]
        if input_id in seen:
            errors.append(f"line {line_number}: duplicate input_id {input_id}")
            continue
        seen.add(input_id)

        if selected_ids is not None and input_id not in selected_ids:
            skipped_by_id += 1
            continue
        if normalised.get("_input_warnings"):
            warning_records += 1
        records.append(normalised)
        if max_records is not None and len(records) >= max_records:
            break

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... {len(errors) - 20} more error(s)"
        raise ValueError(f"Invalid sentence input ({len(errors)} error(s)):\n{preview}{suffix}")
    if not records:
        raise ValueError("No sentence records selected for extraction")
    if selected_ids is not None:
        found_ids = {row["input_id"] for row in records}
        missing = sorted(selected_ids - found_ids)
        if missing:
            preview = ", ".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
            raise ValueError(f"Selected ID(s) not found in sentences input: {preview}{suffix}")

    return records, {
        "source_rows_read": total_rows,
        "records_selected": len(records),
        "records_skipped_by_id_filter": skipped_by_id,
        "records_with_input_warnings": warning_records,
    }


# ---------------------------------------------------------------------------
# Existing output / resume handling
# ---------------------------------------------------------------------------


def _load_existing_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return [], {}
    rows: List[Dict[str, Any]] = []
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(str(path)):
        if not isinstance(row, dict):
            raise ValueError(f"Existing output contains a non-object row: {path}")
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            raise ValueError(f"Existing output row missing input_id: {path}")
        if input_id in indexed:
            raise ValueError(f"Duplicate input_id in existing output: {input_id}")
        rows.append(row)
        indexed[input_id] = row
    return rows, indexed


def _prepare_outputs(
    output: Path,
    graph_dir: Optional[Path],
    mes_output: Optional[Path],
    overwrite: bool,
    resume: bool,
) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    paths = [output]
    if mes_output is not None:
        paths.append(mes_output)
    if not overwrite and not resume:
        existing = [str(path) for path in paths if path.exists()]
        if graph_dir is not None and graph_dir.exists() and any(
            p.is_file() and not p.name.startswith("_") for p in graph_dir.glob("*.json")
        ):
            existing.append(str(graph_dir))
        if existing:
            raise FileExistsError(
                "Output already exists. Use --resume or --overwrite: " + ", ".join(existing)
            )

    if overwrite:
        for path in paths:
            if path.exists():
                path.unlink()
        for suffix in (".summary.json", ".manifest.json"):
            candidate = output.with_suffix(output.suffix + suffix)
            if candidate.exists():
                candidate.unlink()
        if mes_output is not None:
            summary = mes_output.with_suffix(mes_output.suffix + ".summary.json")
            if summary.exists():
                summary.unlink()
        if graph_dir is not None and graph_dir.exists():
            for path in graph_dir.glob("*.json"):
                if path.is_file():
                    path.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    if graph_dir is not None:
        graph_dir.mkdir(parents=True, exist_ok=True)
    if mes_output is not None:
        mes_output.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _record_validation_errors(extraction: Dict[str, Any], valid_ids: set[str]) -> List[str]:
    errors = validate_evidence_ids(extraction, valid_ids)
    previous = extraction.get("_validation_errors") or []
    if not isinstance(previous, list):
        previous = [str(previous)]
    combined = [str(item) for item in previous] + errors
    deduped = list(dict.fromkeys(item for item in combined if item))
    extraction["_validation_errors"] = deduped
    return deduped


def _runtime_signature(extraction: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model/prompt settings that must not be mixed in one output."""
    provenance = extraction.get("_provenance")
    if not isinstance(provenance, Mapping):
        return {"mode": "missing_provenance"}
    keys = (
        "pipeline_version",
        "prompt_version",
        "requested_model",
        "temperature",
        "seed",
        "max_completion_tokens",
        "response_format",
        "max_attempts",
        "retry_base_seconds",
        "mode",
    )
    return {key: provenance.get(key) for key in keys}


def _enforce_extraction_policy(
    extraction: Dict[str, Any],
    valid_ids: set[str],
    *,
    allow_fallback: bool,
    allow_validation_errors: bool,
) -> Tuple[bool, List[str]]:
    validation_errors = _record_validation_errors(extraction, valid_ids)
    used_llm = bool(extraction.get("_used_llm"))
    if not used_llm and not allow_fallback:
        raise ValueError(
            "rule-based fallback was used; rerun with a working LLM client "
            "or pass --allow_fallback for a smoke test"
        )
    if validation_errors and not allow_validation_errors:
        raise ValueError(
            "extraction validation errors: " + "; ".join(validation_errors)
        )
    return used_llm, validation_errors


def _config_signature(args: argparse.Namespace, sentences_path: Path, ids_path: Optional[Path]) -> Dict[str, Any]:
    return {
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "default_model": DEFAULT_MODEL,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_seed": DEFAULT_SEED,
        "default_max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "default_max_attempts": DEFAULT_MAX_ATTEMPTS,
        "default_retry_base_seconds": DEFAULT_RETRY_BASE_SECONDS,
        "local_graph_version": GRAPH_VERSION,
        "mes_algorithm_version": MES_ALGORITHM_VERSION,
        "sentences_sha256": _sha256_file(sentences_path),
        "ids_file_sha256": _sha256_file(ids_path) if ids_path is not None else None,
        "allow_fallback": bool(args.allow_fallback),
        "allow_validation_errors": bool(args.allow_validation_errors),
        "continue_on_error": bool(args.continue_on_error),
        "build_local_graph": args.graph_dir is not None,
        "build_mes": args.mes_output is not None,
        "filter_aka_for_behavior_impact": not bool(args.no_filter_aka),
        "mes_parameters": {
            "max_path_nodes": args.mes_max_path_nodes,
            "exact_cover_limit": args.mes_exact_cover_limit,
            "include_precondition": bool(args.mes_include_precondition),
        },
    }


def _validate_resume_manifest(output: Path, config: Mapping[str, Any]) -> None:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = manifest.get("configuration")
    if isinstance(previous, Mapping) and _sha256_json(previous) != _sha256_json(config):
        raise ValueError(
            "Resume configuration does not match the previous run manifest. "
            "Use the original configuration or start with --overwrite."
        )


def _write_stage_summaries(
    *,
    output: Path,
    graph_dir: Optional[Path],
    mes_output: Optional[Path],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    _write_json_atomic(output.with_suffix(output.suffix + ".summary.json"), summary)
    _write_json_atomic(output.with_suffix(output.suffix + ".manifest.json"), manifest)

    if graph_dir is not None:
        graph_summary = {
            "graph_version": GRAPH_VERSION,
            **dict(summary.get("local_graph", {})),
        }
        _write_json_atomic(graph_dir / "_summary.json", graph_summary)

    if mes_output is not None:
        mes_summary = {
            "algorithm": MES_ALGORITHM_VERSION,
            **dict(summary.get("mes", {})),
        }
        _write_json_atomic(
            mes_output.with_suffix(mes_output.suffix + ".summary.json"),
            mes_summary,
        )


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    sentences_path = Path(args.sentences)
    output = Path(args.output)
    graph_dir = Path(args.graph_dir) if args.graph_dir else None
    mes_output = Path(args.mes_output) if args.mes_output else None
    ids_path = Path(args.ids_file) if args.ids_file else None

    if mes_output is not None and graph_dir is None:
        raise ValueError("--mes_output requires --graph_dir")
    if args.mes_max_path_nodes < 1:
        raise ValueError("--mes_max_path_nodes must be >= 1")
    if args.mes_exact_cover_limit < 1:
        raise ValueError("--mes_exact_cover_limit must be >= 1")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max_records must be >= 1")

    selected_ids = _read_ids(ids_path)
    records, input_stats = _load_sentence_records(
        sentences_path,
        selected_ids=selected_ids,
        max_records=args.max_records,
    )
    sentence_lookup = {row["input_id"]: row["sentences"] for row in records}
    selected_order = [row["input_id"] for row in records]

    _prepare_outputs(
        output=output,
        graph_dir=graph_dir,
        mes_output=mes_output,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )

    config = _config_signature(args, sentences_path, ids_path)
    if args.resume:
        _validate_resume_manifest(output, config)

    _, extraction_by_id = _load_existing_jsonl(output) if args.resume else ([], {})
    if args.resume:
        extra_ids = sorted(set(extraction_by_id) - set(selected_order))
        if extra_ids:
            raise ValueError(
                "Existing extraction output contains IDs outside the selected input: "
                + ", ".join(extra_ids[:20])
            )

    _, mes_by_id = (
        _load_existing_jsonl(mes_output) if args.resume and mes_output is not None else ([], {})
    )

    counters: Dict[str, Any] = {
        "input": input_stats,
        "extraction": {
            "records_selected": len(records),
            "records_reused_on_resume": 0,
            "records_extracted_now": 0,
            "llm_records": 0,
            "fallback_records": 0,
            "validation_error_records": 0,
            "failed_records": 0,
        },
        "local_graph": {
            "graphs_written": 0,
            "graphs_reused_on_resume": 0,
            "complete_entry_behavior_impact_layers": 0,
            "warnings": 0,
            "failed_records": 0,
        },
        "mes": {
            "records_written": 0,
            "records_reused_on_resume": 0,
            "complete_core_chain": 0,
            "partial_mes": 0,
            "empty_mes": 0,
            "warnings": 0,
            "failed_records": 0,
        },
    }
    failures: List[Dict[str, str]] = []
    started_at = time.time()
    runtime_signatures: Dict[str, Dict[str, Any]] = {}

    def register_runtime_signature(input_id: str, extraction: Mapping[str, Any]) -> None:
        signature = _runtime_signature(extraction)
        signature_hash = _sha256_json(signature)
        runtime_signatures.setdefault(signature_hash, signature)
        if len(runtime_signatures) > 1:
            raise ValueError(
                "Extraction output would mix different model/prompt runtime settings. "
                f"Conflicting record: {input_id}. Start a new output with --overwrite."
            )

    def fail(stage: str, input_id: str, exc: BaseException) -> None:
        failures.append({
            "stage": stage,
            "input_id": input_id,
            "error": f"{type(exc).__name__}: {exc}",
        })
        counters[stage]["failed_records"] += 1
        if not args.continue_on_error:
            raise RuntimeError(f"{stage} failed for {input_id}: {exc}") from exc

    fatal_error: Optional[BaseException] = None
    try:
        # Stage 1: extraction.  Existing rows are reused only under --resume.
        for row in tqdm(records, desc="extract", unit="CVE"):
            input_id = row["input_id"]
            if input_id in extraction_by_id:
                extraction = extraction_by_id[input_id]
                try:
                    register_runtime_signature(input_id, extraction)
                    used_llm, validation_errors = _enforce_extraction_policy(
                        extraction,
                        set(row["sentences"]),
                        allow_fallback=bool(args.allow_fallback),
                        allow_validation_errors=bool(args.allow_validation_errors),
                    )
                    counters["extraction"]["records_reused_on_resume"] += 1
                    counters["extraction"]["llm_records" if used_llm else "fallback_records"] += 1
                    counters["extraction"]["validation_error_records"] += int(bool(validation_errors))
                except Exception as exc:
                    fail("extraction", input_id, exc)
                continue
            try:
                extraction = call_llm_extract(input_id, row["sentences"])
                extraction["input_id"] = input_id
                register_runtime_signature(input_id, extraction)
                used_llm, validation_errors = _enforce_extraction_policy(
                    extraction,
                    set(row["sentences"]),
                    allow_fallback=bool(args.allow_fallback),
                    allow_validation_errors=bool(args.allow_validation_errors),
                )

                # Save only after policy and runtime-consistency checks pass.
                _append_jsonl(output, extraction)
                extraction_by_id[input_id] = extraction
                counters["extraction"]["records_extracted_now"] += 1
                counters["extraction"]["llm_records" if used_llm else "fallback_records"] += 1
                counters["extraction"]["validation_error_records"] += int(bool(validation_errors))
            except Exception as exc:
                fail("extraction", input_id, exc)

        # Stage 2 and 3: use the exact extraction records selected above.
        for row in tqdm(records, desc="structure", unit="CVE", disable=graph_dir is None):
            if graph_dir is None:
                break
            input_id = row["input_id"]
            extraction = extraction_by_id.get(input_id)
            if extraction is None:
                fail("local_graph", input_id, ValueError("missing extraction record"))
                continue

            graph_path = graph_dir / f"{_safe_filename(input_id)}.json"
            graph: Optional[Dict[str, Any]] = None
            if args.resume and graph_path.exists():
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                if str(graph.get("input_id", "")) != input_id:
                    fail("local_graph", input_id, ValueError("existing graph input_id mismatch"))
                    continue
                counters["local_graph"]["graphs_reused_on_resume"] += 1
            else:
                try:
                    graph = build_local_graph_for_one(
                        extraction=extraction,
                        sentences_lookup=sentence_lookup,
                        filter_aka_for_behavior_impact=not args.no_filter_aka,
                    )
                    _write_json_atomic(graph_path, graph)
                    counters["local_graph"]["graphs_written"] += 1
                except Exception as exc:
                    fail("local_graph", input_id, exc)
                    continue

            counters["local_graph"]["complete_entry_behavior_impact_layers"] += int(
                bool((graph.get("stats") or {}).get("complete_entry_behavior_impact_layers"))
            )
            counters["local_graph"]["warnings"] += len(graph.get("warnings") or [])

            if mes_output is None:
                continue
            if input_id in mes_by_id:
                counters["mes"]["records_reused_on_resume"] += 1
                mes = mes_by_id[input_id]
            else:
                try:
                    mes = _build_mes_record(
                        graph=graph,
                        max_path_nodes=args.mes_max_path_nodes,
                        exact_cover_limit=args.mes_exact_cover_limit,
                        include_precondition=bool(args.mes_include_precondition),
                    )
                    _append_jsonl(mes_output, mes)
                    mes_by_id[input_id] = mes
                    counters["mes"]["records_written"] += 1
                except Exception as exc:
                    fail("mes", input_id, exc)
                    continue

            status = str(mes.get("status", ""))
            counters["mes"]["complete_core_chain"] += int(status == "complete")
            counters["mes"]["partial_mes"] += int(status == "partial")
            counters["mes"]["empty_mes"] += int(status == "empty")
            counters["mes"]["warnings"] += len(mes.get("warnings") or [])

    except BaseException as exc:
        fatal_error = exc

    elapsed = time.time() - started_at
    output_hash = _sha256_file(output) if output.exists() else None
    mes_hash = _sha256_file(mes_output) if mes_output is not None and mes_output.exists() else None
    graph_hashes: Dict[str, str] = {}
    if graph_dir is not None:
        for input_id in selected_order:
            graph_path = graph_dir / f"{_safe_filename(input_id)}.json"
            if graph_path.exists():
                graph_hashes[input_id] = _sha256_file(graph_path)

    status = "failed" if fatal_error is not None or failures else "passed"
    counters["status"] = status
    counters["elapsed_seconds"] = round(elapsed, 3)
    counters["failure_count"] = len(failures)
    counters["failures"] = failures

    manifest: Dict[str, Any] = {
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "status": status,
        "configuration": config,
        "configuration_sha256": _sha256_json(config),
        "selected_ids_sha256": _sha256_json(selected_order),
        "selected_record_count": len(selected_order),
        "extraction_runtime_signatures": runtime_signatures,
        "extraction_runtime_signature_count": len(runtime_signatures),
        "outputs": {
            "extraction": str(output),
            "extraction_sha256": output_hash,
            "graph_dir": str(graph_dir) if graph_dir is not None else None,
            "graph_file_count": len(graph_hashes),
            "graph_set_sha256": _sha256_json(graph_hashes) if graph_hashes else None,
            "mes": str(mes_output) if mes_output is not None else None,
            "mes_sha256": mes_hash,
        },
        "summary": counters,
    }
    _write_stage_summaries(
        output=output,
        graph_dir=graph_dir,
        mes_output=mes_output,
        summary=counters,
        manifest=manifest,
    )

    if fatal_error is not None:
        raise fatal_error
    if failures:
        raise RuntimeError(f"Pipeline completed with {len(failures)} failed record(s)")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract evidence-linked attack elements and optionally build local graphs and MES records."
        )
    )
    parser.add_argument("--sentences", required=True, help="Input sentences.jsonl")
    parser.add_argument("--output", required=True, help="Output extraction.jsonl")
    parser.add_argument("--graph_dir", help="Optional directory for one local graph JSON per CVE")
    parser.add_argument("--mes_output", help="Optional MES JSONL; requires --graph_dir")
    parser.add_argument("--ids_file", help="Optional fixed list of input_ids to process")
    parser.add_argument("--max_records", type=int, help="Optional smoke-test record limit")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    mode.add_argument("--resume", action="store_true", help="Reuse completed records with matching manifest")

    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow rule-based extraction fallback (smoke tests only; strict by default)",
    )
    parser.add_argument(
        "--allow_validation_errors",
        action="store_true",
        help="Allow extraction records with evidence validation errors",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing after a record fails; run still exits non-zero",
    )
    parser.add_argument(
        "--no_filter_aka",
        action="store_true",
        help='Keep alias-only evidence such as `aka "..."` for Behavior/Impact nodes',
    )
    parser.add_argument("--mes_max_path_nodes", type=int, default=4)
    parser.add_argument("--mes_exact_cover_limit", type=int, default=20)
    parser.add_argument("--mes_include_precondition", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_pipeline(args)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
