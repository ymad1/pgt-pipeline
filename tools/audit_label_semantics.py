#!/usr/bin/env python3
"""Prepare and summarize a blind semantic audit of CVE-to-ATT&CK gold labels.

The audit unit is one ``(CVE, ATT&CK technique)`` gold-label pair.  The tool
never reads MES, candidates, reranking outputs, model predictions, or ranking
metrics.  Reviewers see only the canonical CVE description and the active
ATT&CK technique definition, which keeps the audit independent of downstream
model behavior.

Two workflow commands are provided:

``prepare``
    Validate the fixed dataset, draw a deterministic technique-stratified
    sample, and create two independently ordered reviewer forms plus a coding
    guide and a complete sampling manifest.

``summarize``
    Validate completed reviewer forms, calculate agreement, create an
    adjudication form for disagreements, and (when all disagreements are
    adjudicated) produce weighted label-quality estimates and audit artifacts.

The four allowed semantic decisions are:

``directly_supported``
    The CVE text explicitly states behavior, entry mechanism, or consequence
    that matches the core ATT&CK technique definition.

``inferential_or_plausible``
    The mapping is a reasonable one-step inference, but the decisive behavior
    is not explicitly stated in the CVE description.

``insufficient_text``
    The CVE text lacks enough attack/behavior/impact information to judge the
    mapping (for example, it explicitly says the impact is unknown).

``unsupported``
    The CVE text contains enough information to assess the mapping, but the
    stated behavior does not support the gold technique or points elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "label-semantic-audit-v1.0.0"
AUDIT_PROTOCOL_VERSION = "cve-attck-label-audit-protocol-v1.0.0"
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
DECISIONS: Tuple[str, ...] = (
    "directly_supported",
    "inferential_or_plausible",
    "insufficient_text",
    "unsupported",
)
CONFIDENCE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
FORM_COLUMNS: Tuple[str, ...] = (
    "display_order",
    "sample_id",
    "cve_id",
    "cve_description",
    "technique_id",
    "technique_name",
    "technique_description",
    "parent_technique_id",
    "parent_technique_name",
    "decision",
    "evidence_from_cve",
    "rationale",
    "reviewer_confidence",
)
ADJUDICATION_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "cve_id",
    "cve_description",
    "technique_id",
    "technique_name",
    "technique_description",
    "reviewer_1_decision",
    "reviewer_1_evidence",
    "reviewer_1_rationale",
    "reviewer_2_decision",
    "reviewer_2_evidence",
    "reviewer_2_rationale",
    "adjudicated_decision",
    "adjudication_evidence",
    "adjudication_rationale",
)


class AuditError(RuntimeError):
    """Raised when the audit cannot be prepared or summarized safely."""


# ---------------------------------------------------------------------------
# Deterministic IO helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_hash(seed: int, *parts: str) -> str:
    payload = "\x1f".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"JSONL file does not exist: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(dict(value))
    return rows


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise AuditError(f"ID file does not exist: {path}")
    values: List[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            value = raw.strip()
            if not value:
                continue
            if value in seen:
                raise AuditError(f"Duplicate ID {value!r} in {path}:{line_no}")
            seen.add(value)
            values.append(value)
    if not values:
        raise AuditError(f"ID file is empty: {path}")
    return values


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    _atomic_write_bytes(path, content.encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write_bytes(path, content)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AuditError(f"CSV file has no header: {path}")
        return [{str(k): str(v or "") for k, v in row.items()} for row in reader]


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise AuditError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=False)


# ---------------------------------------------------------------------------
# Input validation and sampling
# ---------------------------------------------------------------------------


def _normalize_labels(value: Any, *, input_id: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise AuditError(f"labels must be a JSON list for {input_id}")
    labels: List[str] = []
    for raw in value:
        technique_id = str(raw).strip().upper()
        if not TECHNIQUE_ID_RE.fullmatch(technique_id):
            raise AuditError(f"Invalid ATT&CK technique ID {raw!r} for {input_id}")
        labels.append(technique_id)
    canonical = tuple(sorted(set(labels)))
    if not canonical:
        raise AuditError(f"No gold labels for {input_id}")
    if tuple(labels) != canonical:
        raise AuditError(
            f"Labels must already be sorted and unique for {input_id}; "
            f"observed={labels!r}, expected={list(canonical)!r}"
        )
    return canonical


def _unique_by_input_id(rows: Sequence[Mapping[str, Any]], *, source: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        input_id = str(row.get("input_id") or "").strip()
        if not input_id:
            raise AuditError(f"Missing input_id in {source} record {index}")
        if input_id in result:
            raise AuditError(f"Duplicate input_id {input_id!r} in {source}")
        result[input_id] = dict(row)
    return result


def _load_techniques(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(path)
    result: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        technique_id = str(row.get("technique_id") or "").strip().upper()
        if not TECHNIQUE_ID_RE.fullmatch(technique_id):
            raise AuditError(f"Invalid technique_id in {path} record {index}: {technique_id!r}")
        if technique_id in result:
            raise AuditError(f"Duplicate technique_id {technique_id!r} in {path}")
        if bool(row.get("revoked")) or bool(row.get("deprecated")):
            raise AuditError(f"Inactive technique {technique_id} appears in active technique index {path}")
        name = str(row.get("name") or "").strip()
        description = str(row.get("description") or "").strip()
        if not name or not description:
            raise AuditError(f"Technique {technique_id} lacks name or description in {path}")
        result[technique_id] = dict(row)
    if not result:
        raise AuditError(f"Technique index is empty: {path}")
    return result


def _load_split_assignments(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path)
    result = _unique_by_input_id(rows, source=path)
    for input_id, row in result.items():
        split = str(row.get("split") or "").strip().lower()
        if split not in {"development", "test"}:
            raise AuditError(f"Invalid split {split!r} for {input_id} in {path}")
        row["split"] = split
    return result


def _load_population(
    *,
    records_path: Path,
    labels_path: Path,
    tech_index_path: Path,
    split_assignments_path: Optional[Path],
    ids_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = _unique_by_input_id(_read_jsonl(records_path), source=records_path)
    labels = _unique_by_input_id(_read_jsonl(labels_path), source=labels_path)
    techniques = _load_techniques(tech_index_path)
    assignments = _load_split_assignments(split_assignments_path)

    if set(records) != set(labels):
        missing_records = sorted(set(labels) - set(records))[:10]
        missing_labels = sorted(set(records) - set(labels))[:10]
        raise AuditError(
            "records/labels input_id mismatch: "
            f"missing_records={missing_records}, missing_labels={missing_labels}"
        )
    if assignments and set(assignments) != set(records):
        missing_assignments = sorted(set(records) - set(assignments))[:10]
        extra_assignments = sorted(set(assignments) - set(records))[:10]
        raise AuditError(
            "split assignments do not align with records: "
            f"missing={missing_assignments}, extra={extra_assignments}"
        )

    selected_ids = set(records)
    ids_hash: Optional[str] = None
    if ids_path is not None:
        ordered_ids = _read_ids(ids_path)
        unknown = [value for value in ordered_ids if value not in records]
        if unknown:
            raise AuditError(f"ID file contains IDs absent from records: {unknown[:10]}")
        selected_ids = set(ordered_ids)
        ids_hash = _sha256_file(ids_path)

    population: List[Dict[str, Any]] = []
    for input_id in sorted(selected_ids):
        record = records[input_id]
        label_row = labels[input_id]
        raw_text = str(record.get("raw_text") or "").strip()
        if not raw_text:
            raise AuditError(f"Blank raw_text for {input_id}")
        gold = _normalize_labels(label_row.get("labels"), input_id=input_id)
        record_labels = record.get("labels")
        if record_labels is not None and _normalize_labels(record_labels, input_id=input_id) != gold:
            raise AuditError(f"records/labels gold mismatch for {input_id}")
        assignment = assignments.get(input_id, {})
        for technique_id in gold:
            technique = techniques.get(technique_id)
            if technique is None:
                raise AuditError(
                    f"Gold technique {technique_id} for {input_id} is missing from active index {tech_index_path}"
                )
            population.append(
                {
                    "input_id": input_id,
                    "raw_text": raw_text,
                    "technique_id": technique_id,
                    "technique_name": str(technique.get("name") or ""),
                    "technique_description": str(technique.get("description") or ""),
                    "parent_technique_id": technique.get("parent_technique_id") or "",
                    "parent_technique_name": technique.get("parent_name") or "",
                    "split": assignment.get("split") or "unspecified",
                    "source_membership": assignment.get("source_membership") or [],
                }
            )

    attack_versions = sorted(
        {
            str(row.get("attack_collection_version") or "unknown")
            for row in techniques.values()
        }
    )
    metadata = {
        "records": len(selected_ids),
        "gold_label_pairs": len(population),
        "techniques_in_gold": len({row["technique_id"] for row in population}),
        "active_techniques_in_index": len(techniques),
        "attack_collection_versions": attack_versions,
        "ids_file_sha256": ids_hash,
    }
    return population, metadata


def _draw_stratified_sample(
    population: Sequence[Mapping[str, Any]],
    *,
    sample_per_technique: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if sample_per_technique < 1:
        raise AuditError("--sample-per-technique must be at least 1")
    by_technique: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in population:
        by_technique[str(row["technique_id"])].append(row)

    sample: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    sample_ids: set[str] = set()
    for technique_id in sorted(by_technique):
        stratum = list(by_technique[technique_id])
        stratum.sort(key=lambda row: _stable_hash(seed, technique_id, str(row["input_id"])))
        population_n = len(stratum)
        sample_n = min(sample_per_technique, population_n)
        probability = sample_n / population_n
        weight = population_n / sample_n
        selected = stratum[:sample_n]
        for row in selected:
            sample_id = "LSA-" + _stable_hash(
                seed, str(row["input_id"]), technique_id
            )[:12].upper()
            if sample_id in sample_ids:
                raise AuditError(f"Sample ID collision: {sample_id}")
            sample_ids.add(sample_id)
            item = dict(row)
            item.update(
                {
                    "sample_id": sample_id,
                    "stratum_population_pairs": population_n,
                    "stratum_sample_pairs": sample_n,
                    "selection_probability": probability,
                    "analysis_weight": weight,
                    "sampling_seed": seed,
                    "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
                }
            )
            sample.append(item)
        summary.append(
            {
                "technique_id": technique_id,
                "technique_name": str(stratum[0]["technique_name"]),
                "population_pairs": population_n,
                "sample_pairs": sample_n,
                "selection_probability": f"{probability:.12g}",
                "analysis_weight": f"{weight:.12g}",
            }
        )

    sample.sort(key=lambda row: (str(row["technique_id"]), str(row["input_id"])))
    return sample, summary


def _reviewer_form_rows(sample: Sequence[Mapping[str, Any]], *, reviewer: int, seed: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        sample,
        key=lambda row: _stable_hash(seed, f"reviewer-{reviewer}", str(row["sample_id"])),
    )
    rows: List[Dict[str, Any]] = []
    for display_order, row in enumerate(ordered, 1):
        rows.append(
            {
                "display_order": display_order,
                "sample_id": row["sample_id"],
                "cve_id": row["input_id"],
                "cve_description": row["raw_text"],
                "technique_id": row["technique_id"],
                "technique_name": row["technique_name"],
                "technique_description": row["technique_description"],
                "parent_technique_id": row.get("parent_technique_id", ""),
                "parent_technique_name": row.get("parent_technique_name", ""),
                "decision": "",
                "evidence_from_cve": "",
                "rationale": "",
                "reviewer_confidence": "",
            }
        )
    return rows


def _coding_guide() -> str:
    decisions = "\n".join(f"- `{value}`" for value in DECISIONS)
    return f"""# CVE–ATT&CK Gold-Label Semantic Audit Guide

Protocol version: `{AUDIT_PROTOCOL_VERSION}`

## Purpose

Judge whether each inherited gold ATT&CK label is supported by the supplied CVE description. The unit of judgment is one CVE–technique pair. Do not consult model predictions, MES records, candidate rankings, evaluation results, or the other reviewer's decisions.

## Allowed decisions

{decisions}

### directly_supported

Choose this when the CVE description explicitly states the core behavior, access mechanism, or consequence represented by the ATT&CK technique. Record the exact supporting phrase in `evidence_from_cve`.

### inferential_or_plausible

Choose this when the mapping is a reasonable one-step inference but the decisive behavior is not directly stated. Do not use multi-step speculation or outside incident knowledge.

### insufficient_text

Choose this when the description is too vague to assess the mapping, such as descriptions saying that the impact or attack vector is unknown. This is different from `unsupported`: the problem is missing information, not contradictory or unrelated information.

### unsupported

Choose this when the CVE description contains enough information to assess the mapping but does not support the gold technique, supports a materially different behavior, or conflicts with the technique definition.

## Review rules

1. Use only the CVE description and ATT&CK definition shown in the form.
2. Do not search for exploit write-ups, vendor advisories, model outputs, or candidate lists during the primary review.
3. Judge the stated text, not what the vulnerability might theoretically enable.
4. For `directly_supported`, quote the shortest decisive CVE phrase.
5. For `inferential_or_plausible`, explain the single inference required.
6. For `insufficient_text`, identify what information is missing.
7. For `unsupported`, explain the mismatch without proposing a replacement label unless useful for adjudication.
8. Complete reviews independently. Discuss disagreements only during adjudication.
9. Use `reviewer_confidence` values: `low`, `medium`, or `high`.

## Independence guarantee

The preparation tool loads only fixed records, inherited gold labels, split metadata, and the active ATT&CK technique index. It does not load extraction, MES, candidate, reranking, prediction, or metric files.
"""


def prepare_audit(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    records_path = Path(args.records).resolve()
    labels_path = Path(args.labels).resolve()
    tech_index_path = Path(args.tech_index).resolve()
    split_path = Path(args.split_assignments).resolve() if args.split_assignments else None
    ids_path = Path(args.ids_file).resolve() if args.ids_file else None

    population, population_metadata = _load_population(
        records_path=records_path,
        labels_path=labels_path,
        tech_index_path=tech_index_path,
        split_assignments_path=split_path,
        ids_path=ids_path,
    )
    sample, sampling_summary = _draw_stratified_sample(
        population,
        sample_per_technique=args.sample_per_technique,
        seed=args.seed,
    )

    sample_path = output_dir / "audit_sample.jsonl"
    reviewer_1_path = output_dir / "audit_form_reviewer1.csv"
    reviewer_2_path = output_dir / "audit_form_reviewer2.csv"
    key_path = output_dir / "audit_key.csv"
    sampling_path = output_dir / "sampling_summary.csv"
    guide_path = output_dir / "AUDIT_GUIDE.md"

    _write_jsonl(sample_path, sample)
    _write_csv(
        reviewer_1_path,
        _reviewer_form_rows(sample, reviewer=1, seed=args.seed),
        FORM_COLUMNS,
    )
    _write_csv(
        reviewer_2_path,
        _reviewer_form_rows(sample, reviewer=2, seed=args.seed),
        FORM_COLUMNS,
    )
    _write_csv(
        key_path,
        (
            {
                "sample_id": row["sample_id"],
                "cve_id": row["input_id"],
                "technique_id": row["technique_id"],
                "split": row["split"],
                "source_membership": "|".join(map(str, row.get("source_membership") or [])),
                "analysis_weight": f"{float(row['analysis_weight']):.12g}",
                "selection_probability": f"{float(row['selection_probability']):.12g}",
            }
            for row in sample
        ),
        (
            "sample_id",
            "cve_id",
            "technique_id",
            "split",
            "source_membership",
            "analysis_weight",
            "selection_probability",
        ),
    )
    _write_csv(
        sampling_path,
        sampling_summary,
        (
            "technique_id",
            "technique_name",
            "population_pairs",
            "sample_pairs",
            "selection_probability",
            "analysis_weight",
        ),
    )
    _atomic_write_bytes(guide_path, _coding_guide().encode("utf-8"))

    output_files = [
        sample_path,
        reviewer_1_path,
        reviewer_2_path,
        key_path,
        sampling_path,
        guide_path,
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "mode": "prepare",
        "blindness": {
            "loaded_inputs": [
                "canonical CVE records",
                "inherited gold labels",
                "active ATT&CK technique index",
                "optional fixed split assignments",
                "optional ID restriction",
            ],
            "forbidden_and_not_loaded": [
                "LLM extraction",
                "local graphs",
                "MES",
                "retrieval candidates",
                "reranking outputs",
                "model predictions",
                "evaluation metrics",
            ],
        },
        "sampling": {
            "unit": "CVE-technique gold-label pair",
            "strategy": "deterministic equal-allocation stratified sampling by gold technique",
            "sample_per_technique_cap": args.sample_per_technique,
            "seed": args.seed,
            "stable_order": "SHA-256(seed, technique_id, input_id)",
            "reviewer_order": "independent SHA-256 order per reviewer",
            "population_pairs": len(population),
            "sample_pairs": len(sample),
            "sampled_techniques": len(sampling_summary),
            "weight": "population pairs in stratum / sampled pairs in stratum",
        },
        "population": population_metadata,
        "decisions": list(DECISIONS),
        "input_files": {
            "records": {"path": str(records_path), "sha256": _sha256_file(records_path)},
            "labels": {"path": str(labels_path), "sha256": _sha256_file(labels_path)},
            "tech_index": {"path": str(tech_index_path), "sha256": _sha256_file(tech_index_path)},
            "split_assignments": (
                {"path": str(split_path), "sha256": _sha256_file(split_path)} if split_path else None
            ),
            "ids_file": {"path": str(ids_path), "sha256": _sha256_file(ids_path)} if ids_path else None,
        },
        "outputs": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    manifest_path = output_dir / "audit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "population_records": population_metadata["records"],
        "population_pairs": len(population),
        "sample_pairs": len(sample),
        "sampled_techniques": len(sampling_summary),
        "reviewer_1_form": str(reviewer_1_path),
        "reviewer_2_form": str(reviewer_2_path),
        "manifest": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Review parsing, agreement, adjudication, and weighted summaries
# ---------------------------------------------------------------------------


def _normalize_decision(value: str, *, source: Path, sample_id: str, allow_blank: bool) -> str:
    decision = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not decision and allow_blank:
        return ""
    if decision not in DECISIONS:
        raise AuditError(
            f"Invalid decision {value!r} for {sample_id} in {source}; "
            f"allowed={list(DECISIONS)}"
        )
    return decision


def _normalize_confidence(value: str, *, source: Path, sample_id: str, allow_blank: bool) -> str:
    confidence = value.strip().lower()
    if not confidence and allow_blank:
        return ""
    if confidence not in CONFIDENCE_LEVELS:
        raise AuditError(
            f"Invalid reviewer_confidence {value!r} for {sample_id} in {source}; "
            f"allowed={list(CONFIDENCE_LEVELS)}"
        )
    return confidence


def _load_reviewer_form(
    path: Path,
    expected_sample: Mapping[str, Mapping[str, Any]],
    *,
    allow_incomplete: bool,
) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(path)
    expected_ids = set(expected_sample)
    result: Dict[str, Dict[str, str]] = {}
    for row_no, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise AuditError(f"Missing sample_id in {path}:{row_no}")
        if sample_id in result:
            raise AuditError(f"Duplicate sample_id {sample_id} in {path}")
        if sample_id not in expected_ids:
            raise AuditError(f"Unknown sample_id {sample_id} in {path}")
        expected = expected_sample[sample_id]
        static_checks = {
            "cve_id": str(expected.get("input_id") or ""),
            "technique_id": str(expected.get("technique_id") or ""),
            "technique_name": str(expected.get("technique_name") or ""),
            "cve_description": str(expected.get("raw_text") or ""),
            "technique_description": str(expected.get("technique_description") or ""),
        }
        for column, expected_value in static_checks.items():
            if row.get(column, "") != expected_value:
                raise AuditError(
                    f"Reviewer form metadata changed for {sample_id} in {path}: "
                    f"column={column!r}"
                )
        decision = _normalize_decision(
            row.get("decision", ""),
            source=path,
            sample_id=sample_id,
            allow_blank=allow_incomplete,
        )
        confidence = _normalize_confidence(
            row.get("reviewer_confidence", ""),
            source=path,
            sample_id=sample_id,
            allow_blank=allow_incomplete,
        )
        if decision and not row.get("rationale", "").strip():
            raise AuditError(f"Completed decision lacks rationale for {sample_id} in {path}")
        if decision == "directly_supported" and not row.get("evidence_from_cve", "").strip():
            raise AuditError(
                f"directly_supported requires evidence_from_cve for {sample_id} in {path}"
            )
        normalized = dict(row)
        normalized["decision"] = decision
        normalized["reviewer_confidence"] = confidence
        result[sample_id] = normalized
    if set(result) != expected_ids:
        missing = sorted(expected_ids - set(result))[:10]
        raise AuditError(f"Reviewer form {path} is missing sample IDs: {missing}")
    return result


def _cohens_kappa(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Optional[float]]:
    if not pairs:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    n = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[d] / n) * (right_counts[d] / n) for d in DECISIONS)
    denominator = 1.0 - expected
    kappa = None if abs(denominator) < 1e-15 else (observed - expected) / denominator
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def _load_adjudication(
    path: Path,
    disagreement_ids: set[str],
    *,
    allow_incomplete: bool,
) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(path)
    result: Dict[str, Dict[str, str]] = {}
    for row_no, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise AuditError(f"Missing sample_id in {path}:{row_no}")
        if sample_id in result:
            raise AuditError(f"Duplicate sample_id {sample_id} in {path}")
        if sample_id not in disagreement_ids:
            raise AuditError(f"Adjudication file contains non-disagreement sample {sample_id}")
        decision = _normalize_decision(
            row.get("adjudicated_decision", ""),
            source=path,
            sample_id=sample_id,
            allow_blank=allow_incomplete,
        )
        if decision and not row.get("adjudication_rationale", "").strip():
            raise AuditError(f"Adjudicated decision lacks rationale for {sample_id} in {path}")
        if decision == "directly_supported" and not row.get("adjudication_evidence", "").strip():
            raise AuditError(
                f"directly_supported adjudication requires adjudication_evidence for {sample_id}"
            )
        normalized = dict(row)
        normalized["adjudicated_decision"] = decision
        result[sample_id] = normalized
    missing_rows = disagreement_ids - set(result)
    if missing_rows and not allow_incomplete:
        raise AuditError(f"Adjudication file is missing disagreements: {sorted(missing_rows)[:10]}")
    return result


def _rate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    raw_counts = Counter(str(row["final_decision"]) for row in rows)
    weighted_counts: Dict[str, float] = {decision: 0.0 for decision in DECISIONS}
    total_weight = 0.0
    for row in rows:
        weight = float(row["analysis_weight"])
        weighted_counts[str(row["final_decision"])] += weight
        total_weight += weight
    raw_n = len(rows)
    return {
        "resolved_pairs": raw_n,
        "raw_counts": {decision: raw_counts.get(decision, 0) for decision in DECISIONS},
        "raw_rates": {
            decision: (raw_counts.get(decision, 0) / raw_n if raw_n else None)
            for decision in DECISIONS
        },
        "weighted_counts": weighted_counts,
        "weighted_rates": {
            decision: (weighted_counts[decision] / total_weight if total_weight else None)
            for decision in DECISIONS
        },
        "weighted_total_pairs": total_weight,
        "direct_or_plausible_raw_rate": (
            (raw_counts.get("directly_supported", 0) + raw_counts.get("inferential_or_plausible", 0)) / raw_n
            if raw_n
            else None
        ),
        "direct_or_plausible_weighted_rate": (
            (
                weighted_counts["directly_supported"]
                + weighted_counts["inferential_or_plausible"]
            )
            / total_weight
            if total_weight
            else None
        ),
    }


def _stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    metrics = [*DECISIONS, "direct_or_plausible"]
    if repetitions <= 0 or not rows:
        return {metric: {"lower": None, "upper": None} for metric in metrics}
    by_technique: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_technique[str(row["technique_id"])].append(row)
    rng = random.Random(seed)
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(repetitions):
        counts: Dict[str, float] = {decision: 0.0 for decision in DECISIONS}
        total = 0.0
        for technique_id in sorted(by_technique):
            stratum = by_technique[technique_id]
            for _index in range(len(stratum)):
                row = stratum[rng.randrange(len(stratum))]
                weight = float(row["analysis_weight"])
                counts[str(row["final_decision"])] += weight
                total += weight
        if total <= 0:
            continue
        for decision in DECISIONS:
            values[decision].append(counts[decision] / total)
        values["direct_or_plausible"].append(
            (counts["directly_supported"] + counts["inferential_or_plausible"]) / total
        )
    alpha = (1.0 - confidence) / 2.0

    def percentile(sorted_values: Sequence[float], probability: float) -> Optional[float]:
        if not sorted_values:
            return None
        position = probability * (len(sorted_values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        fraction = position - lower
        return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for metric, metric_values in values.items():
        ordered = sorted(metric_values)
        result[metric] = {
            "lower": percentile(ordered, alpha),
            "upper": percentile(ordered, 1.0 - alpha),
        }
    return result


def _group_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    group_name_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unspecified")].append(row)
    output: List[Dict[str, Any]] = []
    for group in sorted(grouped):
        group_rows = grouped[group]
        rates = _rate_rows(group_rows)
        item: Dict[str, Any] = {
            group_key: group,
            "resolved_pairs": rates["resolved_pairs"],
            "weighted_total_pairs": f"{float(rates['weighted_total_pairs']):.12g}",
            "direct_or_plausible_raw_rate": rates["direct_or_plausible_raw_rate"],
            "direct_or_plausible_weighted_rate": rates["direct_or_plausible_weighted_rate"],
        }
        if group_name_key:
            item[group_name_key] = str(group_rows[0].get(group_name_key) or "")
        for decision in DECISIONS:
            item[f"{decision}_count"] = rates["raw_counts"][decision]
            item[f"{decision}_raw_rate"] = rates["raw_rates"][decision]
            item[f"{decision}_weighted_rate"] = rates["weighted_rates"][decision]
        output.append(item)
    return output


def summarize_audit(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()

    sample_path = Path(args.sample).resolve()
    reviewer_1_path = Path(args.reviewer1).resolve()
    reviewer_2_path = Path(args.reviewer2).resolve()
    adjudication_path = Path(args.adjudication).resolve() if args.adjudication else None

    sample_rows = _read_jsonl(sample_path)
    sample_by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(sample_rows, 1):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise AuditError(f"Missing sample_id in {sample_path} record {index}")
        if sample_id in sample_by_id:
            raise AuditError(f"Duplicate sample_id {sample_id} in {sample_path}")
        decision_weight = row.get("analysis_weight")
        try:
            weight = float(decision_weight)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"Invalid analysis_weight for {sample_id}: {decision_weight!r}") from exc
        if weight <= 0 or not math.isfinite(weight):
            raise AuditError(f"analysis_weight must be finite and positive for {sample_id}")
        sample_by_id[sample_id] = dict(row)
    if not sample_by_id:
        raise AuditError(f"Audit sample is empty: {sample_path}")

    expected_ids = set(sample_by_id)
    reviewer_1 = _load_reviewer_form(
        reviewer_1_path, sample_by_id, allow_incomplete=args.allow_incomplete
    )
    reviewer_2 = _load_reviewer_form(
        reviewer_2_path, sample_by_id, allow_incomplete=args.allow_incomplete
    )

    completed_pairs: List[Tuple[str, str]] = []
    disagreements: set[str] = set()
    incomplete_ids: set[str] = set()
    confusion: Dict[str, Counter[str]] = {decision: Counter() for decision in DECISIONS}
    for sample_id in sorted(expected_ids):
        decision_1 = reviewer_1[sample_id]["decision"]
        decision_2 = reviewer_2[sample_id]["decision"]
        if not decision_1 or not decision_2:
            incomplete_ids.add(sample_id)
            continue
        completed_pairs.append((decision_1, decision_2))
        confusion[decision_1][decision_2] += 1
        if decision_1 != decision_2:
            disagreements.add(sample_id)

    adjudication: Dict[str, Dict[str, str]] = {}
    if adjudication_path is not None:
        adjudication = _load_adjudication(
            adjudication_path,
            disagreements,
            allow_incomplete=args.allow_incomplete,
        )

    input_hashes = {
        "sample": _sha256_file(sample_path),
        "reviewer1": _sha256_file(reviewer_1_path),
        "reviewer2": _sha256_file(reviewer_2_path),
        "adjudication": _sha256_file(adjudication_path) if adjudication_path else None,
    }
    _prepare_output_dir(output_dir, args.overwrite)

    adjudication_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(disagreements):
        sample = sample_by_id[sample_id]
        r1 = reviewer_1[sample_id]
        r2 = reviewer_2[sample_id]
        existing = adjudication.get(sample_id, {})
        adjudication_rows.append(
            {
                "sample_id": sample_id,
                "cve_id": sample["input_id"],
                "cve_description": sample["raw_text"],
                "technique_id": sample["technique_id"],
                "technique_name": sample["technique_name"],
                "technique_description": sample["technique_description"],
                "reviewer_1_decision": r1["decision"],
                "reviewer_1_evidence": r1.get("evidence_from_cve", ""),
                "reviewer_1_rationale": r1.get("rationale", ""),
                "reviewer_2_decision": r2["decision"],
                "reviewer_2_evidence": r2.get("evidence_from_cve", ""),
                "reviewer_2_rationale": r2.get("rationale", ""),
                "adjudicated_decision": existing.get("adjudicated_decision", ""),
                "adjudication_evidence": existing.get("adjudication_evidence", ""),
                "adjudication_rationale": existing.get("adjudication_rationale", ""),
            }
        )
    adjudication_form_path = output_dir / "adjudication_form.csv"
    _write_csv(adjudication_form_path, adjudication_rows, ADJUDICATION_COLUMNS)

    resolved_rows: List[Dict[str, Any]] = []
    unresolved_ids: set[str] = set(incomplete_ids)
    for sample_id in sorted(expected_ids):
        r1 = reviewer_1[sample_id]
        r2 = reviewer_2[sample_id]
        decision_1 = r1["decision"]
        decision_2 = r2["decision"]
        final_decision = ""
        resolution_source = ""
        final_evidence = ""
        final_rationale = ""
        if not decision_1 or not decision_2:
            unresolved_ids.add(sample_id)
        elif decision_1 == decision_2:
            final_decision = decision_1
            resolution_source = "reviewer_agreement"
            final_evidence = r1.get("evidence_from_cve", "") or r2.get("evidence_from_cve", "")
            final_rationale = r1.get("rationale", "")
        else:
            adjudicated = adjudication.get(sample_id, {})
            final_decision = adjudicated.get("adjudicated_decision", "")
            if final_decision:
                resolution_source = "adjudication"
                final_evidence = adjudicated.get("adjudication_evidence", "")
                final_rationale = adjudicated.get("adjudication_rationale", "")
            else:
                unresolved_ids.add(sample_id)
        if final_decision:
            row = dict(sample_by_id[sample_id])
            row.update(
                {
                    "reviewer_1_decision": decision_1,
                    "reviewer_1_confidence": r1.get("reviewer_confidence", ""),
                    "reviewer_2_decision": decision_2,
                    "reviewer_2_confidence": r2.get("reviewer_confidence", ""),
                    "final_decision": final_decision,
                    "resolution_source": resolution_source,
                    "final_evidence": final_evidence,
                    "final_rationale": final_rationale,
                }
            )
            resolved_rows.append(row)

    agreement = _cohens_kappa(completed_pairs)
    agreement_report = {
        **agreement,
        "total_sample_pairs": len(expected_ids),
        "both_reviewers_complete": len(completed_pairs),
        "incomplete_pairs": len(incomplete_ids),
        "disagreements": len(disagreements),
        "unresolved_after_adjudication": len(unresolved_ids),
        "confusion_matrix": {
            left: {right: confusion[left].get(right, 0) for right in DECISIONS}
            for left in DECISIONS
        },
    }
    agreement_path = output_dir / "reviewer_agreement.json"
    _write_json(agreement_path, agreement_report)

    resolved_path = output_dir / "resolved_audit.jsonl"
    _write_jsonl(resolved_path, resolved_rows)

    overall = _rate_rows(resolved_rows)
    confidence_intervals = _stratified_bootstrap(
        resolved_rows,
        repetitions=args.bootstrap_repetitions,
        confidence=args.confidence,
        seed=args.seed,
    )
    technique_rows = _group_summary_rows(
        resolved_rows, group_key="technique_id", group_name_key="technique_name"
    )
    split_rows = _group_summary_rows(resolved_rows, group_key="split")

    summary_fieldnames = [
        "technique_id",
        "technique_name",
        "resolved_pairs",
        "weighted_total_pairs",
        *[field for decision in DECISIONS for field in (
            f"{decision}_count",
            f"{decision}_raw_rate",
            f"{decision}_weighted_rate",
        )],
        "direct_or_plausible_raw_rate",
        "direct_or_plausible_weighted_rate",
    ]
    technique_path = output_dir / "per_technique_summary.csv"
    _write_csv(technique_path, technique_rows, summary_fieldnames)

    split_fieldnames = [
        "split",
        "resolved_pairs",
        "weighted_total_pairs",
        *[field for decision in DECISIONS for field in (
            f"{decision}_count",
            f"{decision}_raw_rate",
            f"{decision}_weighted_rate",
        )],
        "direct_or_plausible_raw_rate",
        "direct_or_plausible_weighted_rate",
    ]
    split_summary_path = output_dir / "per_split_summary.csv"
    _write_csv(split_summary_path, split_rows, split_fieldnames)

    status = "final" if not unresolved_ids else "provisional"
    summary = {
        "script_version": SCRIPT_VERSION,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "status": status,
        "sample_pairs": len(expected_ids),
        "resolved_pairs": len(resolved_rows),
        "unresolved_pairs": len(unresolved_ids),
        "agreement": agreement_report,
        "overall": overall,
        "weighted_stratified_bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "confidence": args.confidence,
            "seed": args.seed,
            "intervals": confidence_intervals,
        },
        "interpretation": {
            "directly_supported": "explicit textual support",
            "direct_or_plausible": "explicit support plus one-step plausible inference",
            "insufficient_text": "not verifiable from supplied CVE text",
            "unsupported": "text is informative but does not support the inherited gold label",
        },
    }
    summary_name = "audit_summary.json" if status == "final" else "audit_summary_provisional.json"
    summary_path = output_dir / summary_name
    _write_json(summary_path, summary)

    outputs = [
        adjudication_form_path,
        agreement_path,
        resolved_path,
        technique_path,
        split_summary_path,
        summary_path,
    ]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "mode": "summarize",
        "status": status,
        "input_files": {
            "sample": {"path": str(sample_path), "sha256": input_hashes["sample"]},
            "reviewer1": {"path": str(reviewer_1_path), "sha256": input_hashes["reviewer1"]},
            "reviewer2": {"path": str(reviewer_2_path), "sha256": input_hashes["reviewer2"]},
            "adjudication": (
                {"path": str(adjudication_path), "sha256": input_hashes["adjudication"]}
                if adjudication_path
                else None
            ),
        },
        "parameters": {
            "allow_incomplete": args.allow_incomplete,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "confidence": args.confidence,
            "seed": args.seed,
        },
        "outputs": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in outputs
        },
    }
    manifest_path = output_dir / "summary_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "status": status,
        "sample_pairs": len(expected_ids),
        "resolved_pairs": len(resolved_rows),
        "unresolved_pairs": len(unresolved_ids),
        "reviewer_agreement": agreement["observed_agreement"],
        "cohens_kappa": agreement["kappa"],
        "adjudication_form": str(adjudication_form_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or summarize a blind semantic audit of inherited CVE-to-ATT&CK gold labels."
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create deterministic reviewer forms.")
    prepare.add_argument("--records", required=True, help="Canonical records.jsonl containing raw_text.")
    prepare.add_argument("--labels", required=True, help="Aligned labels.jsonl.")
    prepare.add_argument("--tech-index", required=True, help="Active technique_text_index.jsonl.")
    prepare.add_argument(
        "--split-assignments",
        help="Optional split_assignments.jsonl used only for post-audit subgroup summaries.",
    )
    prepare.add_argument("--ids-file", help="Optional ID restriction; no predictions or metrics are read.")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-per-technique", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=20260805)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_audit)

    summarize = subparsers.add_parser(
        "summarize", help="Calculate agreement and produce adjudicated label-quality summaries."
    )
    summarize.add_argument("--sample", required=True, help="audit_sample.jsonl from prepare mode.")
    summarize.add_argument("--reviewer1", required=True, help="Completed reviewer 1 CSV.")
    summarize.add_argument("--reviewer2", required=True, help="Completed reviewer 2 CSV.")
    summarize.add_argument(
        "--adjudication",
        help="Optional completed adjudication CSV. Omit on the first pass to generate the form.",
    )
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--allow-incomplete", action="store_true")
    summarize.add_argument("--bootstrap-repetitions", type=int, default=5000)
    summarize.add_argument("--confidence", type=float, default=0.95)
    summarize.add_argument("--seed", type=int, default=20260805)
    summarize.add_argument("--overwrite", action="store_true")
    summarize.set_defaults(func=summarize_audit)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "confidence", 0.95) <= 0 or getattr(args, "confidence", 0.95) >= 1:
        parser.error("--confidence must be between 0 and 1")
    if getattr(args, "bootstrap_repetitions", 0) < 0:
        parser.error("--bootstrap-repetitions cannot be negative")
    try:
        result = args.func(args)
    except AuditError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
