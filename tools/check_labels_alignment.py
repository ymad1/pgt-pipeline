#!/usr/bin/env python3
"""Audit CVE-to-ATT&CK dataset alignment and split independence.

The checker is designed for the artifacts emitted by
``tools/make_cve2attck_jsonl.py``.  It validates the complete data boundary
before any retrieval or LLM experiment is run:

* labels are always JSON lists containing active ATT&CK technique IDs;
* CVE identifiers are unique after base-CVE normalization;
* ``labels.jsonl``, ``sentences.jsonl``, ``records.jsonl`` and ``ids.txt``
  describe the same records in the same deterministic order;
* output hashes recorded in ``dataset_manifest.json`` still match the files;
* optional X/y source CSVs reconstruct exactly the same per-CVE label unions;
* an optional second split has zero base-CVE overlap with the audited split.

The program writes a machine-readable JSON report and exits with status 1 on
any audit error.  Warnings do not change the exit status unless
``--warnings_as_errors`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd


PIPELINE_VERSION = "labels-alignment-audit-v2.0.0"
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
CVE_RE = re.compile(r"^CVE[-_]\d{4}[-_]\d{4,}(?:[-_].*)?$", re.IGNORECASE)
AUGMENTATION_SUFFIX_RE = re.compile(r"_(?:augumented|augmented)_\d+$", re.IGNORECASE)
KNOWN_ID_COLUMNS = {"input_id", "cve_id", "cve", "id", "name"}
KNOWN_LABEL_COLUMNS = {
    "technique_id",
    "technique",
    "attack_technique",
    "tactic_technique",
    "label",
    "labels",
}


class AuditFailure(RuntimeError):
    """Raised for malformed inputs that prevent a meaningful audit."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_col(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def normalize_input_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "", text)
    return text.upper()


def base_input_id(value: Any) -> str:
    return AUGMENTATION_SUFFIX_RE.sub("", normalize_input_id(value))


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - message path
        raise AuditFailure(f"Cannot parse JSON file {path}: {exc}") from exc


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditFailure(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise AuditFailure(f"Expected an object at {path}:{line_no}")
                obj["__line__"] = line_no
                rows.append(obj)
    except OSError as exc:
        raise AuditFailure(f"Cannot read {path}: {exc}") from exc
    return rows


def _find_explicit_id_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _norm_col(col) in KNOWN_ID_COLUMNS:
            vals = df[col].astype(str).str.strip()
            if vals.map(lambda x: bool(CVE_RE.match(x))).any():
                return str(col)
    best: Optional[str] = None
    best_count = 0
    for col in df.columns:
        vals = df[col].astype(str).str.strip()
        count = int(vals.map(lambda x: bool(CVE_RE.match(x))).sum())
        if count > best_count:
            best, best_count = str(col), count
    return best if best_count else None


def _detect_label_col(df: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    for col in df.columns:
        if col != id_col and _norm_col(col) in KNOWN_LABEL_COLUMNS:
            return str(col)
    return None


def _is_active_onehot(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def load_attack_catalog(path: Path) -> Tuple[Dict[str, str], Set[str]]:
    data = _json_load(path)
    by_name: MutableMapping[str, Set[str]] = defaultdict(set)
    active_ids: Set[str] = set()
    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if bool(obj.get("revoked")) or bool(obj.get("x_mitre_deprecated")):
            continue
        tid: Optional[str] = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                candidate = str(ref["external_id"]).strip().upper()
                if TECHNIQUE_ID_RE.fullmatch(candidate):
                    tid = candidate
                    break
        name = str(obj.get("name") or "").strip()
        if tid and name:
            active_ids.add(tid)
            by_name[_norm_col(name)].add(tid)

    # ATT&CK can contain duplicate display names (for example across platforms or
    # object versions).  Keep only names that resolve uniquely.  If an ambiguous
    # name is actually used by a source label, _resolve_label will reject it as
    # unmapped instead of silently choosing one technique.
    unique_name_to_id = {name: next(iter(ids)) for name, ids in by_name.items() if len(ids) == 1}
    return unique_name_to_id, active_ids


def _resolve_label(value: Any, name_to_id: Mapping[str, str], active_ids: Set[str]) -> Optional[str]:
    label = str(value).strip()
    if not label or label.casefold() == "nan":
        return None
    upper = label.upper()
    if TECHNIQUE_ID_RE.fullmatch(upper):
        if upper not in active_ids:
            raise AuditFailure(f"Inactive or unknown ATT&CK ID in source labels: {label}")
        return upper
    mapped = name_to_id.get(_norm_col(label))
    if mapped is None:
        raise AuditFailure(f"Cannot map source label name to an active ATT&CK ID: {label}")
    return mapped


def reconstruct_source_labels(
    x_csv: Path,
    y_csv: Path,
    enterprise_attack: Path,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Reconstruct the base-CVE label union from the original paired CSVs."""
    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)
    x_id_col = _find_explicit_id_col(X)
    if x_id_col is None:
        raise AuditFailure(f"No CVE identifier column found in {x_csv}")
    x_ids = [normalize_input_id(x) for x in X[x_id_col].tolist()]
    if any(not x for x in x_ids):
        raise AuditFailure(f"Blank CVE identifier found in {x_csv}")

    y_id_col = _find_explicit_id_col(y)
    label_col = _detect_label_col(y, y_id_col)
    name_to_id, active_ids = load_attack_catalog(enterprise_attack)

    row_labels: List[Set[str]] = []
    alignment: str
    if label_col is not None:
        if y_id_col is not None:
            alignment = "explicit_id"
            by_id: MutableMapping[str, Set[str]] = defaultdict(set)
            for _, row in y.iterrows():
                iid = normalize_input_id(row[y_id_col])
                tid = _resolve_label(row[label_col], name_to_id, active_ids)
                if iid and tid:
                    by_id[iid].add(tid)
            row_labels = [set(by_id.get(iid, set())) for iid in x_ids]
        else:
            alignment = "row_order"
            if len(X) != len(y):
                raise AuditFailure(f"X/y row mismatch: X={len(X)}, y={len(y)}")
            for value in y[label_col].tolist():
                tid = _resolve_label(value, name_to_id, active_ids)
                row_labels.append({tid} if tid else set())
    else:
        alignment = "explicit_id" if y_id_col else "row_order"
        onehot_cols = [str(c) for c in y.columns if str(c) != y_id_col]
        col_to_tid: Dict[str, str] = {}
        for col in onehot_cols:
            col_to_tid[col] = _resolve_label(col, name_to_id, active_ids) or ""
        if y_id_col is not None:
            by_id: MutableMapping[str, Set[str]] = defaultdict(set)
            for _, row in y.iterrows():
                iid = normalize_input_id(row[y_id_col])
                if not iid:
                    continue
                for col, tid in col_to_tid.items():
                    if tid and _is_active_onehot(row[col]):
                        by_id[iid].add(tid)
            row_labels = [set(by_id.get(iid, set())) for iid in x_ids]
        else:
            if len(X) != len(y):
                raise AuditFailure(f"X/y row mismatch: X={len(X)}, y={len(y)}")
            for _, row in y.iterrows():
                labels = {tid for col, tid in col_to_tid.items() if tid and _is_active_onehot(row[col])}
                row_labels.append(labels)

    unions: MutableMapping[str, Set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    for iid, labels in zip(x_ids, row_labels):
        bid = base_input_id(iid)
        unions[bid].update(labels)
        source_counts[bid] += 1

    expected = {iid: sorted(labels) for iid, labels in sorted(unions.items())}
    return expected, {
        "x_rows": int(len(X)),
        "y_rows": int(len(y)),
        "x_id_column": x_id_col,
        "y_id_column": y_id_col,
        "y_label_column": label_col,
        "alignment": alignment,
        "base_cves": len(expected),
        "duplicate_base_cves": sum(1 for count in source_counts.values() if count > 1),
        "empty_label_base_cves": sum(1 for labels in expected.values() if not labels),
    }


def _add_error(report: Dict[str, Any], code: str, message: str, **details: Any) -> None:
    item = {"code": code, "message": message}
    if details:
        item["details"] = details
    report["errors"].append(item)


def _add_warning(report: Dict[str, Any], code: str, message: str, **details: Any) -> None:
    item = {"code": code, "message": message}
    if details:
        item["details"] = details
    report["warnings"].append(item)


def _index_unique_rows(
    rows: Iterable[Dict[str, Any]],
    source_name: str,
    report: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    index: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        iid = normalize_input_id(row.get("input_id", ""))
        line_no = row.get("__line__")
        if not iid:
            _add_error(report, "blank_input_id", f"Blank input_id in {source_name}", line=line_no)
            continue
        if iid in index:
            _add_error(
                report,
                "duplicate_input_id",
                f"Duplicate input_id {iid} in {source_name}",
                first_line=index[iid].get("__line__"),
                duplicate_line=line_no,
            )
            continue
        row = dict(row)
        row["input_id"] = iid
        index[iid] = row
        order.append(iid)
    return index, order


def _load_ids(path: Path, report: Dict[str, Any], source_name: str) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        iid = normalize_input_id(raw)
        if not iid:
            continue
        if iid in seen:
            _add_error(report, "duplicate_id_line", f"Duplicate ID {iid} in {source_name}", line=line_no)
        else:
            seen.add(iid)
            ids.append(iid)
    return ids


def _dataset_ids_from_path(path: Path, report: Dict[str, Any], label: str) -> List[str]:
    if path.is_dir():
        ids_file = path / "ids.txt"
        labels_file = path / "labels.jsonl"
        if ids_file.exists():
            return _load_ids(ids_file, report, f"{label}/ids.txt")
        if labels_file.exists():
            rows = _read_jsonl(labels_file)
            _, order = _index_unique_rows(rows, f"{label}/labels.jsonl", report)
            return order
        raise AuditFailure(f"Neither ids.txt nor labels.jsonl exists in {path}")
    if path.suffix.casefold() == ".jsonl":
        rows = _read_jsonl(path)
        _, order = _index_unique_rows(rows, str(path), report)
        return order
    return _load_ids(path, report, str(path))


def audit_dataset(
    dataset_dir: Path,
    *,
    enterprise_attack: Path,
    x_csv: Optional[Path],
    y_csv: Optional[Path],
    other_split: Optional[Path],
    warnings_as_errors: bool,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_dir": str(dataset_dir.resolve()),
        "status": "UNKNOWN",
        "errors": [],
        "warnings": [],
        "statistics": {},
        "checks": {},
        "inputs": {},
    }

    required = {
        "labels": dataset_dir / "labels.jsonl",
        "sentences": dataset_dir / "sentences.jsonl",
        "records": dataset_dir / "records.jsonl",
        "ids": dataset_dir / "ids.txt",
        "manifest": dataset_dir / "dataset_manifest.json",
    }
    for name, path in required.items():
        if not path.exists():
            _add_error(report, "missing_required_file", f"Missing required file: {path}", artifact=name)
    if report["errors"]:
        report["status"] = "FAIL"
        return report

    name_to_id, active_ids = load_attack_catalog(enterprise_attack)
    del name_to_id
    report["inputs"]["enterprise_attack"] = {
        "path": str(enterprise_attack.resolve()),
        "sha256": _sha256_file(enterprise_attack),
        "active_technique_ids": len(active_ids),
    }

    label_rows = _read_jsonl(required["labels"])
    sentence_rows = _read_jsonl(required["sentences"])
    record_rows = _read_jsonl(required["records"])
    labels_idx, labels_order = _index_unique_rows(label_rows, "labels.jsonl", report)
    sentences_idx, sentences_order = _index_unique_rows(sentence_rows, "sentences.jsonl", report)
    records_idx, records_order = _index_unique_rows(record_rows, "records.jsonl", report)
    ids_order = _load_ids(required["ids"], report, "ids.txt")

    canonical_labels: Dict[str, List[str]] = {}
    label_frequency: Counter[str] = Counter()
    string_label_rows = 0
    empty_label_rows = 0
    invalid_label_rows = 0
    for iid, row in labels_idx.items():
        labels = row.get("labels")
        if isinstance(labels, str):
            string_label_rows += 1
            _add_error(report, "labels_not_list", f"labels is a string for {iid}", line=row.get("__line__"))
            labels_list = [labels]
        elif isinstance(labels, list):
            labels_list = labels
        else:
            _add_error(report, "labels_not_list", f"labels is not a list for {iid}", line=row.get("__line__"))
            labels_list = []

        normalized: List[str] = []
        for value in labels_list:
            tid = str(value).strip().upper()
            if not TECHNIQUE_ID_RE.fullmatch(tid):
                invalid_label_rows += 1
                _add_error(report, "invalid_technique_id", f"Invalid ATT&CK technique ID {value!r} for {iid}")
                continue
            if tid not in active_ids:
                invalid_label_rows += 1
                _add_error(report, "inactive_technique_id", f"Inactive or unknown ATT&CK ID {tid} for {iid}")
                continue
            normalized.append(tid)
        deduped = sorted(set(normalized))
        if normalized != deduped:
            _add_error(report, "labels_not_canonical", f"Labels are not sorted unique for {iid}", observed=normalized, expected=deduped)
        if not deduped:
            empty_label_rows += 1
            _add_error(report, "empty_labels", f"No valid labels for {iid}")
        canonical_labels[iid] = deduped
        label_frequency.update(deduped)

    base_counts = Counter(base_input_id(iid) for iid in labels_idx)
    duplicate_base_ids = sorted(iid for iid, count in base_counts.items() if count > 1)
    if duplicate_base_ids:
        _add_error(
            report,
            "duplicate_base_cve",
            "Multiple output records collapse to the same base CVE",
            count=len(duplicate_base_ids),
            examples=duplicate_base_ids[:20],
        )

    expected_set = set(labels_idx)
    for artifact_name, artifact_order in (
        ("sentences.jsonl", sentences_order),
        ("records.jsonl", records_order),
        ("ids.txt", ids_order),
    ):
        actual_set = set(artifact_order)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            _add_error(
                report,
                "artifact_id_mismatch",
                f"ID set mismatch between labels.jsonl and {artifact_name}",
                missing_count=len(missing),
                extra_count=len(extra),
                missing_examples=missing[:20],
                extra_examples=extra[:20],
            )
        if artifact_order != labels_order:
            _add_error(
                report,
                "artifact_order_mismatch",
                f"Record order differs between labels.jsonl and {artifact_name}",
            )

    malformed_sentence_rows = 0
    for iid, row in sentences_idx.items():
        raw_text = row.get("raw_text")
        sentences = row.get("sentences")
        if not isinstance(raw_text, str) or not raw_text.strip():
            malformed_sentence_rows += 1
            _add_error(report, "blank_raw_text", f"Blank raw_text for {iid} in sentences.jsonl")
        if not isinstance(sentences, dict) or not sentences:
            malformed_sentence_rows += 1
            _add_error(report, "invalid_sentences", f"sentences must be a non-empty object for {iid}")
        else:
            for eid, text in sentences.items():
                if not re.fullmatch(r"E\d+", str(eid)):
                    malformed_sentence_rows += 1
                    _add_error(report, "invalid_evidence_id", f"Invalid evidence ID {eid!r} for {iid}")
                if not isinstance(text, str) or not text.strip():
                    malformed_sentence_rows += 1
                    _add_error(report, "blank_evidence_text", f"Blank evidence text {eid!r} for {iid}")

    record_mismatches = 0
    for iid in sorted(set(records_idx) & set(labels_idx) & set(sentences_idx)):
        record = records_idx[iid]
        rec_labels = record.get("labels")
        if not isinstance(rec_labels, list) or sorted(set(str(x).upper() for x in rec_labels)) != canonical_labels[iid]:
            record_mismatches += 1
            _add_error(report, "record_label_mismatch", f"records.jsonl labels differ for {iid}")
        if str(record.get("raw_text") or "") != str(sentences_idx[iid].get("raw_text") or ""):
            record_mismatches += 1
            _add_error(report, "record_text_mismatch", f"records.jsonl raw_text differs for {iid}")
        if record.get("sentences") != sentences_idx[iid].get("sentences"):
            record_mismatches += 1
            _add_error(report, "record_sentences_mismatch", f"records.jsonl sentences differ for {iid}")

    manifest = _json_load(required["manifest"])
    output_hash_mismatches = 0
    manifest_outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    if not isinstance(manifest_outputs, dict):
        _add_error(report, "manifest_outputs_missing", "dataset_manifest.json does not contain an outputs object")
    else:
        for filename, metadata in manifest_outputs.items():
            path = dataset_dir / filename
            if not path.exists():
                output_hash_mismatches += 1
                _add_error(report, "manifest_output_missing", f"Manifest output is missing: {filename}")
                continue
            expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
            actual_hash = _sha256_file(path)
            if expected_hash != actual_hash:
                output_hash_mismatches += 1
                _add_error(
                    report,
                    "manifest_hash_mismatch",
                    f"SHA-256 mismatch for {filename}",
                    expected=expected_hash,
                    actual=actual_hash,
                )

    source_check: Optional[Dict[str, Any]] = None
    if (x_csv is None) != (y_csv is None):
        _add_error(report, "incomplete_source_pair", "Both --x_csv and --y_csv are required for source reconstruction")
    elif x_csv is not None and y_csv is not None:
        expected_labels, source_stats = reconstruct_source_labels(x_csv, y_csv, enterprise_attack)
        source_check = source_stats
        generated_ids = set(canonical_labels)
        source_ids = set(expected_labels)
        missing = sorted(source_ids - generated_ids)
        extra = sorted(generated_ids - source_ids)
        mismatches: List[Dict[str, Any]] = []
        for iid in sorted(source_ids & generated_ids):
            if expected_labels[iid] != canonical_labels[iid]:
                mismatches.append({
                    "input_id": iid,
                    "source": expected_labels[iid],
                    "generated": canonical_labels[iid],
                })
        if missing or extra or mismatches:
            _add_error(
                report,
                "source_reconstruction_mismatch",
                "Generated labels do not exactly match the union reconstructed from X/y",
                missing_count=len(missing),
                extra_count=len(extra),
                mismatch_count=len(mismatches),
                missing_examples=missing[:20],
                extra_examples=extra[:20],
                mismatch_examples=mismatches[:20],
            )
        report["inputs"]["x_csv"] = {"path": str(x_csv.resolve()), "sha256": _sha256_file(x_csv)}
        report["inputs"]["y_csv"] = {"path": str(y_csv.resolve()), "sha256": _sha256_file(y_csv)}

    split_overlap: List[str] = []
    if other_split is not None:
        other_ids = _dataset_ids_from_path(other_split, report, "other_split")
        current_base = {base_input_id(iid) for iid in labels_order}
        other_base = {base_input_id(iid) for iid in other_ids}
        split_overlap = sorted(current_base & other_base)
        if split_overlap:
            _add_error(
                report,
                "split_leakage",
                "Base CVEs overlap between the audited dataset and the other split",
                overlap_count=len(split_overlap),
                examples=split_overlap[:50],
            )

    report["statistics"] = {
        "records": len(labels_idx),
        "unique_base_cves": len(base_counts),
        "labels_total_assignments": int(sum(label_frequency.values())),
        "labels_unique_techniques": len(label_frequency),
        "string_label_rows": string_label_rows,
        "empty_label_rows": empty_label_rows,
        "invalid_label_values": invalid_label_rows,
        "duplicate_base_cves": len(duplicate_base_ids),
        "malformed_sentence_items": malformed_sentence_rows,
        "record_mismatches": record_mismatches,
        "manifest_hash_mismatches": output_hash_mismatches,
        "other_split_overlap": len(split_overlap),
        "source_reconstruction": source_check,
        "label_frequency": dict(sorted(label_frequency.items())),
    }
    report["checks"] = {
        "labels_are_lists": string_label_rows == 0,
        "labels_are_active_ids": invalid_label_rows == 0,
        "labels_nonempty": empty_label_rows == 0,
        "base_cves_unique": not duplicate_base_ids,
        "artifact_ids_and_order_aligned": labels_order == sentences_order == records_order == ids_order,
        "records_content_aligned": record_mismatches == 0,
        "sentences_valid": malformed_sentence_rows == 0,
        "manifest_hashes_valid": output_hash_mismatches == 0,
        "source_labels_exact": source_check is None or not any(
            err["code"] == "source_reconstruction_mismatch" for err in report["errors"]
        ),
        "split_independent": other_split is None or not split_overlap,
    }

    if warnings_as_errors and report["warnings"]:
        for warning in report["warnings"]:
            _add_error(report, "warning_promoted", warning["message"], original_code=warning["code"])
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CVE/ATT&CK labels, artifacts and split independence.")
    parser.add_argument("dataset_dir", type=Path, help="Directory emitted by make_cve2attck_jsonl.py")
    parser.add_argument("--enterprise_attack", type=Path, required=True)
    parser.add_argument("--x_csv", type=Path)
    parser.add_argument("--y_csv", type=Path)
    parser.add_argument(
        "--other_split",
        type=Path,
        help="Other dataset directory, ids.txt, or labels.jsonl used to detect base-CVE leakage.",
    )
    parser.add_argument("--report", type=Path, help="Output JSON report path")
    parser.add_argument("--warnings_as_errors", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = audit_dataset(
            args.dataset_dir,
            enterprise_attack=args.enterprise_attack,
            x_csv=args.x_csv,
            y_csv=args.y_csv,
            other_split=args.other_split,
            warnings_as_errors=bool(args.warnings_as_errors),
        )
    except AuditFailure as exc:
        report = {
            "pipeline_version": PIPELINE_VERSION,
            "dataset_dir": str(args.dataset_dir.resolve()),
            "status": "FAIL",
            "errors": [{"code": "audit_failure", "message": str(exc)}],
            "warnings": [],
            "statistics": {},
            "checks": {},
            "inputs": {},
        }

    report_path = args.report or (args.dataset_dir / "alignment_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"status: {report['status']}")
    print(f"report: {report_path}")
    stats = report.get("statistics") or {}
    if stats:
        print(
            "records={records}, techniques={techniques}, errors={errors}, warnings={warnings}".format(
                records=stats.get("records", 0),
                techniques=stats.get("labels_unique_techniques", 0),
                errors=len(report.get("errors", [])),
                warnings=len(report.get("warnings", [])),
            )
        )
    if report.get("errors"):
        for item in report["errors"][:20]:
            print(f"ERROR [{item['code']}]: {item['message']}")
        if len(report["errors"]) > 20:
            print(f"... {len(report['errors']) - 20} additional errors in the JSON report")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
