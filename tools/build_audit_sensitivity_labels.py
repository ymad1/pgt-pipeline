#!/usr/bin/env python3
"""Build auditable label subsets for semantic-quality sensitivity analysis.

This tool never changes the inherited gold labels in place. It intersects an
existing labels.jsonl file with a completed semantic-audit CSV and writes
separate JSONL label files for:

* audited_all
* directly_supported
* supported_or_plausible
* insufficient_or_unsupported (diagnostic only)

The unit of filtering is a CVE-technique gold-label pair. CVEs with no labels
remaining in a subset are omitted from that subset. The output is suitable for
pgt.compare_rankers with ``--id_policy intersection``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

SCRIPT_VERSION = "audit-sensitivity-labels-v1.0.0"
VALID_DECISIONS = {
    "directly_supported",
    "inferential_or_plausible",
    "insufficient_text",
    "unsupported",
}
SUBSETS: Mapping[str, Set[str]] = {
    "audited_all": set(VALID_DECISIONS),
    "directly_supported": {"directly_supported"},
    "supported_or_plausible": {
        "directly_supported",
        "inferential_or_plausible",
    },
    "insufficient_or_unsupported": {
        "insufficient_text",
        "unsupported",
    },
}


class SensitivityLabelError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SensitivityLabelError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise SensitivityLabelError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def _normalize_labels(value: Any, *, input_id: str) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise SensitivityLabelError(
            f"labels must be a string or list for {input_id}; got {type(value).__name__}"
        )
    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        technique_id = str(raw).strip().upper()
        if not technique_id:
            continue
        if technique_id in seen:
            continue
        seen.add(technique_id)
        normalized.append(technique_id)
    if not normalized:
        raise SensitivityLabelError(f"No valid labels for {input_id}")
    return normalized


def _load_labels(path: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    labels: Dict[str, List[str]] = {}
    order: List[str] = []
    for row_no, row in enumerate(_read_jsonl(path), start=1):
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            raise SensitivityLabelError(f"Missing input_id in {path}, record {row_no}")
        if input_id in labels:
            raise SensitivityLabelError(f"Duplicate input_id in labels: {input_id}")
        labels[input_id] = _normalize_labels(row.get("labels"), input_id=input_id)
        order.append(input_id)
    if not labels:
        raise SensitivityLabelError(f"No label records found in {path}")
    return labels, order


def _decision_column(fieldnames: Sequence[str]) -> str:
    for name in ("ai_decision", "final_decision", "decision"):
        if name in fieldnames:
            return name
    raise SensitivityLabelError(
        "Audit CSV must contain one of: ai_decision, final_decision, decision"
    )


def _load_audit(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], str]:
    pairs: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SensitivityLabelError(f"Audit CSV has no header: {path}")
        decision_field = _decision_column(reader.fieldnames)
        for line_no, row in enumerate(reader, start=2):
            input_id = str(row.get("cve_id") or row.get("input_id") or "").strip()
            technique_id = str(row.get("technique_id") or "").strip().upper()
            decision = str(row.get(decision_field) or "").strip().lower()
            if not input_id or not technique_id:
                raise SensitivityLabelError(
                    f"Missing cve_id/input_id or technique_id at {path}:{line_no}"
                )
            if decision not in VALID_DECISIONS:
                raise SensitivityLabelError(
                    f"Invalid decision {decision!r} for {input_id}/{technique_id} "
                    f"at {path}:{line_no}"
                )
            key = (input_id, technique_id)
            if key in pairs:
                raise SensitivityLabelError(
                    f"Duplicate audited pair {input_id}/{technique_id} in {path}"
                )
            pairs[key] = {
                "decision": decision,
                "sample_id": str(row.get("sample_id") or "").strip(),
                "split": str(row.get("split") or "").strip(),
                "ai_confidence": str(
                    row.get("ai_confidence") or row.get("reviewer_confidence") or ""
                ).strip(),
            }
    if not pairs:
        raise SensitivityLabelError(f"No audit decisions found in {path}")
    return pairs, decision_field


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _write_ids(path: Path, values: Sequence[str]) -> None:
    _atomic_write_text(path, "".join(f"{value}\n" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create semantic-audit label subsets without modifying inherited gold labels."
    )
    parser.add_argument("--labels", required=True, help="Original labels.jsonl for a split.")
    parser.add_argument("--audit_csv", required=True, help="Completed semantic-audit CSV.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--expected_split",
        choices=("development", "test"),
        help="When audit rows carry split values, fail if matching CVEs use another split.",
    )
    parser.add_argument(
        "--allow_unmatched_audit_pairs",
        action="store_true",
        help="Allow audit pairs whose CVE is present in labels but whose technique is not a gold label.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels).resolve()
    audit_path = Path(args.audit_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace generated files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, label_order = _load_labels(labels_path)
    audit_pairs, decision_field = _load_audit(audit_path)

    input_pairs = {(input_id, technique_id) for input_id, values in labels.items() for technique_id in values}
    audited_pairs_in_split: Dict[Tuple[str, str], Dict[str, str]] = {}
    unmatched_same_cve: List[Tuple[str, str]] = []
    split_mismatches: List[Tuple[str, str, str]] = []

    for pair, metadata in audit_pairs.items():
        input_id, technique_id = pair
        if input_id not in labels:
            continue
        if args.expected_split and metadata.get("split") and metadata["split"] != args.expected_split:
            split_mismatches.append((input_id, technique_id, metadata["split"]))
        if pair not in input_pairs:
            unmatched_same_cve.append(pair)
            continue
        audited_pairs_in_split[pair] = metadata

    if split_mismatches:
        raise SensitivityLabelError(
            "Audit split mismatch; examples: " + ", ".join(
                f"{cve}/{technique}={split}" for cve, technique, split in split_mismatches[:10]
            )
        )
    if unmatched_same_cve and not args.allow_unmatched_audit_pairs:
        raise SensitivityLabelError(
            "Audited pair is not an inherited gold pair in the supplied labels; examples: "
            + ", ".join(f"{cve}/{technique}" for cve, technique in unmatched_same_cve[:10])
        )
    if not audited_pairs_in_split:
        raise SensitivityLabelError(
            "No audited CVE-technique pairs overlap the supplied labels file."
        )

    output_files: Dict[str, Dict[str, Any]] = {}
    audit_pair_rows: List[Dict[str, Any]] = []
    for (input_id, technique_id), metadata in sorted(audited_pairs_in_split.items()):
        audit_pair_rows.append(
            {
                "input_id": input_id,
                "technique_id": technique_id,
                "decision": metadata["decision"],
                "sample_id": metadata.get("sample_id", ""),
                "ai_confidence": metadata.get("ai_confidence", ""),
            }
        )
    pair_table_path = output_dir / "audited_pairs_in_split.csv"
    with pair_table_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("input_id", "technique_id", "decision", "sample_id", "ai_confidence"),
        )
        writer.writeheader()
        writer.writerows(audit_pair_rows)

    subset_summary: Dict[str, Any] = {}
    for subset_name, allowed in SUBSETS.items():
        selected: MutableMapping[str, List[str]] = defaultdict(list)
        for input_id in label_order:
            for technique_id in labels[input_id]:
                metadata = audited_pairs_in_split.get((input_id, technique_id))
                if metadata and metadata["decision"] in allowed:
                    selected[input_id].append(technique_id)

        rows = [
            {"input_id": input_id, "labels": selected[input_id]}
            for input_id in label_order
            if selected.get(input_id)
        ]
        ids = [row["input_id"] for row in rows]
        label_path = output_dir / f"labels_{subset_name}.jsonl"
        ids_path = output_dir / f"ids_{subset_name}.txt"
        _write_jsonl(label_path, rows)
        _write_ids(ids_path, ids)
        assignment_count = sum(len(row["labels"]) for row in rows)
        subset_summary[subset_name] = {
            "allowed_decisions": sorted(allowed),
            "cves": len(rows),
            "label_assignments": assignment_count,
            "labels_file": label_path.name,
            "ids_file": ids_path.name,
        }
        output_files[label_path.name] = {
            "sha256": _sha256_file(label_path),
            "bytes": label_path.stat().st_size,
        }
        output_files[ids_path.name] = {
            "sha256": _sha256_file(ids_path),
            "bytes": ids_path.stat().st_size,
        }

    summary = {
        "script_version": SCRIPT_VERSION,
        "labels_file": str(labels_path),
        "audit_csv": str(audit_path),
        "decision_field": decision_field,
        "input_cves": len(labels),
        "input_label_assignments": sum(len(values) for values in labels.values()),
        "audited_pairs_overlapping_split": len(audited_pairs_in_split),
        "audited_cves_overlapping_split": len({pair[0] for pair in audited_pairs_in_split}),
        "unmatched_audit_pairs_for_present_cves": len(unmatched_same_cve),
        "subsets": subset_summary,
        "evaluation_note": (
            "Use these files only for sensitivity analysis. Keep the original labels file "
            "as the primary evaluation and use pgt.compare_rankers --id_policy intersection."
        ),
    }
    summary_path = output_dir / "semantic_sensitivity_summary.json"
    _atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    output_files[pair_table_path.name] = {
        "sha256": _sha256_file(pair_table_path),
        "bytes": pair_table_path.stat().st_size,
    }
    output_files[summary_path.name] = {
        "sha256": _sha256_file(summary_path),
        "bytes": summary_path.stat().st_size,
    }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "input_sha256": {
            "labels": _sha256_file(labels_path),
            "audit_csv": _sha256_file(audit_path),
        },
        "configuration": {
            "expected_split": args.expected_split,
            "allow_unmatched_audit_pairs": args.allow_unmatched_audit_pairs,
        },
        "output_sha256": output_files,
    }
    manifest_path = output_dir / "semantic_sensitivity_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
