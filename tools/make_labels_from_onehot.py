#!/usr/bin/env python3
"""Deprecated compatibility entry point for one-hot CVE labels.

The authoritative data-construction implementation is
``tools/make_cve2attck_jsonl.py``.  This wrapper preserves the historical
four-positional-argument command while delegating every transformation to the
canonical builder.  It therefore inherits the same guarantees:

* active ATT&CK technique-name resolution;
* explicit X/y alignment checks;
* base-CVE normalization;
* union of labels across duplicate and augmented rows;
* list-valued labels for every record;
* deterministic provenance and input/output hashes.

Unlike the historical script, this command never writes an unaudited labels
file by itself.  It always creates a complete sidecar dataset directory and a
compatibility manifest next to the requested labels output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Running from the repository root: python -m tools.make_labels_from_onehot
    from tools.make_cve2attck_jsonl import build_dataset
except ImportError:  # Running directly: python tools/make_labels_from_onehot.py
    from make_cve2attck_jsonl import build_dataset


WRAPPER_VERSION = "onehot-label-compat-v2.0.0"
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _default_artifact_dir(out_labels: Path) -> Path:
    return out_labels.parent / f"{out_labels.stem}.dataset"


def _default_manifest_path(out_labels: Path) -> Path:
    return out_labels.parent / f"{out_labels.name}.manifest.json"


def _validate_labels_file(path: Path, *, allow_empty_labels: bool) -> Dict[str, Any]:
    seen_ids: set[str] = set()
    technique_ids: set[str] = set()
    records = 0
    assignments = 0
    empty_records: List[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in generated labels at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Generated labels line {line_number} is not a JSON object."
                )

            input_id = row.get("input_id")
            labels = row.get("labels")
            if not isinstance(input_id, str) or not input_id:
                raise ValueError(
                    f"Generated labels line {line_number} has no valid input_id."
                )
            if input_id in seen_ids:
                raise ValueError(f"Duplicate generated input_id: {input_id}")
            seen_ids.add(input_id)

            if not isinstance(labels, list):
                raise ValueError(
                    f"Generated labels for {input_id} are not a JSON list."
                )
            if not labels:
                empty_records.append(input_id)
            if labels != sorted(set(labels)):
                raise ValueError(
                    f"Generated labels for {input_id} are not sorted and unique."
                )
            for label in labels:
                if not isinstance(label, str) or not TECHNIQUE_ID_RE.fullmatch(label):
                    raise ValueError(
                        f"Generated label {label!r} for {input_id} is not an ATT&CK technique ID."
                    )
                technique_ids.add(label)
                assignments += 1
            records += 1

    if records == 0:
        raise ValueError("The canonical builder produced an empty labels file.")
    if empty_records and not allow_empty_labels:
        examples = ", ".join(empty_records[:10])
        raise ValueError(
            f"Generated labels contain {len(empty_records)} empty-label records. "
            f"Examples: {examples}. Use --allow_empty_labels only for diagnosis."
        )

    return {
        "records": records,
        "techniques": len(technique_ids),
        "label_assignments": assignments,
        "empty_label_records": len(empty_records),
    }


def _write_temp_file(parent: Path, prefix: str, content: bytes) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=prefix, dir=str(parent))
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _transactional_install(
    prepared: Sequence[Tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    """Install prepared files/directories with rollback on failure.

    ``prepared`` contains ``(temporary_path, destination_path)`` pairs.  Each
    temporary path must be on the same filesystem as its destination.
    """

    destinations = [destination for _, destination in prepared]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate destination in compatibility output transaction.")

    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite: {names}")

    backups: Dict[Path, Path] = {}
    installed: List[Path] = []
    try:
        for destination in existing:
            backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
            backups[destination] = backup

        for temporary, destination in prepared:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            if destination.is_dir():
                shutil.rmtree(destination, ignore_errors=True)
            else:
                destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            if backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)


def build_compatible_labels(
    x_csv: Path,
    y_csv: Path,
    enterprise_attack: Path,
    out_labels: Path,
    *,
    artifact_dir: Optional[Path] = None,
    split_name: str = "legacy_compat",
    keep_augmented: bool = False,
    allow_unmapped_labels: bool = False,
    allow_nonbinary_onehot: bool = False,
    allow_empty_labels: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build a canonical dataset and install a backwards-compatible labels file."""

    x_csv = x_csv.resolve()
    y_csv = y_csv.resolve()
    enterprise_attack = enterprise_attack.resolve()
    out_labels = out_labels.resolve()
    artifact_dir = (artifact_dir or _default_artifact_dir(out_labels)).resolve()
    manifest_path = _default_manifest_path(out_labels).resolve()

    for source in (x_csv, y_csv, enterprise_attack):
        if not source.is_file():
            raise FileNotFoundError(source)

    if out_labels == manifest_path or out_labels == artifact_dir:
        raise ValueError("Compatibility output paths overlap.")
    if artifact_dir == manifest_path:
        raise ValueError("Artifact directory and compatibility manifest paths overlap.")
    if artifact_dir in out_labels.parents or out_labels in artifact_dir.parents:
        raise ValueError("Labels output and artifact directory must not contain one another.")

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    temp_artifact = Path(
        tempfile.mkdtemp(prefix=f".{artifact_dir.name}.tmp-", dir=str(artifact_dir.parent))
    )
    temp_label: Optional[Path] = None
    temp_manifest: Optional[Path] = None

    try:
        canonical_manifest = build_dataset(
            x_csv,
            y_csv,
            temp_artifact,
            enterprise_attack,
            split_name=split_name,
            deduplicate_base_cve=not keep_augmented,
            allow_unmapped_labels=allow_unmapped_labels,
            allow_nonbinary_onehot=allow_nonbinary_onehot,
            overwrite=False,
        )
        canonical_labels = temp_artifact / "labels.jsonl"
        audit = _validate_labels_file(
            canonical_labels, allow_empty_labels=allow_empty_labels
        )

        label_bytes = canonical_labels.read_bytes()
        temp_label = _write_temp_file(
            out_labels.parent, f".{out_labels.name}.tmp-", label_bytes
        )

        compatibility_manifest: Dict[str, Any] = {
            "wrapper_version": WRAPPER_VERSION,
            "deprecated_entry_point": True,
            "canonical_entry_point": "tools/make_cve2attck_jsonl.py",
            "message": (
                "This labels file was produced by the canonical traceable dataset builder. "
                "Use the sidecar dataset directory for formal experiments."
            ),
            "configuration": {
                "split_name": split_name,
                "deduplicate_base_cve": not keep_augmented,
                "allow_unmapped_labels": allow_unmapped_labels,
                "allow_nonbinary_onehot": allow_nonbinary_onehot,
                "allow_empty_labels": allow_empty_labels,
            },
            "inputs": {
                "x_csv": {"name": x_csv.name, "sha256": _sha256_file(x_csv)},
                "y_csv": {"name": y_csv.name, "sha256": _sha256_file(y_csv)},
                "enterprise_attack": {
                    "name": enterprise_attack.name,
                    "sha256": _sha256_file(enterprise_attack),
                },
            },
            "outputs": {
                "labels": {
                    "name": out_labels.name,
                    "sha256": hashlib.sha256(label_bytes).hexdigest(),
                    "bytes": len(label_bytes),
                },
                "artifact_directory": artifact_dir.name,
                "canonical_dataset_manifest_sha256": _sha256_file(
                    temp_artifact / "dataset_manifest.json"
                ),
            },
            "audit": audit,
            "canonical_statistics": canonical_manifest["statistics"],
        }
        temp_manifest = _write_temp_file(
            manifest_path.parent,
            f".{manifest_path.name}.tmp-",
            _canonical_json_bytes(compatibility_manifest),
        )

        _transactional_install(
            [
                (temp_artifact, artifact_dir),
                (temp_label, out_labels),
                (temp_manifest, manifest_path),
            ],
            overwrite=overwrite,
        )
        temp_label = None
        temp_manifest = None
        return compatibility_manifest
    finally:
        if temp_artifact.exists():
            shutil.rmtree(temp_artifact, ignore_errors=True)
        if temp_label is not None:
            temp_label.unlink(missing_ok=True)
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated compatibility wrapper. Builds a complete canonical CVE/ATT&CK "
            "dataset and copies its audited labels.jsonl to the historical output path."
        )
    )
    parser.add_argument("x_csv", type=Path)
    parser.add_argument("y_csv", type=Path)
    parser.add_argument("enterprise_attack", type=Path)
    parser.add_argument("out_labels", type=Path)
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=None,
        help="Complete canonical dataset directory (default: <out-label-stem>.dataset).",
    )
    parser.add_argument("--split_name", default="legacy_compat")
    parser.add_argument("--keep_augmented", action="store_true")
    parser.add_argument("--allow_unmapped_labels", action="store_true")
    parser.add_argument("--allow_nonbinary_onehot", action="store_true")
    parser.add_argument("--allow_empty_labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(
        "NOTICE: tools/make_labels_from_onehot.py is deprecated; "
        "delegating to tools/make_cve2attck_jsonl.py.",
        file=sys.stderr,
    )
    manifest = build_compatible_labels(
        args.x_csv,
        args.y_csv,
        args.enterprise_attack,
        args.out_labels,
        artifact_dir=args.artifact_dir,
        split_name=str(args.split_name),
        keep_augmented=bool(args.keep_augmented),
        allow_unmapped_labels=bool(args.allow_unmapped_labels),
        allow_nonbinary_onehot=bool(args.allow_nonbinary_onehot),
        allow_empty_labels=bool(args.allow_empty_labels),
        overwrite=bool(args.overwrite),
    )
    audit = manifest["audit"]
    print(f"OK: {args.out_labels}")
    print(
        f"records={audit['records']}, techniques={audit['techniques']}, "
        f"assignments={audit['label_assignments']}, "
        f"empty_labels={audit['empty_label_records']}"
    )
    print(
        "Canonical dataset: "
        f"{(args.artifact_dir or _default_artifact_dir(args.out_labels)).resolve()}"
    )
    print(f"Compatibility manifest: {_default_manifest_path(args.out_labels).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
