#!/usr/bin/env python3
"""Build traceable CVE-to-ATT&CK JSONL files from paired X/y CSV files.

This tool is deliberately conservative:

* X and y are aligned by an explicit CVE identifier when y has one; otherwise
  row alignment is accepted only when X and y have the same number of rows.
* one-hot label names are resolved to active MITRE ATT&CK technique IDs using
  a supplied Enterprise ATT&CK STIX bundle.
* duplicate and augmented rows are collapsed to one base CVE by default.
* labels from every row belonging to the same base CVE are unioned before a
  single record is retained.
* every output label is a JSON list, including single-label records.
* deterministic audit artifacts record source rows, text conflicts, label
  unions, input hashes, and the exact transformation configuration.

The generated files are suitable as an auditable input layer. They do not
create a development/test split; ``ids.txt`` and provenance fields are emitted
so that a separate, fixed split step can be performed without reconstructing
source membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd


PIPELINE_VERSION = "cve2attck-jsonl-v2.0.0"
CVE_RE = re.compile(r"^CVE[-_]\d{4}[-_]\d{4,}(?:[-_].*)?$", re.IGNORECASE)
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
AUGMENTATION_SUFFIX_RE = re.compile(r"_(?:augumented|augmented)_\d+$", re.IGNORECASE)
KNOWN_ID_COLUMNS = {"input_id", "cve_id", "cve", "id", "name"}
KNOWN_TEXT_COLUMNS = {"raw_text", "text", "description", "sentence", "summary"}
KNOWN_LABEL_COLUMNS = {
    "technique_id",
    "technique",
    "attack_technique",
    "tactic_technique",
    "label",
    "labels",
}


@dataclass(frozen=True)
class SourceRow:
    source_row: int
    source_id: str
    base_input_id: str
    text: str
    labels: Tuple[str, ...]
    is_augmented: bool
    augmentation_index: Optional[int]


@dataclass(frozen=True)
class AttackCatalog:
    name_to_id: Mapping[str, str]
    ambiguous_names: Mapping[str, Tuple[str, ...]]
    active_ids: Set[str]
    source_sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def normalize_input_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "", text)
    return text.upper()


def base_input_id(input_id: str) -> str:
    return AUGMENTATION_SUFFIX_RE.sub("", input_id)


def _augmentation_index(input_id: str) -> Optional[int]:
    match = re.search(r"_(?:augumented|augmented)_(\d+)$", input_id, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _find_explicit_id_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _normalize_column_name(col) in KNOWN_ID_COLUMNS:
            values = df[col].astype(str).str.strip()
            if values.map(lambda x: bool(CVE_RE.match(x))).any():
                return str(col)

    best_col: Optional[str] = None
    best_count = 0
    for col in df.columns:
        values = df[col].astype(str).str.strip()
        count = int(values.map(lambda x: bool(CVE_RE.match(x))).sum())
        if count > best_count:
            best_col = str(col)
            best_count = count
    return best_col if best_count > 0 else None


def pick_id_col(df: pd.DataFrame) -> str:
    col = _find_explicit_id_col(df)
    if col is None:
        raise ValueError("No CVE identifier column was found in X.")
    return col


def pick_text_col(df: pd.DataFrame, id_col: str) -> str:
    for col in df.columns:
        if col != id_col and _normalize_column_name(col) in KNOWN_TEXT_COLUMNS:
            return str(col)

    string_cols: List[str] = []
    for col in df.columns:
        if col == id_col:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            string_cols.append(str(col))
    if not string_cols:
        raise ValueError("No text/string column was found in X.")

    return max(
        string_cols,
        key=lambda col: float(df[col].fillna("").astype(str).str.len().mean()),
    )


def load_attack_catalog(path: Path) -> AttackCatalog:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    by_name: MutableMapping[str, Set[str]] = defaultdict(set)
    active_ids: Set[str] = set()

    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if bool(obj.get("revoked")) or bool(obj.get("x_mitre_deprecated")):
            continue

        technique_id: Optional[str] = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                candidate = str(ref["external_id"]).strip().upper()
                if TECHNIQUE_ID_RE.fullmatch(candidate):
                    technique_id = candidate
                    break
        name = str(obj.get("name") or "").strip()
        if not name or technique_id is None:
            continue

        active_ids.add(technique_id)
        by_name[_normalize_column_name(name)].add(technique_id)

    name_to_id: Dict[str, str] = {}
    ambiguous: Dict[str, Tuple[str, ...]] = {}
    for name, ids in by_name.items():
        ordered = tuple(sorted(ids))
        if len(ordered) == 1:
            name_to_id[name] = ordered[0]
        else:
            ambiguous[name] = ordered

    return AttackCatalog(
        name_to_id=name_to_id,
        ambiguous_names=ambiguous,
        active_ids=active_ids,
        source_sha256=_sha256_file(path),
    )


def _resolve_label(raw_label: Any, catalog: AttackCatalog) -> Tuple[Optional[str], Optional[str]]:
    label = str(raw_label).strip()
    if not label or label.casefold() == "nan":
        return None, None

    upper = label.upper()
    if TECHNIQUE_ID_RE.fullmatch(upper):
        if upper in catalog.active_ids:
            return upper, None
        return None, f"inactive_or_unknown_id:{label}"

    key = _normalize_column_name(label)
    if key in catalog.name_to_id:
        return catalog.name_to_id[key], None
    if key in catalog.ambiguous_names:
        ids = ",".join(catalog.ambiguous_names[key])
        return None, f"ambiguous_name:{label}:{ids}"
    return None, f"unmapped_name:{label}"


def _is_binary_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number in (0.0, 1.0)


def _is_active_onehot(value: Any, *, allow_nonbinary: bool) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if allow_nonbinary:
        return number > 0.0
    return number == 1.0


def _detect_label_column(y: pd.DataFrame, y_id_col: Optional[str]) -> Optional[str]:
    for col in y.columns:
        if col == y_id_col:
            continue
        if _normalize_column_name(col) in KNOWN_LABEL_COLUMNS:
            return str(col)
    return None


def _parse_y_labels(
    y: pd.DataFrame,
    x_ids: Sequence[str],
    catalog: AttackCatalog,
    *,
    allow_unmapped_labels: bool,
    allow_nonbinary_onehot: bool,
) -> Tuple[List[Set[str]], Dict[str, Any]]:
    """Return one label set per X row and a deterministic mapping audit."""

    y_id_col = _find_explicit_id_col(y)
    label_col = _detect_label_column(y, y_id_col)
    mapping_audit: Dict[str, Any] = {
        "alignment": "explicit_id" if y_id_col else "row_order",
        "y_id_column": y_id_col,
        "label_column": label_col,
        "column_to_technique_id": {},
        "unmapped_labels": [],
        "nonbinary_columns": [],
    }

    unresolved: Set[str] = set()

    if label_col is not None:
        labels_by_id: MutableMapping[str, Set[str]] = defaultdict(set)
        row_labels: List[Set[str]] = []

        for row_idx, row in y.iterrows():
            technique_id, error = _resolve_label(row[label_col], catalog)
            if error:
                unresolved.add(error)
            labels = {technique_id} if technique_id else set()
            if y_id_col is None:
                row_labels.append(labels)
            else:
                iid = normalize_input_id(row[y_id_col])
                if iid:
                    labels_by_id[iid].update(labels)

        if y_id_col is None:
            if len(row_labels) != len(x_ids):
                raise ValueError(
                    f"X/y row count mismatch without y IDs: X={len(x_ids)}, y={len(row_labels)}"
                )
            result = row_labels
        else:
            result = [set(labels_by_id.get(iid, set())) for iid in x_ids]

    else:
        excluded = {y_id_col} if y_id_col else set()
        technique_columns: List[Tuple[str, Optional[str]]] = []

        for col in y.columns:
            if col in excluded:
                continue
            series = y[col]
            if not series.map(_is_binary_value).all():
                if allow_nonbinary_onehot and pd.to_numeric(series, errors="coerce").notna().all():
                    mapping_audit["nonbinary_columns"].append(str(col))
                else:
                    continue

            technique_id, error = _resolve_label(col, catalog)
            if error:
                unresolved.add(error)
                technique_columns.append((str(col), None))
            else:
                technique_columns.append((str(col), technique_id))
                mapping_audit["column_to_technique_id"][str(col)] = technique_id

        if not technique_columns:
            raise ValueError("No row-label column or ATT&CK one-hot columns were found in y.")

        def labels_for_row(row: pd.Series) -> Set[str]:
            labels: Set[str] = set()
            for col, technique_id in technique_columns:
                if technique_id and _is_active_onehot(
                    row[col], allow_nonbinary=allow_nonbinary_onehot
                ):
                    labels.add(technique_id)
            return labels

        if y_id_col is None:
            if len(y) != len(x_ids):
                raise ValueError(f"X/y row count mismatch: X={len(x_ids)}, y={len(y)}")
            result = [labels_for_row(row) for _, row in y.iterrows()]
        else:
            labels_by_id = defaultdict(set)
            for _, row in y.iterrows():
                iid = normalize_input_id(row[y_id_col])
                if iid:
                    labels_by_id[iid].update(labels_for_row(row))
            result = [set(labels_by_id.get(iid, set())) for iid in x_ids]

    mapping_audit["unmapped_labels"] = sorted(unresolved)
    if unresolved and not allow_unmapped_labels:
        examples = "; ".join(sorted(unresolved)[:10])
        raise ValueError(
            f"Unresolved ATT&CK labels ({len(unresolved)}). "
            f"Use an exact active technique ID/name or --allow_unmapped_labels. Examples: {examples}"
        )

    return result, mapping_audit


def _choose_canonical_row(rows: Sequence[SourceRow]) -> SourceRow:
    usable = [row for row in rows if row.text and row.text.casefold() != "nan"]
    if not usable:
        raise ValueError(f"No usable CVE text for {rows[0].base_input_id}")

    def key(row: SourceRow) -> Tuple[int, int, int, str]:
        # Prefer an original base-CVE row. If none exists, use the lowest
        # augmentation index and then the earliest source row.
        original_rank = 0 if not row.is_augmented else 1
        augmentation_rank = row.augmentation_index if row.augmentation_index is not None else -1
        return (original_rank, augmentation_rank, row.source_row, row.source_id)

    return min(usable, key=key)


def _build_records(
    X: pd.DataFrame,
    row_labels: Sequence[Set[str]],
    *,
    x_id_col: str,
    x_text_col: str,
    split_name: str,
    deduplicate_base_cve: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_rows: List[SourceRow] = []
    dropped_invalid_rows = 0

    for zero_idx, (_, row) in enumerate(X.iterrows()):
        iid = normalize_input_id(row[x_id_col])
        text = str(row[x_text_col]).strip()
        if not iid or not CVE_RE.match(iid) or not text or text.casefold() == "nan":
            dropped_invalid_rows += 1
            continue
        base_id = base_input_id(iid)
        aug_idx = _augmentation_index(iid)
        source_rows.append(
            SourceRow(
                source_row=zero_idx + 2,  # CSV header is line 1
                source_id=iid,
                base_input_id=base_id,
                text=text,
                labels=tuple(sorted(row_labels[zero_idx])),
                is_augmented=aug_idx is not None,
                augmentation_index=aug_idx,
            )
        )

    groups: MutableMapping[str, List[SourceRow]] = defaultdict(list)
    for row in source_rows:
        key = row.base_input_id if deduplicate_base_cve else row.source_id
        groups[key].append(row)

    output_records: List[Dict[str, Any]] = []
    duplicate_audit: List[Dict[str, Any]] = []
    label_union_count = 0
    text_conflict_groups = 0
    augmented_only_groups = 0

    for output_id in sorted(groups):
        rows = sorted(groups[output_id], key=lambda item: (item.source_row, item.source_id))
        canonical = _choose_canonical_row(rows)
        labels: Set[str] = set()
        individual_label_sets: Set[Tuple[str, ...]] = set()
        for source in rows:
            labels.update(source.labels)
            individual_label_sets.add(source.labels)

        if len(individual_label_sets) > 1:
            label_union_count += 1

        unique_texts = sorted({source.text for source in rows})
        text_conflict = len(unique_texts) > 1
        if text_conflict:
            text_conflict_groups += 1
        if all(source.is_augmented for source in rows):
            augmented_only_groups += 1

        provenance = {
            "source_split": split_name,
            "canonical_source_id": canonical.source_id,
            "canonical_source_row": canonical.source_row,
            "source_ids": [source.source_id for source in rows],
            "source_rows": [source.source_row for source in rows],
            "source_record_count": len(rows),
            "augmented_source_count": sum(source.is_augmented for source in rows),
            "labels_unioned": len(individual_label_sets) > 1,
            "text_conflict": text_conflict,
            "source_text_sha256": [
                hashlib.sha256(source.text.encode("utf-8")).hexdigest() for source in rows
            ],
        }

        record = {
            "input_id": output_id,
            "raw_text": canonical.text,
            "sentences": {"E1": canonical.text},
            "labels": sorted(labels),
            "provenance": provenance,
        }
        output_records.append(record)

        if len(rows) > 1 or text_conflict or provenance["labels_unioned"]:
            duplicate_audit.append(
                {
                    "input_id": output_id,
                    "selected_source_id": canonical.source_id,
                    "selected_source_row": canonical.source_row,
                    "source_ids": provenance["source_ids"],
                    "source_rows": provenance["source_rows"],
                    "source_labels": [list(source.labels) for source in rows],
                    "union_labels": sorted(labels),
                    "labels_unioned": provenance["labels_unioned"],
                    "text_conflict": text_conflict,
                    "unique_text_count": len(unique_texts),
                }
            )

    stats = {
        "input_x_rows": int(len(X)),
        "usable_source_rows": len(source_rows),
        "dropped_invalid_x_rows": dropped_invalid_rows,
        "output_records": len(output_records),
        "collapsed_source_rows": len(source_rows) - len(output_records),
        "augmented_source_rows": sum(row.is_augmented for row in source_rows),
        "duplicate_groups": sum(len(rows) > 1 for rows in groups.values()),
        "label_union_groups": label_union_count,
        "text_conflict_groups": text_conflict_groups,
        "augmented_only_groups": augmented_only_groups,
        "empty_label_records": sum(not record["labels"] for record in output_records),
        "multi_label_records": sum(len(record["labels"]) > 1 for record in output_records),
    }
    return output_records, duplicate_audit, stats


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    expected = [
        "sentences.jsonl",
        "labels.jsonl",
        "records.jsonl",
        "ids.txt",
        "duplicate_audit.jsonl",
        "label_mapping.json",
        "dataset_manifest.json",
    ]
    existing = [out_dir / name for name in expected if (out_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use --overwrite: {joined}")
    out_dir.mkdir(parents=True, exist_ok=True)


def build_dataset(
    x_csv: Path,
    y_csv: Path,
    out_dir: Path,
    enterprise_attack: Path,
    *,
    split_name: str,
    deduplicate_base_cve: bool,
    allow_unmapped_labels: bool,
    allow_nonbinary_onehot: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    _prepare_output_dir(out_dir, overwrite)

    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)
    x_id_col = pick_id_col(X)
    x_text_col = pick_text_col(X, x_id_col)
    x_ids = [normalize_input_id(value) for value in X[x_id_col].tolist()]

    catalog = load_attack_catalog(enterprise_attack)
    row_labels, label_mapping = _parse_y_labels(
        y,
        x_ids,
        catalog,
        allow_unmapped_labels=allow_unmapped_labels,
        allow_nonbinary_onehot=allow_nonbinary_onehot,
    )

    records, duplicate_audit, stats = _build_records(
        X,
        row_labels,
        x_id_col=x_id_col,
        x_text_col=x_text_col,
        split_name=split_name,
        deduplicate_base_cve=deduplicate_base_cve,
    )

    if not records:
        raise ValueError("No valid output records were produced.")

    sentences_rows = [
        {
            "input_id": record["input_id"],
            "raw_text": record["raw_text"],
            "sentences": record["sentences"],
            "provenance": record["provenance"],
        }
        for record in records
    ]
    labels_rows = [
        {
            "input_id": record["input_id"],
            "labels": record["labels"],
            "provenance": {
                "source_split": record["provenance"]["source_split"],
                "source_ids": record["provenance"]["source_ids"],
                "source_rows": record["provenance"]["source_rows"],
                "labels_unioned": record["provenance"]["labels_unioned"],
            },
        }
        for record in records
    ]

    _write_jsonl(out_dir / "sentences.jsonl", sentences_rows)
    _write_jsonl(out_dir / "labels.jsonl", labels_rows)
    _write_jsonl(out_dir / "records.jsonl", records)
    _write_jsonl(out_dir / "duplicate_audit.jsonl", duplicate_audit)
    (out_dir / "ids.txt").write_text(
        "".join(f"{record['input_id']}\n" for record in records), encoding="utf-8", newline="\n"
    )

    label_mapping_path = out_dir / "label_mapping.json"
    label_mapping_path.write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    label_frequency = Counter(
        label for record in records for label in record["labels"]
    )
    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "split_name": split_name,
        "configuration": {
            "deduplicate_base_cve": deduplicate_base_cve,
            "augmentation_suffixes": ["_augumented_N", "_augmented_N"],
            "canonical_text_rule": "original base row, otherwise lowest augmentation index, then earliest CSV row",
            "duplicate_label_rule": "set union before retaining one output CVE",
            "output_order": "input_id ascending",
            "allow_unmapped_labels": allow_unmapped_labels,
            "allow_nonbinary_onehot": allow_nonbinary_onehot,
        },
        "columns": {
            "x_id": x_id_col,
            "x_text": x_text_col,
            "y_alignment": label_mapping["alignment"],
            "y_id": label_mapping["y_id_column"],
            "y_label": label_mapping["label_column"],
        },
        "statistics": stats,
        "label_frequency": dict(sorted(label_frequency.items())),
        "inputs": {
            "x_csv": {"name": x_csv.name, "sha256": _sha256_file(x_csv), "rows": int(len(X))},
            "y_csv": {"name": y_csv.name, "sha256": _sha256_file(y_csv), "rows": int(len(y))},
            "enterprise_attack": {
                "name": enterprise_attack.name,
                "sha256": catalog.source_sha256,
                "active_technique_ids": len(catalog.active_ids),
            },
        },
        "outputs": {},
    }

    output_paths = [
        out_dir / "sentences.jsonl",
        out_dir / "labels.jsonl",
        out_dir / "records.jsonl",
        out_dir / "ids.txt",
        out_dir / "duplicate_audit.jsonl",
        out_dir / "label_mapping.json",
    ]
    manifest["outputs"] = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in output_paths
    }
    manifest["manifest_content_sha256"] = _canonical_json_sha256(manifest)

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return manifest


def _default_enterprise_attack() -> Optional[Path]:
    candidates = [
        Path("data/attack/enterprise-attack.json"),
        Path(__file__).resolve().parents[1] / "data" / "attack" / "enterprise-attack.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deduplicated, traceable CVE-to-ATT&CK JSONL inputs."
    )
    parser.add_argument("x_csv", type=Path)
    parser.add_argument("y_csv", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--enterprise_attack",
        type=Path,
        default=_default_enterprise_attack(),
        help="Enterprise ATT&CK STIX bundle used to map technique names to active IDs.",
    )
    parser.add_argument("--split_name", default="unspecified")
    parser.add_argument(
        "--keep_augmented",
        action="store_true",
        help="Keep augmented rows as separate records instead of collapsing to base CVEs.",
    )
    parser.add_argument("--allow_unmapped_labels", action="store_true")
    parser.add_argument("--allow_nonbinary_onehot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.enterprise_attack is None:
        parser.error(
            "--enterprise_attack is required because no default data/attack/enterprise-attack.json was found."
        )

    manifest = build_dataset(
        args.x_csv,
        args.y_csv,
        args.out_dir,
        args.enterprise_attack,
        split_name=str(args.split_name),
        deduplicate_base_cve=not args.keep_augmented,
        allow_unmapped_labels=bool(args.allow_unmapped_labels),
        allow_nonbinary_onehot=bool(args.allow_nonbinary_onehot),
        overwrite=bool(args.overwrite),
    )
    stats = manifest["statistics"]
    print(f"OK: {args.out_dir}")
    print(
        "rows: "
        f"X={stats['input_x_rows']} -> output={stats['output_records']} "
        f"(collapsed={stats['collapsed_source_rows']}, augmented={stats['augmented_source_rows']})"
    )
    print(
        "audit: "
        f"duplicate_groups={stats['duplicate_groups']}, "
        f"label_union_groups={stats['label_union_groups']}, "
        f"text_conflicts={stats['text_conflict_groups']}, "
        f"empty_labels={stats['empty_label_records']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
