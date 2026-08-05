"""Audit ATT&CK gold-label availability and candidate coverage.

This module replaces the former Oracle@K-oriented diagnostic with a deterministic
coverage audit that separates three questions:

1. Are the gold ATT&CK technique identifiers present in the exact, versioned
   technique index used by retrieval?
2. Does the fixed candidate list contain at least one gold technique at each K?
3. How do those answers change after an explicitly reported sub-technique to
   parent-technique normalization?

The script never reads reranking predictions and therefore cannot classify a
reranker error as a candidate-generation error.  It supports an index-only mode
when ``--candidates`` is omitted and a full candidate audit when it is supplied.
All reports are deterministic and accompanied by SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

try:  # package execution: python -m pgt.analyze_missing_gold
    from .io import read_jsonl
except ImportError:  # direct execution from repository root
    from pgt.io import read_jsonl  # type: ignore


AUDIT_VERSION = "missing-gold-audit-v2.0.0"
_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


# ---------------------------------------------------------------------------
# Deterministic file helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: Optional[int] = None) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value, indent=2))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = bytearray()
    for row in rows:
        payload.extend(_json_bytes(dict(row), indent=None))
    _atomic_write_bytes(path, bytes(payload))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _parse_ks(raw: str) -> List[int]:
    values: Set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid K value: {item!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("All K values must be positive")
        values.add(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one K value is required")
    return sorted(values)


def _load_id_file(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    ids: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            input_id = line.strip()
            if not input_id:
                continue
            if input_id in seen:
                raise ValueError(f"Duplicate input_id in {path} at line {line_number}: {input_id}")
            seen.add(input_id)
            ids.append(input_id)
    if not ids:
        raise ValueError(f"ID file is empty: {path}")
    return ids


# ---------------------------------------------------------------------------
# ATT&CK catalogue and label loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TechniqueCatalogue:
    rows: Dict[str, Dict[str, Any]]
    active_ids: Set[str]
    parent_by_id: Dict[str, str]
    name_by_id: Dict[str, str]
    collection_versions: Tuple[str, ...]
    index_schema_versions: Tuple[str, ...]

    def parent_id(self, technique_id: str) -> str:
        """Normalize a technique to its parent with a documented fallback.

        The versioned index relationship is authoritative.  For a gold
        sub-technique absent from the active index, the syntactic ``Txxxx``
        prefix is used only to determine whether an active parent exists.
        """
        if technique_id in self.parent_by_id:
            return self.parent_by_id[technique_id]
        if "." in technique_id:
            prefix = technique_id.split(".", 1)[0]
            if prefix in self.active_ids:
                return prefix
        return technique_id


def _load_technique_catalogue(path: Path) -> TechniqueCatalogue:
    rows: Dict[str, Dict[str, Any]] = {}
    parent_by_id: Dict[str, str] = {}
    name_by_id: Dict[str, str] = {}
    collection_versions: Set[str] = set()
    schema_versions: Set[str] = set()

    for line_number, row in enumerate(
        read_jsonl(path, record_kind=None, validate=False, enforce_unique_input_ids=False),
        start=1,
    ):
        technique_id = str(
            row.get("technique_id") or row.get("id") or row.get("technique") or ""
        ).strip()
        if not _TECHNIQUE_RE.fullmatch(technique_id):
            raise ValueError(
                f"Invalid or missing technique_id in {path} at record {line_number}: {technique_id!r}"
            )
        if technique_id in rows:
            raise ValueError(f"Duplicate technique_id in {path}: {technique_id}")
        if bool(row.get("revoked", False)) or bool(row.get("deprecated", False)):
            raise ValueError(
                f"The retrieval index must contain active techniques only; found inactive {technique_id}"
            )
        rows[technique_id] = dict(row)
        name_by_id[technique_id] = str(row.get("name") or "").strip()
        parent = str(row.get("parent_technique_id") or "").strip()
        if parent:
            if not _TECHNIQUE_RE.fullmatch(parent) or "." in parent:
                raise ValueError(f"Invalid parent_technique_id for {technique_id}: {parent!r}")
            parent_by_id[technique_id] = parent
        version = str(row.get("attack_collection_version") or "").strip()
        if version:
            collection_versions.add(version)
        schema_version = str(row.get("index_schema_version") or "").strip()
        if schema_version:
            schema_versions.add(schema_version)

    if not rows:
        raise ValueError(f"Technique index is empty: {path}")
    missing_parents = sorted({parent for parent in parent_by_id.values() if parent not in rows})
    if missing_parents:
        raise ValueError(
            "Technique index contains sub-techniques whose parents are absent: "
            + ", ".join(missing_parents[:20])
        )

    return TechniqueCatalogue(
        rows=rows,
        active_ids=set(rows),
        parent_by_id=parent_by_id,
        name_by_id=name_by_id,
        collection_versions=tuple(sorted(collection_versions)),
        index_schema_versions=tuple(sorted(schema_versions)),
    )


def _normalize_label_value(value: Any, *, input_id: str) -> Tuple[List[str], bool]:
    repaired_string = False
    if isinstance(value, str):
        labels = [value]
        repaired_string = True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        labels = list(value)
    elif value is None:
        labels = []
    else:
        raise ValueError(
            f"labels for {input_id!r} must be a string or list, got {type(value).__name__}"
        )

    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in labels:
        if not isinstance(raw, str):
            raise ValueError(f"Non-string gold label for {input_id!r}: {raw!r}")
        technique_id = raw.strip()
        if not technique_id:
            continue
        if not _TECHNIQUE_RE.fullmatch(technique_id):
            raise ValueError(f"Invalid ATT&CK technique ID for {input_id!r}: {technique_id!r}")
        if technique_id not in seen:
            seen.add(technique_id)
            normalized.append(technique_id)
    return sorted(normalized), repaired_string


def _load_gold_labels(
    path: Path,
    selected_ids: Optional[List[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    gold_map: Dict[str, Set[str]] = defaultdict(set)
    row_count = 0
    string_repairs = 0
    duplicate_rows = 0
    seen_rows: Set[str] = set()

    for row in read_jsonl(
        path, record_kind=None, validate=False, enforce_unique_input_ids=False
    ):
        row_count += 1
        input_id = str(row.get("input_id") or "").strip()
        if not input_id:
            raise ValueError(f"Missing input_id in {path} at record {row_count}")
        labels, repaired = _normalize_label_value(row.get("labels"), input_id=input_id)
        if repaired:
            string_repairs += 1
        if input_id in seen_rows:
            duplicate_rows += 1
        seen_rows.add(input_id)
        gold_map[input_id].update(labels)

    if selected_ids is not None:
        selected_set = set(selected_ids)
        missing = [input_id for input_id in selected_ids if input_id not in gold_map]
        if missing:
            raise ValueError(
                f"ID file contains {len(missing)} CVEs absent from labels; examples: {missing[:10]}"
            )
        extra = sorted(set(gold_map) - selected_set)
        gold_map = {input_id: gold_map[input_id] for input_id in selected_ids}
    else:
        extra = []
        gold_map = {input_id: gold_map[input_id] for input_id in sorted(gold_map)}

    empty_ids = sorted(input_id for input_id, labels in gold_map.items() if not labels)
    if empty_ids:
        raise ValueError(
            f"Gold-label file contains {len(empty_ids)} CVEs with no labels; examples: {empty_ids[:10]}"
        )

    return gold_map, {
        "source_rows": row_count,
        "unique_input_ids": len(gold_map),
        "duplicate_rows_unioned": duplicate_rows,
        "string_label_rows_repaired": string_repairs,
        "extra_label_ids_excluded_by_id_file": len(extra),
    }


def _load_stix_status(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    bundle = json.loads(path.read_text(encoding="utf-8-sig"))
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"ATT&CK STIX bundle has no objects list: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping) or obj.get("type") != "attack-pattern":
            continue
        technique_id = ""
        for ref in obj.get("external_references", []) or []:
            if not isinstance(ref, Mapping) or ref.get("source_name") != "mitre-attack":
                continue
            candidate = str(ref.get("external_id") or "").strip()
            if _TECHNIQUE_RE.fullmatch(candidate):
                technique_id = candidate
                break
        if not technique_id:
            continue
        out[technique_id] = {
            "name": str(obj.get("name") or "").strip(),
            "revoked": bool(obj.get("revoked", False)),
            "deprecated": bool(obj.get("x_mitre_deprecated", False)),
            "stix_id": obj.get("id"),
        }
    return out


# ---------------------------------------------------------------------------
# Candidate loading and coverage logic
# ---------------------------------------------------------------------------


def _load_candidates(
    path: Path,
    catalogue: TechniqueCatalogue,
    selected_ids: Optional[List[str]],
    *,
    validate_schema: bool,
    allow_unknown_candidate_ids: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    candidates_map: Dict[str, List[str]] = {}
    unknown_counter: Counter[str] = Counter()
    list_lengths: Counter[int] = Counter()

    record_kind = "candidates" if validate_schema else "none"
    for record_number, row in enumerate(
        read_jsonl(
            path,
            record_kind=record_kind,
            validate=validate_schema,
            enforce_unique_input_ids=True,
        ),
        start=1,
    ):
        input_id = str(row.get("input_id") or "").strip()
        if not input_id:
            raise ValueError(f"Missing input_id in candidates at record {record_number}")
        if input_id in candidates_map:
            raise ValueError(f"Duplicate candidate record for {input_id}")
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError(f"candidates for {input_id} must be a list")
        technique_ids: List[str] = []
        seen: Set[str] = set()
        for position, candidate in enumerate(raw_candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"Candidate {position} for {input_id} is not an object")
            technique_id = str(candidate.get("technique_id") or "").strip()
            if not _TECHNIQUE_RE.fullmatch(technique_id):
                raise ValueError(
                    f"Invalid candidate technique ID for {input_id} at rank {position}: {technique_id!r}"
                )
            if technique_id in seen:
                raise ValueError(f"Duplicate candidate {technique_id} for {input_id}")
            seen.add(technique_id)
            technique_ids.append(technique_id)
            if technique_id not in catalogue.active_ids:
                unknown_counter[technique_id] += 1
        list_lengths[len(technique_ids)] += 1
        candidates_map[input_id] = technique_ids

    if unknown_counter and not allow_unknown_candidate_ids:
        examples = unknown_counter.most_common(10)
        raise ValueError(
            "Candidate file contains technique IDs absent from the active retrieval index; "
            f"examples: {examples}. Use --allow_unknown_candidate_ids only for a legacy audit."
        )

    if selected_ids is not None:
        selected_set = set(selected_ids)
        extra_ids = sorted(set(candidates_map) - selected_set)
        candidates_map = {
            input_id: candidates_map[input_id]
            for input_id in selected_ids
            if input_id in candidates_map
        }
    else:
        extra_ids = []

    return candidates_map, {
        "candidate_records": len(candidates_map),
        "candidate_list_length_distribution": {
            str(length): count for length, count in sorted(list_lengths.items())
        },
        "unknown_candidate_ids": dict(sorted(unknown_counter.items())),
        "extra_candidate_ids_excluded_by_id_file": len(extra_ids),
    }


def _first_hit_rank(gold: Set[str], candidates: Sequence[str]) -> Optional[int]:
    for rank, technique_id in enumerate(candidates, start=1):
        if technique_id in gold:
            return rank
    return None


def _parent_dedupe(ids: Sequence[str], catalogue: TechniqueCatalogue) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for technique_id in ids:
        parent = catalogue.parent_id(technique_id)
        if parent not in seen:
            seen.add(parent)
            normalized.append(parent)
    return normalized


def _status_for_rank(
    *,
    first_hit_rank: Optional[int],
    candidate_count: int,
    cutoff: int,
    all_gold_missing_index: bool,
    candidate_record_missing: bool,
) -> str:
    if all_gold_missing_index:
        return "all_gold_missing_from_active_index"
    if candidate_record_missing:
        return "candidate_record_missing"
    if candidate_count == 0:
        return "candidate_list_empty"
    if first_hit_rank is None:
        return "gold_absent_from_candidate_universe"
    if first_hit_rank <= cutoff:
        return "covered_at_k"
    return "gold_only_beyond_k"


# ---------------------------------------------------------------------------
# Audit engine
# ---------------------------------------------------------------------------


def run_audit(
    *,
    labels_path: Path,
    technique_index_path: Path,
    output_dir: Path,
    candidates_path: Optional[Path] = None,
    id_file: Optional[Path] = None,
    attack_stix_path: Optional[Path] = None,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    validate_candidate_schema: bool = True,
    allow_id_mismatch: bool = False,
    allow_unknown_candidate_ids: bool = False,
) -> Dict[str, Any]:
    ks = sorted({int(value) for value in ks})
    if not ks or min(ks) <= 0:
        raise ValueError("ks must contain positive integers")

    selected_ids = _load_id_file(id_file)
    catalogue = _load_technique_catalogue(technique_index_path)
    gold_map, label_stats = _load_gold_labels(labels_path, selected_ids)
    stix_status = _load_stix_status(attack_stix_path)

    ordered_ids = selected_ids if selected_ids is not None else sorted(gold_map)

    candidates_map: Optional[Dict[str, List[str]]] = None
    candidate_stats: Dict[str, Any] = {}
    if candidates_path is not None:
        candidates_map, candidate_stats = _load_candidates(
            candidates_path,
            catalogue,
            selected_ids,
            validate_schema=validate_candidate_schema,
            allow_unknown_candidate_ids=allow_unknown_candidate_ids,
        )
        missing_candidate_ids = [input_id for input_id in ordered_ids if input_id not in candidates_map]
        extra_candidate_ids = sorted(set(candidates_map) - set(ordered_ids))
        if (missing_candidate_ids or extra_candidate_ids) and not allow_id_mismatch:
            raise ValueError(
                "Labels/candidates ID mismatch: "
                f"missing_candidate_records={len(missing_candidate_ids)}, "
                f"extra_candidate_records={len(extra_candidate_ids)}. "
                "Use --allow_id_mismatch only to diagnose an incomplete legacy run."
            )
    else:
        missing_candidate_ids = []
        extra_candidate_ids = []

    # Gold-index audit is performed independently of candidate availability.
    unique_gold = sorted({label for labels in gold_map.values() for label in labels})
    gold_support = Counter(label for labels in gold_map.values() for label in labels)
    index_rows: List[Dict[str, Any]] = []
    missing_index_rows: List[Dict[str, Any]] = []
    index_status_counter: Counter[str] = Counter()

    for technique_id in unique_gold:
        exact_active = technique_id in catalogue.active_ids
        parent_id = catalogue.parent_id(technique_id)
        parent_active = parent_id in catalogue.active_ids
        if exact_active:
            index_status = "active_exact"
        elif parent_active and parent_id != technique_id:
            index_status = "missing_exact_parent_active"
        else:
            stix = stix_status.get(technique_id, {})
            if stix.get("revoked"):
                index_status = "revoked_in_source_stix"
            elif stix.get("deprecated"):
                index_status = "deprecated_in_source_stix"
            elif stix:
                index_status = "active_in_stix_but_missing_from_index"
            else:
                index_status = "not_found_in_source_stix_or_index"
        index_status_counter[index_status] += 1
        row = {
            "technique_id": technique_id,
            "technique_name": catalogue.name_by_id.get(technique_id, stix_status.get(technique_id, {}).get("name", "")),
            "support_cves": gold_support[technique_id],
            "exact_active_index": exact_active,
            "parent_technique_id": parent_id,
            "parent_active_index": parent_active,
            "index_status": index_status,
        }
        index_rows.append(row)
        if not exact_active:
            missing_index_rows.append(row)

    per_cve_rows: List[Dict[str, Any]] = []
    coverage_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    status_counts: Dict[Tuple[str, int, str], int] = defaultdict(int)
    technique_coverage: Dict[str, Dict[str, Any]] = {
        technique_id: {
            "technique_id": technique_id,
            "technique_name": catalogue.name_by_id.get(technique_id, ""),
            "parent_technique_id": catalogue.parent_id(technique_id),
            "support_cves": gold_support[technique_id],
            "exact_active_index": technique_id in catalogue.active_ids,
            **{f"exact_covered_at_{k}": 0 for k in ks},
            **{f"parent_covered_at_{k}": 0 for k in ks},
        }
        for technique_id in unique_gold
    }

    if candidates_map is not None:
        max_k = max(ks)
        for input_id in ordered_ids:
            gold_exact = set(gold_map[input_id])
            gold_parent = {catalogue.parent_id(label) for label in gold_exact}
            active_gold_exact = gold_exact & catalogue.active_ids
            active_gold_parent = gold_parent & catalogue.active_ids
            missing_gold_exact = gold_exact - catalogue.active_ids

            record_missing = input_id not in candidates_map
            candidate_exact = candidates_map.get(input_id, [])
            candidate_parent = _parent_dedupe(candidate_exact, catalogue)
            exact_rank = _first_hit_rank(gold_exact, candidate_exact)
            parent_rank = _first_hit_rank(gold_parent, candidate_parent)

            exact_status_max = _status_for_rank(
                first_hit_rank=exact_rank,
                candidate_count=len(candidate_exact),
                cutoff=max_k,
                all_gold_missing_index=not active_gold_exact,
                candidate_record_missing=record_missing,
            )
            parent_status_max = _status_for_rank(
                first_hit_rank=parent_rank,
                candidate_count=len(candidate_parent),
                cutoff=max_k,
                all_gold_missing_index=not active_gold_parent,
                candidate_record_missing=record_missing,
            )

            exact_hits_by_k: Dict[str, bool] = {}
            parent_hits_by_k: Dict[str, bool] = {}
            for k in ks:
                exact_hit = exact_rank is not None and exact_rank <= k
                parent_hit = parent_rank is not None and parent_rank <= k
                exact_hits_by_k[str(k)] = exact_hit
                parent_hits_by_k[str(k)] = parent_hit
                if exact_hit:
                    coverage_counts[("exact", k)] += 1
                if parent_hit:
                    coverage_counts[("parent", k)] += 1
                exact_status = _status_for_rank(
                    first_hit_rank=exact_rank,
                    candidate_count=len(candidate_exact),
                    cutoff=k,
                    all_gold_missing_index=not active_gold_exact,
                    candidate_record_missing=record_missing,
                )
                parent_status = _status_for_rank(
                    first_hit_rank=parent_rank,
                    candidate_count=len(candidate_parent),
                    cutoff=k,
                    all_gold_missing_index=not active_gold_parent,
                    candidate_record_missing=record_missing,
                )
                status_counts[("exact", k, exact_status)] += 1
                status_counts[("parent", k, parent_status)] += 1

                for technique_id in gold_exact:
                    if technique_id in set(candidate_exact[:k]):
                        technique_coverage[technique_id][f"exact_covered_at_{k}"] += 1
                    parent_id = catalogue.parent_id(technique_id)
                    if parent_id in set(candidate_parent[:k]):
                        technique_coverage[technique_id][f"parent_covered_at_{k}"] += 1

            per_cve_rows.append(
                {
                    "input_id": input_id,
                    "gold_exact": sorted(gold_exact),
                    "gold_parent": sorted(gold_parent),
                    "active_gold_exact": sorted(active_gold_exact),
                    "active_gold_parent": sorted(active_gold_parent),
                    "gold_missing_from_active_index": sorted(missing_gold_exact),
                    "candidate_record_missing": record_missing,
                    "candidate_count_exact": len(candidate_exact),
                    "candidate_count_parent_deduplicated": len(candidate_parent),
                    "first_exact_hit_rank": exact_rank,
                    "first_parent_hit_rank": parent_rank,
                    "exact_hits_by_k": exact_hits_by_k,
                    "parent_hits_by_k": parent_hits_by_k,
                    "exact_status_at_max_k": exact_status_max,
                    "parent_status_at_max_k": parent_status_max,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    gold_index_path = output_dir / "gold_index_audit.csv"
    missing_index_path = output_dir / "missing_gold_from_active_index.csv"
    coverage_path = output_dir / "coverage_by_k.csv"
    status_path = output_dir / "failure_reasons_by_k.csv"
    per_cve_path = output_dir / "per_cve_diagnostics.jsonl"
    per_technique_path = output_dir / "per_technique_coverage.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "audit_manifest.json"

    index_fields = [
        "technique_id",
        "technique_name",
        "support_cves",
        "exact_active_index",
        "parent_technique_id",
        "parent_active_index",
        "index_status",
    ]
    _write_csv(gold_index_path, index_fields, index_rows)
    _write_csv(missing_index_path, index_fields, missing_index_rows)

    coverage_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    if candidates_map is not None:
        denominator = len(ordered_ids)
        for view in ("exact", "parent"):
            for k in ks:
                covered = coverage_counts[(view, k)]
                coverage_rows.append(
                    {
                        "view": view,
                        "k": k,
                        "cves_evaluated": denominator,
                        "covered_cves": covered,
                        "coverage_rate": covered / denominator if denominator else 0.0,
                    }
                )
                statuses = sorted(
                    {
                        status
                        for (stored_view, stored_k, status), count in status_counts.items()
                        if stored_view == view and stored_k == k and count
                    }
                )
                for status in statuses:
                    count = status_counts[(view, k, status)]
                    failure_rows.append(
                        {
                            "view": view,
                            "k": k,
                            "status": status,
                            "cve_count": count,
                            "proportion": count / denominator if denominator else 0.0,
                        }
                    )
        _write_jsonl(per_cve_path, per_cve_rows)

        technique_fields = [
            "technique_id",
            "technique_name",
            "parent_technique_id",
            "support_cves",
            "exact_active_index",
            *[f"exact_covered_at_{k}" for k in ks],
            *[f"parent_covered_at_{k}" for k in ks],
        ]
        _write_csv(
            per_technique_path,
            technique_fields,
            [technique_coverage[tid] for tid in sorted(technique_coverage)],
        )
    else:
        _write_jsonl(per_cve_path, [])
        _write_csv(
            per_technique_path,
            [
                "technique_id",
                "technique_name",
                "parent_technique_id",
                "support_cves",
                "exact_active_index",
            ],
            [
                {
                    key: technique_coverage[tid][key]
                    for key in (
                        "technique_id",
                        "technique_name",
                        "parent_technique_id",
                        "support_cves",
                        "exact_active_index",
                    )
                }
                for tid in sorted(technique_coverage)
            ],
        )

    _write_csv(
        coverage_path,
        ["view", "k", "cves_evaluated", "covered_cves", "coverage_rate"],
        coverage_rows,
    )
    _write_csv(
        status_path,
        ["view", "k", "status", "cve_count", "proportion"],
        failure_rows,
    )

    summary: Dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "mode": "candidate_coverage" if candidates_map is not None else "gold_index_only",
        "cves_in_scope": len(ordered_ids),
        "label_loading": label_stats,
        "gold_techniques": {
            "unique": len(unique_gold),
            "assignments": sum(gold_support.values()),
            "index_status_counts": dict(sorted(index_status_counter.items())),
            "missing_exact_active_index": len(missing_index_rows),
            "missing_exact_but_parent_active": sum(
                1 for row in missing_index_rows if row["parent_active_index"]
            ),
        },
        "technique_index": {
            "active_techniques": len(catalogue.active_ids),
            "collection_versions": list(catalogue.collection_versions),
            "index_schema_versions": list(catalogue.index_schema_versions),
        },
        "normalization": {
            "exact_view": "no normalization",
            "parent_view": (
                "use parent_technique_id from the versioned index; for a gold "
                "sub-technique absent from the index, use its Txxxx prefix only "
                "when that parent is active"
            ),
            "parent_candidate_duplicates": "retain the earliest ranked occurrence",
        },
        "ks": list(ks),
    }
    if candidates_map is not None:
        summary["candidate_loading"] = {
            **candidate_stats,
            "missing_candidate_records": len(missing_candidate_ids),
            "extra_candidate_records": len(extra_candidate_ids),
        }
        summary["coverage"] = {
            f"{row['view']}_at_{row['k']}": {
                "covered_cves": row["covered_cves"],
                "cves_evaluated": row["cves_evaluated"],
                "rate": row["coverage_rate"],
            }
            for row in coverage_rows
        }
    _write_json(summary_path, summary)

    output_paths = [
        summary_path,
        gold_index_path,
        missing_index_path,
        coverage_path,
        status_path,
        per_cve_path,
        per_technique_path,
    ]
    manifest = {
        "audit_version": AUDIT_VERSION,
        "parameters": {
            "ks": list(ks),
            "validate_candidate_schema": validate_candidate_schema,
            "allow_id_mismatch": allow_id_mismatch,
            "allow_unknown_candidate_ids": allow_unknown_candidate_ids,
        },
        "inputs": {
            "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "technique_index": {
                "path": str(technique_index_path),
                "sha256": sha256_file(technique_index_path),
            },
            "candidates": (
                {"path": str(candidates_path), "sha256": sha256_file(candidates_path)}
                if candidates_path is not None
                else None
            ),
            "id_file": (
                {"path": str(id_file), "sha256": sha256_file(id_file)}
                if id_file is not None
                else None
            ),
            "attack_stix": (
                {"path": str(attack_stix_path), "sha256": sha256_file(attack_stix_path)}
                if attack_stix_path is not None
                else None
            ),
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    _write_json(manifest_path, manifest)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit gold ATT&CK IDs against the versioned active index and, when "
            "provided, measure exact and parent-normalized candidate coverage."
        )
    )
    parser.add_argument("--labels", required=True, type=Path, help="labels.jsonl")
    parser.add_argument(
        "--tech_index", required=True, type=Path, help="versioned technique_text_index.jsonl"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="optional candidates.jsonl; omit for a gold-index-only audit",
    )
    parser.add_argument(
        "--id_file",
        type=Path,
        default=None,
        help="optional fixed split ids.txt; also fixes output order and scope",
    )
    parser.add_argument(
        "--attack_stix",
        type=Path,
        default=None,
        help="optional full enterprise-attack.json for revoked/deprecated classification",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--ks",
        default="1,3,5,10,20",
        help="comma-separated candidate cutoffs (default: 1,3,5,10,20)",
    )
    parser.add_argument(
        "--skip_candidate_schema_validation",
        action="store_true",
        help="legacy diagnostics only; formal runs should keep schema validation enabled",
    )
    parser.add_argument(
        "--allow_id_mismatch",
        action="store_true",
        help="diagnose incomplete legacy runs instead of failing on label/candidate ID mismatch",
    )
    parser.add_argument(
        "--allow_unknown_candidate_ids",
        action="store_true",
        help="legacy diagnostics only; formal candidates must come from the active index",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ks = _parse_ks(args.ks)
    summary = run_audit(
        labels_path=args.labels,
        technique_index_path=args.tech_index,
        candidates_path=args.candidates,
        id_file=args.id_file,
        attack_stix_path=args.attack_stix,
        output_dir=args.output_dir,
        ks=ks,
        validate_candidate_schema=not args.skip_candidate_schema_validation,
        allow_id_mismatch=args.allow_id_mismatch,
        allow_unknown_candidate_ids=args.allow_unknown_candidate_ids,
    )
    print(f"[OK] audit written to: {args.output_dir}")
    print(
        f"mode={summary['mode']} cves={summary['cves_in_scope']} "
        f"gold_techniques={summary['gold_techniques']['unique']} "
        f"missing_from_active_index={summary['gold_techniques']['missing_exact_active_index']}"
    )
    if "coverage" in summary:
        for key, value in sorted(summary["coverage"].items()):
            print(f"{key}: {value['covered_cves']}/{value['cves_evaluated']} ({value['rate']:.4f})")


if __name__ == "__main__":
    main()
