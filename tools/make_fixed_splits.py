#!/usr/bin/env python3
"""Create deterministic, leakage-free development and test splits.

The input to this tool is one or more traceable dataset directories generated
by ``tools/make_cve2attck_jsonl.py``.  Source directories are merged at the
base-CVE level before splitting, so a CVE that appears in more than one source
partition can never leak across the new development and test sets.

Key guarantees
--------------
* duplicate CVEs across source directories are collapsed before splitting;
* labels are unioned across every occurrence of the same base CVE;
* the canonical text is selected by a deterministic, label-independent rule;
* development and test contain exactly the requested number of unique CVEs;
* multilabel frequencies are balanced with deterministic iterative stratification
  followed by an optional bounded pair-swap refinement;
* every technique with support >= ``--min_support_both`` is required to appear
  in both splits when that constraint is feasible;
* all artifacts, input hashes, split IDs, configuration, and label statistics
  are recorded in machine-readable manifests;
* the generated ``combined``, ``development``, and ``test`` directories are
  compatible with ``tools/check_labels_alignment.py``.

The split algorithm never looks at model predictions or downstream metrics.
Only CVE identifiers and gold labels are used for stratification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


PIPELINE_VERSION = "fixed-multilabel-split-v1.0.0"
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
AUGMENTATION_SUFFIX_RE = re.compile(r"_(?:augumented|augmented)_\d+$", re.IGNORECASE)
REQUIRED_DATASET_FILES = (
    "records.jsonl",
    "labels.jsonl",
    "sentences.jsonl",
    "ids.txt",
    "dataset_manifest.json",
)


class SplitError(RuntimeError):
    """Raised when a reproducible split cannot be constructed safely."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    position: int


@dataclass
class SourceRecord:
    source_name: str
    source_position: int
    source_record_order: int
    record: Dict[str, Any]
    record_sha256: str


@dataclass
class MergedRecord:
    input_id: str
    labels: Tuple[str, ...]
    raw_text: str
    sentences: Dict[str, str]
    provenance: Dict[str, Any]

    def as_record(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "labels": list(self.labels),
            "provenance": self.provenance,
            "raw_text": self.raw_text,
            "sentences": self.sentences,
        }


# ---------------------------------------------------------------------------
# Generic IO and canonicalization helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_json_sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def _stable_tie(seed: int, *parts: str) -> str:
    payload = "\x1f".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_input_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "", text)
    return text.upper()


def base_input_id(value: Any) -> str:
    return AUGMENTATION_SUFFIX_RE.sub("", normalize_input_id(value))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise SplitError(f"Cannot parse JSON file {path}: {exc}") from exc


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
                    raise SplitError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise SplitError(f"Expected a JSON object at {path}:{line_no}")
                obj = dict(obj)
                obj["__line__"] = line_no
                rows.append(obj)
    except OSError as exc:
        raise SplitError(f"Cannot read {path}: {exc}") from exc
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_labels(value: Any, *, source: str, input_id: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise SplitError(f"labels must be a JSON list for {input_id} in {source}")
    labels: List[str] = []
    for raw in value:
        tid = str(raw).strip().upper()
        if not TECHNIQUE_ID_RE.fullmatch(tid):
            raise SplitError(f"Invalid ATT&CK technique ID {raw!r} for {input_id} in {source}")
        labels.append(tid)
    canonical = tuple(sorted(set(labels)))
    if not canonical:
        raise SplitError(f"No labels for {input_id} in {source}")
    if list(value) != list(canonical):
        raise SplitError(
            f"Labels must already be sorted and unique for {input_id} in {source}; "
            f"observed={value!r}, expected={list(canonical)!r}"
        )
    return canonical


def _normalize_sentences(value: Any, *, source: str, input_id: str) -> Dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise SplitError(f"sentences must be a non-empty object for {input_id} in {source}")
    items: List[Tuple[int, str, str]] = []
    for eid, text in value.items():
        match = re.fullmatch(r"E(\d+)", str(eid))
        if not match:
            raise SplitError(f"Invalid evidence ID {eid!r} for {input_id} in {source}")
        clean = str(text).strip()
        if not clean:
            raise SplitError(f"Blank evidence text {eid!r} for {input_id} in {source}")
        items.append((int(match.group(1)), f"E{int(match.group(1))}", clean))
    items.sort(key=lambda x: x[0])
    expected = list(range(1, len(items) + 1))
    observed = [x[0] for x in items]
    if observed != expected:
        raise SplitError(
            f"Evidence identifiers must be contiguous E1..En for {input_id} in {source}; observed={observed}"
        )
    return {eid: text for _, eid, text in items}


def _validate_source_dir(spec: SourceSpec) -> Dict[str, Any]:
    if not spec.path.is_dir():
        raise SplitError(f"Source directory does not exist: {spec.path}")
    missing = [name for name in REQUIRED_DATASET_FILES if not (spec.path / name).exists()]
    if missing:
        raise SplitError(f"Source {spec.name} is missing required files: {', '.join(missing)}")

    manifest = _read_json(spec.path / "dataset_manifest.json")
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    if not isinstance(outputs, dict):
        raise SplitError(f"Source manifest lacks an outputs object: {spec.path / 'dataset_manifest.json'}")

    for filename, meta in outputs.items():
        path = spec.path / filename
        if not path.exists():
            raise SplitError(f"Source manifest references a missing file: {path}")
        expected = meta.get("sha256") if isinstance(meta, dict) else None
        actual = _sha256_file(path)
        if expected != actual:
            raise SplitError(
                f"Source hash mismatch for {path}: expected={expected}, actual={actual}. "
                "Run tools/check_labels_alignment.py before splitting."
            )

    return manifest


# ---------------------------------------------------------------------------
# Source merge
# ---------------------------------------------------------------------------


def _parse_source_spec(text: str, position: int) -> SourceSpec:
    if "=" not in text:
        raise SplitError("Each --source must use NAME=DATASET_DIR syntax")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SplitError(f"Invalid source name {name!r}; use letters, digits, dot, underscore, or hyphen")
    path = Path(raw_path.strip()).expanduser().resolve()
    return SourceSpec(name=name, path=path, position=position)


def load_and_merge_sources(specs: Sequence[SourceSpec]) -> Tuple[List[MergedRecord], Dict[str, Any]]:
    if not specs:
        raise SplitError("At least one --source is required")
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise SplitError("--source names must be unique")

    by_cve: MutableMapping[str, List[SourceRecord]] = defaultdict(list)
    source_summaries: Dict[str, Any] = {}

    for spec in specs:
        manifest = _validate_source_dir(spec)
        rows = _read_jsonl(spec.path / "records.jsonl")
        seen: Set[str] = set()
        for order, row in enumerate(rows):
            iid = base_input_id(row.get("input_id", ""))
            if not iid:
                raise SplitError(f"Blank input_id in {spec.path / 'records.jsonl'} at line {row.get('__line__')}")
            if iid in seen:
                raise SplitError(f"Duplicate base CVE {iid} within source {spec.name}")
            seen.add(iid)

            labels = _normalize_labels(row.get("labels"), source=spec.name, input_id=iid)
            raw_text = str(row.get("raw_text") or "").strip()
            if not raw_text:
                raise SplitError(f"Blank raw_text for {iid} in source {spec.name}")
            sentences = _normalize_sentences(row.get("sentences"), source=spec.name, input_id=iid)

            clean_row = {
                "input_id": iid,
                "labels": list(labels),
                "provenance": row.get("provenance") if isinstance(row.get("provenance"), dict) else {},
                "raw_text": raw_text,
                "sentences": sentences,
            }
            by_cve[iid].append(
                SourceRecord(
                    source_name=spec.name,
                    source_position=spec.position,
                    source_record_order=order,
                    record=clean_row,
                    record_sha256=_canonical_json_sha256(clean_row),
                )
            )

        source_summaries[spec.name] = {
            "path": str(spec.path),
            "position": spec.position,
            "records": len(rows),
            "manifest_sha256": _sha256_file(spec.path / "dataset_manifest.json"),
            "records_sha256": _sha256_file(spec.path / "records.jsonl"),
            "labels_sha256": _sha256_file(spec.path / "labels.jsonl"),
            "sentences_sha256": _sha256_file(spec.path / "sentences.jsonl"),
            "ids_sha256": _sha256_file(spec.path / "ids.txt"),
            "source_pipeline_version": manifest.get("pipeline_version") if isinstance(manifest, dict) else None,
        }

    merged: List[MergedRecord] = []
    cross_source_duplicates: List[Dict[str, Any]] = []
    label_union_groups = 0
    text_conflict_groups = 0

    for iid in sorted(by_cve):
        variants = by_cve[iid]
        all_labels: Set[str] = set()
        label_sets: Set[Tuple[str, ...]] = set()
        text_hashes: Set[str] = set()
        for variant in variants:
            labels = tuple(variant.record["labels"])
            label_sets.add(labels)
            all_labels.update(labels)
            text_hashes.add(hashlib.sha256(variant.record["raw_text"].encode("utf-8")).hexdigest())

        # The rule is label-independent: choose the longest non-empty canonical
        # text, then source argument order, then source record order, then hash.
        canonical = min(
            variants,
            key=lambda v: (
                -len(v.record["raw_text"]),
                v.source_position,
                v.source_record_order,
                v.record_sha256,
            ),
        )

        labels = tuple(sorted(all_labels))
        labels_unioned = any(tuple(v.record["labels"]) != labels for v in variants)
        text_conflict = len(text_hashes) > 1
        if labels_unioned:
            label_union_groups += 1
        if text_conflict:
            text_conflict_groups += 1

        merge_provenance = {
            "pipeline_version": PIPELINE_VERSION,
            "canonical_text_rule": "longest raw_text, then source argument order, source record order, record SHA-256",
            "canonical_source": canonical.source_name,
            "canonical_record_sha256": canonical.record_sha256,
            "source_membership": sorted({v.source_name for v in variants}),
            "source_occurrences": [
                {
                    "source": v.source_name,
                    "source_position": v.source_position,
                    "source_record_order": v.source_record_order,
                    "record_sha256": v.record_sha256,
                    "labels": v.record["labels"],
                    "raw_text_sha256": hashlib.sha256(v.record["raw_text"].encode("utf-8")).hexdigest(),
                    "original_provenance": v.record.get("provenance", {}),
                }
                for v in sorted(
                    variants,
                    key=lambda x: (x.source_position, x.source_record_order, x.record_sha256),
                )
            ],
            "source_occurrence_count": len(variants),
            "labels_unioned_across_sources": labels_unioned,
            "text_conflict_across_sources": text_conflict,
        }

        merged.append(
            MergedRecord(
                input_id=iid,
                labels=labels,
                raw_text=canonical.record["raw_text"],
                sentences=dict(canonical.record["sentences"]),
                provenance={
                    "source_record_provenance": canonical.record.get("provenance", {}),
                    "fixed_split_merge": merge_provenance,
                },
            )
        )

        if len(variants) > 1:
            cross_source_duplicates.append(
                {
                    "input_id": iid,
                    "source_membership": merge_provenance["source_membership"],
                    "source_occurrence_count": len(variants),
                    "labels_by_source": {
                        v.source_name: v.record["labels"]
                        for v in sorted(variants, key=lambda x: (x.source_position, x.source_record_order))
                    },
                    "merged_labels": list(labels),
                    "labels_unioned_across_sources": labels_unioned,
                    "text_conflict_across_sources": text_conflict,
                    "canonical_source": canonical.source_name,
                }
            )

    if not merged:
        raise SplitError("No usable records were loaded")

    audit = {
        "sources": source_summaries,
        "source_record_total": sum(x["records"] for x in source_summaries.values()),
        "unique_base_cves": len(merged),
        "collapsed_cross_source_occurrences": sum(len(v) - 1 for v in by_cve.values()),
        "cross_source_duplicate_groups": len(cross_source_duplicates),
        "cross_source_label_union_groups": label_union_groups,
        "cross_source_text_conflict_groups": text_conflict_groups,
        "cross_source_duplicates": cross_source_duplicates,
    }
    return merged, audit


# ---------------------------------------------------------------------------
# Deterministic multilabel split
# ---------------------------------------------------------------------------


def _label_frequency(records: Sequence[MergedRecord]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for record in records:
        freq.update(record.labels)
    return freq


def _target_dev_counts(
    frequencies: Mapping[str, int],
    dev_fraction: float,
    min_support_both: int,
) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    for label, support in sorted(frequencies.items()):
        target = int(round(support * dev_fraction))
        if support >= min_support_both:
            target = max(1, min(support - 1, target))
        else:
            # Singleton and otherwise infeasible labels remain in test by
            # default.  This protects the final evaluation label space.
            target = 0
        targets[label] = target
    return targets


def _split_objective(
    counts: Mapping[str, int],
    frequencies: Mapping[str, int],
    targets: Mapping[str, int],
    min_support_both: int,
) -> float:
    value = 0.0
    for label, support in frequencies.items():
        current = int(counts.get(label, 0))
        target = int(targets[label])
        weight = 1.0 / max(1.0, float(support))
        value += weight * float((current - target) ** 2)
        if support >= min_support_both and (current == 0 or current == support):
            value += 1_000_000.0
    return value


def _iterative_initial_dev(
    records: Sequence[MergedRecord],
    dev_size: int,
    seed: int,
) -> Set[str]:
    """Deterministic two-fold iterative multilabel stratification.

    This is a dependency-free implementation of the rarest-label-first idea:
    repeatedly choose the label with the fewest unassigned examples and assign
    one such example to the fold with the greatest remaining demand for that
    label. Exact fold capacities are enforced throughout.
    """
    by_id = {r.input_id: r for r in records}
    total = len(records)
    folds = ("development", "test")
    capacity: Dict[str, int] = {
        "development": dev_size,
        "test": total - dev_size,
    }
    desired_samples: Dict[str, float] = {fold: float(value) for fold, value in capacity.items()}

    frequencies = _label_frequency(records)
    remaining_by_label: MutableMapping[str, Set[str]] = defaultdict(set)
    for record in records:
        for label in record.labels:
            remaining_by_label[label].add(record.input_id)

    desired_label: Dict[str, Dict[str, float]] = {
        fold: {
            label: support * (capacity[fold] / total)
            for label, support in frequencies.items()
        }
        for fold in folds
    }

    unassigned: Set[str] = set(by_id)
    assignment: Dict[str, str] = {}
    while unassigned:
        active_labels = [label for label, ids in remaining_by_label.items() if ids]
        if active_labels:
            label = min(active_labels, key=lambda x: (len(remaining_by_label[x]), x))
            iid = min(
                remaining_by_label[label],
                key=lambda x: (_stable_tie(seed, "iterative-sample", label, x), x),
            )
        else:
            # The normal data path never reaches this branch because every CVE
            # has at least one label, but retaining it makes the implementation
            # total and deterministic.
            label = None
            iid = min(unassigned, key=lambda x: (_stable_tie(seed, "unlabelled", x), x))

        eligible = [fold for fold in folds if capacity[fold] > 0]
        if not eligible:
            raise SplitError("Iterative stratification exhausted fold capacity unexpectedly")

        if label is None:
            chosen_fold = max(
                eligible,
                key=lambda fold: (desired_samples[fold], -folds.index(fold)),
            )
        else:
            chosen_fold = max(
                eligible,
                key=lambda fold: (
                    desired_label[fold][label],
                    desired_samples[fold],
                    -folds.index(fold),
                ),
            )

        assignment[iid] = chosen_fold
        capacity[chosen_fold] -= 1
        desired_samples[chosen_fold] -= 1.0
        for item_label in by_id[iid].labels:
            desired_label[chosen_fold][item_label] -= 1.0
            remaining_by_label[item_label].discard(iid)
        unassigned.remove(iid)

    if capacity["development"] != 0 or capacity["test"] != 0:
        raise SplitError(f"Iterative stratification did not fill exact capacities: {capacity}")
    return {iid for iid, fold in assignment.items() if fold == "development"}


def _label_component(
    label: str,
    count: int,
    frequencies: Mapping[str, int],
    targets: Mapping[str, int],
    min_support_both: int,
) -> float:
    support = frequencies[label]
    target = targets[label]
    value = (1.0 / max(1.0, float(support))) * float((count - target) ** 2)
    if support >= min_support_both and (count == 0 or count == support):
        value += 1_000_000.0
    return value


def _refine_by_swaps(
    records: Sequence[MergedRecord],
    dev_ids: Set[str],
    frequencies: Mapping[str, int],
    targets: Mapping[str, int],
    min_support_both: int,
    seed: int,
    max_passes: int,
    pool_size: int = 256,
) -> Tuple[Set[str], Dict[str, Any]]:
    """Improve label balance with bounded deterministic pair swaps.

    Only the most useful development and test candidates are considered in
    each pass, which keeps runtime bounded for the full corpus. A swap is
    accepted only if it strictly decreases the complete weighted objective.
    """
    by_id = {r.input_id: r for r in records}
    all_ids = set(by_id)
    counts: Counter[str] = Counter()
    for iid in dev_ids:
        counts.update(by_id[iid].labels)

    current_obj = _split_objective(counts, frequencies, targets, min_support_both)
    swaps: List[Dict[str, Any]] = []

    for pass_no in range(1, max_passes + 1):
        deviations = {label: counts[label] - targets[label] for label in frequencies}
        if all(value == 0 for value in deviations.values()):
            break

        def out_utility(iid: str) -> float:
            score = 0.0
            for label in by_id[iid].labels:
                support = frequencies[label]
                deviation = deviations[label]
                score += max(deviation, 0) / support
                score -= max(-deviation, 0) / support
            return score

        def in_utility(iid: str) -> float:
            score = 0.0
            for label in by_id[iid].labels:
                support = frequencies[label]
                deviation = deviations[label]
                score += max(-deviation, 0) / support
                score -= max(deviation, 0) / support
            return score

        test_ids = all_ids - dev_ids
        ordered_dev = sorted(
            dev_ids,
            key=lambda iid: (-out_utility(iid), _stable_tie(seed, "swap-out", iid), iid),
        )[:pool_size]
        ordered_test = sorted(
            test_ids,
            key=lambda iid: (-in_utility(iid), _stable_tie(seed, "swap-in", iid), iid),
        )[:pool_size]

        best_delta = 0.0
        best_pair: Optional[Tuple[str, str, str]] = None
        for out_id in ordered_dev:
            out_labels = set(by_id[out_id].labels)
            for in_id in ordered_test:
                in_labels = set(by_id[in_id].labels)
                changed = out_labels | in_labels
                if not changed or out_labels == in_labels:
                    continue
                delta = 0.0
                for label in changed:
                    old_count = counts[label]
                    new_count = old_count - int(label in out_labels) + int(label in in_labels)
                    delta += _label_component(
                        label, new_count, frequencies, targets, min_support_both
                    ) - _label_component(
                        label, old_count, frequencies, targets, min_support_both
                    )
                tie = _stable_tie(seed, "swap", out_id, in_id)
                if delta < best_delta - 1e-12 or (
                    abs(delta - best_delta) <= 1e-12
                    and best_pair is not None
                    and (tie, out_id, in_id) < best_pair
                ):
                    best_delta = delta
                    best_pair = (tie, out_id, in_id)

        if best_pair is None:
            break
        _, out_id, in_id = best_pair
        before = current_obj
        counts.subtract(by_id[out_id].labels)
        counts.update(by_id[in_id].labels)
        dev_ids.remove(out_id)
        dev_ids.add(in_id)
        current_obj += best_delta
        swaps.append(
            {
                "pass": pass_no,
                "development_to_test": out_id,
                "test_to_development": in_id,
                "objective_before": before,
                "objective_after": current_obj,
            }
        )

    return dev_ids, {
        "swap_count": len(swaps),
        "swaps": swaps,
        "final_objective": current_obj,
        "passes_requested": max_passes,
        "candidate_pool_size": pool_size,
    }

def make_split(
    records: Sequence[MergedRecord],
    *,
    dev_size: int,
    seed: int,
    min_support_both: int,
    max_swap_passes: int,
) -> Tuple[List[MergedRecord], List[MergedRecord], Dict[str, Any]]:
    if not (1 <= dev_size < len(records)):
        raise SplitError(f"development size must be in [1, {len(records) - 1}], got {dev_size}")

    frequencies = _label_frequency(records)
    if not frequencies:
        raise SplitError("No technique labels were found")
    targets = _target_dev_counts(frequencies, dev_size / len(records), min_support_both)

    initial_ids = _iterative_initial_dev(records, dev_size, seed)
    refined_ids, refinement = _refine_by_swaps(
        records,
        set(initial_ids),
        frequencies,
        targets,
        min_support_both,
        seed,
        max_swap_passes,
    )

    development = sorted((r for r in records if r.input_id in refined_ids), key=lambda r: r.input_id)
    test = sorted((r for r in records if r.input_id not in refined_ids), key=lambda r: r.input_id)

    dev_freq = _label_frequency(development)
    test_freq = _label_frequency(test)
    infeasible_or_failed: List[Dict[str, Any]] = []
    for label, support in sorted(frequencies.items()):
        if support >= min_support_both and (dev_freq[label] == 0 or test_freq[label] == 0):
            infeasible_or_failed.append(
                {
                    "technique_id": label,
                    "support": support,
                    "development": dev_freq[label],
                    "test": test_freq[label],
                }
            )
    if infeasible_or_failed:
        raise SplitError(
            "Could not satisfy label presence in both splits for supported labels: "
            + json.dumps(infeasible_or_failed, ensure_ascii=False)
        )

    overlap = sorted(set(r.input_id for r in development) & set(r.input_id for r in test))
    if overlap:
        raise SplitError(f"Internal error: split overlap detected: {overlap[:20]}")
    if len(development) + len(test) != len(records):
        raise SplitError("Internal error: split sizes do not sum to the combined corpus")

    label_rows: List[Dict[str, Any]] = []
    for label in sorted(frequencies):
        total = frequencies[label]
        label_rows.append(
            {
                "technique_id": label,
                "total_support": total,
                "target_development_support": targets[label],
                "development_support": dev_freq[label],
                "test_support": test_freq[label],
                "development_share": dev_freq[label] / total,
                "absolute_target_error": abs(dev_freq[label] - targets[label]),
                "present_in_both": bool(dev_freq[label] and test_freq[label]),
            }
        )

    audit = {
        "algorithm": "deterministic rarest-label-first iterative multilabel stratification plus bounded pair-swap refinement",
        "algorithm_uses_model_predictions": False,
        "algorithm_uses_downstream_metrics": False,
        "seed": seed,
        "development_size": len(development),
        "test_size": len(test),
        "development_fraction": len(development) / len(records),
        "min_support_both": min_support_both,
        "initial_development_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(initial_ids)) + "\n").encode("utf-8")
        ).hexdigest(),
        "refinement": refinement,
        "label_distribution": label_rows,
        "development_ids_sha256": hashlib.sha256(
            ("\n".join(r.input_id for r in development) + "\n").encode("utf-8")
        ).hexdigest(),
        "test_ids_sha256": hashlib.sha256(
            ("\n".join(r.input_id for r in test) + "\n").encode("utf-8")
        ).hexdigest(),
        "overlap_count": 0,
    }
    return development, test, audit


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _artifact_meta(path: Path) -> Dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_dataset_dir(
    path: Path,
    records: Sequence[MergedRecord],
    *,
    split_name: str,
    parent_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    ordered = sorted(records, key=lambda r: r.input_id)

    label_rows = [{"input_id": r.input_id, "labels": list(r.labels)} for r in ordered]
    sentence_rows = [
        {"input_id": r.input_id, "raw_text": r.raw_text, "sentences": r.sentences}
        for r in ordered
    ]
    record_rows = [r.as_record() for r in ordered]

    _write_jsonl(path / "labels.jsonl", label_rows)
    _write_jsonl(path / "sentences.jsonl", sentence_rows)
    _write_jsonl(path / "records.jsonl", record_rows)
    (path / "ids.txt").write_text("".join(f"{r.input_id}\n" for r in ordered), encoding="utf-8")

    label_freq = _label_frequency(ordered)
    source_membership: Counter[str] = Counter()
    multi_source_records = 0
    for r in ordered:
        memberships = (
            r.provenance.get("fixed_split_merge", {}).get("source_membership", [])
            if isinstance(r.provenance, dict)
            else []
        )
        if len(memberships) > 1:
            multi_source_records += 1
        for source in memberships:
            source_membership[str(source)] += 1

    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "split_name": split_name,
        "statistics": {
            "output_records": len(ordered),
            "multi_label_records": sum(1 for r in ordered if len(r.labels) > 1),
            "technique_count": len(label_freq),
            "multi_source_records": multi_source_records,
            "source_membership_counts": dict(sorted(source_membership.items())),
        },
        "label_frequency": dict(sorted(label_freq.items())),
        "configuration": parent_manifest["configuration"],
        "inputs": parent_manifest["inputs"],
        "parent_split_manifest_sha256": parent_manifest["manifest_content_sha256"],
        "outputs": {},
    }
    for filename in ("labels.jsonl", "sentences.jsonl", "records.jsonl", "ids.txt"):
        manifest["outputs"][filename] = _artifact_meta(path / filename)
    manifest["manifest_content_sha256"] = _canonical_json_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    _write_json(path / "dataset_manifest.json", manifest)
    return manifest


def _write_label_distribution(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "technique_id",
        "total_support",
        "target_development_support",
        "development_support",
        "test_support",
        "development_share",
        "absolute_target_error",
        "present_in_both",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SplitError(f"Output directory already exists: {path}; use --overwrite to replace it")
        if path.resolve() == Path(path.anchor).resolve():
            raise SplitError("Refusing to remove a filesystem root")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def build_outputs(
    specs: Sequence[SourceSpec],
    output_dir: Path,
    *,
    dev_fraction: float,
    dev_size_override: Optional[int],
    seed: int,
    min_support_both: int,
    max_swap_passes: int,
    overwrite: bool,
) -> Dict[str, Any]:
    if not (0.0 < dev_fraction < 1.0):
        raise SplitError("--dev_fraction must be between 0 and 1")
    if min_support_both < 2:
        raise SplitError("--min_support_both must be at least 2")
    if max_swap_passes < 0:
        raise SplitError("--max_swap_passes cannot be negative")

    records, merge_audit = load_and_merge_sources(specs)
    dev_size = dev_size_override if dev_size_override is not None else int(round(len(records) * dev_fraction))
    development, test, split_audit = make_split(
        records,
        dev_size=dev_size,
        seed=seed,
        min_support_both=min_support_both,
        max_swap_passes=max_swap_passes,
    )

    _prepare_output_dir(output_dir, overwrite)

    inputs = {
        spec.name: merge_audit["sources"][spec.name]
        for spec in specs
    }
    configuration = {
        "source_merge_level": "base CVE before splitting",
        "cross_source_label_rule": "set union",
        "canonical_text_rule": "longest raw_text, then source argument order, source record order, record SHA-256",
        "output_order": "input_id ascending",
        "development_size_requested": dev_size,
        "development_fraction_requested": dev_fraction,
        "seed": seed,
        "min_support_both": min_support_both,
        "max_swap_passes": max_swap_passes,
        "stratification": split_audit["algorithm"],
        "singleton_policy": "test",
        "uses_model_predictions": False,
        "uses_downstream_metrics": False,
    }

    split_manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "inputs": inputs,
        "configuration": configuration,
        "merge_audit": {k: v for k, v in merge_audit.items() if k != "cross_source_duplicates"},
        "split_audit": {k: v for k, v in split_audit.items() if k != "label_distribution"},
        "outputs": {},
    }
    split_manifest["manifest_content_sha256"] = _canonical_json_sha256(split_manifest)

    combined_manifest = _write_dataset_dir(
        output_dir / "combined",
        records,
        split_name="combined_unique_base_cves",
        parent_manifest=split_manifest,
    )
    development_manifest = _write_dataset_dir(
        output_dir / "development",
        development,
        split_name="development",
        parent_manifest=split_manifest,
    )
    test_manifest = _write_dataset_dir(
        output_dir / "test",
        test,
        split_name="test",
        parent_manifest=split_manifest,
    )

    _write_jsonl(output_dir / "cross_source_duplicates.jsonl", merge_audit["cross_source_duplicates"])
    assignments = []
    dev_ids = {r.input_id for r in development}
    for r in records:
        assignments.append(
            {
                "input_id": r.input_id,
                "split": "development" if r.input_id in dev_ids else "test",
                "labels": list(r.labels),
                "source_membership": r.provenance["fixed_split_merge"]["source_membership"],
            }
        )
    _write_jsonl(output_dir / "split_assignments.jsonl", assignments)
    _write_label_distribution(output_dir / "label_distribution.csv", split_audit["label_distribution"])
    _write_json(
        output_dir / "overlap_audit.json",
        {
            "development_records": len(development),
            "test_records": len(test),
            "overlap_count": 0,
            "overlap_ids": [],
            "development_ids_sha256": split_audit["development_ids_sha256"],
            "test_ids_sha256": split_audit["test_ids_sha256"],
        },
    )

    split_manifest["dataset_manifests"] = {
        "combined": {
            "records": combined_manifest["statistics"]["output_records"],
            "sha256": _sha256_file(output_dir / "combined" / "dataset_manifest.json"),
        },
        "development": {
            "records": development_manifest["statistics"]["output_records"],
            "sha256": _sha256_file(output_dir / "development" / "dataset_manifest.json"),
        },
        "test": {
            "records": test_manifest["statistics"]["output_records"],
            "sha256": _sha256_file(output_dir / "test" / "dataset_manifest.json"),
        },
    }
    for filename in (
        "cross_source_duplicates.jsonl",
        "split_assignments.jsonl",
        "label_distribution.csv",
        "overlap_audit.json",
    ):
        split_manifest["outputs"][filename] = _artifact_meta(output_dir / filename)
    split_manifest["manifest_content_sha256"] = _canonical_json_sha256(
        {k: v for k, v in split_manifest.items() if k != "manifest_content_sha256"}
    )
    _write_json(output_dir / "split_manifest.json", split_manifest)

    # Final hard checks after all writes.
    dev_written = set((output_dir / "development" / "ids.txt").read_text(encoding="utf-8").split())
    test_written = set((output_dir / "test" / "ids.txt").read_text(encoding="utf-8").split())
    combined_written = set((output_dir / "combined" / "ids.txt").read_text(encoding="utf-8").split())
    if dev_written & test_written:
        raise SplitError("Post-write validation found development/test overlap")
    if dev_written | test_written != combined_written:
        raise SplitError("Post-write validation found missing or extra split IDs")

    return {
        "output_dir": str(output_dir),
        "combined_records": len(records),
        "development_records": len(development),
        "test_records": len(test),
        "cross_source_duplicate_groups": merge_audit["cross_source_duplicate_groups"],
        "cross_source_label_union_groups": merge_audit["cross_source_label_union_groups"],
        "cross_source_text_conflict_groups": merge_audit["cross_source_text_conflict_groups"],
        "technique_count": len(_label_frequency(records)),
        "swap_count": split_audit["refinement"]["swap_count"],
        "overlap_count": 0,
        "split_manifest_sha256": _sha256_file(output_dir / "split_manifest.json"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=DATASET_DIR",
        help="Traceable source dataset directory; repeat for multiple sources",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dev_fraction", type=float, default=0.20)
    parser.add_argument(
        "--dev_size",
        type=int,
        default=None,
        help="Exact development size; overrides the rounded --dev_fraction size",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--min_support_both",
        type=int,
        default=2,
        help="Require labels with at least this total support to appear in both splits",
    )
    parser.add_argument("--max_swap_passes", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        specs = [_parse_source_spec(text, i) for i, text in enumerate(args.source)]
        result = build_outputs(
            specs,
            args.output_dir.expanduser().resolve(),
            dev_fraction=args.dev_fraction,
            dev_size_override=args.dev_size,
            seed=args.seed,
            min_support_both=args.min_support_both,
            max_swap_passes=args.max_swap_passes,
            overwrite=args.overwrite,
        )
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
