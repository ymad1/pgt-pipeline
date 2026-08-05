"""Development-only selection of retrieval fusion weight and candidate budget.

The selector reuses the canonical TF--IDF retrieval implementation and evaluates
only a fixed development split.  A held-out test ID file may be supplied solely
to verify that the two splits are disjoint; test labels and test metrics are
never read.

For each ``alpha`` and candidate budget ``Top-N`` in the declared grid, the
script reports candidate coverage, sample-macro recall, precision, AP, and MRR.
The selected configuration follows a predeclared cost-aware rule:

1. retain configurations within ``primary_tolerance`` of the best development
   value of the primary metric;
2. choose the smallest candidate budget among the retained configurations;
3. break remaining ties by candidate coverage, MRR, macro recall, closeness of
   alpha to 0.5, and finally the smaller alpha.

This rule prevents the test set from influencing retrieval configuration and
avoids automatically choosing the largest candidate budget for a negligible
coverage gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .compare_rankers import load_labels
from .retrieve_candidates import (
    build_idf,
    evidence_query,
    load_mes,
    load_technique_index,
    mes_query,
    rank_one,
    tfidf_vector,
    tokenize,
)

SCRIPT_VERSION = "retrieval-selection-v1.0.0"
DEFAULT_ALPHAS = "0.00,0.20,0.40,0.50,0.60,0.80,1.00"
DEFAULT_TOPNS = "5,10,15,20,30,50"
PRIMARY_METRICS = {
    "candidate_coverage",
    "macro_recall",
    "mrr",
    "map",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_stable_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            yield row


def _read_ids(path: Path) -> List[str]:
    values: List[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        if value in seen:
            raise ValueError(f"Duplicate ID {value!r} in {path} at line {line_number}")
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError(f"ID file is empty: {path}")
    return values


def _parse_float_grid(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Alpha outside [0,1]: {value}")
        values.append(round(value, 12))
    result = sorted(set(values))
    if not result:
        raise ValueError("Alpha grid is empty")
    return result


def _parse_int_grid(raw: str) -> List[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("Top-N grid must contain positive integers")
    return values


def _average_precision(ranking: Sequence[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    hits = 0
    total = 0.0
    for index, technique_id in enumerate(ranking, start=1):
        if technique_id in gold:
            hits += 1
            total += hits / index
    return total / len(gold)


def _sample_metrics(ranking: Sequence[str], gold: set[str]) -> Dict[str, float]:
    hit_positions = [index for index, technique_id in enumerate(ranking, start=1) if technique_id in gold]
    hits = len(set(ranking) & gold)
    return {
        "candidate_coverage": float(bool(hit_positions)),
        "macro_recall": hits / len(gold) if gold else 0.0,
        "precision": hits / len(ranking) if ranking else 0.0,
        "map": _average_precision(ranking, gold),
        "mrr": (1.0 / min(hit_positions)) if hit_positions else 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    probability = max(0.0, min(1.0, probability))
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if repetitions <= 0 or len(values) == 1:
        point = _mean(values)
        return point, point
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        _mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(repetitions)
    ]
    estimates.sort()
    alpha = 1.0 - confidence
    return _quantile(estimates, alpha / 2.0), _quantile(estimates, 1.0 - alpha / 2.0)


def _index_sentence_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            raise ValueError(f"Sentence record missing input_id in {path}")
        if input_id in result:
            raise ValueError(f"Duplicate sentence input_id: {input_id}")
        result[input_id] = row
    return result


def _selected_row(rows: Sequence[Mapping[str, Any]], primary_metric: str, tolerance: float) -> Dict[str, Any]:
    best_value = max(float(row[primary_metric]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row[primary_metric]) >= best_value - tolerance - 1e-15
    ]
    eligible.sort(
        key=lambda row: (
            int(row["topn"]),
            -float(row["candidate_coverage"]),
            -float(row["mrr"]),
            -float(row["macro_recall"]),
            abs(float(row["alpha"]) - 0.5),
            float(row["alpha"]),
        )
    )
    selected = eligible[0]
    selected["best_primary_value"] = best_value
    selected["eligible_configuration_count"] = len(eligible)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select retrieval alpha and Top-N using only a fixed development split."
    )
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--mes", required=True)
    parser.add_argument("--tech_index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev_ids", required=True)
    parser.add_argument(
        "--test_ids",
        default=None,
        help="Optional held-out IDs used only for a split-overlap check; no test metrics are computed.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--topns", default=DEFAULT_TOPNS)
    parser.add_argument("--score_normalization", choices=("none", "minmax"), default="none")
    parser.add_argument("--normalize_to_parent", action="store_true")
    parser.add_argument("--primary_metric", choices=sorted(PRIMARY_METRICS), default="candidate_coverage")
    parser.add_argument(
        "--primary_tolerance",
        type=float,
        default=0.005,
        help="Absolute development-metric tolerance used before preferring a smaller Top-N.",
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--allow_zero_primary",
        action="store_true",
        help="Allow selection when every development configuration has zero primary performance; intended only for interface smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.primary_tolerance < 0.0:
        raise ValueError("--primary_tolerance must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be within (0,1)")
    if args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap_repetitions must be non-negative")

    sentences_path = Path(args.sentences)
    mes_path = Path(args.mes)
    tech_index_path = Path(args.tech_index)
    labels_path = Path(args.labels)
    dev_ids_path = Path(args.dev_ids)
    test_ids_path = Path(args.test_ids) if args.test_ids else None
    output_dir = Path(args.output_dir)
    sweep_path = output_dir / "retrieval_sweep.csv"
    selected_path = output_dir / "selected_retrieval.json"
    manifest_path = output_dir / "retrieval_selection_manifest.json"

    outputs = [sweep_path, selected_path, manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}; use --overwrite")

    alphas = _parse_float_grid(args.alphas)
    topns = _parse_int_grid(args.topns)
    dev_ids = _read_ids(dev_ids_path)
    dev_set = set(dev_ids)
    overlap: List[str] = []
    if test_ids_path is not None:
        test_ids = _read_ids(test_ids_path)
        overlap = sorted(dev_set & set(test_ids))
        if overlap:
            raise ValueError(
                f"Development/test overlap detected ({len(overlap)} IDs), examples: {overlap[:5]}"
            )

    labels, label_statistics = load_labels(
        labels_path,
        parent=bool(args.normalize_to_parent),
        fail_on_invalid=True,
    )
    missing_labels = [input_id for input_id in dev_ids if input_id not in labels]
    if missing_labels:
        raise ValueError(
            f"Labels missing for {len(missing_labels)} development IDs; examples: {missing_labels[:5]}"
        )

    sentences = _index_sentence_rows(sentences_path)
    missing_sentences = [input_id for input_id in dev_ids if input_id not in sentences]
    if missing_sentences:
        raise ValueError(
            f"Sentences missing for {len(missing_sentences)} development IDs; examples: {missing_sentences[:5]}"
        )

    mes_by_id, mes_status_counts = load_mes(mes_path)
    missing_mes = [input_id for input_id in dev_ids if input_id not in mes_by_id]
    if missing_mes:
        raise ValueError(
            f"MES missing for {len(missing_mes)} development IDs; examples: {missing_mes[:5]}"
        )

    technique_ids, technique_docs, technique_metadata = load_technique_index(tech_index_path)
    if max(topns) > len(technique_ids):
        raise ValueError(
            f"Top-N grid requests {max(topns)} candidates but the index contains only {len(technique_ids)} techniques"
        )
    technique_tokens = [tokenize(document) for document in technique_docs]
    idf = build_idf(technique_tokens)
    technique_vectors = [tfidf_vector(tokens, idf) for tokens in technique_tokens]

    query_cache: Dict[str, Tuple[str, str]] = {}
    mes_usage_counts: Dict[str, int] = defaultdict(int)
    for input_id in dev_ids:
        text_query, _ = evidence_query(sentences[input_id])
        structure_query = mes_query(mes_by_id[input_id])
        query_cache[input_id] = (text_query, structure_query)
        mes_usage_counts[str(mes_by_id[input_id].get("status", "unknown"))] += 1

    rows: List[Dict[str, Any]] = []
    max_topn = max(topns)
    for alpha_index, alpha in enumerate(alphas):
        full_rankings: Dict[str, List[str]] = {}
        for input_id in dev_ids:
            text_query, structure_query = query_cache[input_id]
            candidates = rank_one(
                query_text=text_query,
                query_structure=structure_query,
                technique_ids=technique_ids,
                technique_vectors=technique_vectors,
                idf=idf,
                alpha=alpha,
                normalization=str(args.score_normalization),
                parent_normalization=bool(args.normalize_to_parent),
                topn=max_topn,
            )
            full_rankings[input_id] = [str(item["technique_id"]) for item in candidates]

        for topn in topns:
            vectors: Dict[str, List[float]] = defaultdict(list)
            for input_id in dev_ids:
                metrics = _sample_metrics(full_rankings[input_id][:topn], labels[input_id])
                for metric_name, value in metrics.items():
                    vectors[metric_name].append(float(value))
            primary_values = vectors[str(args.primary_metric)]
            ci_low, ci_high = _bootstrap_ci(
                primary_values,
                repetitions=int(args.bootstrap_repetitions),
                confidence=float(args.confidence),
                seed=int(args.seed) + alpha_index * 1000 + topn,
            )
            rows.append(
                {
                    "alpha": alpha,
                    "topn": topn,
                    "candidate_coverage": round(_mean(vectors["candidate_coverage"]), 12),
                    "macro_recall": round(_mean(vectors["macro_recall"]), 12),
                    "precision": round(_mean(vectors["precision"]), 12),
                    "map": round(_mean(vectors["map"]), 12),
                    "mrr": round(_mean(vectors["mrr"]), 12),
                    "primary_metric": str(args.primary_metric),
                    "primary_value": round(_mean(primary_values), 12),
                    "primary_ci_low": round(ci_low, 12),
                    "primary_ci_high": round(ci_high, 12),
                    "development_cves": len(dev_ids),
                }
            )

    # Duplicate the selected primary metric under its canonical key so the
    # selection helper can operate directly on all supported metrics.
    best_primary = max(float(row[str(args.primary_metric)]) for row in rows)
    if best_primary <= 0.0 and not args.allow_zero_primary:
        raise RuntimeError(
            "Every retrieval configuration has zero development performance for "
            f"{args.primary_metric}; refusing to select an arbitrary configuration. "
            "Use --allow_zero_primary only for an interface smoke test."
        )
    selected = _selected_row(rows, str(args.primary_metric), float(args.primary_tolerance))
    selected_alpha = float(selected["alpha"])
    selected_topn = int(selected["topn"])

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(
        sweep_path,
        rows,
        (
            "alpha",
            "topn",
            "candidate_coverage",
            "macro_recall",
            "precision",
            "map",
            "mrr",
            "primary_metric",
            "primary_value",
            "primary_ci_low",
            "primary_ci_high",
            "development_cves",
        ),
    )

    selected_payload = {
        "version": SCRIPT_VERSION,
        "selected_alpha": selected_alpha,
        "selected_topn": selected_topn,
        "score_normalization": str(args.score_normalization),
        "parent_normalization": bool(args.normalize_to_parent),
        "development_metrics": {
            key: selected[key]
            for key in (
                "candidate_coverage",
                "macro_recall",
                "precision",
                "map",
                "mrr",
                "primary_ci_low",
                "primary_ci_high",
            )
        },
        "selection_rule": {
            "primary_metric": str(args.primary_metric),
            "primary_tolerance": float(args.primary_tolerance),
            "best_primary_value": selected["best_primary_value"],
            "eligible_configuration_count": selected["eligible_configuration_count"],
            "tie_break_order": [
                "smallest_topn",
                "highest_candidate_coverage",
                "highest_mrr",
                "highest_macro_recall",
                "alpha_closest_to_0.5",
                "smaller_alpha",
            ],
        },
        "split_guard": {
            "development_id_count": len(dev_ids),
            "test_id_file_supplied": test_ids_path is not None,
            "development_test_overlap": len(overlap),
            "test_labels_read": False,
            "test_metrics_computed": False,
        },
    }
    _write_json_atomic(selected_path, selected_payload)

    manifest = {
        "version": SCRIPT_VERSION,
        "configuration": {
            "alphas": alphas,
            "topns": topns,
            "score_normalization": str(args.score_normalization),
            "parent_normalization": bool(args.normalize_to_parent),
            "primary_metric": str(args.primary_metric),
            "primary_tolerance": float(args.primary_tolerance),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "confidence": float(args.confidence),
            "seed": int(args.seed),
            "heuristic_boosting": False,
            "allow_zero_primary": bool(args.allow_zero_primary),
        },
        "inputs": {
            "sentences": {"path": str(sentences_path), "sha256": _sha256_file(sentences_path)},
            "mes": {"path": str(mes_path), "sha256": _sha256_file(mes_path)},
            "tech_index": {"path": str(tech_index_path), "sha256": _sha256_file(tech_index_path)},
            "labels": {"path": str(labels_path), "sha256": _sha256_file(labels_path)},
            "dev_ids": {"path": str(dev_ids_path), "sha256": _sha256_file(dev_ids_path)},
            "test_ids": (
                {"path": str(test_ids_path), "sha256": _sha256_file(test_ids_path)}
                if test_ids_path is not None
                else None
            ),
        },
        "counts": {
            "development_cves": len(dev_ids),
            "technique_corpus": technique_metadata,
            "mes_status_in_file": mes_status_counts,
            "mes_status_used": dict(sorted(mes_usage_counts.items())),
            "label_statistics": label_statistics,
            "grid_configurations": len(rows),
        },
        "selection": selected_payload,
        "outputs": {
            "retrieval_sweep": {"path": str(sweep_path), "sha256": _sha256_file(sweep_path)},
            "selected_retrieval": {"path": str(selected_path), "sha256": _sha256_file(selected_path)},
        },
    }
    _write_json_atomic(manifest_path, manifest)

    print(
        json.dumps(
            {
                "development_cves": len(dev_ids),
                "grid_configurations": len(rows),
                "selected_alpha": selected_alpha,
                "selected_topn": selected_topn,
                "selected": str(selected_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
