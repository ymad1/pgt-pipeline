"""Publication-oriented evaluation for CVE-to-ATT&CK ranking experiments.

This module evaluates one or more controlled reranking conditions, including
repeated runs of the same condition.  It is designed for the Reviewer-2
revision and therefore emphasizes strict comparability and reproducibility:

* duplicate CVE rows in the label file are unioned rather than overwritten;
* accidental string-valued labels are normalized to singleton lists and
  reported in the manifest;
* all compared runs must contain the same CVE IDs and identical candidate
  sets/retrieval orders by default;
* sample-macro, label-micro, and technique-macro results are reported;
* per-technique and head/mid/tail recall analyses are exported;
* paired bootstrap confidence intervals and paired significance tests are
  computed on the same CVEs; and
* every input and generated report is hashed in an evaluation manifest.

Example condition arguments use ``NAME=PATH`` and may be repeated, e.g. two
independent runs of the full method can both be supplied as ``full=...``.
The command-line examples are intentionally omitted from this source header;
see the revised README once the complete pipeline has been updated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_VERSION = "compare-rankers-v2.0.0"
TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
DEFAULT_KS = (1, 3, 5, 10, 20)
DEFAULT_TEST_METRICS = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _sample_sd(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    probability = max(0.0, min(1.0, probability))
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if repetitions <= 0 or len(values) == 1:
        point = _mean(values)
        return point, point
    rng = random.Random(seed)
    n = len(values)
    estimates: List[float] = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(_mean(sample))
    estimates.sort()
    alpha = 1.0 - confidence
    return (
        _quantile(estimates, alpha / 2.0),
        _quantile(estimates, 1.0 - alpha / 2.0),
    )


def _bootstrap_difference_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("Paired vectors have different lengths.")
    differences = [a - b for a, b in zip(left, right)]
    return _bootstrap_ci(
        differences,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
    )


def _exact_mcnemar_p(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right):
        raise ValueError("Paired vectors have different lengths.")
    if not all(value in (0.0, 1.0) for value in [*left, *right]):
        return None
    left_only = sum(1 for a, b in zip(left, right) if a == 1.0 and b == 0.0)
    right_only = sum(1 for a, b in zip(left, right) if a == 0.0 and b == 1.0)
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_sign_flip_p(
    left: Sequence[float],
    right: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> float:
    if len(left) != len(right):
        raise ValueError("Paired vectors have different lengths.")
    differences = [a - b for a, b in zip(left, right)]
    nonzero = [value for value in differences if abs(value) > 1e-15]
    if not nonzero:
        return 1.0
    observed = abs(_mean(nonzero))

    if len(nonzero) <= 20:
        extreme = 0
        total = 1 << len(nonzero)
        for mask in range(total):
            signed = [
                value if (mask >> index) & 1 else -value
                for index, value in enumerate(nonzero)
            ]
            if abs(_mean(signed)) >= observed - 1e-15:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    for _ in range(max(1, repetitions)):
        signed = [value if rng.random() < 0.5 else -value for value in nonzero]
        if abs(_mean(signed)) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1.0) / (max(1, repetitions) + 1.0)


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm--Bonferroni adjusted p-values in original order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _to_parent(technique_id: str) -> str:
    return technique_id.split(".", 1)[0]


def _normalize_technique_id(value: Any, *, parent: bool) -> Optional[str]:
    if not isinstance(value, str):
        return None
    technique_id = value.strip().upper()
    if not technique_id or not TECHNIQUE_RE.fullmatch(technique_id):
        return None
    return _to_parent(technique_id) if parent else technique_id


def _label_values(raw: Any) -> Tuple[List[Any], bool]:
    """Return label values and whether a string-valued label was repaired."""
    if raw is None:
        return [], False
    if isinstance(raw, str):
        # A single ATT&CK ID is the legacy error observed in the deduplicated
        # file.  Comma/semicolon separated strings are accepted defensively.
        values = [part.strip() for part in re.split(r"[,;\s]+", raw) if part.strip()]
        return values, True
    if isinstance(raw, (list, tuple, set)):
        return list(raw), False
    return [raw], False


def load_labels(
    path: str | Path,
    *,
    parent: bool,
    fail_on_invalid: bool,
) -> Tuple[Dict[str, set[str]], Dict[str, Any]]:
    labels: Dict[str, set[str]] = defaultdict(set)
    statistics_out: Dict[str, Any] = {
        "rows": 0,
        "unique_input_ids": 0,
        "duplicate_input_rows_unioned": 0,
        "string_valued_labels_repaired": 0,
        "invalid_labels_removed": 0,
        "empty_label_rows": 0,
    }
    seen_rows: set[str] = set()

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            statistics_out["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc
            input_id = row.get("input_id")
            if not isinstance(input_id, str) or not input_id.strip():
                raise ValueError(f"Missing input_id in {path} at line {line_number}.")
            input_id = input_id.strip()
            if input_id in seen_rows:
                statistics_out["duplicate_input_rows_unioned"] += 1
            seen_rows.add(input_id)

            raw_values, repaired_string = _label_values(row.get("labels"))
            statistics_out["string_valued_labels_repaired"] += int(repaired_string)
            valid_count = 0
            for raw_value in raw_values:
                technique_id = _normalize_technique_id(raw_value, parent=parent)
                if technique_id is None:
                    statistics_out["invalid_labels_removed"] += 1
                    if fail_on_invalid:
                        raise ValueError(
                            f"Invalid ATT&CK label {raw_value!r} for {input_id} "
                            f"in {path} at line {line_number}."
                        )
                    continue
                labels[input_id].add(technique_id)
                valid_count += 1
            if valid_count == 0:
                statistics_out["empty_label_rows"] += 1

    normalized = {input_id: value for input_id, value in labels.items() if value}
    statistics_out["unique_input_ids"] = len(normalized)
    statistics_out["unique_techniques"] = len(
        {technique_id for values in normalized.values() for technique_id in values}
    )
    statistics_out["label_assignments"] = sum(len(values) for values in normalized.values())
    return normalized, statistics_out


# ---------------------------------------------------------------------------
# Ranking files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    method: str
    path: Path
    run_index: int


@dataclass
class RankingRun:
    spec: RunSpec
    rankings: Dict[str, List[str]]
    retrieval_rankings: Dict[str, List[str]]
    candidate_signatures: Dict[str, str]
    candidate_ids: Dict[str, Tuple[str, ...]]
    metadata_modes: set[str]


def _parse_run_specs(values: Sequence[str]) -> List[RunSpec]:
    counters: Dict[str, int] = defaultdict(int)
    specs: List[RunSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run must use NAME=PATH format: {value!r}")
        method, raw_path = value.split("=", 1)
        method = method.strip()
        path = Path(raw_path.strip())
        if not method:
            raise ValueError(f"Empty method name in run specification: {value!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        counters[method] += 1
        specs.append(RunSpec(method=method, path=path, run_index=counters[method]))
    if not specs:
        raise ValueError("At least one --run NAME=PATH is required.")
    return specs


def _score(value: Any, default: float = float("-inf")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _candidate_order(
    candidates: Sequence[Mapping[str, Any]],
    *,
    parent: bool,
    allow_short: bool,
    required_length: int,
) -> Tuple[List[str], List[str], Tuple[str, ...], str]:
    valid_candidates: List[Dict[str, Any]] = []
    for original_position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError("Candidate list contains a non-object value.")
        raw_technique_id = _normalize_technique_id(
            candidate.get("technique_id"), parent=False
        )
        if raw_technique_id is None:
            raise ValueError(f"Invalid candidate technique_id: {candidate.get('technique_id')!r}")
        technique_id = _to_parent(raw_technique_id) if parent else raw_technique_id
        item = dict(candidate)
        item["_raw_id"] = raw_technique_id
        item["_normalized_id"] = technique_id
        item["_original_position"] = original_position
        valid_candidates.append(item)

    if not allow_short and len(valid_candidates) < required_length:
        raise ValueError(
            f"Ranking contains {len(valid_candidates)} candidates; at least "
            f"{required_length} are required."
        )

    retrieval_sorted = sorted(
        valid_candidates,
        key=lambda item: (
            int(item.get("retrieval_rank") or item.get("rank") or item["_original_position"]),
            -_score(item.get("retrieval_score", item.get("score_fused")), 0.0),
            str(item["_normalized_id"]),
        ),
    )
    reranked_sorted = sorted(
        valid_candidates,
        key=lambda item: (
            int(item.get("rerank_rank") or 10**9),
            -_score(item.get("final_score"), 0.0),
            -_score(item.get("llm_score"), 0.0),
            int(item.get("retrieval_rank") or item.get("rank") or item["_original_position"]),
            str(item["_normalized_id"]),
        ),
    )
    # Legacy outputs without rerank_rank are ordered by final score.
    if not any(item.get("rerank_rank") is not None for item in valid_candidates):
        reranked_sorted = sorted(
            valid_candidates,
            key=lambda item: (
                -_score(item.get("final_score"), 0.0),
                -_score(item.get("llm_score"), 0.0),
                -_score(item.get("score_fused"), 0.0),
                int(item["_original_position"]),
                str(item["_normalized_id"]),
            ),
        )

    def deduplicate(items: Sequence[Mapping[str, Any]]) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for item in items:
            technique_id = str(item["_normalized_id"])
            if technique_id not in seen:
                seen.add(technique_id)
                output.append(technique_id)
        return output

    retrieval = deduplicate(retrieval_sorted)
    reranked = deduplicate(reranked_sorted)

    # Fairness is checked on the original candidate IDs before optional parent
    # normalization, so different sub-technique sets cannot be hidden by both
    # collapsing to the same parent technique.
    raw_retrieval_order = [str(item["_raw_id"]) for item in retrieval_sorted]
    raw_candidate_set = tuple(sorted(set(raw_retrieval_order)))
    signature = _sha256_bytes(
        _canonical_json(
            {
                "raw_retrieval_order": raw_retrieval_order,
                "raw_candidate_set": raw_candidate_set,
            }
        ).encode("utf-8")
    )
    return reranked, retrieval, raw_candidate_set, signature


def load_ranking_run(
    spec: RunSpec,
    *,
    parent: bool,
    allow_short: bool,
    required_length: int,
) -> RankingRun:
    rankings: Dict[str, List[str]] = {}
    retrieval_rankings: Dict[str, List[str]] = {}
    candidate_signatures: Dict[str, str] = {}
    candidate_ids: Dict[str, Tuple[str, ...]] = {}
    metadata_modes: set[str] = set()

    with spec.path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {spec.path} at line {line_number}.") from exc
            input_id = row.get("input_id")
            if not isinstance(input_id, str) or not input_id.strip():
                raise ValueError(f"Missing input_id in {spec.path} at line {line_number}.")
            input_id = input_id.strip()
            if input_id in rankings:
                raise ValueError(f"Duplicate input_id {input_id!r} in {spec.path}.")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Empty candidate list for {input_id} in {spec.path}.")
            reranked, retrieval, candidate_set, signature = _candidate_order(
                candidates,
                parent=parent,
                allow_short=allow_short,
                required_length=required_length,
            )
            rankings[input_id] = reranked
            retrieval_rankings[input_id] = retrieval
            candidate_ids[input_id] = candidate_set
            candidate_signatures[input_id] = signature
            metadata = row.get("rerank_metadata")
            if isinstance(metadata, Mapping) and metadata.get("mode"):
                metadata_modes.add(str(metadata["mode"]))

    if not rankings:
        raise ValueError(f"No ranking rows found in {spec.path}.")
    return RankingRun(
        spec=spec,
        rankings=rankings,
        retrieval_rankings=retrieval_rankings,
        candidate_signatures=candidate_signatures,
        candidate_ids=candidate_ids,
        metadata_modes=metadata_modes,
    )


def _resolve_evaluation_ids(
    labels: Mapping[str, set[str]],
    runs: Sequence[RankingRun],
    *,
    policy: str,
) -> Tuple[List[str], Dict[str, Any]]:
    label_ids = set(labels)
    run_id_sets = [set(run.rankings) for run in runs]
    report: Dict[str, Any] = {
        "label_ids": len(label_ids),
        "run_ids": {
            f"{run.spec.method}#{run.spec.run_index}": len(run.rankings) for run in runs
        },
    }

    if policy == "strict":
        reference = run_id_sets[0]
        for run, ids in zip(runs, run_id_sets):
            if ids != reference:
                missing = sorted(reference - ids)[:10]
                extra = sorted(ids - reference)[:10]
                raise ValueError(
                    f"Run {run.spec.method}#{run.spec.run_index} has different CVE IDs. "
                    f"Missing examples: {missing}; extra examples: {extra}."
                )
        missing_labels = sorted(reference - label_ids)
        if missing_labels:
            raise ValueError(
                f"{len(missing_labels)} evaluated CVEs have no valid labels; examples: "
                f"{missing_labels[:10]}"
            )
        evaluation_ids = sorted(reference)
    elif policy == "intersection":
        common = set(label_ids)
        for ids in run_id_sets:
            common.intersection_update(ids)
        evaluation_ids = sorted(common)
        if not evaluation_ids:
            raise ValueError("No common labeled CVEs across the supplied runs.")
    else:
        raise ValueError(f"Unknown ID policy: {policy}")

    report["evaluation_ids"] = len(evaluation_ids)
    report["excluded_label_only_ids"] = len(label_ids - set(evaluation_ids))
    report["excluded_by_run"] = {
        f"{run.spec.method}#{run.spec.run_index}": len(set(run.rankings) - set(evaluation_ids))
        for run in runs
    }
    return evaluation_ids, report


def _verify_candidate_fairness(
    runs: Sequence[RankingRun],
    evaluation_ids: Sequence[str],
    *,
    strict: bool,
) -> Dict[str, Any]:
    reference = runs[0]
    mismatches: List[Dict[str, Any]] = []
    for run in runs[1:]:
        for input_id in evaluation_ids:
            if run.candidate_signatures[input_id] != reference.candidate_signatures[input_id]:
                mismatches.append(
                    {
                        "input_id": input_id,
                        "reference": f"{reference.spec.method}#{reference.spec.run_index}",
                        "other": f"{run.spec.method}#{run.spec.run_index}",
                        "reference_candidates": list(reference.candidate_ids[input_id]),
                        "other_candidates": list(run.candidate_ids[input_id]),
                    }
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    if mismatches and strict:
        first = mismatches[0]
        raise ValueError(
            "Candidate-set/retrieval-order mismatch across conditions for "
            f"{first['input_id']} ({first['reference']} vs {first['other']})."
        )
    return {
        "strict": strict,
        "reference_run": f"{reference.spec.method}#{reference.spec.run_index}",
        "mismatch_count_capped": len(mismatches),
        "mismatch_examples": mismatches,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _average_precision_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    relevant_seen = 0
    precision_sum = 0.0
    for rank, technique_id in enumerate(ranking[:k], start=1):
        if technique_id in gold:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    return precision_sum / min(len(gold), k)


def _sample_metrics(ranking: Sequence[str], gold: set[str], ks: Sequence[int]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    first_relevant_rank: Optional[int] = None
    for rank, technique_id in enumerate(ranking, start=1):
        if technique_id in gold:
            first_relevant_rank = rank
            break
    result["mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    for k in ks:
        topk = list(ranking[:k])
        intersection = len(gold.intersection(topk))
        result[f"hit@{k}"] = 1.0 if intersection else 0.0
        result[f"precision@{k}"] = intersection / k
        result[f"recall@{k}"] = intersection / len(gold)
        result[f"ap@{k}"] = _average_precision_at_k(ranking, gold, k)
        result[f"retrieved_labels@{k}"] = float(intersection)
    return result


def _evaluate_run(
    run: RankingRun,
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    return {
        input_id: _sample_metrics(run.rankings[input_id], labels[input_id], ks)
        for input_id in evaluation_ids
    }


def _aggregate_method_samples(
    run_samples: Sequence[Mapping[str, Mapping[str, float]]],
    evaluation_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    metric_names = sorted(next(iter(run_samples[0].values())).keys())
    aggregate: Dict[str, Dict[str, float]] = {}
    for input_id in evaluation_ids:
        aggregate[input_id] = {
            metric: _mean([samples[input_id][metric] for samples in run_samples])
            for metric in metric_names
        }
    return aggregate


def _metric_summary(
    values: Sequence[float],
    *,
    bootstrap_repetitions: int,
    confidence: float,
    seed: int,
) -> Dict[str, float]:
    lower, upper = _bootstrap_ci(
        values,
        repetitions=bootstrap_repetitions,
        confidence=confidence,
        seed=seed,
    )
    return {
        "estimate": _mean(values),
        "ci_lower": lower,
        "ci_upper": upper,
        "sample_sd": _sample_sd(values),
    }


def _method_summary(
    method: str,
    runs: Sequence[RankingRun],
    run_samples: Sequence[Mapping[str, Mapping[str, float]]],
    aggregate_samples: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
    ks: Sequence[int],
    *,
    bootstrap_repetitions: int,
    confidence: float,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metric_names = sorted(next(iter(aggregate_samples.values())).keys())
    metrics: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for metric_index, metric in enumerate(metric_names):
        values = [aggregate_samples[input_id][metric] for input_id in evaluation_ids]
        run_estimates = [
            _mean([samples[input_id][metric] for input_id in evaluation_ids])
            for samples in run_samples
        ]
        summary = _metric_summary(
            values,
            bootstrap_repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + metric_index * 1009,
        )
        summary["run_mean"] = _mean(run_estimates)
        summary["run_sd"] = _sample_sd(run_estimates)
        metrics[metric] = summary
        summary_rows.append(
            {
                "method": method,
                "runs": len(runs),
                "metric": metric,
                **summary,
                "n_cves": len(evaluation_ids),
            }
        )

    # Label-micro precision/recall and technique-macro recall.
    for k in ks:
        retrieved_sum = sum(
            aggregate_samples[input_id][f"retrieved_labels@{k}"]
            for input_id in evaluation_ids
        )
        total_gold = sum(len(labels[input_id]) for input_id in evaluation_ids)
        metrics[f"micro_precision@{k}"] = {
            "estimate": retrieved_sum / (len(evaluation_ids) * k)
        }
        metrics[f"micro_recall@{k}"] = {"estimate": retrieved_sum / total_gold}

    return {
        "runs": len(runs),
        "run_files": [str(run.spec.path) for run in runs],
        "metadata_modes": sorted({mode for run in runs for mode in run.metadata_modes}),
        "metrics": metrics,
    }, summary_rows


def _retrieval_coverage(
    reference: RankingRun,
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for k in ks:
        values = [
            1.0
            if labels[input_id].intersection(reference.retrieval_rankings[input_id][:k])
            else 0.0
            for input_id in evaluation_ids
        ]
        output[f"candidate_coverage@{k}"] = _mean(values)
    full_values = [
        1.0
        if labels[input_id].intersection(reference.retrieval_rankings[input_id])
        else 0.0
        for input_id in evaluation_ids
    ]
    output["candidate_coverage@all"] = _mean(full_values)
    return output


# ---------------------------------------------------------------------------
# Per-technique and long-tail analyses
# ---------------------------------------------------------------------------


def _bucket_for_support(support: int, *, tail_max: int, head_min: int) -> str:
    if support <= tail_max:
        return "tail"
    if support >= head_min:
        return "head"
    return "mid"


def _per_technique_rows(
    method: str,
    runs: Sequence[RankingRun],
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
    ks: Sequence[int],
    *,
    tail_max: int,
    head_min: int,
) -> List[Dict[str, Any]]:
    support_ids: Dict[str, List[str]] = defaultdict(list)
    for input_id in evaluation_ids:
        for technique_id in labels[input_id]:
            support_ids[technique_id].append(input_id)

    rows: List[Dict[str, Any]] = []
    for technique_id in sorted(support_ids):
        ids = support_ids[technique_id]
        row: Dict[str, Any] = {
            "method": method,
            "technique_id": technique_id,
            "support": len(ids),
            "frequency_bucket": _bucket_for_support(
                len(ids), tail_max=tail_max, head_min=head_min
            ),
        }
        for k in ks:
            run_recalls = []
            for run in runs:
                hits = sum(
                    1 for input_id in ids if technique_id in run.rankings[input_id][:k]
                )
                run_recalls.append(hits / len(ids))
            row[f"recall@{k}"] = _mean(run_recalls)
            row[f"run_sd@{k}"] = _sample_sd(run_recalls)
        rows.append(row)
    return rows


def _long_tail_summary(
    per_technique_rows: Sequence[Mapping[str, Any]], ks: Sequence[int]
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_technique_rows:
        grouped[(str(row["method"]), str(row["frequency_bucket"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (method, bucket), rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "method": method,
            "frequency_bucket": bucket,
            "techniques": len(rows),
            "label_assignments": sum(int(row["support"]) for row in rows),
        }
        for k in ks:
            summary[f"technique_macro_recall@{k}"] = _mean(
                [float(row[f"recall@{k}"]) for row in rows]
            )
        output.append(summary)
    return output


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------


def _pairwise_rows(
    aggregate_samples_by_method: Mapping[str, Mapping[str, Mapping[str, float]]],
    evaluation_ids: Sequence[str],
    metrics: Sequence[str],
    *,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    confidence: float,
    seed: int,
    reference_method: Optional[str],
) -> List[Dict[str, Any]]:
    methods = sorted(aggregate_samples_by_method)
    pairs: List[Tuple[str, str]] = []
    if reference_method:
        if reference_method not in aggregate_samples_by_method:
            raise ValueError(f"Unknown reference method: {reference_method}")
        pairs = [(reference_method, method) for method in methods if method != reference_method]
    else:
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                pairs.append((left, right))

    rows: List[Dict[str, Any]] = []
    for pair_index, (left_method, right_method) in enumerate(pairs):
        for metric_index, metric in enumerate(metrics):
            try:
                left = [
                    aggregate_samples_by_method[left_method][input_id][metric]
                    for input_id in evaluation_ids
                ]
                right = [
                    aggregate_samples_by_method[right_method][input_id][metric]
                    for input_id in evaluation_ids
                ]
            except KeyError as exc:
                raise ValueError(f"Unknown test metric: {metric}") from exc

            row_seed = seed + pair_index * 100003 + metric_index * 1009
            ci_lower, ci_upper = _bootstrap_difference_ci(
                left,
                right,
                repetitions=bootstrap_repetitions,
                confidence=confidence,
                seed=row_seed,
            )
            mcnemar_p = _exact_mcnemar_p(left, right)
            if mcnemar_p is not None:
                p_value = mcnemar_p
                test_name = "exact_mcnemar"
            else:
                p_value = _paired_sign_flip_p(
                    left,
                    right,
                    repetitions=permutation_repetitions,
                    seed=row_seed,
                )
                test_name = "paired_sign_flip"
            differences = [a - b for a, b in zip(left, right)]
            rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "left_estimate": _mean(left),
                    "right_estimate": _mean(right),
                    "difference_left_minus_right": _mean(differences),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "test": test_name,
                    "p_value": p_value,
                    "wins": sum(value > 1e-15 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "losses": sum(value < -1e-15 for value in differences),
                    "n_cves": len(evaluation_ids),
                }
            )

    adjusted = _holm_adjust([float(row["p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = adjusted_p
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict repeated-run evaluation for controlled CVE-to-ATT&CK rankers."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for every condition/run; identical NAME values are repeated runs.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parent", action="store_true")
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--id_policy", choices=("strict", "intersection"), default="strict")
    parser.add_argument(
        "--strict_candidate_fairness",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow_short_rankings", action="store_true")
    parser.add_argument(
        "--include_retrieval_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive a retrieval-only ranking from the shared retrieval_rank fields.",
    )
    parser.add_argument("--retrieval_method_name", default="retrieval")
    parser.add_argument(
        "--fail_on_invalid_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=5000)
    parser.add_argument("--permutation_repetitions", type=int, default=20000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--tail_max_support", type=int, default=5)
    parser.add_argument("--head_min_support", type=int, default=21)
    parser.add_argument(
        "--test_metrics", default=",".join(DEFAULT_TEST_METRICS)
    )
    parser.add_argument(
        "--reference_method",
        default="full",
        help="Pair every method against this method; use 'all' for all pairs.",
    )
    args = parser.parse_args()

    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if not ks or min(ks) < 1:
        parser.error("--ks must contain positive integers.")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap_repetitions must be non-negative.")
    if args.permutation_repetitions < 1:
        parser.error("--permutation_repetitions must be positive.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be in (0,1).")
    if args.tail_max_support < 1:
        parser.error("--tail_max_support must be positive.")
    if args.head_min_support <= args.tail_max_support:
        parser.error("--head_min_support must exceed --tail_max_support.")

    run_specs = _parse_run_specs(args.run)
    labels, label_statistics = load_labels(
        args.labels,
        parent=args.parent,
        fail_on_invalid=args.fail_on_invalid_labels,
    )
    required_length = max(ks)
    runs = [
        load_ranking_run(
            spec,
            parent=args.parent,
            allow_short=args.allow_short_rankings,
            required_length=required_length,
        )
        for spec in run_specs
    ]
    evaluation_ids, id_report = _resolve_evaluation_ids(
        labels, runs, policy=args.id_policy
    )
    fairness_report = _verify_candidate_fairness(
        runs,
        evaluation_ids,
        strict=args.strict_candidate_fairness,
    )

    evaluation_runs = list(runs)
    if args.include_retrieval_baseline:
        existing_methods = {run.spec.method for run in runs}
        if args.retrieval_method_name in existing_methods:
            raise ValueError(
                f"Retrieval baseline name {args.retrieval_method_name!r} conflicts "
                "with a supplied method name."
            )
        reference_run = runs[0]
        evaluation_runs.append(
            RankingRun(
                spec=RunSpec(
                    method=args.retrieval_method_name,
                    path=reference_run.spec.path,
                    run_index=1,
                ),
                rankings=dict(reference_run.retrieval_rankings),
                retrieval_rankings=dict(reference_run.retrieval_rankings),
                candidate_signatures=dict(reference_run.candidate_signatures),
                candidate_ids=dict(reference_run.candidate_ids),
                metadata_modes={"retrieval_only_derived"},
            )
        )

    runs_by_method: Dict[str, List[RankingRun]] = defaultdict(list)
    for run in evaluation_runs:
        runs_by_method[run.spec.method].append(run)

    run_samples_by_method: Dict[str, List[Dict[str, Dict[str, float]]]] = {}
    aggregate_samples_by_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    method_report: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for method_index, method in enumerate(sorted(runs_by_method)):
        method_runs = runs_by_method[method]
        run_samples = [
            _evaluate_run(run, labels, evaluation_ids, ks) for run in method_runs
        ]
        aggregate_samples = _aggregate_method_samples(run_samples, evaluation_ids)
        run_samples_by_method[method] = run_samples
        aggregate_samples_by_method[method] = aggregate_samples
        report, rows = _method_summary(
            method,
            method_runs,
            run_samples,
            aggregate_samples,
            labels,
            evaluation_ids,
            ks,
            bootstrap_repetitions=args.bootstrap_repetitions,
            confidence=args.confidence,
            seed=args.seed + method_index * 100003,
        )
        method_report[method] = report
        summary_rows.extend(rows)

    per_technique_rows: List[Dict[str, Any]] = []
    for method in sorted(runs_by_method):
        per_technique_rows.extend(
            _per_technique_rows(
                method,
                runs_by_method[method],
                labels,
                evaluation_ids,
                ks,
                tail_max=args.tail_max_support,
                head_min=args.head_min_support,
            )
        )
    long_tail_rows = _long_tail_summary(per_technique_rows, ks)

    # Overall technique-macro recall gives each ATT&CK technique equal weight,
    # complementing the CVE-macro and label-micro summaries above.
    per_technique_by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_technique_rows:
        per_technique_by_method[str(row["method"])].append(row)
    for method in sorted(per_technique_by_method):
        rows = per_technique_by_method[method]
        for k in ks:
            metric_name = f"technique_macro_recall@{k}"
            estimate = _mean([float(row[f"recall@{k}"]) for row in rows])
            method_report[method]["metrics"][metric_name] = {"estimate": estimate}
            summary_rows.append(
                {
                    "method": method,
                    "runs": len(runs_by_method[method]),
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_lower": "",
                    "ci_upper": "",
                    "sample_sd": "",
                    "run_mean": estimate,
                    "run_sd": "",
                    "n_cves": len(evaluation_ids),
                }
            )

    test_metrics = [
        value.strip() for value in args.test_metrics.split(",") if value.strip()
    ]
    reference_method = None if args.reference_method.lower() == "all" else args.reference_method
    pairwise_rows = _pairwise_rows(
        aggregate_samples_by_method,
        evaluation_ids,
        test_metrics,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        confidence=args.confidence,
        seed=args.seed,
        reference_method=reference_method,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metric_summary.csv"
    pairwise_path = output_dir / "pairwise_tests.csv"
    per_technique_path = output_dir / "per_technique_recall.csv"
    long_tail_path = output_dir / "long_tail_summary.csv"
    report_path = output_dir / "evaluation_report.json"
    manifest_path = output_dir / "evaluation_manifest.json"

    _write_csv(
        summary_path,
        summary_rows,
        fieldnames=(
            "method",
            "runs",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "sample_sd",
            "run_mean",
            "run_sd",
            "n_cves",
        ),
    )
    _write_csv(
        pairwise_path,
        pairwise_rows,
        fieldnames=(
            "left_method",
            "right_method",
            "metric",
            "left_estimate",
            "right_estimate",
            "difference_left_minus_right",
            "ci_lower",
            "ci_upper",
            "test",
            "p_value",
            "p_value_holm",
            "wins",
            "ties",
            "losses",
            "n_cves",
        ),
    )
    per_technique_fields = [
        "method",
        "technique_id",
        "support",
        "frequency_bucket",
    ] + [field for k in ks for field in (f"recall@{k}", f"run_sd@{k}")]
    _write_csv(per_technique_path, per_technique_rows, per_technique_fields)
    long_tail_fields = [
        "method",
        "frequency_bucket",
        "techniques",
        "label_assignments",
    ] + [f"technique_macro_recall@{k}" for k in ks]
    _write_csv(long_tail_path, long_tail_rows, long_tail_fields)

    coverage = _retrieval_coverage(runs[0], labels, evaluation_ids, ks)
    report: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": _utc_now(),
        "configuration": {
            "parent_normalization": args.parent,
            "ks": ks,
            "id_policy": args.id_policy,
            "strict_candidate_fairness": args.strict_candidate_fairness,
            "allow_short_rankings": args.allow_short_rankings,
            "include_retrieval_baseline": args.include_retrieval_baseline,
            "retrieval_method_name": args.retrieval_method_name,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "permutation_repetitions": args.permutation_repetitions,
            "confidence": args.confidence,
            "seed": args.seed,
            "tail_max_support": args.tail_max_support,
            "head_min_support": args.head_min_support,
            "test_metrics": test_metrics,
            "reference_method": reference_method,
        },
        "label_statistics": label_statistics,
        "id_report": id_report,
        "candidate_fairness": fairness_report,
        "candidate_coverage": coverage,
        "methods": method_report,
        "long_tail": long_tail_rows,
        "pairwise_tests": pairwise_rows,
    }
    _write_json(report_path, report)

    input_hashes = {"labels": _sha256_file(args.labels)}
    for run in runs:
        input_hashes[f"{run.spec.method}#{run.spec.run_index}"] = _sha256_file(run.spec.path)
    manifest: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": _sha256_file(__file__),
        "generated_utc": _utc_now(),
        "python_version": sys.version,
        "input_sha256": input_hashes,
        "output_sha256": {
            summary_path.name: _sha256_file(summary_path),
            pairwise_path.name: _sha256_file(pairwise_path),
            per_technique_path.name: _sha256_file(per_technique_path),
            long_tail_path.name: _sha256_file(long_tail_path),
            report_path.name: _sha256_file(report_path),
        },
        "run_signature": _sha256_bytes(
            _canonical_json(
                {
                    "configuration": report["configuration"],
                    "input_sha256": input_hashes,
                    "evaluation_ids": evaluation_ids,
                }
            ).encode("utf-8")
        ),
    }
    _write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "methods": {method: len(values) for method, values in runs_by_method.items()},
                "evaluation_cves": len(evaluation_ids),
                "candidate_fairness_mismatches": fairness_report["mismatch_count_capped"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
