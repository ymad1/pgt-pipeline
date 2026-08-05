"""Development-only selection of the retrieval/LLM fusion weight beta.

This module selects ``beta`` using a fixed development split and never reports
or inspects final-test performance.  It is intended for the Reviewer-2 revision
of the CVE-to-ATT&CK pipeline, where hyperparameters must be chosen independently
from the held-out test set.

The input reranking files must retain both ``score_fused`` (retrieval score) and
``llm_score`` for every candidate.  Multiple reranking files may be supplied to
select beta from the mean development-set performance across repeated LLM runs.

Generated files:

* ``beta_sweep.csv``: one row per beta with the primary development metric,
  confidence interval, run-to-run variability, and secondary metrics;
* ``selected_beta.json``: the deterministic selection decision and tie-break
  trace;
* ``beta_selection_manifest.json``: input hashes, split provenance, validation
  counts, and output hashes.

No test-set metric is computed by this module.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_VERSION = "sweep-beta-offline-v2.0.0"
TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
DEFAULT_KS = (1, 3, 5, 10, 20)
DEFAULT_BETAS = tuple(round(index / 20.0, 2) for index in range(21))
SUPPORTED_METRICS = {
    "hit",
    "precision",
    "recall",
    "ap",
    "mrr",
}


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


def _normalise_technique(value: Any, *, parent: bool) -> Optional[str]:
    if not isinstance(value, str):
        return None
    technique = value.strip().upper()
    if not technique:
        return None
    if parent:
        technique = technique.split(".", 1)[0]
    if not TECHNIQUE_RE.fullmatch(technique):
        return None
    return technique


def _deduplicate_keep_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_int_list(text: str) -> Tuple[int, ...]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--ks must contain positive integers.")
    return tuple(values)


def _parse_beta_list(text: str) -> Tuple[float, ...]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Beta must be in [0, 1], got {value}.")
        values.append(round(value, 10))
    if not values:
        raise ValueError("No beta values were supplied.")
    return tuple(sorted(set(values)))


def _parse_primary_metric(text: str) -> Tuple[str, Optional[int]]:
    metric = text.strip().lower()
    if metric == "mrr":
        return "mrr", None
    match = re.fullmatch(r"(hit|precision|recall|ap)@(\d+)", metric)
    if not match:
        raise ValueError(
            "--primary_metric must be mrr or one of hit@K, precision@K, "
            "recall@K, ap@K."
        )
    family = match.group(1)
    k = int(match.group(2))
    if family not in SUPPORTED_METRICS or k <= 0:
        raise ValueError(f"Unsupported primary metric: {text}")
    return family, k


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixed split and labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelLoadReport:
    rows: int
    unique_ids: int
    duplicate_rows: int
    string_labels_fixed: int
    invalid_labels_dropped: int
    empty_label_ids: int


def _extract_ids_from_json(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        for key in ("input_ids", "ids", "cve_ids", "development_ids", "dev_ids"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [str(item).strip() for item in candidate if str(item).strip()]
        if "input_id" in value:
            item = str(value["input_id"]).strip()
            return [item] if item else []
    return []


def load_split_ids(path: str | Path) -> Tuple[List[str], Dict[str, Any]]:
    """Load a fixed split from txt, JSON, JSONL, or CSV.

    The returned list is deduplicated while preserving the file order.  Empty
    files and duplicate IDs are rejected/reported rather than silently used.
    """

    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    suffix = split_path.suffix.lower()
    raw_ids: List[str] = []

    if suffix == ".json":
        raw_ids.extend(_extract_ids_from_json(json.loads(split_path.read_text(encoding="utf-8-sig"))))
    elif suffix == ".jsonl":
        with split_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                extracted = _extract_ids_from_json(value)
                if not extracted:
                    raise ValueError(
                        f"No input_id-like field at {split_path}:{line_number}."
                    )
                raw_ids.extend(extracted)
    elif suffix == ".csv":
        with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV split file has no header: {split_path}")
            key = next(
                (
                    name
                    for name in ("input_id", "cve_id", "id")
                    if name in reader.fieldnames
                ),
                None,
            )
            if key is None:
                raise ValueError(
                    f"CSV split file must contain input_id, cve_id, or id: {split_path}"
                )
            for row in reader:
                item = str(row.get(key, "")).strip()
                if item:
                    raw_ids.append(item)
    else:
        with split_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                item = line.strip()
                if item:
                    raw_ids.append(item)

    unique_ids = _deduplicate_keep_order(raw_ids)
    if not unique_ids:
        raise ValueError(f"Development split is empty: {split_path}")

    report = {
        "path": str(split_path.resolve()),
        "sha256": _sha256_file(split_path),
        "raw_id_count": len(raw_ids),
        "unique_id_count": len(unique_ids),
        "duplicate_id_count": len(raw_ids) - len(unique_ids),
    }
    return unique_ids, report


def _coerce_labels(value: Any) -> Tuple[List[Any], bool]:
    if value is None:
        return [], False
    if isinstance(value, str):
        return [value], True
    if isinstance(value, (list, tuple, set)):
        return list(value), False
    return [value], False


def load_labels(
    path: str | Path,
    *,
    parent: bool,
) -> Tuple[Dict[str, set[str]], LabelLoadReport]:
    labels_path = Path(path)
    mapping: Dict[str, set[str]] = {}
    rows = 0
    duplicate_rows = 0
    string_labels_fixed = 0
    invalid_labels_dropped = 0

    with labels_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            row = json.loads(line)
            input_id = row.get("input_id") or row.get("cve_id") or row.get("id")
            if not isinstance(input_id, str) or not input_id.strip():
                raise ValueError(f"Missing input_id at {labels_path}:{line_number}.")
            input_id = input_id.strip()

            raw_labels = row.get("labels")
            if raw_labels is None:
                raw_labels = row.get("gold")
            values, was_string = _coerce_labels(raw_labels)
            if was_string:
                string_labels_fixed += 1

            normalised: set[str] = set()
            for value in values:
                technique = _normalise_technique(value, parent=parent)
                if technique is None:
                    invalid_labels_dropped += 1
                else:
                    normalised.add(technique)

            if input_id in mapping:
                duplicate_rows += 1
                mapping[input_id].update(normalised)
            else:
                mapping[input_id] = set(normalised)

    empty_label_ids = sum(1 for labels in mapping.values() if not labels)
    return mapping, LabelLoadReport(
        rows=rows,
        unique_ids=len(mapping),
        duplicate_rows=duplicate_rows,
        string_labels_fixed=string_labels_fixed,
        invalid_labels_dropped=invalid_labels_dropped,
        empty_label_ids=empty_label_ids,
    )


# ---------------------------------------------------------------------------
# Reranking input and fairness validation
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    name: str
    path: Path
    rows: Dict[str, List[Dict[str, Any]]]
    candidate_signatures: Dict[str, str]
    retrieval_signatures: Dict[str, str]
    metadata: Dict[str, Any]


def _retrieval_ordered(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.get("retrieval_rank", 10**9)),
            str(candidate.get("technique_id", "")),
        ),
    )


def _candidate_signature(candidates: Sequence[Mapping[str, Any]]) -> str:
    ids = [
        str(candidate.get("technique_id", ""))
        for candidate in _retrieval_ordered(candidates)
    ]
    return _sha256_bytes(_canonical_json(ids).encode("utf-8"))


def _retrieval_signature(candidates: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "technique_id": candidate.get("technique_id"),
            "score_fused": candidate.get("score_fused"),
            "retrieval_rank": candidate.get("retrieval_rank"),
        }
        for candidate in _retrieval_ordered(candidates)
    ]
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def load_reranked_run(path: str | Path, *, run_index: int) -> RunData:
    run_path = Path(path)
    rows: Dict[str, List[Dict[str, Any]]] = {}
    candidate_signatures: Dict[str, str] = {}
    retrieval_signatures: Dict[str, str] = {}
    line_count = 0
    duplicate_ids: List[str] = []

    with run_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            row = json.loads(line)
            input_id = row.get("input_id")
            candidates = row.get("candidates")
            if not isinstance(input_id, str) or not input_id.strip():
                raise ValueError(f"Missing input_id at {run_path}:{line_number}.")
            input_id = input_id.strip()
            if input_id in rows:
                duplicate_ids.append(input_id)
                continue
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Empty candidate list for {input_id} in {run_path}.")

            validated: List[Dict[str, Any]] = []
            seen_raw_ids: set[str] = set()
            seen_retrieval_ranks: set[int] = set()
            for candidate_index, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, Mapping):
                    raise ValueError(
                        f"Candidate {candidate_index} for {input_id} is not an object."
                    )
                raw_tid = candidate.get("technique_id")
                technique = _normalise_technique(raw_tid, parent=False)
                if technique is None:
                    raise ValueError(
                        f"Invalid technique_id {raw_tid!r} for {input_id} in {run_path}."
                    )
                if technique in seen_raw_ids:
                    raise ValueError(
                        f"Duplicate raw candidate {technique} for {input_id} in {run_path}."
                    )
                seen_raw_ids.add(technique)

                score_fused = candidate.get("score_fused")
                llm_score = candidate.get("llm_score")
                if score_fused is None or llm_score is None:
                    raise ValueError(
                        f"Candidate {technique} for {input_id} must contain both "
                        "score_fused and llm_score for offline beta selection."
                    )
                try:
                    retrieval_score = float(score_fused)
                    model_score = float(llm_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Non-numeric score for {input_id}/{technique} in {run_path}."
                    ) from exc
                if not math.isfinite(retrieval_score) or not math.isfinite(model_score):
                    raise ValueError(
                        f"Non-finite score for {input_id}/{technique} in {run_path}."
                    )
                if not 0.0 <= model_score <= 1.0:
                    raise ValueError(
                        f"llm_score outside [0,1] for {input_id}/{technique}: {model_score}"
                    )

                raw_retrieval_rank = (
                    candidate.get("retrieval_rank")
                    or candidate.get("rank")
                    or candidate_index
                )
                try:
                    retrieval_rank = int(raw_retrieval_rank)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid retrieval_rank for {input_id}/{technique}: "
                        f"{raw_retrieval_rank!r}"
                    ) from exc
                if retrieval_rank <= 0 or retrieval_rank in seen_retrieval_ranks:
                    raise ValueError(
                        f"Non-positive or duplicate retrieval_rank for "
                        f"{input_id}/{technique}: {retrieval_rank}"
                    )
                seen_retrieval_ranks.add(retrieval_rank)

                validated.append(
                    {
                        "technique_id": technique,
                        "score_fused": retrieval_score,
                        "llm_score": model_score,
                        "retrieval_rank": retrieval_rank,
                    }
                )

            rows[input_id] = validated
            candidate_signatures[input_id] = _candidate_signature(validated)
            retrieval_signatures[input_id] = _retrieval_signature(validated)

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate input_id rows in {run_path}: {preview}")
    if not rows:
        raise ValueError(f"No reranking rows found in {run_path}.")

    return RunData(
        name=f"run_{run_index + 1}",
        path=run_path,
        rows=rows,
        candidate_signatures=candidate_signatures,
        retrieval_signatures=retrieval_signatures,
        metadata={
            "path": str(run_path.resolve()),
            "sha256": _sha256_file(run_path),
            "line_count": line_count,
            "unique_id_count": len(rows),
        },
    )


def validate_dev_inputs(
    runs: Sequence[RunData],
    dev_ids: Sequence[str],
    labels: Mapping[str, set[str]],
    *,
    minimum_candidates: int,
) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one reranking file is required.")

    missing_labels = [input_id for input_id in dev_ids if not labels.get(input_id)]
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(
            f"{len(missing_labels)} development IDs have no valid gold label: {preview}"
        )

    reference = runs[0]
    missing_by_run: Dict[str, List[str]] = {}
    fairness_mismatches: List[Dict[str, str]] = []
    short_rankings: List[Tuple[str, str, int]] = []

    for run in runs:
        missing = [input_id for input_id in dev_ids if input_id not in run.rows]
        if missing:
            missing_by_run[run.name] = missing
        for input_id in dev_ids:
            candidates = run.rows.get(input_id)
            if candidates is not None and len(candidates) < minimum_candidates:
                short_rankings.append((run.name, input_id, len(candidates)))

    if missing_by_run:
        summary = "; ".join(
            f"{name}: {len(ids)} missing" for name, ids in missing_by_run.items()
        )
        raise ValueError(f"Development split is incomplete across reranking files: {summary}")

    if short_rankings:
        name, input_id, count = short_rankings[0]
        raise ValueError(
            f"{name}/{input_id} has {count} candidates; at least "
            f"{minimum_candidates} are required."
        )

    for run in runs[1:]:
        for input_id in dev_ids:
            if run.candidate_signatures[input_id] != reference.candidate_signatures[input_id]:
                fairness_mismatches.append(
                    {
                        "run": run.name,
                        "input_id": input_id,
                        "kind": "candidate_order",
                    }
                )
                continue
            if run.retrieval_signatures[input_id] != reference.retrieval_signatures[input_id]:
                fairness_mismatches.append(
                    {
                        "run": run.name,
                        "input_id": input_id,
                        "kind": "retrieval_score",
                    }
                )

    if fairness_mismatches:
        first = fairness_mismatches[0]
        raise ValueError(
            "Repeated runs do not use identical candidate sets/retrieval scores. "
            f"First mismatch: {first}."
        )

    return {
        "development_id_count": len(dev_ids),
        "run_count": len(runs),
        "minimum_candidates": minimum_candidates,
        "missing_label_count": 0,
        "missing_rerank_count": 0,
        "candidate_fairness_mismatch_count": 0,
    }


# ---------------------------------------------------------------------------
# Ranking and metrics
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    beta: float,
    parent: bool,
) -> List[str]:
    scored: List[Tuple[float, float, float, int, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        technique = _normalise_technique(candidate.get("technique_id"), parent=parent)
        if technique is None:
            continue
        retrieval_score = float(candidate["score_fused"])
        llm_score = float(candidate["llm_score"])
        final_score = beta * retrieval_score + (1.0 - beta) * llm_score
        retrieval_rank = int(candidate.get("retrieval_rank") or index)
        scored.append(
            (final_score, llm_score, retrieval_score, retrieval_rank, technique)
        )

    scored.sort(
        key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4])
    )
    return _deduplicate_keep_order(item[4] for item in scored)


def _sample_metrics(
    ranking: Sequence[str],
    gold: set[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    reciprocal_rank = 0.0
    for rank, technique in enumerate(ranking, start=1):
        if technique in gold:
            reciprocal_rank = 1.0 / rank
            break
    result["mrr"] = reciprocal_rank

    for k in ks:
        top_k = list(ranking[:k])
        relevant_positions = [
            index
            for index, technique in enumerate(top_k, start=1)
            if technique in gold
        ]
        intersection_count = len(relevant_positions)
        result[f"hit@{k}"] = 1.0 if intersection_count else 0.0
        result[f"precision@{k}"] = intersection_count / k
        result[f"recall@{k}"] = intersection_count / len(gold)

        if relevant_positions:
            running_relevant = 0
            precision_sum = 0.0
            for index, technique in enumerate(top_k, start=1):
                if technique in gold:
                    running_relevant += 1
                    precision_sum += running_relevant / index
            result[f"ap@{k}"] = precision_sum / min(len(gold), k)
        else:
            result[f"ap@{k}"] = 0.0

    return result


def _metric_key(family: str, k: Optional[int]) -> str:
    return family if family == "mrr" else f"{family}@{k}"


@dataclass
class BetaEvaluation:
    beta: float
    run_metrics: List[Dict[str, float]]
    per_cve_metrics: Dict[str, Dict[str, float]]
    summary: Dict[str, float]


def evaluate_beta(
    runs: Sequence[RunData],
    dev_ids: Sequence[str],
    labels: Mapping[str, set[str]],
    *,
    beta: float,
    parent: bool,
    ks: Sequence[int],
    primary_key: str,
    bootstrap_repetitions: int,
    confidence: float,
    bootstrap_seed: int,
) -> BetaEvaluation:
    per_run_per_cve: List[Dict[str, Dict[str, float]]] = []
    run_metrics: List[Dict[str, float]] = []

    for run in runs:
        sample_map: Dict[str, Dict[str, float]] = {}
        for input_id in dev_ids:
            ranking = rank_candidates(run.rows[input_id], beta=beta, parent=parent)
            sample_map[input_id] = _sample_metrics(ranking, labels[input_id], ks)
        per_run_per_cve.append(sample_map)

        keys = sorted(next(iter(sample_map.values())).keys())
        run_metrics.append(
            {
                key: _mean([sample_map[input_id][key] for input_id in dev_ids])
                for key in keys
            }
        )

    # Average each CVE's metric across repeated stochastic runs.  The bootstrap
    # resamples CVEs, not individual candidates or runs.
    per_cve_metrics: Dict[str, Dict[str, float]] = {}
    keys = sorted(next(iter(per_run_per_cve[0].values())).keys())
    for input_id in dev_ids:
        per_cve_metrics[input_id] = {
            key: _mean([sample_map[input_id][key] for sample_map in per_run_per_cve])
            for key in keys
        }

    primary_values = [per_cve_metrics[input_id][primary_key] for input_id in dev_ids]
    ci_low, ci_high = _bootstrap_ci(
        primary_values,
        repetitions=bootstrap_repetitions,
        confidence=confidence,
        seed=bootstrap_seed + int(round(beta * 10000)),
    )

    summary: Dict[str, float] = {
        key: _mean([per_cve_metrics[input_id][key] for input_id in dev_ids])
        for key in keys
    }
    summary.update(
        {
            "primary_ci_low": ci_low,
            "primary_ci_high": ci_high,
            "primary_run_sd": _sample_sd([metrics[primary_key] for metrics in run_metrics]),
        }
    )
    return BetaEvaluation(
        beta=beta,
        run_metrics=run_metrics,
        per_cve_metrics=per_cve_metrics,
        summary=summary,
    )


def _selection_key(evaluation: BetaEvaluation, primary_key: str) -> Tuple[float, ...]:
    """Deterministic, predeclared tie-breaking.

    1. maximise the selected primary development metric;
    2. maximise MRR;
    3. maximise Hit@1;
    4. minimise run-to-run standard deviation;
    5. prefer the larger beta (more retrieval weight / less LLM dependence).

    The final preference is only reached under exact metric ties and is recorded
    in ``selected_beta.json``.
    """

    summary = evaluation.summary
    return (
        summary[primary_key],
        summary.get("mrr", float("-inf")),
        summary.get("hit@1", float("-inf")),
        -summary["primary_run_sd"],
        evaluation.beta,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_sweep_csv(
    path: Path,
    evaluations: Sequence[BetaEvaluation],
    *,
    primary_key: str,
    ks: Sequence[int],
) -> None:
    metric_columns = ["mrr"]
    for k in ks:
        metric_columns.extend(
            [f"hit@{k}", f"precision@{k}", f"recall@{k}", f"ap@{k}"]
        )

    fieldnames = [
        "beta",
        "primary_metric",
        "primary_value",
        "primary_ci_low",
        "primary_ci_high",
        "primary_run_sd",
        *metric_columns,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation in evaluations:
            row: Dict[str, Any] = {
                "beta": f"{evaluation.beta:.10g}",
                "primary_metric": primary_key,
                "primary_value": f"{evaluation.summary[primary_key]:.10f}",
                "primary_ci_low": f"{evaluation.summary['primary_ci_low']:.10f}",
                "primary_ci_high": f"{evaluation.summary['primary_ci_high']:.10f}",
                "primary_run_sd": f"{evaluation.summary['primary_run_sd']:.10f}",
            }
            for column in metric_columns:
                row[column] = f"{evaluation.summary[column]:.10f}"
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select beta on a fixed development split only."
    )
    parser.add_argument(
        "--reranked",
        required=True,
        nargs="+",
        help="One or more repeated reranking JSONL files for the same condition.",
    )
    parser.add_argument("--labels", required=True, help="Gold-label JSONL file.")
    parser.add_argument(
        "--dev_ids",
        required=True,
        help="Fixed development ID file (txt, JSON, JSONL, or CSV).",
    )
    parser.add_argument(
        "--test_ids",
        default=None,
        help="Optional held-out test ID file used only to verify zero overlap; never evaluated.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parent", action="store_true")
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument(
        "--betas",
        default=",".join(f"{value:.2f}" for value in DEFAULT_BETAS),
        help="Comma-separated beta grid in [0,1].",
    )
    parser.add_argument(
        "--primary_metric",
        default="hit@1",
        help="Development-only selection metric: mrr or hit/precision/recall/ap@K.",
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap_repetitions must be non-negative.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1.")

    ks = _parse_int_list(args.ks)
    betas = _parse_beta_list(args.betas)
    family, metric_k = _parse_primary_metric(args.primary_metric)
    primary_key = _metric_key(family, metric_k)
    if metric_k is not None and metric_k not in ks:
        ks = tuple(sorted({*ks, metric_k}))

    output_dir = Path(args.output_dir)
    output_files = {
        "sweep": output_dir / "beta_sweep.csv",
        "selection": output_dir / "selected_beta.json",
        "manifest": output_dir / "beta_selection_manifest.json",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "Output files already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_ids, dev_report = load_split_ids(args.dev_ids)
    test_report: Optional[Dict[str, Any]] = None
    if args.test_ids:
        test_ids, test_report = load_split_ids(args.test_ids)
        overlap = sorted(set(dev_ids).intersection(test_ids))
        if overlap:
            preview = ", ".join(overlap[:10])
            raise ValueError(
                f"Development/test split overlap detected ({len(overlap)} IDs): {preview}"
            )
        test_report["overlap_with_development"] = 0
        # Deliberately discard the test IDs after the overlap check.  They are
        # never joined with labels or reranking rows below.
        del test_ids

    labels, label_report = load_labels(args.labels, parent=args.parent)
    runs = [
        load_reranked_run(path, run_index=index)
        for index, path in enumerate(args.reranked)
    ]
    validation_report = validate_dev_inputs(
        runs,
        dev_ids,
        labels,
        minimum_candidates=max(ks),
    )

    evaluations = [
        evaluate_beta(
            runs,
            dev_ids,
            labels,
            beta=beta,
            parent=args.parent,
            ks=ks,
            primary_key=primary_key,
            bootstrap_repetitions=args.bootstrap_repetitions,
            confidence=args.confidence,
            bootstrap_seed=args.seed,
        )
        for beta in betas
    ]
    selected = max(evaluations, key=lambda item: _selection_key(item, primary_key))

    write_sweep_csv(
        output_files["sweep"],
        evaluations,
        primary_key=primary_key,
        ks=ks,
    )

    tie_break_trace = [
        {
            "beta": evaluation.beta,
            "selection_key": list(_selection_key(evaluation, primary_key)),
            "primary_value": evaluation.summary[primary_key],
            "mrr": evaluation.summary.get("mrr"),
            "hit@1": evaluation.summary.get("hit@1"),
            "primary_run_sd": evaluation.summary["primary_run_sd"],
        }
        for evaluation in sorted(
            evaluations,
            key=lambda item: _selection_key(item, primary_key),
            reverse=True,
        )
    ]

    selection_payload = {
        "script_version": SCRIPT_VERSION,
        "selection_scope": "fixed_development_split_only",
        "test_metrics_computed": False,
        "selected_beta": selected.beta,
        "primary_metric": primary_key,
        "primary_value": selected.summary[primary_key],
        "primary_confidence_interval": {
            "confidence": args.confidence,
            "lower": selected.summary["primary_ci_low"],
            "upper": selected.summary["primary_ci_high"],
        },
        "run_count": len(runs),
        "development_size": len(dev_ids),
        "secondary_metrics": {
            "mrr": selected.summary.get("mrr"),
            "hit@1": selected.summary.get("hit@1"),
            "primary_run_sd": selected.summary["primary_run_sd"],
        },
        "tie_break_policy": [
            "maximise the declared primary development metric",
            "maximise development MRR",
            "maximise development Hit@1",
            "minimise run-to-run standard deviation",
            "prefer larger beta under an otherwise exact tie",
        ],
        "ranked_beta_trace": tie_break_trace,
    }
    _write_json(output_files["selection"], selection_payload)

    manifest_payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": _utc_now(),
        "selection_scope": "development_only",
        "test_metrics_computed": False,
        "configuration": {
            "parent_normalisation": bool(args.parent),
            "ks": list(ks),
            "beta_grid": list(betas),
            "primary_metric": primary_key,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "confidence": args.confidence,
            "seed": args.seed,
            "score_formula": "final_score = beta * score_fused + (1-beta) * llm_score",
            "candidate_sort_tie_break": [
                "descending final_score",
                "descending llm_score",
                "descending score_fused",
                "ascending original retrieval rank",
                "ascending technique_id",
            ],
        },
        "development_split": dev_report,
        "held_out_test_split_overlap_check": test_report,
        "labels": {
            "path": str(Path(args.labels).resolve()),
            "sha256": _sha256_file(args.labels),
            **label_report.__dict__,
        },
        "reranking_runs": [run.metadata for run in runs],
        "validation": validation_report,
        "selected_beta": selected.beta,
        "outputs": {},
    }

    # Hash the two substantive outputs before writing the manifest itself.
    manifest_payload["outputs"] = {
        "beta_sweep.csv": _sha256_file(output_files["sweep"]),
        "selected_beta.json": _sha256_file(output_files["selection"]),
    }
    _write_json(output_files["manifest"], manifest_payload)

    print(
        f"Selected beta={selected.beta:.4f} by {primary_key}="
        f"{selected.summary[primary_key]:.6f} on {len(dev_ids)} development CVEs "
        f"across {len(runs)} run(s)."
    )
    print("No held-out test metric was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
