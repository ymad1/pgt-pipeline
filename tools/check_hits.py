#!/usr/bin/env python3
"""Quick, deterministic ranking-metric audit for CVE-to-ATT&CK experiments.

This tool is intentionally a *thin wrapper* around ``pgt.compare_rankers``.
It does not maintain a second implementation of Hit@K, Precision@K,
Recall@K, AP@K, MRR, label-micro recall, or technique-macro recall.  The same
loaders, candidate-order reconstruction, parent normalization, ID alignment,
candidate-fairness checks, and per-CVE metric functions used by the formal
publication evaluator are imported directly.

Use this script for a fast point-estimate check while experiments are running.
Use ``python -m pgt.compare_rankers`` for confidence intervals, repeated-run
uncertainty, paired significance tests, and long-tail report files.

The old version of this script contained hard-coded paths, read ``pred`` and
``gold`` from legacy result files, silently skipped duplicate base CVEs, and
used a separate metric implementation.  Those behaviors are deliberately
removed so that quick checks cannot drift away from the paper's definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ``python tools/check_hits.py`` places tools/ rather than the repository root
# on sys.path.  Add the root only when the normal package import is unavailable.
try:
    from pgt.compare_rankers import (
        RankingRun,
        RunSpec,
        _aggregate_method_samples,
        _evaluate_run,
        _mean,
        _parse_run_specs,
        _per_technique_rows,
        _resolve_evaluation_ids,
        _retrieval_coverage,
        _sample_sd,
        _verify_candidate_fairness,
        load_labels,
        load_ranking_run,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation location
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from pgt.compare_rankers import (  # type: ignore[no-redef]
        RankingRun,
        RunSpec,
        _aggregate_method_samples,
        _evaluate_run,
        _mean,
        _parse_run_specs,
        _per_technique_rows,
        _resolve_evaluation_ids,
        _retrieval_coverage,
        _sample_sd,
        _verify_candidate_fairness,
        load_labels,
        load_ranking_run,
    )


TOOL_VERSION = "check-hits-v2.0.0"
DEFAULT_KS = (1, 3, 5, 10, 20)
FIRST_HIT_BUCKETS = ("1", "2-3", "4-5", "6-10", "11-20", ">20", "miss")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ks(raw: str) -> List[int]:
    values: set[int] = set()
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


def _read_fixed_ids(path: Path) -> List[str]:
    identifiers: List[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            input_id = line.strip()
            if not input_id:
                continue
            if input_id in seen:
                raise ValueError(
                    f"Duplicate input_id in {path} at line {line_number}: {input_id}"
                )
            seen.add(input_id)
            identifiers.append(input_id)
    if not identifiers:
        raise ValueError(f"ID file is empty: {path}")
    return identifiers


def _apply_fixed_ids(
    inferred_ids: Sequence[str],
    fixed_ids: Sequence[str],
    labels: Mapping[str, set[str]],
    runs: Sequence[RankingRun],
    *,
    policy: str,
) -> Tuple[List[str], Dict[str, Any]]:
    inferred_set = set(inferred_ids)
    fixed_set = set(fixed_ids)

    missing_labels = [input_id for input_id in fixed_ids if input_id not in labels]
    missing_by_run = {
        f"{run.spec.method}#{run.spec.run_index}": [
            input_id for input_id in fixed_ids if input_id not in run.rankings
        ]
        for run in runs
    }
    missing_by_run = {key: value for key, value in missing_by_run.items() if value}
    if missing_labels or missing_by_run:
        raise ValueError(
            "The fixed ID file references unavailable records. "
            f"Missing labels (first 10): {missing_labels[:10]}; "
            f"missing by run: "
            f"{ {key: value[:10] for key, value in missing_by_run.items()} }"
        )

    extra_in_inferred = sorted(inferred_set - fixed_set)
    absent_from_inferred = sorted(fixed_set - inferred_set)
    if absent_from_inferred:
        raise ValueError(
            f"{len(absent_from_inferred)} fixed IDs are outside the aligned evaluation set; "
            f"examples: {absent_from_inferred[:10]}"
        )
    if policy == "exact" and extra_in_inferred:
        raise ValueError(
            "--id_file_policy exact requires the fixed ID file to equal the aligned "
            f"run IDs. Found {len(extra_in_inferred)} extra aligned IDs; examples: "
            f"{extra_in_inferred[:10]}"
        )
    if policy not in {"exact", "subset"}:
        raise ValueError(f"Unknown fixed ID policy: {policy}")

    return list(fixed_ids), {
        "path_policy": policy,
        "fixed_id_count": len(fixed_ids),
        "excluded_aligned_ids": len(extra_in_inferred),
    }


def _derived_retrieval_run(reference: RankingRun, method_name: str) -> RankingRun:
    return RankingRun(
        spec=RunSpec(method=method_name, path=reference.spec.path, run_index=1),
        rankings=dict(reference.retrieval_rankings),
        retrieval_rankings=dict(reference.retrieval_rankings),
        candidate_signatures=dict(reference.candidate_signatures),
        candidate_ids=dict(reference.candidate_ids),
        metadata_modes={"retrieval_only_derived"},
    )


def _first_hit_bucket(ranking: Sequence[str], gold: set[str]) -> str:
    for rank, technique_id in enumerate(ranking, start=1):
        if technique_id in gold:
            if rank == 1:
                return "1"
            if rank <= 3:
                return "2-3"
            if rank <= 5:
                return "4-5"
            if rank <= 10:
                return "6-10"
            if rank <= 20:
                return "11-20"
            return ">20"
    return "miss"


def _first_hit_distribution(
    runs: Sequence[RankingRun],
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
) -> Dict[str, float]:
    # Repeated runs are weighted equally, matching the repeated-run averaging
    # convention in compare_rankers.
    per_run: List[Dict[str, float]] = []
    for run in runs:
        counts = Counter(
            _first_hit_bucket(run.rankings[input_id], labels[input_id])
            for input_id in evaluation_ids
        )
        per_run.append(
            {bucket: counts[bucket] / len(evaluation_ids) for bucket in FIRST_HIT_BUCKETS}
        )
    return {
        bucket: _mean([distribution[bucket] for distribution in per_run])
        for bucket in FIRST_HIT_BUCKETS
    }


def _summarize_method(
    method: str,
    runs: Sequence[RankingRun],
    labels: Mapping[str, set[str]],
    evaluation_ids: Sequence[str],
    ks: Sequence[int],
    *,
    tail_max: int,
    head_min: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run_samples = [_evaluate_run(run, labels, evaluation_ids, ks) for run in runs]
    aggregate_samples = _aggregate_method_samples(run_samples, evaluation_ids)
    metric_names = sorted(next(iter(aggregate_samples.values())).keys())

    cve_macro: Dict[str, float] = {}
    run_sd: Dict[str, float] = {}
    for metric in metric_names:
        if metric.startswith("retrieved_labels@"):
            continue
        cve_macro[metric] = _mean(
            [aggregate_samples[input_id][metric] for input_id in evaluation_ids]
        )
        run_estimates = [
            _mean([sample[input_id][metric] for input_id in evaluation_ids])
            for sample in run_samples
        ]
        run_sd[metric] = _sample_sd(run_estimates)

    total_gold = sum(len(labels[input_id]) for input_id in evaluation_ids)
    label_micro: Dict[str, float] = {}
    for k in ks:
        retrieved = sum(
            aggregate_samples[input_id][f"retrieved_labels@{k}"]
            for input_id in evaluation_ids
        )
        label_micro[f"micro_precision@{k}"] = retrieved / (len(evaluation_ids) * k)
        label_micro[f"micro_recall@{k}"] = retrieved / total_gold

    per_technique = _per_technique_rows(
        method,
        runs,
        labels,
        evaluation_ids,
        ks,
        tail_max=tail_max,
        head_min=head_min,
    )
    technique_macro = {
        f"technique_macro_recall@{k}": _mean(
            [float(row[f"recall@{k}"]) for row in per_technique]
        )
        for k in ks
    }

    summary = {
        "runs": len(runs),
        "run_files": [str(run.spec.path) for run in runs],
        "metadata_modes": sorted({mode for run in runs for mode in run.metadata_modes}),
        "cve_macro": cve_macro,
        "label_micro": label_micro,
        "technique_macro": technique_macro,
        "run_sd": run_sd,
        "first_hit_rank_distribution": _first_hit_distribution(
            runs, labels, evaluation_ids
        ),
    }

    rows: List[Dict[str, Any]] = []
    for averaging, metrics in (
        ("cve_macro", cve_macro),
        ("label_micro", label_micro),
        ("technique_macro", technique_macro),
    ):
        for metric, estimate in sorted(metrics.items()):
            rows.append(
                {
                    "method": method,
                    "runs": len(runs),
                    "averaging": averaging,
                    "metric": metric,
                    "estimate": estimate,
                    "run_sd": run_sd.get(metric, ""),
                    "n_cves": len(evaluation_ids),
                }
            )
    return summary, rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "runs",
        "averaging",
        "metric",
        "estimate",
        "delta_vs_reference",
        "run_sd",
        "n_cves",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _print_summary(
    methods: Mapping[str, Mapping[str, Any]],
    ks: Sequence[int],
    reference_method: str,
) -> None:
    print(f"Evaluation reference: {reference_method}")
    print("Metrics use the exact definitions implemented in pgt.compare_rankers.\n")
    for method in sorted(methods):
        summary = methods[method]
        print(f"[{method}] runs={summary['runs']}")
        cve_macro = summary["cve_macro"]
        for k in ks:
            print(
                f"  K={k:>2}: "
                f"Hit={cve_macro[f'hit@{k}']:.4f}  "
                f"P={cve_macro[f'precision@{k}']:.4f}  "
                f"R={cve_macro[f'recall@{k}']:.4f}  "
                f"AP={cve_macro[f'ap@{k}']:.4f}"
            )
        print(f"  MRR={cve_macro['mrr']:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast point-estimate audit using the exact metric implementation "
            "from pgt.compare_rankers."
        )
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each condition; identical NAME values are repeated runs.",
    )
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument(
        "--id_file_policy", choices=("exact", "subset"), default="exact"
    )
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
    )
    parser.add_argument("--retrieval_method_name", default="retrieval")
    parser.add_argument("--reference_method", default="retrieval")
    parser.add_argument(
        "--fail_on_invalid_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tail_max_support", type=int, default=5)
    parser.add_argument("--head_min_support", type=int, default=21)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    if args.tail_max_support < 1:
        parser.error("--tail_max_support must be positive")
    if args.head_min_support <= args.tail_max_support:
        parser.error("--head_min_support must exceed --tail_max_support")

    run_specs = _parse_run_specs(args.run)
    labels, label_statistics = load_labels(
        args.labels,
        parent=args.parent,
        fail_on_invalid=args.fail_on_invalid_labels,
    )
    required_length = max(ks)
    supplied_runs = [
        load_ranking_run(
            spec,
            parent=args.parent,
            allow_short=args.allow_short_rankings,
            required_length=required_length,
        )
        for spec in run_specs
    ]

    evaluation_ids, id_report = _resolve_evaluation_ids(
        labels, supplied_runs, policy=args.id_policy
    )
    fixed_id_report: Optional[Dict[str, Any]] = None
    if args.ids is not None:
        fixed_ids = _read_fixed_ids(args.ids)
        evaluation_ids, fixed_id_report = _apply_fixed_ids(
            evaluation_ids,
            fixed_ids,
            labels,
            supplied_runs,
            policy=args.id_file_policy,
        )

    fairness_report = _verify_candidate_fairness(
        supplied_runs,
        evaluation_ids,
        strict=args.strict_candidate_fairness,
    )

    evaluation_runs = list(supplied_runs)
    if args.include_retrieval_baseline:
        existing = {run.spec.method for run in supplied_runs}
        if args.retrieval_method_name in existing:
            raise ValueError(
                f"Retrieval method name {args.retrieval_method_name!r} conflicts "
                "with a supplied method name."
            )
        evaluation_runs.append(
            _derived_retrieval_run(supplied_runs[0], args.retrieval_method_name)
        )

    runs_by_method: Dict[str, List[RankingRun]] = defaultdict(list)
    for run in evaluation_runs:
        runs_by_method[run.spec.method].append(run)

    method_summaries: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    for method in sorted(runs_by_method):
        summary, rows = _summarize_method(
            method,
            runs_by_method[method],
            labels,
            evaluation_ids,
            ks,
            tail_max=args.tail_max_support,
            head_min=args.head_min_support,
        )
        method_summaries[method] = summary
        metric_rows.extend(rows)

    if args.reference_method not in method_summaries:
        raise ValueError(
            f"Reference method {args.reference_method!r} was not evaluated. "
            f"Available methods: {sorted(method_summaries)}"
        )

    reference_metrics: Dict[Tuple[str, str], float] = {}
    reference_summary = method_summaries[args.reference_method]
    for averaging in ("cve_macro", "label_micro", "technique_macro"):
        for metric, estimate in reference_summary[averaging].items():
            reference_metrics[(averaging, metric)] = float(estimate)

    for row in metric_rows:
        reference = reference_metrics.get((str(row["averaging"]), str(row["metric"])))
        row["delta_vs_reference"] = (
            float(row["estimate"]) - reference if reference is not None else ""
        )

    candidate_coverage = _retrieval_coverage(
        supplied_runs[0], labels, evaluation_ids, ks
    )
    report: Dict[str, Any] = {
        "tool_version": TOOL_VERSION,
        "metric_implementation": "pgt.compare_rankers",
        "intended_use": (
            "Quick deterministic point-estimate audit only; use pgt.compare_rankers "
            "for confidence intervals and statistical tests."
        ),
        "configuration": {
            "ks": ks,
            "parent_normalization": bool(args.parent),
            "id_policy": args.id_policy,
            "fixed_id_policy": args.id_file_policy if args.ids else None,
            "strict_candidate_fairness": bool(args.strict_candidate_fairness),
            "allow_short_rankings": bool(args.allow_short_rankings),
            "include_retrieval_baseline": bool(args.include_retrieval_baseline),
            "reference_method": args.reference_method,
            "tail_max_support": args.tail_max_support,
            "head_min_support": args.head_min_support,
        },
        "inputs": {
            "labels": {
                "path": str(Path(args.labels)),
                "sha256": _sha256_file(Path(args.labels)),
            },
            "ids": (
                {"path": str(args.ids), "sha256": _sha256_file(args.ids)}
                if args.ids
                else None
            ),
            "runs": [
                {
                    "method": spec.method,
                    "run_index": spec.run_index,
                    "path": str(spec.path),
                    "sha256": _sha256_file(spec.path),
                }
                for spec in run_specs
            ],
        },
        "label_statistics": label_statistics,
        "id_alignment": id_report,
        "fixed_id_alignment": fixed_id_report,
        "candidate_fairness": fairness_report,
        "evaluation_cves": len(evaluation_ids),
        "candidate_coverage": candidate_coverage,
        "methods": method_summaries,
        "metric_rows": metric_rows,
    }

    if args.output_json is not None:
        _write_json(args.output_json, report)
    if args.output_csv is not None:
        _write_csv(args.output_csv, metric_rows)

    print(f"Aligned CVEs: {len(evaluation_ids)}")
    print(f"Candidate fairness mismatches: {fairness_report['mismatch_count_capped']}")
    _print_summary(method_summaries, ks, args.reference_method)
    if args.output_json is not None:
        print(f"JSON report: {args.output_json}")
    if args.output_csv is not None:
        print(f"CSV report: {args.output_csv}")


if __name__ == "__main__":
    main()
