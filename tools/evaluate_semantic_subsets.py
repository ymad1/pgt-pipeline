#!/usr/bin/env python3
"""Evaluate rerankers on AI-audited semantic label subsets.

The tool first calls ``build_audit_sensitivity_labels.py`` to construct
relation-level label subsets without changing the inherited gold labels. It
then reuses already-generated reranking outputs and calls
``pgt.compare_rankers`` for every subset with enough CVEs.

No OpenAI API call is made by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCRIPT_VERSION = "semantic-subset-evaluation-v1.0.0"
VALID_SUBSETS = {
    "audited_all",
    "directly_supported",
    "supported_or_plausible",
    "insufficient_or_unsupported",
}
VALID_MODES = {"generic", "evidence", "structure", "full"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_csv_values(text: str) -> List[str]:
    values: List[str] = []
    seen = set()
    for raw in str(text).split(","):
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _run(command: Sequence[str], *, cwd: Path) -> Dict[str, Any]:
    print("  " + subprocess.list2cmdline([str(value) for value in command]))
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd),
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            + subprocess.list2cmdline([str(value) for value in command])
        )
    return {
        "command": [str(value) for value in command],
        "returncode": completed.returncode,
    }


def _file_metadata(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for path in sorted(set(paths), key=lambda item: str(item)):
        if path.is_file():
            metadata[str(path)] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--audit_csv", required=True)
    parser.add_argument("--rerank_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--modes", default="generic,evidence,structure,full")
    parser.add_argument("--seeds", default="20260805,20260806,20260807")
    parser.add_argument(
        "--subsets",
        default="audited_all,directly_supported,supported_or_plausible",
    )
    parser.add_argument("--min_cves", type=int, default=20)
    parser.add_argument("--ks", default="1,3,5,10,20")
    parser.add_argument("--bootstrap_repetitions", type=int, default=5000)
    parser.add_argument("--permutation_repetitions", type=int, default=20000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--tail_max_support", type=int, default=5)
    parser.add_argument("--head_min_support", type=int, default=21)
    parser.add_argument("--reference_method", default="full")
    parser.add_argument("--parent", action="store_true")
    parser.add_argument(
        "--include_retrieval_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retrieval_method_name", default="retrieval")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.min_cves < 1:
        parser.error("--min_cves must be at least 1")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap_repetitions must be non-negative")
    if args.permutation_repetitions < 1:
        parser.error("--permutation_repetitions must be positive")

    project_root = Path(__file__).resolve().parents[1]
    labels_path = Path(args.labels).resolve()
    audit_path = Path(args.audit_csv).resolve()
    rerank_root = Path(args.rerank_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    for required in (labels_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not rerank_root.is_dir():
        raise FileNotFoundError(rerank_root)

    modes = _parse_csv_values(args.modes)
    unknown_modes = sorted(set(modes) - VALID_MODES)
    if unknown_modes:
        parser.error(f"Unknown modes: {unknown_modes}")
    if not modes:
        parser.error("--modes cannot be empty")

    try:
        seeds = [int(value) for value in _parse_csv_values(args.seeds)]
    except ValueError as exc:
        parser.error(f"Invalid --seeds value: {exc}")
    if not seeds:
        parser.error("--seeds cannot be empty")
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds must contain unique integers")

    subsets = _parse_csv_values(args.subsets)
    unknown_subsets = sorted(set(subsets) - VALID_SUBSETS)
    if unknown_subsets:
        parser.error(f"Unknown subsets: {unknown_subsets}")
    if not subsets:
        parser.error("--subsets cannot be empty")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"

    builder = project_root / "tools" / "build_audit_sensitivity_labels.py"
    if not builder.is_file():
        raise FileNotFoundError(builder)

    commands: List[Dict[str, Any]] = []
    build_command = [
        sys.executable,
        str(builder),
        "--labels",
        str(labels_path),
        "--audit_csv",
        str(audit_path),
        "--output_dir",
        str(labels_dir),
        "--expected_split",
        "test",
        "--overwrite",
    ]
    print("[semantic] build audited label subsets")
    commands.append(_run(build_command, cwd=project_root))

    subset_summary_path = labels_dir / "semantic_sensitivity_summary.json"
    subset_summary = _read_json(subset_summary_path)

    results: Dict[str, Any] = {}
    input_run_files: List[Path] = []
    output_files: List[Path] = [
        subset_summary_path,
        labels_dir / "semantic_sensitivity_manifest.json",
        labels_dir / "audited_pairs_in_split.csv",
    ]

    for subset in subsets:
        subset_meta = dict(subset_summary.get("subsets", {}).get(subset, {}))
        n_cves = int(subset_meta.get("cves", 0))
        result: Dict[str, Any] = {
            "cves": n_cves,
            "label_assignments": int(subset_meta.get("label_assignments", 0)),
            "status": "skipped_insufficient_cves" if n_cves < args.min_cves else "pending",
            "min_cves": args.min_cves,
        }
        results[subset] = result
        if n_cves < args.min_cves:
            print(
                f"[semantic] skip {subset}: {n_cves} CVEs < minimum {args.min_cves}"
            )
            continue

        label_file = labels_dir / str(subset_meta["labels_file"])
        eval_dir = output_dir / subset
        command: List[str] = [
            sys.executable,
            "-m",
            "pgt.compare_rankers",
            "--labels",
            str(label_file),
        ]
        run_files: List[Path] = []
        for mode in modes:
            for run_seed in seeds:
                run_file = rerank_root / mode / f"seed_{run_seed}.jsonl"
                if not run_file.is_file():
                    raise FileNotFoundError(run_file)
                run_files.append(run_file)
                command += ["--run", f"{mode}={run_file}"]
        command += [
            "--output_dir",
            str(eval_dir),
            "--id_policy",
            "intersection",
            "--ks",
            str(args.ks),
            "--bootstrap_repetitions",
            str(args.bootstrap_repetitions),
            "--permutation_repetitions",
            str(args.permutation_repetitions),
            "--confidence",
            str(args.confidence),
            "--seed",
            str(args.seed),
            "--tail_max_support",
            str(args.tail_max_support),
            "--head_min_support",
            str(args.head_min_support),
            "--reference_method",
            str(args.reference_method),
        ]
        if args.parent:
            command.append("--parent")
        if args.include_retrieval_baseline:
            command += [
                "--include_retrieval_baseline",
                "--retrieval_method_name",
                str(args.retrieval_method_name),
            ]

        print(f"[semantic] evaluate {subset} ({n_cves} CVEs)")
        commands.append(_run(command, cwd=project_root))
        expected = [
            eval_dir / "metric_summary.csv",
            eval_dir / "pairwise_tests.csv",
            eval_dir / "evaluation_report.json",
            eval_dir / "evaluation_manifest.json",
        ]
        missing = [path for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing semantic evaluation outputs: {missing}")
        output_files.extend(expected)
        input_run_files.extend(run_files)
        result.update(
            {
                "status": "evaluated",
                "output_dir": str(eval_dir),
                "labels_file": str(label_file),
            }
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "completed_utc": _utc_now(),
        "labels": str(labels_path),
        "audit_csv": str(audit_path),
        "rerank_root": str(rerank_root),
        "audited_cves_overlapping_split": int(
            subset_summary.get("audited_cves_overlapping_split", 0)
        ),
        "audited_pairs_overlapping_split": int(
            subset_summary.get("audited_pairs_overlapping_split", 0)
        ),
        "results": results,
        "no_api_calls": True,
    }
    summary_path = output_dir / "semantic_evaluation_summary.json"
    _atomic_write_json(summary_path, summary)
    output_files.append(summary_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "completed_utc": _utc_now(),
        "configuration": {
            "modes": modes,
            "seeds": seeds,
            "subsets": subsets,
            "min_cves": args.min_cves,
            "ks": str(args.ks),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "permutation_repetitions": args.permutation_repetitions,
            "confidence": args.confidence,
            "seed": args.seed,
            "tail_max_support": args.tail_max_support,
            "head_min_support": args.head_min_support,
            "reference_method": args.reference_method,
            "parent": bool(args.parent),
            "include_retrieval_baseline": bool(args.include_retrieval_baseline),
            "retrieval_method_name": args.retrieval_method_name,
        },
        "inputs": _file_metadata([labels_path, audit_path, *input_run_files]),
        "outputs": _file_metadata(output_files),
        "commands": commands,
        "no_api_calls": True,
    }
    manifest_path = output_dir / "semantic_evaluation_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "audited_cves": summary["audited_cves_overlapping_split"],
                "evaluated_subsets": [
                    name for name, value in results.items() if value["status"] == "evaluated"
                ],
                "skipped_subsets": [
                    name
                    for name, value in results.items()
                    if value["status"] == "skipped_insufficient_cves"
                ],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
