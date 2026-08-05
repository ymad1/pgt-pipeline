#!/usr/bin/env python3
"""Fail-fast orchestration for the revised CVE-to-ATT&CK pipeline.

The driver keeps data construction, split creation, evidence segmentation,
LLM extraction, local graph/MES construction, retrieval, controlled reranking,
development-only beta selection, held-out test evaluation, and audit outputs in
one traceable run directory.

The script intentionally delegates scientific logic to the canonical modules
instead of reimplementing it. Its responsibilities are orchestration,
input/output isolation, split-safe subsetting, command/provenance recording, and
stopping immediately when any stage fails.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pgt.io import read_jsonl as validated_read_jsonl
from pgt.io import write_jsonl as validated_write_jsonl

SCRIPT_VERSION = "reviewer2-pipeline-v1.2.0"
DEFAULT_CONFIG_NAME = "pipeline_config.json"
STAGE_ORDER = (
    "data",
    "attack",
    "segment",
    "extract",
    "select_retrieval",
    "retrieve",
    "rerank_dev",
    "select_beta",
    "rerank_test",
    "evaluate",
    "audit",
)
RERANK_MODES = ("generic", "evidence", "structure", "full")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    payload = _stable_json_bytes(value)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_ids(path: Path) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        if value in seen:
            raise ValueError(f"Duplicate ID {value!r} in {path} at line {line_number}")
        seen.add(value)
        ids.append(value)
    if not ids:
        raise ValueError(f"ID file is empty: {path}")
    return ids


def _write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in ids), encoding="utf-8", newline="\n")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _path_metadata(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in sorted(set(paths), key=lambda item: str(item)):
        if path.is_file():
            result[str(path)] = {"type": "file", "size": path.stat().st_size, "sha256": _sha256_file(path)}
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            digest = hashlib.sha256()
            for item in files:
                relative = item.relative_to(path).as_posix().encode("utf-8")
                digest.update(relative + b"\0" + _sha256_file(item).encode("ascii") + b"\n")
            result[str(path)] = {
                "type": "directory",
                "file_count": len(files),
                "tree_sha256": digest.hexdigest(),
            }
        else:
            result[str(path)] = {"type": "missing"}
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_CONFIG: Dict[str, Any] = {
    "workspace": "runs/reviewer2_v2",
    "smoke_workspace": "runs/reviewer2_v2_smoke",
    "data": {
        "x_train": "data/cve2attck_src_20260107/X_train.csv",
        "y_train": "data/cve2attck_src_20260107/y_train.csv",
        "x_test": "data/cve2attck_src_20260107/X_test.csv",
        "y_test": "data/cve2attck_src_20260107/y_test.csv",
        "enterprise_attack": "data/attack/enterprise-attack.json",
    },
    "split": {
        "dev_fraction": 0.20,
        "dev_size": None,
        "seed": 20260805,
        "min_support_both": 2,
        "max_swap_passes": 20,
    },
    "segmentation": {
        "aggressive_split": True,
        "max_chars": 420,
        "min_chars": 24,
        "max_evidence": 12,
    },
    "extraction": {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.0,
        "seed": 20260805,
        "max_tokens": 1400,
        "attempts": 2,
        "retry_base_seconds": 0.8,
        "allow_fallback": False,
        "allow_validation_errors": False,
        "mes_max_path_nodes": 4,
        "mes_exact_cover_limit": 20,
        "mes_include_precondition": False,
    },
    "retrieval_selection": {
        "alphas": "0.00,0.20,0.40,0.50,0.60,0.80,1.00",
        "topns": "5,10,15,20,30,50",
        "score_normalization": "none",
        "normalize_to_parent": True,
        "primary_metric": "candidate_coverage",
        "primary_tolerance": 0.005,
        "bootstrap_repetitions": 2000,
        "confidence": 0.95,
        "seed": 20260805,
    },
    "retrieval": {
        "fallback_topn_for_plan": 20,
        "fallback_alpha_for_plan": 0.60,
    },
    "reranking": {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.2,
        "seeds": [20260805, 20260806, 20260807],
        "max_tokens": 1800,
        "attempts": 3,
        "retry_backoff": 1.0,
        "initial_beta_for_dev": 0.0,
        "dev_modes": ["full"],
        "test_modes": ["generic", "evidence", "structure", "full"],
        "allow_empty_mes": True,
        "require_complete_mes": False,
    },
    "beta_selection": {
        "betas": "0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00",
        "primary_metric": "hit@1",
        "bootstrap_repetitions": 2000,
        "confidence": 0.95,
        "seed": 20260805,
    },
    "evaluation": {
        "parent": True,
        "ks": "1,3,5,10,20",
        "bootstrap_repetitions": 5000,
        "permutation_repetitions": 20000,
        "confidence": 0.95,
        "seed": 20260805,
        "tail_max_support": 5,
        "head_min_support": 21,
        "reference_method": "full",
        "include_retrieval_baseline": True,
        "retrieval_method_name": "retrieval",
    },
}


@dataclass(frozen=True)
class Paths:
    root: Path
    workspace: Path
    source_train: Path
    source_test: Path
    fixed_split: Path
    attack_cache: Path
    inputs: Path
    pipeline: Path
    retrieval_selection: Path
    rerank_dev: Path
    beta: Path
    rerank_test: Path
    evaluation: Path
    audit: Path
    state: Path

    @classmethod
    def from_config(cls, root: Path, config: Mapping[str, Any]) -> "Paths":
        workspace = _resolve(root, str(config["workspace"]))
        return cls(
            root=root,
            workspace=workspace,
            source_train=workspace / "datasets" / "source_train",
            source_test=workspace / "datasets" / "source_test",
            fixed_split=workspace / "fixed_split",
            attack_cache=workspace / "attack_cache",
            inputs=workspace / "inputs",
            pipeline=workspace / "pipeline",
            retrieval_selection=workspace / "retrieval_selection",
            rerank_dev=workspace / "reranking" / "development",
            beta=workspace / "beta_selection",
            rerank_test=workspace / "reranking" / "test",
            evaluation=workspace / "evaluation" / "test",
            audit=workspace / "audit",
            state=workspace / "pipeline_state.json",
        )


# ---------------------------------------------------------------------------
# Runner and state
# ---------------------------------------------------------------------------


class PipelineRunner:
    def __init__(
        self,
        *,
        root: Path,
        config: Mapping[str, Any],
        paths: Paths,
        plan_only: bool,
        overwrite: bool,
        resume: bool,
        smoke_records_per_split: int,
    ) -> None:
        self.root = root
        self.config = config
        self.paths = paths
        self.plan_only = plan_only
        self.overwrite = overwrite
        self.resume = resume
        self.smoke_records_per_split = smoke_records_per_split
        self.python = sys.executable
        self.state: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "project_root": str(root),
            "workspace": str(paths.workspace),
            "configuration_sha256": _sha256_bytes(_stable_json_bytes(config)),
            "smoke_records_per_split": smoke_records_per_split,
            "run_scope": "smoke" if smoke_records_per_split > 0 else "formal",
            "eligible_for_formal_reporting": smoke_records_per_split == 0,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "stages": {},
        }
        if paths.state.exists() and resume:
            previous = _read_json(paths.state)
            if previous.get("configuration_sha256") != self.state["configuration_sha256"]:
                raise RuntimeError(
                    "Cannot resume: configuration hash differs from the existing pipeline state."
                )
            if int(previous.get("smoke_records_per_split", -1)) != smoke_records_per_split:
                raise RuntimeError(
                    "Cannot resume: smoke/full scope differs from the existing pipeline state."
                )
            self.state = previous

    def save_state(self) -> None:
        if self.plan_only:
            return
        self.state["updated_utc"] = _utc_now()
        _write_json_atomic(self.paths.state, self.state)

    def run_command(
        self,
        *,
        stage: str,
        name: str,
        command: Sequence[str],
        expected_outputs: Sequence[Path],
        env_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        stage_state = self.state.setdefault("stages", {}).setdefault(stage, {"commands": []})
        command_record = {
            "name": name,
            "command": [str(value) for value in command],
            "command_text": _command_text(command),
            "expected_outputs": [str(path) for path in expected_outputs],
        }

        outputs_exist = bool(expected_outputs) and all(path.exists() for path in expected_outputs)
        previous_commands = stage_state.get("commands", [])
        previous_record = next(
            (
                item
                for item in reversed(previous_commands)
                if item.get("name") == name and item.get("status") == "succeeded"
            ),
            None,
        )
        if self.resume and outputs_exist and previous_record is not None:
            current_outputs = _path_metadata(expected_outputs)
            if previous_record.get("outputs") != current_outputs:
                raise RuntimeError(
                    f"Cannot resume {stage}/{name}: existing output hashes differ "
                    "from the recorded successful run."
                )
            print(f"[resume] {stage}/{name}")
            return

        print(f"[{stage}] {name}\n  {_command_text(command)}")
        if self.plan_only:
            return

        if outputs_exist and not self.overwrite and not self.resume:
            raise FileExistsError(
                f"Outputs already exist for {stage}/{name}. Use --resume or --overwrite."
            )

        env = os.environ.copy()
        if env_overrides:
            env.update({str(key): str(value) for key, value in env_overrides.items()})

        command_record["started_utc"] = _utc_now()
        command_record["status"] = "running"
        stage_state.setdefault("commands", []).append(command_record)
        self.save_state()

        try:
            completed = subprocess.run(
                list(command),
                cwd=self.root,
                env=env,
                check=False,
                text=True,
            )
            command_record["return_code"] = completed.returncode
            if completed.returncode != 0:
                command_record["status"] = "failed"
                raise RuntimeError(
                    f"Stage {stage}/{name} failed with exit code {completed.returncode}."
                )
            missing = [path for path in expected_outputs if not path.exists()]
            if missing:
                command_record["status"] = "failed"
                raise RuntimeError(
                    f"Stage {stage}/{name} returned success but outputs are missing: {missing}"
                )
            command_record["status"] = "succeeded"
            command_record["outputs"] = _path_metadata(expected_outputs)
        except Exception as exc:
            command_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.save_state()
            raise
        finally:
            command_record["finished_utc"] = _utc_now()
            self.save_state()

    def record_internal(
        self,
        *,
        stage: str,
        name: str,
        expected_outputs: Sequence[Path],
        action: Any,
    ) -> None:
        stage_state = self.state.setdefault("stages", {}).setdefault(stage, {"commands": []})
        outputs_exist = bool(expected_outputs) and all(path.exists() for path in expected_outputs)
        previous_record = next(
            (
                item
                for item in reversed(stage_state.get("commands", []))
                if item.get("name") == name and item.get("status") == "succeeded"
            ),
            None,
        )
        if self.resume and outputs_exist and previous_record is not None:
            current_outputs = _path_metadata(expected_outputs)
            if previous_record.get("outputs") != current_outputs:
                raise RuntimeError(
                    f"Cannot resume {stage}/{name}: existing output hashes differ "
                    "from the recorded successful run."
                )
            print(f"[resume] {stage}/{name}")
            return

        print(f"[{stage}] {name}")
        if self.plan_only:
            return
        if outputs_exist and not self.overwrite and not self.resume:
            raise FileExistsError(
                f"Outputs already exist for {stage}/{name}. Use --resume or --overwrite."
            )

        record = {
            "name": name,
            "command_text": "internal deterministic operation",
            "expected_outputs": [str(path) for path in expected_outputs],
            "started_utc": _utc_now(),
            "status": "running",
        }
        stage_state.setdefault("commands", []).append(record)
        self.save_state()
        try:
            action()
            missing = [path for path in expected_outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"Internal stage did not produce expected outputs: {missing}")
            record["status"] = "succeeded"
            record["outputs"] = _path_metadata(expected_outputs)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["finished_utc"] = _utc_now()
            self.save_state()


# ---------------------------------------------------------------------------
# Split-safe file operations
# ---------------------------------------------------------------------------


def _subset_jsonl(
    source: Path,
    ids: Sequence[str],
    destination: Path,
    *,
    record_kind: Optional[str],
) -> None:
    if record_kind is None:
        rows = _read_jsonl(source)
    else:
        rows = list(validated_read_jsonl(source, record_kind=record_kind))

    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        input_id = row.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError(f"Missing input_id in {source}")
        if input_id in by_id:
            raise ValueError(f"Duplicate input_id {input_id!r} in {source}")
        by_id[input_id] = row
    missing = [input_id for input_id in ids if input_id not in by_id]
    if missing:
        raise ValueError(f"{source} is missing {len(missing)} requested IDs; examples: {missing[:5]}")

    selected_rows = (by_id[input_id] for input_id in ids)
    if record_kind is None:
        _write_jsonl_atomic(destination, selected_rows)
    else:
        validated_write_jsonl(
            destination,
            selected_rows,
            record_kind=record_kind,
            enforce_unique_input_ids=True,
        )


def _prepare_active_ids(paths: Paths, smoke_records_per_split: int) -> Dict[str, Path]:
    official_dev = paths.fixed_split / "development" / "ids.txt"
    official_test = paths.fixed_split / "test" / "ids.txt"
    dev_ids = _read_ids(official_dev)
    test_ids = _read_ids(official_test)

    if smoke_records_per_split > 0:
        dev_ids = dev_ids[:smoke_records_per_split]
        test_ids = test_ids[:smoke_records_per_split]
        if not dev_ids or not test_ids:
            raise ValueError("Smoke mode requires at least one development and one test record.")

    combined_ids = dev_ids + test_ids
    if len(set(combined_ids)) != len(combined_ids):
        raise RuntimeError("Development and test IDs overlap.")

    output = {
        "development": paths.inputs / "development_ids.txt",
        "test": paths.inputs / "test_ids.txt",
        "combined": paths.inputs / "combined_ids.txt",
    }
    _write_ids(output["development"], dev_ids)
    _write_ids(output["test"], test_ids)
    _write_ids(output["combined"], combined_ids)
    return output


def _prepare_split_inputs(paths: Paths) -> None:
    ids = {
        "development": _read_ids(paths.inputs / "development_ids.txt"),
        "test": _read_ids(paths.inputs / "test_ids.txt"),
    }
    sources = {
        "sentences": (paths.pipeline / "sentences.jsonl", "sentences"),
        "mes": (paths.pipeline / "mes.jsonl", "mes"),
        "candidates": (paths.pipeline / "candidates.jsonl", "candidates"),
        "labels": (paths.fixed_split / "combined" / "labels.jsonl", None),
    }
    for split_name, split_ids in ids.items():
        split_dir = paths.inputs / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, (source, record_kind) in sources.items():
            _subset_jsonl(
                source,
                split_ids,
                split_dir / f"{name}.jsonl",
                record_kind=record_kind,
            )
        _write_ids(split_dir / "ids.txt", split_ids)


def _selected_beta(path: Path) -> float:
    payload = _read_json(path)
    beta = float(payload["selected_beta"])
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"Selected beta outside [0,1]: {beta}")
    return beta


def _selected_retrieval(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    alpha = float(payload["selected_alpha"])
    topn = int(payload["selected_topn"])
    normalization = str(payload["score_normalization"])
    parent_normalization = bool(payload["parent_normalization"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Selected alpha outside [0,1]: {alpha}")
    if topn <= 0:
        raise ValueError(f"Selected Top-N must be positive: {topn}")
    if normalization not in {"none", "minmax"}:
        raise ValueError(f"Unsupported selected score normalization: {normalization}")
    return {
        "alpha": alpha,
        "topn": topn,
        "score_normalization": normalization,
        "normalize_to_parent": parent_normalization,
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _openai_preflight(runner: PipelineRunner, stage: str) -> None:
    runner.run_command(
        stage=stage,
        name="openai_client_preflight",
        command=[
            runner.python,
            "-c",
            (
                "from pgt.openai_client import get_openai_client; "
                "get_openai_client(); print('OpenAI client preflight: OK')"
            ),
        ],
        expected_outputs=[],
    )


def stage_data(runner: PipelineRunner) -> None:
    cfg = runner.config
    data = cfg["data"]
    split = cfg["split"]
    p = runner.paths
    attack = _resolve(runner.root, data["enterprise_attack"])

    for name, x_key, y_key, destination in (
        ("source_train", "x_train", "y_train", p.source_train),
        ("source_test", "x_test", "y_test", p.source_test),
    ):
        command = [
            runner.python,
            str(runner.root / "tools" / "make_cve2attck_jsonl.py"),
            str(_resolve(runner.root, data[x_key])),
            str(_resolve(runner.root, data[y_key])),
            str(destination),
            "--enterprise_attack",
            str(attack),
            "--split_name",
            name,
        ]
        if runner.overwrite or runner.resume:
            command.append("--overwrite")
        runner.run_command(
            stage="data",
            name=f"build_{name}",
            command=command,
            expected_outputs=[destination / "records.jsonl", destination / "labels.jsonl", destination / "dataset_manifest.json"],
        )

    command = [
        runner.python,
        str(runner.root / "tools" / "make_fixed_splits.py"),
        "--source",
        f"source_train={p.source_train}",
        "--source",
        f"source_test={p.source_test}",
        "--output_dir",
        str(p.fixed_split),
        "--dev_fraction",
        str(split["dev_fraction"]),
        "--seed",
        str(split["seed"]),
        "--min_support_both",
        str(split["min_support_both"]),
        "--max_swap_passes",
        str(split["max_swap_passes"]),
    ]
    if split.get("dev_size") is not None:
        command += ["--dev_size", str(split["dev_size"])]
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="data",
        name="make_fixed_splits",
        command=command,
        expected_outputs=[
            p.fixed_split / "combined" / "records.jsonl",
            p.fixed_split / "development" / "ids.txt",
            p.fixed_split / "test" / "ids.txt",
            p.fixed_split / "split_manifest.json",
        ],
    )

    for split_name in ("combined", "development", "test"):
        dataset_dir = p.fixed_split / split_name
        command = [
            runner.python,
            str(runner.root / "tools" / "check_labels_alignment.py"),
            str(dataset_dir),
            "--enterprise_attack",
            str(attack),
            "--report",
            str(p.audit / f"dataset_{split_name}.json"),
            "--warnings_as_errors",
        ]
        runner.run_command(
            stage="data",
            name=f"audit_{split_name}",
            command=command,
            expected_outputs=[p.audit / f"dataset_{split_name}.json"],
        )

    runner.record_internal(
        stage="data",
        name="prepare_active_ids",
        expected_outputs=[
            p.inputs / "development_ids.txt",
            p.inputs / "test_ids.txt",
            p.inputs / "combined_ids.txt",
        ],
        action=lambda: _prepare_active_ids(p, runner.smoke_records_per_split),
    )


def stage_attack(runner: PipelineRunner) -> None:
    data = runner.config["data"]
    p = runner.paths
    command = [
        runner.python,
        "-m",
        "pgt.export_attack_stix",
        "--stix_bundle",
        str(_resolve(runner.root, data["enterprise_attack"])),
        "--attack_kg",
        str(p.attack_cache / "attack_kg.json"),
        "--tech_index",
        str(p.attack_cache / "technique_text_index.jsonl"),
        "--manifest",
        str(p.attack_cache / "attack_manifest.json"),
    ]
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="attack",
        name="export_active_attack",
        command=command,
        expected_outputs=[
            p.attack_cache / "attack_kg.json",
            p.attack_cache / "technique_text_index.jsonl",
            p.attack_cache / "attack_manifest.json",
        ],
    )


def stage_segment(runner: PipelineRunner) -> None:
    cfg = runner.config["segmentation"]
    p = runner.paths
    command = [
        runner.python,
        "-m",
        "pgt.split_sentences",
        "--input",
        str(p.fixed_split / "combined" / "records.jsonl"),
        "--output",
        str(p.pipeline / "sentences.jsonl"),
        "--ids",
        str(p.inputs / "combined_ids.txt"),
        "--max_chars",
        str(cfg["max_chars"]),
        "--min_chars",
        str(cfg["min_chars"]),
        "--max_evidence",
        str(cfg["max_evidence"]),
    ]
    if cfg.get("aggressive_split", False):
        command.append("--aggressive_split")
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="segment",
        name="segment_evidence",
        command=command,
        expected_outputs=[
            p.pipeline / "sentences.jsonl",
            p.pipeline / "sentences.jsonl.manifest.json",
        ],
    )


def stage_extract(runner: PipelineRunner) -> None:
    cfg = runner.config["extraction"]
    p = runner.paths
    if not cfg.get("allow_fallback", False):
        _openai_preflight(runner, "extract")
    command = [
        runner.python,
        "-m",
        "pgt.extract",
        "--sentences",
        str(p.pipeline / "sentences.jsonl"),
        "--output",
        str(p.pipeline / "extraction.jsonl"),
        "--graph_dir",
        str(p.pipeline / "local_graphs"),
        "--mes_output",
        str(p.pipeline / "mes.jsonl"),
        "--ids_file",
        str(p.inputs / "combined_ids.txt"),
        "--mes_max_path_nodes",
        str(cfg["mes_max_path_nodes"]),
        "--mes_exact_cover_limit",
        str(cfg["mes_exact_cover_limit"]),
    ]
    if cfg.get("allow_fallback", False):
        command.append("--allow_fallback")
    if cfg.get("allow_validation_errors", False):
        command.append("--allow_validation_errors")
    if cfg.get("mes_include_precondition", False):
        command.append("--mes_include_precondition")
    if runner.overwrite:
        command.append("--overwrite")
    elif runner.resume:
        command.append("--resume")

    env = {
        "OPENAI_EXTRACTION_MODEL": str(cfg["model"]),
        "OPENAI_EXTRACTION_TEMPERATURE": str(cfg["temperature"]),
        "OPENAI_EXTRACTION_SEED": str(cfg["seed"]),
        "OPENAI_EXTRACTION_MAX_TOKENS": str(cfg["max_tokens"]),
        "OPENAI_EXTRACTION_MAX_ATTEMPTS": str(cfg["attempts"]),
        "OPENAI_EXTRACTION_RETRY_BASE_SECONDS": str(cfg["retry_base_seconds"]),
    }
    runner.run_command(
        stage="extract",
        name="extract_graph_mes",
        command=command,
        expected_outputs=[
            p.pipeline / "extraction.jsonl",
            p.pipeline / "local_graphs" / "_summary.json",
            p.pipeline / "mes.jsonl",
            p.pipeline / "extraction.jsonl.manifest.json",
        ],
        env_overrides=env,
    )


def stage_select_retrieval(runner: PipelineRunner) -> None:
    cfg = runner.config["retrieval_selection"]
    p = runner.paths
    command = [
        runner.python,
        "-m",
        "pgt.sweep_retrieval",
        "--sentences",
        str(p.pipeline / "sentences.jsonl"),
        "--mes",
        str(p.pipeline / "mes.jsonl"),
        "--tech_index",
        str(p.attack_cache / "technique_text_index.jsonl"),
        "--labels",
        str(p.fixed_split / "development" / "labels.jsonl"),
        "--dev_ids",
        str(p.inputs / "development_ids.txt"),
        "--test_ids",
        str(p.inputs / "test_ids.txt"),
        "--output_dir",
        str(p.retrieval_selection),
        "--alphas",
        str(cfg["alphas"]),
        "--topns",
        str(cfg["topns"]),
        "--score_normalization",
        str(cfg["score_normalization"]),
        "--primary_metric",
        str(cfg["primary_metric"]),
        "--primary_tolerance",
        str(cfg["primary_tolerance"]),
        "--bootstrap_repetitions",
        str(cfg["bootstrap_repetitions"]),
        "--confidence",
        str(cfg["confidence"]),
        "--seed",
        str(cfg["seed"]),
    ]
    if cfg.get("normalize_to_parent", False):
        command.append("--normalize_to_parent")
    if runner.smoke_records_per_split > 0:
        command.append("--allow_zero_primary")
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="select_retrieval",
        name="select_retrieval_on_development",
        command=command,
        expected_outputs=[
            p.retrieval_selection / "selected_retrieval.json",
            p.retrieval_selection / "retrieval_sweep.csv",
            p.retrieval_selection / "retrieval_selection_manifest.json",
        ],
    )


def stage_retrieve(runner: PipelineRunner) -> None:
    p = runner.paths
    selected_path = p.retrieval_selection / "selected_retrieval.json"
    if runner.plan_only and not selected_path.exists():
        fallback = runner.config["retrieval"]
        selected = {
            "alpha": float(fallback["fallback_alpha_for_plan"]),
            "topn": int(fallback["fallback_topn_for_plan"]),
            "score_normalization": str(runner.config["retrieval_selection"]["score_normalization"]),
            "normalize_to_parent": bool(runner.config["retrieval_selection"].get("normalize_to_parent", False)),
        }
        print(
            "[plan] selected retrieval configuration unavailable; "
            f"plan-preview alpha={selected['alpha']}, Top-N={selected['topn']} shown"
        )
    else:
        selected = _selected_retrieval(selected_path)
    command = [
        runner.python,
        "-m",
        "pgt.retrieve_candidates",
        "--sentences",
        str(p.pipeline / "sentences.jsonl"),
        "--mes",
        str(p.pipeline / "mes.jsonl"),
        "--tech_index",
        str(p.attack_cache / "technique_text_index.jsonl"),
        "--output",
        str(p.pipeline / "candidates.jsonl"),
        "--topn",
        str(selected["topn"]),
        "--alpha",
        str(selected["alpha"]),
        "--score_normalization",
        str(selected["score_normalization"]),
    ]
    if selected.get("normalize_to_parent", False):
        command.append("--normalize_to_parent")
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="retrieve",
        name="retrieve_candidates",
        command=command,
        expected_outputs=[
            p.pipeline / "candidates.jsonl",
            p.pipeline / "candidates.jsonl.summary.json",
        ],
    )

    runner.record_internal(
        stage="retrieve",
        name="prepare_split_inputs",
        expected_outputs=[
            p.inputs / "development" / "sentences.jsonl",
            p.inputs / "development" / "mes.jsonl",
            p.inputs / "development" / "candidates.jsonl",
            p.inputs / "development" / "labels.jsonl",
            p.inputs / "test" / "sentences.jsonl",
            p.inputs / "test" / "mes.jsonl",
            p.inputs / "test" / "candidates.jsonl",
            p.inputs / "test" / "labels.jsonl",
        ],
        action=lambda: _prepare_split_inputs(p),
    )


def _rerank_command(
    runner: PipelineRunner,
    *,
    split_name: str,
    mode: str,
    seed: int,
    beta: float,
    output: Path,
) -> List[str]:
    cfg = runner.config["reranking"]
    p = runner.paths
    split_dir = p.inputs / split_name
    command = [
        runner.python,
        "-m",
        "pgt.rerank",
        "--sentences",
        str(split_dir / "sentences.jsonl"),
        "--candidates",
        str(split_dir / "candidates.jsonl"),
        "--tech_index",
        str(p.attack_cache / "technique_text_index.jsonl"),
        "--output",
        str(output),
        "--mode",
        mode,
        "--topk",
        str(
            _selected_retrieval(p.retrieval_selection / "selected_retrieval.json")["topn"]
            if (p.retrieval_selection / "selected_retrieval.json").exists()
            else runner.config["retrieval"]["fallback_topn_for_plan"]
        ),
        "--beta",
        str(beta),
        "--model",
        str(cfg["model"]),
        "--temperature",
        str(cfg["temperature"]),
        "--seed",
        str(seed),
        "--max_tokens",
        str(cfg["max_tokens"]),
        "--attempts",
        str(cfg["attempts"]),
        "--retry_backoff",
        str(cfg["retry_backoff"]),
        "--manifest",
        str(output) + ".manifest.json",
    ]
    if mode in {"structure", "full"}:
        command += ["--mes", str(split_dir / "mes.jsonl")]
    if cfg.get("allow_empty_mes", False):
        command.append("--allow_empty_mes")
    if cfg.get("require_complete_mes", False):
        command.append("--require_complete_mes")
    if runner.overwrite:
        command.append("--overwrite")
    elif runner.resume:
        command.append("--resume")
    return command


def stage_rerank_dev(runner: PipelineRunner) -> None:
    _openai_preflight(runner, "rerank_dev")
    cfg = runner.config["reranking"]
    modes = [str(value) for value in cfg["dev_modes"]]
    for mode in modes:
        if mode not in RERANK_MODES:
            raise ValueError(f"Unknown development reranking mode: {mode}")
    for mode in modes:
        for seed in [int(value) for value in cfg["seeds"]]:
            output = runner.paths.rerank_dev / mode / f"seed_{seed}.jsonl"
            runner.run_command(
                stage="rerank_dev",
                name=f"{mode}_seed_{seed}",
                command=_rerank_command(
                    runner,
                    split_name="development",
                    mode=mode,
                    seed=seed,
                    beta=float(cfg["initial_beta_for_dev"]),
                    output=output,
                ),
                expected_outputs=[output, Path(str(output) + ".manifest.json")],
            )


def stage_select_beta(runner: PipelineRunner) -> None:
    cfg = runner.config["beta_selection"]
    p = runner.paths
    seeds = [int(value) for value in runner.config["reranking"]["seeds"]]
    full_runs = [p.rerank_dev / "full" / f"seed_{seed}.jsonl" for seed in seeds]
    if not full_runs:
        raise RuntimeError("At least one full development reranking run is required.")
    command = [runner.python, "-m", "pgt.sweep_beta_offline", "--reranked"]
    command.extend(str(path) for path in full_runs)
    command += [
        "--labels",
        str(p.inputs / "development" / "labels.jsonl"),
        "--dev_ids",
        str(p.inputs / "development" / "ids.txt"),
        "--test_ids",
        str(p.inputs / "test" / "ids.txt"),
        "--output_dir",
        str(p.beta),
        "--betas",
        str(cfg["betas"]),
        "--primary_metric",
        str(cfg["primary_metric"]),
        "--bootstrap_repetitions",
        str(cfg["bootstrap_repetitions"]),
        "--confidence",
        str(cfg["confidence"]),
        "--seed",
        str(cfg["seed"]),
    ]
    if runner.config["evaluation"].get("parent", False):
        command.append("--parent")
    if runner.overwrite or runner.resume:
        command.append("--overwrite")
    runner.run_command(
        stage="select_beta",
        name="select_beta_on_development",
        command=command,
        expected_outputs=[
            p.beta / "selected_beta.json",
            p.beta / "beta_sweep.csv",
            p.beta / "beta_selection_manifest.json",
        ],
    )


def stage_rerank_test(runner: PipelineRunner) -> None:
    _openai_preflight(runner, "rerank_test")
    cfg = runner.config["reranking"]
    beta_path = runner.paths.beta / "selected_beta.json"
    if runner.plan_only and not beta_path.exists():
        beta = float(runner.config["reranking"].get("initial_beta_for_dev", 0.0))
        print(f"[plan] selected beta unavailable; plan-preview beta={beta} shown for test commands")
    else:
        beta = _selected_beta(beta_path)
    modes = [str(value) for value in cfg["test_modes"]]
    for mode in modes:
        if mode not in RERANK_MODES:
            raise ValueError(f"Unknown test reranking mode: {mode}")
    for mode in modes:
        for seed in [int(value) for value in cfg["seeds"]]:
            output = runner.paths.rerank_test / mode / f"seed_{seed}.jsonl"
            runner.run_command(
                stage="rerank_test",
                name=f"{mode}_seed_{seed}",
                command=_rerank_command(
                    runner,
                    split_name="test",
                    mode=mode,
                    seed=seed,
                    beta=beta,
                    output=output,
                ),
                expected_outputs=[output, Path(str(output) + ".manifest.json")],
            )


def stage_evaluate(runner: PipelineRunner) -> None:
    cfg = runner.config["evaluation"]
    rerank_cfg = runner.config["reranking"]
    p = runner.paths
    command = [
        runner.python,
        "-m",
        "pgt.compare_rankers",
        "--labels",
        str(p.inputs / "test" / "labels.jsonl"),
    ]
    for mode in [str(value) for value in rerank_cfg["test_modes"]]:
        for seed in [int(value) for value in rerank_cfg["seeds"]]:
            command += ["--run", f"{mode}={p.rerank_test / mode / f'seed_{seed}.jsonl'}"]
    command += [
        "--output_dir",
        str(p.evaluation),
        "--ks",
        str(cfg["ks"]),
        "--bootstrap_repetitions",
        str(cfg["bootstrap_repetitions"]),
        "--permutation_repetitions",
        str(cfg["permutation_repetitions"]),
        "--confidence",
        str(cfg["confidence"]),
        "--seed",
        str(cfg["seed"]),
        "--tail_max_support",
        str(cfg["tail_max_support"]),
        "--head_min_support",
        str(cfg["head_min_support"]),
        "--reference_method",
        str(cfg["reference_method"]),
    ]
    if cfg.get("parent", False):
        command.append("--parent")
    if cfg.get("include_retrieval_baseline", False):
        command += [
            "--include_retrieval_baseline",
            "--retrieval_method_name",
            str(cfg.get("retrieval_method_name", "retrieval")),
        ]
    runner.run_command(
        stage="evaluate",
        name="held_out_test_evaluation",
        command=command,
        expected_outputs=[
            p.evaluation / "metric_summary.csv",
            p.evaluation / "pairwise_tests.csv",
            p.evaluation / "per_technique_recall.csv",
            p.evaluation / "long_tail_summary.csv",
            p.evaluation / "evaluation_report.json",
            p.evaluation / "evaluation_manifest.json",
        ],
    )


def stage_audit(runner: PipelineRunner) -> None:
    p = runner.paths
    command = [
        runner.python,
        "-m",
        "pgt.analyze_missing_gold",
        "--labels",
        str(p.inputs / "test" / "labels.jsonl"),
        "--candidates",
        str(p.inputs / "test" / "candidates.jsonl"),
        "--tech_index",
        str(p.attack_cache / "technique_text_index.jsonl"),
        "--attack_stix",
        str(_resolve(runner.root, runner.config["data"]["enterprise_attack"])),
        "--id_file",
        str(p.inputs / "test" / "ids.txt"),
        "--output_dir",
        str(p.audit / "test_candidate_coverage"),
    ]
    runner.run_command(
        stage="audit",
        name="test_candidate_coverage",
        command=command,
        expected_outputs=[
            p.audit / "test_candidate_coverage" / "summary.json",
            p.audit / "test_candidate_coverage" / "coverage_by_k.csv",
            p.audit / "test_candidate_coverage" / "audit_manifest.json",
        ],
    )

    def final_manifest() -> None:
        required = [
            p.fixed_split / "split_manifest.json",
            p.attack_cache / "attack_manifest.json",
            p.pipeline / "sentences.jsonl.manifest.json",
            p.pipeline / "extraction.jsonl.manifest.json",
            p.pipeline / "candidates.jsonl.summary.json",
            p.retrieval_selection / "selected_retrieval.json",
            p.retrieval_selection / "retrieval_selection_manifest.json",
            p.beta / "selected_beta.json",
            p.evaluation / "evaluation_manifest.json",
            p.audit / "test_candidate_coverage" / "audit_manifest.json",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot finalize pipeline; missing artifacts: {missing}")
        payload = {
            "script_version": SCRIPT_VERSION,
            "completed_utc": _utc_now(),
            "configuration_sha256": runner.state["configuration_sha256"],
            "smoke_records_per_split": runner.smoke_records_per_split,
            "run_scope": "smoke" if runner.smoke_records_per_split > 0 else "formal",
            "eligible_for_formal_reporting": runner.smoke_records_per_split == 0,
            "selected_retrieval": _selected_retrieval(p.retrieval_selection / "selected_retrieval.json"),
            "selected_beta": _selected_beta(p.beta / "selected_beta.json"),
            "artifacts": _path_metadata(required),
        }
        _write_json_atomic(p.workspace / "final_run_manifest.json", payload)

    runner.record_internal(
        stage="audit",
        name="final_run_manifest",
        expected_outputs=[p.workspace / "final_run_manifest.json"],
        action=final_manifest,
    )


STAGE_FUNCTIONS = {
    "data": stage_data,
    "attack": stage_attack,
    "segment": stage_segment,
    "extract": stage_extract,
    "select_retrieval": stage_select_retrieval,
    "retrieve": stage_retrieve,
    "rerank_dev": stage_rerank_dev,
    "select_beta": stage_select_beta,
    "rerank_test": stage_rerank_test,
    "evaluate": stage_evaluate,
    "audit": stage_audit,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_project(root: Path) -> None:
    required = [
        root / "pgt" / "__init__.py",
        root / "pgt" / "extract.py",
        root / "pgt" / "retrieve_candidates.py",
        root / "pgt" / "sweep_retrieval.py",
        root / "pgt" / "rerank.py",
        root / "pgt" / "compare_rankers.py",
        root / "tools" / "make_cve2attck_jsonl.py",
        root / "tools" / "make_fixed_splits.py",
        root / "tools" / "check_labels_alignment.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Project root is missing required files: {missing}")


def _load_config(root: Path, config_path: Optional[Path]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        supplied = _read_json(config_path)
        if not isinstance(supplied, dict):
            raise ValueError("Pipeline configuration must be a JSON object.")
        _deep_merge(config, supplied)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    workspace = str(config.get("workspace", "")).strip()
    smoke_workspace = str(config.get("smoke_workspace", "")).strip()
    if not workspace or not smoke_workspace:
        raise ValueError("Both workspace and smoke_workspace must be configured.")
    if Path(workspace) == Path(smoke_workspace):
        raise ValueError("workspace and smoke_workspace must be different paths.")

    extraction = config["extraction"]
    if float(extraction["retry_base_seconds"]) < 0:
        raise ValueError("extraction.retry_base_seconds must be non-negative.")

    reranking = config["reranking"]
    seeds = [int(value) for value in reranking["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("reranking.seeds must be a non-empty list of unique integers.")
    dev_modes = [str(value) for value in reranking["dev_modes"]]
    test_modes = [str(value) for value in reranking["test_modes"]]
    if "full" not in dev_modes:
        raise ValueError("reranking.dev_modes must include 'full' for beta selection.")
    unknown = sorted((set(dev_modes) | set(test_modes)) - set(RERANK_MODES))
    if unknown:
        raise ValueError(f"Unknown reranking modes in configuration: {unknown}")
    missing_test = sorted(set(RERANK_MODES) - set(test_modes))
    if missing_test:
        raise ValueError(
            "reranking.test_modes must include all four controlled modes; "
            f"missing: {missing_test}"
        )

    evaluation = config["evaluation"]
    if evaluation.get("include_retrieval_baseline", False):
        retrieval_name = str(evaluation.get("retrieval_method_name", "")).strip()
        if not retrieval_name:
            raise ValueError("evaluation.retrieval_method_name must be non-empty.")
        if retrieval_name in set(test_modes):
            raise ValueError(
                "evaluation.retrieval_method_name must not conflict with a reranking mode."
            )


def _selected_stages(stage: str, through: Optional[str]) -> List[str]:
    if through is not None:
        if through not in STAGE_ORDER:
            raise ValueError(f"Unknown --through stage: {through}")
        return list(STAGE_ORDER[: STAGE_ORDER.index(through) + 1])
    if stage == "all":
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage}")
    return [stage]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Optional JSON configuration; defaults are used when omitted ({DEFAULT_CONFIG_NAME}).",
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing pgt/ and tools/ (default: current directory).",
    )
    parser.add_argument("--stage", choices=("all",) + STAGE_ORDER, default="all")
    parser.add_argument(
        "--through",
        choices=STAGE_ORDER,
        default=None,
        help="Run all dependency stages from data through the named stage.",
    )
    parser.add_argument("--plan", action="store_true", help="Print the complete execution plan without running commands.")
    parser.add_argument("--overwrite", action="store_true", help="Replace stage outputs where supported.")
    parser.add_argument("--resume", action="store_true", help="Resume a matching interrupted run.")
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="Use the first N development and N test CVEs after fixed splitting.",
    )
    parser.add_argument(
        "--write_default_config",
        type=Path,
        default=None,
        help="Write the built-in configuration template and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive.")
    if args.smoke < 0:
        parser.error("--smoke must be non-negative.")

    if args.write_default_config is not None:
        _write_json_atomic(args.write_default_config.resolve(), DEFAULT_CONFIG)
        print(f"Wrote default configuration: {args.write_default_config.resolve()}")
        return 0

    root = args.project_root.resolve()
    _validate_project(root)
    config_path = args.config.resolve() if args.config is not None else None
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _load_config(root, config_path)
    _validate_config(config)
    if args.smoke > 0:
        config["workspace"] = str(config["smoke_workspace"])
    paths = Paths.from_config(root, config)

    runner = PipelineRunner(
        root=root,
        config=config,
        paths=paths,
        plan_only=bool(args.plan),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        smoke_records_per_split=int(args.smoke),
    )

    stages = _selected_stages(args.stage, args.through)
    print(f"Pipeline version: {SCRIPT_VERSION}")
    print(f"Project root: {root}")
    print(f"Workspace: {paths.workspace}")
    print(f"Stages: {', '.join(stages)}")
    if args.plan:
        print("Mode: plan only")
    elif args.smoke:
        print(f"Mode: smoke ({args.smoke} development + {args.smoke} test records)")
    else:
        print("Mode: full")

    if not args.plan:
        paths.workspace.mkdir(parents=True, exist_ok=True)
    runner.save_state()
    for stage in stages:
        STAGE_FUNCTIONS[stage](runner)
        stage_state = runner.state.setdefault("stages", {}).setdefault(stage, {})
        if not runner.plan_only:
            stage_state["status"] = "succeeded"
            stage_state["finished_utc"] = _utc_now()
            runner.save_state()

    print("Pipeline plan completed." if args.plan else "Pipeline stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
