"""Reproducible LLM reranking with explicit ablation modes and MES injection.

This revision addresses the reproducibility and baseline concerns raised by
Reviewer 2.  It uses a pinned model snapshot by default, exposes every decoding
and fusion parameter on the command line, logs a run manifest, uses strict JSON
schema parsing, supports fixed seeds and retry accounting, and separates four
LLM reranking conditions that operate on the identical candidate list:

    generic   raw CVE text + candidate descriptions
    evidence  evidence units + candidate descriptions
    structure MES structural chain + candidate descriptions
    full      evidence units + MES + candidate descriptions

Technique-specific post-calibration is disabled by default.  Optional rules can
be supplied as a versioned JSON file and are then hashed in the run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from .io import read_jsonl

SCRIPT_VERSION = "rerank-v2.0.0"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
VALID_MODES = ("generic", "evidence", "structure", "full")


def _load_secrets() -> Dict[str, str]:
    path = Path("secrets.json")
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


_SECRETS = _load_secrets()


def _get_cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    if _SECRETS.get(name):
        return _SECRETS[name]
    value = os.getenv(name)
    return value if value else default


def _get_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    api_key = _get_cfg("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in secrets.json or an environment variable."
        )
    base_url = _get_cfg("OPENAI_BASE_URL")
    timeout_s = float(_get_cfg("OPENAI_TIMEOUT", "90"))
    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _sha256_bytes(p.read_bytes())


def _append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def _index_by_input_id(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        input_id = row.get("input_id")
        if isinstance(input_id, str) and input_id:
            out[input_id] = row
    return out


def _load_done_ids(path: str) -> set[str]:
    done: set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            input_id = obj.get("input_id")
            if isinstance(input_id, str) and input_id:
                done.add(input_id)
    return done


def _load_tech_index(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        tid = row.get("technique_id") or row.get("id") or row.get("technique")
        if tid:
            out[str(tid)] = row
    return out


def _technique_text(row: Mapping[str, Any]) -> str:
    name = row.get("name") or row.get("title")
    description = row.get("description") or row.get("text") or row.get("content")
    if not name and isinstance(row.get("fields"), Mapping):
        fields = row["fields"]
        name = fields.get("name")
        description = fields.get("description") or fields.get("text")
    parts = [str(x).strip() for x in (name, description) if isinstance(x, str) and x.strip()]
    if parts:
        return " - ".join(parts)
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True)


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _rerank_schema() -> Dict[str, Any]:
    return {
        "name": "rerank_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "technique_id": {"type": "string"},
                            "llm_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["technique_id", "llm_score", "reason", "evidence_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranking"],
            "additionalProperties": False,
        },
    }


def _candidate_payload(
    candidates: Sequence[Mapping[str, Any]],
    tech_index: Mapping[str, Mapping[str, Any]],
    topk: int,
) -> List[Dict[str, Any]]:
    packed: List[Dict[str, Any]] = []
    for candidate in candidates[:topk]:
        tid = candidate.get("technique_id")
        if not isinstance(tid, str):
            continue
        packed.append(
            {
                "technique_id": tid,
                "retrieval_score": candidate.get("score_fused"),
                "text_score": candidate.get("score_text"),
                "structure_score": candidate.get("score_graph"),
                "technique_text": _truncate(_technique_text(tech_index.get(tid, {})), 700),
            }
        )
    return packed


def _mes_structural_payload(mes_row: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = []
    for node in mes_row.get("nodes") or []:
        if not isinstance(node, Mapping) or node.get("type") in ("cve", "evidence"):
            continue
        nodes.append(
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "text": node.get("text"),
                "evidence_ids": node.get("evidence_ids") or [],
            }
        )
    edges = []
    for edge in mes_row.get("edges") or []:
        if isinstance(edge, Mapping) and edge.get("type") == "typed_sequence":
            edges.append(
                {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "type": edge.get("type"),
                }
            )
    return {
        "algorithm": mes_row.get("algorithm"),
        "complete_core_chain": mes_row.get("complete_core_chain"),
        "chain": mes_row.get("chain") or [],
        "nodes": nodes,
        "typed_sequence_edges": edges,
        "compact_text": mes_row.get("compact_text") or "",
        "mes_sha256": mes_row.get("mes_sha256"),
    }


def _build_messages(
    input_id: str,
    raw_text: str,
    sentences: Mapping[str, str],
    mes_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    tech_index: Mapping[str, Mapping[str, Any]],
    topk: int,
    mode: str,
) -> Tuple[List[Dict[str, str]], str]:
    candidate_json = json.dumps(
        _candidate_payload(candidates, tech_index, topk),
        ensure_ascii=False,
        sort_keys=True,
    )
    evidence_lines = "\n".join(f"{eid}: {text}" for eid, text in sentences.items())
    mes_json = json.dumps(_mes_structural_payload(mes_row), ensure_ascii=False, sort_keys=True)

    system_parts = [
        "You are a cybersecurity mapping assistant.",
        "Rerank the supplied MITRE ATT&CK technique candidates for one CVE.",
        "Return every supplied candidate exactly once and sort by llm_score descending.",
        "Use the full score range: 0.85-1.00 direct mechanism support; 0.55-0.80 strong support; "
        "0.30-0.50 weak or generic overlap; 0.00-0.25 unsupported.",
        "Do not introduce candidate identifiers or evidence identifiers that were not supplied.",
    ]

    user_parts = [f"input_id: {input_id}"]
    if mode == "generic":
        system_parts.append(
            "Use only the raw CVE text and technique descriptions. Do not use an evidence-unit or MES constraint. "
            "The evidence_ids array must be empty for every candidate."
        )
        user_parts.extend(["Raw CVE text:", raw_text])
    elif mode == "evidence":
        system_parts.append(
            "Use only the supplied evidence units and technique descriptions. Every positive rationale must cite "
            "one or more supplied evidence identifiers."
        )
        user_parts.extend(["Evidence units:", evidence_lines])
    elif mode == "structure":
        system_parts.append(
            "Use only the supplied MES structural chain and technique descriptions. Do not rely on raw CVE text. "
            "Cite only evidence identifiers attached to MES structural nodes."
        )
        user_parts.extend(["MES structural representation:", mes_json])
    elif mode == "full":
        system_parts.append(
            "Use only the supplied evidence units, MES structural chain, and technique descriptions. A high score "
            "requires both mechanism compatibility and explicit evidence support."
        )
        user_parts.extend(
            [
                "Evidence units:",
                evidence_lines,
                "MES structural representation:",
                mes_json,
            ]
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    user_parts.extend(
        [
            "Candidates:",
            candidate_json,
            "Return a ranking for all candidates.",
        ]
    )
    system = "\n".join(system_parts)
    user = "\n\n".join(user_parts)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_hash = _sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    return messages, prompt_hash


def _valid_evidence_ids_for_mode(
    mode: str,
    sentences: Mapping[str, str],
    mes_row: Mapping[str, Any],
) -> set[str]:
    if mode == "generic":
        return set()
    if mode == "evidence" or mode == "full":
        return set(sentences.keys())
    return {str(eid) for eid in (mes_row.get("evidence_ids") or []) if isinstance(eid, str)}


def _sanitize_ranking(
    ranking: Any,
    candidate_ids: Sequence[str],
    valid_evidence_ids: set[str],
    generic_mode: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    allowed = set(candidate_ids)
    result: Dict[str, Dict[str, Any]] = {}
    stats = {"unknown_candidates": 0, "duplicate_candidates": 0, "invalid_evidence_ids": 0}
    if not isinstance(ranking, list):
        ranking = []
    for item in ranking:
        if not isinstance(item, Mapping):
            continue
        tid = item.get("technique_id")
        if not isinstance(tid, str) or tid not in allowed:
            stats["unknown_candidates"] += 1
            continue
        if tid in result:
            stats["duplicate_candidates"] += 1
            continue
        try:
            score = max(0.0, min(1.0, float(item.get("llm_score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        raw_eids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        eids: List[str] = []
        if not generic_mode:
            for eid in raw_eids:
                if isinstance(eid, str) and eid in valid_evidence_ids and eid not in eids:
                    eids.append(eid)
                else:
                    stats["invalid_evidence_ids"] += 1
        result[tid] = {
            "llm_score": score,
            "reason": reason.strip(),
            "evidence_ids": eids,
        }
    return result, stats


def _load_calibration_rules(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Calibration rule file must contain a JSON array.")
    return [dict(rule) for rule in data if isinstance(rule, Mapping)]


def _regex_matches(pattern: Optional[str], text: str) -> bool:
    if not pattern:
        return True
    import re

    return bool(re.search(pattern, text, flags=re.I))


def _apply_calibration_rules(
    tid: str,
    score: float,
    reason: str,
    evidence_text: str,
    rules: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    applied: List[str] = []
    current = score
    current_reason = reason
    for idx, rule in enumerate(rules):
        if not _regex_matches(str(rule.get("technique_regex") or ""), tid):
            continue
        if rule.get("evidence_regex") and not _regex_matches(str(rule["evidence_regex"]), evidence_text):
            continue
        if rule.get("not_evidence_regex") and _regex_matches(str(rule["not_evidence_regex"]), evidence_text):
            continue
        if rule.get("reason_regex") and not _regex_matches(str(rule["reason_regex"]), current_reason):
            continue
        if rule.get("not_reason_regex") and _regex_matches(str(rule["not_reason_regex"]), current_reason):
            continue
        before = current
        if rule.get("floor") is not None:
            current = max(current, float(rule["floor"]))
        if rule.get("cap") is not None:
            current = min(current, float(rule["cap"]))
        current = max(0.0, min(1.0, current))
        if current != before:
            name = str(rule.get("name") or f"rule_{idx}")
            applied.append(name)
            suffix = str(rule.get("reason_suffix") or "").strip()
            if suffix:
                current_reason = (current_reason + " " + suffix).strip()
    return current, current_reason, applied


def _safe_retrieval_score(candidate: Mapping[str, Any], strict: bool) -> Tuple[float, bool]:
    try:
        value = float(candidate.get("score_fused", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    out_of_range = not 0.0 <= value <= 1.0
    if strict and out_of_range:
        raise ValueError(f"score_fused outside [0,1]: {value}")
    return max(0.0, min(1.0, value)), out_of_range


def _write_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible ATT&CK candidate reranking.")
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tech_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mes", default=None, help="Required for structure/full modes")
    parser.add_argument("--mode", choices=VALID_MODES, default="full")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_backoff", type=float, default=1.0)
    parser.add_argument("--calibration_rules", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--strict_scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode in ("structure", "full") and not args.mes:
        parser.error("--mes is required for structure/full modes")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be in [0,1]")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be in [0,2]")

    output = Path(args.output)
    if output.exists() and args.overwrite:
        output.unlink()
    elif output.exists() and not args.overwrite:
        # Resume mode is intentional and is recorded in the manifest.
        pass

    model = args.model or _get_cfg("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    topk = max(1, min(100, args.topk))
    beta = float(args.beta)
    max_tokens = max(1, args.max_tokens)
    retries = max(0, args.retries)
    manifest_path = args.manifest or str(output) + ".manifest.json"

    sentences_map = _index_by_input_id(args.sentences)
    mes_map = _index_by_input_id(args.mes)
    tech_index = _load_tech_index(args.tech_index)
    candidate_rows = list(read_jsonl(args.candidates))
    calibration_rules = _load_calibration_rules(args.calibration_rules)
    done_ids = _load_done_ids(str(output))

    try:
        import openai  # type: ignore

        openai_version = getattr(openai, "__version__", "unknown")
    except Exception:
        openai_version = "unavailable"

    manifest: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": _sha256_file(__file__),
        "created_unix": time.time(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_package_version": openai_version,
        "configuration": {
            "mode": args.mode,
            "model": model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_completion_tokens": max_tokens,
            "topk": topk,
            "beta": beta,
            "retries": retries,
            "retry_backoff": args.retry_backoff,
            "strict_json_schema": True,
            "strict_scores": args.strict_scores,
            "score_normalization": "candidate score_fused must already be in [0,1]; reranker clamps only when strict_scores=false",
            "final_score": "beta*score_fused + (1-beta)*llm_score",
            "calibration_rules": args.calibration_rules,
            "calibration_rules_sha256": _sha256_file(args.calibration_rules),
        },
        "input_sha256": {
            "sentences": _sha256_file(args.sentences),
            "candidates": _sha256_file(args.candidates),
            "tech_index": _sha256_file(args.tech_index),
            "mes": _sha256_file(args.mes),
        },
        "resume_existing_rows": len(done_ids),
        "counters": {
            "candidate_rows": len(candidate_rows),
            "processed": 0,
            "skipped_resume": 0,
            "api_attempts": 0,
            "api_retries": 0,
            "api_failures": 0,
            "missing_candidates_in_llm_output": 0,
            "score_clamps": 0,
            "calibration_applications": 0,
            "missing_mes": 0,
            "missing_sentences": 0,
        },
    }
    _write_manifest(manifest_path, manifest)

    client = _get_openai_client()
    schema = _rerank_schema()

    for row in tqdm(candidate_rows, desc=f"rerank[{args.mode}]"):
        input_id = row.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            continue
        if input_id in done_ids:
            manifest["counters"]["skipped_resume"] += 1
            continue

        candidates = row.get("candidates") or []
        if not isinstance(candidates, list):
            candidates = []
        srow = sentences_map.get(input_id, {})
        sentences = srow.get("sentences") or {}
        if not isinstance(sentences, Mapping):
            sentences = {}
        raw_text = srow.get("raw_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raw_text = " ".join(str(v) for v in sentences.values())
        mes_row = mes_map.get(input_id, {})

        if not sentences:
            manifest["counters"]["missing_sentences"] += 1
        if args.mode in ("structure", "full") and not mes_row:
            manifest["counters"]["missing_mes"] += 1

        candidate_ids = [
            c.get("technique_id")
            for c in candidates[:topk]
            if isinstance(c, Mapping) and isinstance(c.get("technique_id"), str)
        ]
        messages, prompt_hash = _build_messages(
            input_id=input_id,
            raw_text=raw_text,
            sentences=sentences,
            mes_row=mes_row,
            candidates=candidates,
            tech_index=tech_index,
            topk=topk,
            mode=args.mode,
        )

        llm_map: Dict[str, Dict[str, Any]] = {}
        parse_stats: Dict[str, int] = {}
        last_error: Optional[Exception] = None
        response_meta: Dict[str, Any] = {}
        attempts_used = 0

        for attempt in range(retries + 1):
            attempts_used += 1
            manifest["counters"]["api_attempts"] += 1
            if attempt > 0:
                manifest["counters"]["api_retries"] += 1
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    seed=args.seed,
                    response_format={"type": "json_schema", "json_schema": schema},
                    max_completion_tokens=max_tokens,
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                valid_eids = _valid_evidence_ids_for_mode(args.mode, sentences, mes_row)
                llm_map, parse_stats = _sanitize_ranking(
                    parsed.get("ranking"),
                    candidate_ids=candidate_ids,
                    valid_evidence_ids=valid_eids,
                    generic_mode=args.mode == "generic",
                )
                response_meta = {
                    "returned_model": getattr(response, "model", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "request_id": getattr(response, "_request_id", None),
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                }
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(max(0.0, args.retry_backoff) * (attempt + 1))

        if last_error is not None:
            manifest["counters"]["api_failures"] += 1

        for tid in candidate_ids:
            if tid not in llm_map:
                manifest["counters"]["missing_candidates_in_llm_output"] += 1
                llm_map[tid] = {
                    "llm_score": 0.0,
                    "reason": "Missing from LLM output; deterministic fallback applied.",
                    "evidence_ids": [],
                }

        evidence_text = "\n".join(f"{eid}: {text}" for eid, text in sentences.items())
        merged: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            tid = candidate.get("technique_id")
            if not isinstance(tid, str):
                continue
            retrieval_score, clamped = _safe_retrieval_score(candidate, strict=args.strict_scores)
            manifest["counters"]["score_clamps"] += int(clamped)
            out_candidate = dict(candidate)
            if tid in llm_map:
                llm_score = float(llm_map[tid]["llm_score"])
                reason = str(llm_map[tid]["reason"])
                applied_rules: List[str] = []
                if calibration_rules:
                    llm_score, reason, applied_rules = _apply_calibration_rules(
                        tid,
                        llm_score,
                        reason,
                        evidence_text,
                        calibration_rules,
                    )
                    manifest["counters"]["calibration_applications"] += len(applied_rules)
                out_candidate.update(
                    {
                        "llm_score": llm_score,
                        "final_score": beta * retrieval_score + (1.0 - beta) * llm_score,
                        "reason": reason,
                        "evidence_ids": llm_map[tid]["evidence_ids"],
                        "calibration_rules_applied": applied_rules,
                    }
                )
            else:
                out_candidate.update(
                    {
                        "llm_score": None,
                        "final_score": retrieval_score,
                        "reason": None,
                        "evidence_ids": None,
                        "calibration_rules_applied": [],
                    }
                )
            merged.append(out_candidate)

        merged.sort(
            key=lambda item: (
                float(item.get("final_score", 0.0) or 0.0),
                float(item.get("score_fused", 0.0) or 0.0),
                str(item.get("technique_id", "")),
            ),
            reverse=True,
        )

        output_row: Dict[str, Any] = {
            "input_id": input_id,
            "candidates": merged,
            "_meta": {
                "script_version": SCRIPT_VERSION,
                "mode": args.mode,
                "model_requested": model,
                "temperature": args.temperature,
                "seed": args.seed,
                "max_completion_tokens": max_tokens,
                "topk": topk,
                "beta": beta,
                "attempts_used": attempts_used,
                "prompt_sha256": prompt_hash,
                "mes_sha256": mes_row.get("mes_sha256") if isinstance(mes_row, Mapping) else None,
                "parse_stats": parse_stats,
                **response_meta,
            },
        }
        if last_error is not None:
            output_row["_rerank_error"] = f"{type(last_error).__name__}: {last_error}"
        _append_jsonl(str(output), output_row)
        done_ids.add(input_id)
        manifest["counters"]["processed"] += 1
        _write_manifest(manifest_path, manifest)

    manifest["finished_unix"] = time.time()
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest["counters"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
