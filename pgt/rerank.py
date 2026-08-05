"""Reproducible LLM reranking for CVE-to-ATT&CK candidate lists.

The reranker implements four controlled conditions over an identical candidate
set:

``generic``
    Raw CVE text + ATT&CK candidate descriptions.
``evidence``
    Stable evidence units + ATT&CK candidate descriptions.
``structure``
    The Minimal Explainable Subgraph (MES) + candidate descriptions.
``full``
    Evidence units + MES + candidate descriptions.

The implementation is intentionally strict for publication experiments:

* the model snapshot and all decoding/fusion parameters are explicit;
* candidate sets must contain the configured number of unique candidates;
* every candidate must be returned exactly once by the LLM;
* evidence identifiers are validated against the supplied context;
* technique-specific score calibration is not applied;
* missing inputs and exhausted API retries fail rather than silently changing
  the experiment; and
* a manifest records configuration, input hashes, prompt hashes, retries,
  response metadata, and output hashes.

The module uses :mod:`pgt.openai_client`; no API key, proxy, or endpoint is
hard-coded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from .io import read_jsonl
from .openai_client import get_openai_client, get_openai_runtime_config

SCRIPT_VERSION = "rerank-v2.2.0"
PROMPT_VERSION = "rerank-prompt-v2.2.0"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
VALID_MODES = ("generic", "evidence", "structure", "full")
STRUCTURAL_NODE_TYPES = {"Precondition", "Entry", "VulnType", "Behavior", "Impact"}
TRACEABILITY_EDGE_TYPES = {"mentions", "supported_by"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return _sha256_bytes(p.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _index_by_input_id(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        input_id = row.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError(f"Missing input_id in {path}: {row!r}")
        if input_id in result:
            raise ValueError(f"Duplicate input_id {input_id!r} in {path}")
        result[input_id] = row
    return result


def _load_done_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with p.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cannot resume from corrupt JSONL {p} at line {line_number}."
                ) from exc
            input_id = row.get("input_id")
            if isinstance(input_id, str) and input_id:
                done.add(input_id)
    return done


# ---------------------------------------------------------------------------
# ATT&CK and input normalization
# ---------------------------------------------------------------------------


def _load_tech_index(path: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        technique_id = row.get("technique_id") or row.get("id") or row.get("technique")
        if not technique_id:
            continue
        tid = str(technique_id)
        if tid in result:
            raise ValueError(f"Duplicate technique_id {tid!r} in {path}")
        result[tid] = row
    if not result:
        raise ValueError(f"No ATT&CK techniques found in {path}")
    return result


def _technique_text(row: Mapping[str, Any]) -> str:
    direct_text = row.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    name = row.get("name") or row.get("title")
    description = row.get("description") or row.get("content")
    fields = row.get("fields")
    if isinstance(fields, Mapping):
        if not name:
            name = fields.get("name") or fields.get("title")
        if not description:
            description = fields.get("description") or fields.get("text")

    parts = [
        value.strip()
        for value in (name, description)
        if isinstance(value, str) and value.strip()
    ]
    return "\n".join(parts)


def _truncate(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _ordered_sentences(row: Mapping[str, Any]) -> Dict[str, str]:
    raw = row.get("sentences") or {}
    if not isinstance(raw, Mapping):
        return {}

    def order_key(item: Tuple[Any, Any]) -> Tuple[int, str]:
        key = str(item[0])
        digits = "".join(ch for ch in key if ch.isdigit())
        return (int(digits) if digits else 10**9, key)

    result: Dict[str, str] = {}
    for evidence_id, text in sorted(raw.items(), key=order_key):
        eid = str(evidence_id).strip()
        value = str(text).strip() if text is not None else ""
        if eid and value:
            result[eid] = value
    return result


def _raw_text(row: Mapping[str, Any], sentences: Mapping[str, str]) -> str:
    raw = row.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return " ".join(sentences.values()).strip()


def _select_candidates(
    row: Mapping[str, Any],
    *,
    topk: int,
    require_exact_topk: bool,
) -> List[Dict[str, Any]]:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"Candidate row {row.get('input_id')!r} has no candidate list.")

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for position, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Candidate row {row.get('input_id')!r} contains a non-object candidate."
            )
        technique_id = candidate.get("technique_id")
        if not isinstance(technique_id, str) or not technique_id.strip():
            raise ValueError(
                f"Candidate row {row.get('input_id')!r} has a candidate without technique_id."
            )
        tid = technique_id.strip()
        if tid in seen:
            raise ValueError(
                f"Candidate row {row.get('input_id')!r} contains duplicate {tid}."
            )
        seen.add(tid)
        normalized = dict(candidate)
        normalized["technique_id"] = tid
        normalized.setdefault("rank", position)
        selected.append(normalized)
        if len(selected) == topk:
            break

    if require_exact_topk and len(selected) != topk:
        raise ValueError(
            f"Candidate row {row.get('input_id')!r} contains {len(selected)} unique "
            f"candidates; exactly {topk} are required."
        )
    if not selected:
        raise ValueError(f"Candidate row {row.get('input_id')!r} is empty.")
    return selected


def _retrieval_score(candidate: Mapping[str, Any], *, strict: bool) -> Tuple[float, bool]:
    try:
        value = float(candidate.get("score_fused", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    out_of_range = not 0.0 <= value <= 1.0
    if out_of_range and strict:
        raise ValueError(
            f"score_fused for {candidate.get('technique_id')} is outside [0,1]: {value}"
        )
    return max(0.0, min(1.0, value)), out_of_range


# ---------------------------------------------------------------------------
# MES serialization
# ---------------------------------------------------------------------------


def _mes_structural_payload(mes_row: Mapping[str, Any]) -> Dict[str, Any]:
    structural_nodes: List[Dict[str, Any]] = []
    structural_node_ids: set[str] = set()

    for node in mes_row.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        node_type = str(node.get("type") or "")
        node_id = str(node.get("id") or "")
        if node_type not in STRUCTURAL_NODE_TYPES or not node_id:
            continue
        evidence_ids = [
            str(eid)
            for eid in (node.get("evidence_ids") or [])
            if isinstance(eid, str) and eid
        ]
        structural_nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "text": str(node.get("text") or "").strip(),
                "evidence_ids": evidence_ids,
                "confidence": node.get("confidence"),
            }
        )
        structural_node_ids.add(node_id)

    structural_edges: List[Dict[str, Any]] = []
    for edge in mes_row.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        edge_type = str(edge.get("type") or "")
        src = str(edge.get("src") or edge.get("source") or "")
        dst = str(edge.get("dst") or edge.get("target") or "")
        if (
            not src
            or not dst
            or edge_type in TRACEABILITY_EDGE_TYPES
            or src not in structural_node_ids
            or dst not in structural_node_ids
        ):
            continue
        structural_edges.append(
            {
                "src": src,
                "dst": dst,
                "type": edge_type,
                "origin": edge.get("origin"),
                "shared_evidence_ids": edge.get("shared_evidence_ids") or [],
                "structural_score": edge.get("structural_score"),
            }
        )

    structural_nodes.sort(key=lambda item: (str(item["type"]), str(item["id"])))
    structural_edges.sort(
        key=lambda item: (str(item["src"]), str(item["type"]), str(item["dst"]))
    )

    evidence_ids = sorted(
        {
            str(eid)
            for eid in (mes_row.get("evidence_ids") or [])
            if isinstance(eid, str) and eid
        }
    )

    return {
        "algorithm": mes_row.get("algorithm"),
        "source_graph_version": mes_row.get("source_graph_version"),
        "status": mes_row.get("status"),
        "complete_core_chain": bool(mes_row.get("complete_core_chain")),
        "chain": mes_row.get("chain") or [],
        "chain_types": mes_row.get("chain_types") or [],
        "compact_text": str(mes_row.get("compact_text") or ""),
        "evidence_ids": evidence_ids,
        "structural_nodes": structural_nodes,
        "structural_edges": structural_edges,
        "mes_sha256": mes_row.get("mes_sha256"),
    }


def _validate_mes_for_mode(
    input_id: str,
    mes_row: Mapping[str, Any],
    *,
    mode: str,
    allow_empty_mes: bool,
    require_complete_mes: bool,
) -> None:
    if mode not in {"structure", "full"}:
        return
    if not mes_row:
        raise ValueError(f"{input_id}: MES record is required for mode={mode}.")

    payload = _mes_structural_payload(mes_row)
    if require_complete_mes and not payload["complete_core_chain"]:
        raise ValueError(f"{input_id}: complete MES core chain is required.")
    if not allow_empty_mes and not payload["structural_nodes"]:
        raise ValueError(f"{input_id}: MES contains no structural nodes.")


# ---------------------------------------------------------------------------
# Prompt and response schema
# ---------------------------------------------------------------------------


def _score_item_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "llm_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reason": {"type": "string"},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["llm_score", "reason", "evidence_ids"],
        "additionalProperties": False,
    }


def _rerank_schema(candidate_ids: Sequence[str]) -> Dict[str, Any]:
    """Build a candidate-specific strict response contract.

    An array schema can constrain item shape but cannot reliably force the
    model to emit every distinct candidate exactly once.  The keyed object
    below makes every supplied technique ID a required property and rejects
    additions, so omission and duplication are prevented at the structured
    output boundary rather than repaired after generation.
    """

    ordered_ids = [str(tid) for tid in candidate_ids]
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Response schema requires a non-empty unique candidate list.")

    score_properties = {tid: _score_item_schema() for tid in ordered_ids}
    schema_hash = _sha256_text(_canonical_json(ordered_ids))[:12]
    return {
        "name": f"cve_attck_rerank_{schema_hash}",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": score_properties,
                    "required": ordered_ids,
                    "additionalProperties": False,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }


def _candidate_payload(
    candidates: Sequence[Mapping[str, Any]],
    tech_index: Mapping[str, Mapping[str, Any]],
    *,
    max_technique_chars: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for candidate in candidates:
        tid = str(candidate["technique_id"])
        technique_text = _technique_text(tech_index.get(tid, {}))
        if not technique_text:
            raise ValueError(f"No ATT&CK name/description found for candidate {tid}.")
        result.append(
            {
                "technique_id": tid,
                "technique_text": _truncate(technique_text, max_technique_chars),
            }
        )
    return result


def _build_messages(
    *,
    input_id: str,
    mode: str,
    raw_text: str,
    sentences: Mapping[str, str],
    mes_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    tech_index: Mapping[str, Mapping[str, Any]],
    max_technique_chars: int,
) -> Tuple[List[Dict[str, str]], str]:
    packed_candidates = _candidate_payload(
        candidates,
        tech_index,
        max_technique_chars=max_technique_chars,
    )
    candidate_json = json.dumps(packed_candidates, ensure_ascii=False, sort_keys=True)
    evidence_lines = "\n".join(f"{eid}: {text}" for eid, text in sentences.items())
    mes_payload = _mes_structural_payload(mes_row)
    mes_json = json.dumps(mes_payload, ensure_ascii=False, sort_keys=True)

    system_lines = [
        "You are performing controlled candidate discrimination for CVE-to-MITRE ATT&CK mapping.",
        "Evaluate only the candidate techniques supplied in the user message.",
        "Return one result for every supplied technique_id; do not add or omit candidates.",
        "The top-level scores object must use each supplied technique_id as an exact property key.",
        "The llm_score is a support score, not a calibrated probability.",
        "Use this rubric consistently:",
        "- 0.85-1.00: the supplied context directly states or clearly instantiates the candidate mechanism.",
        "- 0.55-0.80: strong mechanism support, although one detail may be implicit.",
        "- 0.30-0.50: weak, indirect, or generic overlap.",
        "- 0.00-0.25: unsupported, contradicted, or dependent on missing conditions.",
        "Keep each reason concise (normally no more than 40 words).",
        "Do not use technique-specific scoring shortcuts or external vulnerability facts.",
        "Output must follow the JSON schema exactly.",
    ]

    user_sections = [f"input_id: {input_id}", f"condition: {mode}"]

    if mode == "generic":
        system_lines.extend(
            [
                "Use only the raw CVE text and the supplied ATT&CK descriptions.",
                "Do not impose evidence-unit or MES constraints.",
                "Return an empty evidence_ids array for every candidate.",
            ]
        )
        user_sections.extend(["Raw CVE text:", raw_text])
    elif mode == "evidence":
        system_lines.extend(
            [
                "Use only the supplied evidence units and ATT&CK descriptions.",
                "For scores above 0.25, cite at least one supplied evidence identifier.",
            ]
        )
        user_sections.extend(["Evidence units:", evidence_lines])
    elif mode == "structure":
        system_lines.extend(
            [
                "Use only the supplied MES and ATT&CK descriptions; do not use raw CVE text.",
                "For scores above 0.25, cite at least one evidence identifier attached to the MES.",
            ]
        )
        user_sections.extend(["Minimal Explainable Subgraph (MES):", mes_json])
    elif mode == "full":
        system_lines.extend(
            [
                "Use only the supplied evidence units, MES, and ATT&CK descriptions.",
                "A score above 0.25 requires valid evidence citation.",
                "A strong score should be compatible with the MES chain and cite at least one MES-linked evidence identifier.",
            ]
        )
        user_sections.extend(
            [
                "Evidence units:",
                evidence_lines,
                "Minimal Explainable Subgraph (MES):",
                mes_json,
            ]
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    user_sections.extend(
        [
            "Candidate techniques (the order is the common retrieval order used in every condition):",
            candidate_json,
            "Return the complete scores object now.",
        ]
    )

    messages = [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]
    prompt_hash = _sha256_text(_canonical_json(messages))
    return messages, prompt_hash


# ---------------------------------------------------------------------------
# Strict parsing and evidence constraints
# ---------------------------------------------------------------------------


def _context_evidence_ids(
    *,
    mode: str,
    sentences: Mapping[str, str],
    mes_row: Mapping[str, Any],
) -> Tuple[set[str], set[str]]:
    sentence_ids = set(sentences.keys())
    mes_ids = {
        str(eid)
        for eid in (mes_row.get("evidence_ids") or [])
        if isinstance(eid, str) and eid
    }
    if mode == "generic":
        return set(), mes_ids
    if mode == "structure":
        return mes_ids, mes_ids
    return sentence_ids, mes_ids


def _parse_complete_ranking(
    response_items: Any,
    *,
    candidate_ids: Sequence[str],
    valid_evidence_ids: set[str],
    mes_evidence_ids: set[str],
    mode: str,
    citation_threshold: float,
    enforce_evidence_citations: bool,
    require_mes_citation_for_strong_score: bool,
    strong_score_threshold: float,
    max_reason_chars: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    expected = list(candidate_ids)
    expected_set = set(expected)
    parsed: Dict[str, Dict[str, Any]] = {}
    stats = {
        "invalid_evidence_ids_removed": 0,
        "generic_evidence_ids_ignored": 0,
        "citation_caps": 0,
        "mes_overlap_caps": 0,
        "reason_truncations": 0,
        "keyed_response_records": 0,
        "legacy_array_response_records": 0,
    }

    normalized_items: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(response_items, Mapping):
        stats["keyed_response_records"] = 1
        unknown = sorted(str(key) for key in response_items if str(key) not in expected_set)
        if unknown:
            raise ValueError(f"LLM returned unknown technique_id(s): {', '.join(unknown)}")
        missing = [tid for tid in expected if tid not in response_items]
        if missing:
            raise ValueError(f"LLM omitted candidate(s): {', '.join(missing)}")
        for tid in expected:
            item = response_items.get(tid)
            if not isinstance(item, Mapping):
                raise ValueError(f"LLM result for {tid} is not an object.")
            normalized_items.append((tid, item))
    elif isinstance(response_items, list):
        # Backward-compatible parser for old saved/mock responses. New API
        # requests use the keyed strict schema above.
        stats["legacy_array_response_records"] = 1
        for item in response_items:
            if not isinstance(item, Mapping):
                raise ValueError("LLM ranking contains a non-object item.")
            tid = item.get("technique_id")
            if not isinstance(tid, str) or tid not in expected_set:
                raise ValueError(f"LLM returned unknown technique_id: {tid!r}")
            normalized_items.append((tid, item))
    else:
        raise ValueError("LLM response must contain a keyed 'scores' object.")

    for tid, item in normalized_items:
        if tid in parsed:
            raise ValueError(f"LLM returned duplicate technique_id: {tid}")

        try:
            score = float(item.get("llm_score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid llm_score for {tid}") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"llm_score for {tid} is outside [0,1]: {score}")

        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        reason = reason.strip()
        if len(reason) > max_reason_chars:
            reason = _truncate(reason, max_reason_chars)
            stats["reason_truncations"] += 1

        raw_eids = item.get("evidence_ids")
        if not isinstance(raw_eids, list):
            raise ValueError(f"evidence_ids for {tid} is not an array.")

        evidence_ids: List[str] = []
        if mode == "generic":
            stats["generic_evidence_ids_ignored"] += len(raw_eids)
        else:
            for raw_eid in raw_eids:
                if not isinstance(raw_eid, str) or raw_eid not in valid_evidence_ids:
                    stats["invalid_evidence_ids_removed"] += 1
                    continue
                if raw_eid not in evidence_ids:
                    evidence_ids.append(raw_eid)

        constraints: List[str] = []
        if (
            enforce_evidence_citations
            and mode != "generic"
            and score > citation_threshold
            and not evidence_ids
        ):
            score = citation_threshold
            constraints.append("missing_valid_evidence_citation")
            stats["citation_caps"] += 1

        if (
            require_mes_citation_for_strong_score
            and mode in {"structure", "full"}
            and score >= strong_score_threshold
            and mes_evidence_ids
            and not (set(evidence_ids) & mes_evidence_ids)
        ):
            score = min(score, strong_score_threshold - 0.01)
            constraints.append("missing_mes_evidence_overlap")
            stats["mes_overlap_caps"] += 1

        if constraints:
            suffix = "Constraint applied: " + ", ".join(constraints) + "."
            reason = f"{reason} {suffix}".strip()

        parsed[tid] = {
            "llm_score": round(score, 12),
            "reason": reason,
            "evidence_ids": evidence_ids,
            "constraint_flags": constraints,
        }

    missing = [tid for tid in expected if tid not in parsed]
    if missing:
        raise ValueError(f"LLM omitted candidate(s): {', '.join(missing)}")
    if len(parsed) != len(expected):
        raise ValueError("LLM candidate cardinality does not match the supplied set.")

    return parsed, stats


# ---------------------------------------------------------------------------
# API interaction and score fusion
# ---------------------------------------------------------------------------


def _response_metadata(response: Any) -> Dict[str, Any]:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    return {
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "_request_id", None),
        "returned_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        "created": getattr(response, "created", None),
    }


def _call_reranker(
    *,
    client: Any,
    model: str,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    temperature: float,
    seed: int,
    max_tokens: int,
) -> Tuple[Any, Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        seed=seed,
        response_format={"type": "json_schema", "json_schema": dict(schema)},
        max_completion_tokens=max_tokens,
    )
    if not getattr(response, "choices", None):
        raise RuntimeError("OpenAI response contains no choices.")
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("OpenAI response content is empty.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI response is not valid JSON.") from exc
    return parsed, _response_metadata(response)


def _merge_scores(
    *,
    candidates: Sequence[Mapping[str, Any]],
    llm_map: Mapping[str, Mapping[str, Any]],
    beta: float,
    strict_scores: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    merged: List[Dict[str, Any]] = []
    clamp_count = 0

    for original_position, candidate in enumerate(candidates, start=1):
        tid = str(candidate["technique_id"])
        retrieval_score, clamped = _retrieval_score(candidate, strict=strict_scores)
        clamp_count += int(clamped)
        llm_result = llm_map[tid]
        llm_score = float(llm_result["llm_score"])
        final_score = beta * retrieval_score + (1.0 - beta) * llm_score

        output_candidate = dict(candidate)
        output_candidate.update(
            {
                "retrieval_rank": int(candidate.get("rank") or original_position),
                "retrieval_score": round(retrieval_score, 12),
                "llm_score": round(llm_score, 12),
                "final_score": round(final_score, 12),
                "reason": llm_result["reason"],
                "evidence_ids": list(llm_result["evidence_ids"]),
                "constraint_flags": list(llm_result["constraint_flags"]),
            }
        )
        merged.append(output_candidate)

    merged.sort(
        key=lambda item: (
            -float(item["final_score"]),
            -float(item["llm_score"]),
            -float(item["retrieval_score"]),
            int(item["retrieval_rank"]),
            str(item["technique_id"]),
        )
    )
    for rank, candidate in enumerate(merged, start=1):
        candidate["rerank_rank"] = rank
    return merged, clamp_count


# ---------------------------------------------------------------------------
# Manifest and CLI
# ---------------------------------------------------------------------------


def _configuration(args: argparse.Namespace, model: str, topk: int) -> Dict[str, Any]:
    return {
        "mode": args.mode,
        "model": model,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_completion_tokens": args.max_tokens,
        "application_max_attempts": args.attempts,
        "retry_backoff_seconds": args.retry_backoff,
        "topk": topk,
        "require_exact_topk": args.require_exact_topk,
        "beta": args.beta,
        "final_score_formula": "beta*score_fused + (1-beta)*llm_score",
        "strict_json_schema": True,
        "response_contract": "candidate_keyed_required_object-v1",
        "retry_seed_policy": "base_seed_plus_attempt_index",
        "strict_retrieval_scores": args.strict_scores,
        "allow_empty_mes": args.allow_empty_mes,
        "require_complete_mes": args.require_complete_mes,
        "citation_threshold": args.citation_threshold,
        "enforce_evidence_citations": args.enforce_evidence_citations,
        "require_mes_citation_for_strong_score": args.require_mes_citation_for_strong_score,
        "strong_score_threshold": args.strong_score_threshold,
        "max_reason_chars": args.max_reason_chars,
        "max_technique_chars": args.max_technique_chars,
        "technique_specific_calibration": False,
        "api_failure_policy": "fail_after_exhausted_attempts",
        "incomplete_response_policy": "schema_prevent_then_retry_with_distinct_seed",
    }


def _run_signature(
    configuration: Mapping[str, Any], input_hashes: Mapping[str, Any]
) -> str:
    return _sha256_text(
        _canonical_json({"configuration": configuration, "input_sha256": input_hashes})
    )


def _prepare_output(
    *,
    output_path: Path,
    manifest_path: Path,
    overwrite: bool,
    resume: bool,
    run_signature: str,
) -> set[str]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if output_path.exists() and overwrite:
        output_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        return set()

    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite or --resume explicitly."
        )

    if resume:
        output_exists = output_path.exists()
        manifest_exists = manifest_path.exists()

        # ``--resume`` is also used by the end-to-end orchestrator when a
        # stage has not started yet. In that case neither artifact exists, so
        # begin a fresh run rather than treating the absence as corruption.
        if not output_exists and not manifest_exists:
            return set()

        # A manifest can legitimately exist before the first result row is
        # appended (for example, interruption during the first API call).
        # Validate its signature and safely restart from zero completed rows.
        if manifest_exists and not output_exists:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest.get("run_signature") != run_signature:
                # No result row exists, so there is nothing to mix with the
                # new run. Remove the stale pre-first-row manifest and start
                # cleanly under the current response contract.
                manifest_path.unlink()
            return set()

        # An output without its manifest cannot be verified against the
        # current configuration and inputs, so fail rather than mixing runs.
        if output_exists and not manifest_exists:
            raise FileNotFoundError(
                "Cannot resume: output JSONL exists but its manifest is missing."
            )

        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_signature") != run_signature:
            raise RuntimeError(
                "Resume configuration/input hashes do not match the existing run manifest."
            )
        return _load_done_ids(output_path)

    return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible LLM reranking with generic/evidence/MES/full conditions."
    )
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tech_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mes", default=None, help="Required for structure/full modes.")
    parser.add_argument("--mode", choices=VALID_MODES, default="full")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--require_exact_topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max_tokens", type=int, default=1800)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry_backoff", type=float, default=1.0)
    parser.add_argument("--strict_scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_empty_mes", action="store_true")
    parser.add_argument("--require_complete_mes", action="store_true")
    parser.add_argument("--citation_threshold", type=float, default=0.25)
    parser.add_argument(
        "--enforce_evidence_citations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require_mes_citation_for_strong_score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strong_score_threshold", type=float, default=0.55)
    parser.add_argument("--max_reason_chars", type=int, default=600)
    parser.add_argument("--max_technique_chars", type=int, default=700)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode in {"structure", "full"} and not args.mes:
        parser.error("--mes is required for structure/full modes.")
    if args.topk < 1:
        parser.error("--topk must be at least 1.")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be in [0,1].")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be in [0,2].")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be positive.")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1.")
    if args.retry_backoff < 0:
        parser.error("--retry_backoff must be non-negative.")
    if not 0.0 <= args.citation_threshold <= 1.0:
        parser.error("--citation_threshold must be in [0,1].")
    if not 0.0 < args.strong_score_threshold <= 1.0:
        parser.error("--strong_score_threshold must be in (0,1].")

    output_path = Path(args.output)
    manifest_path = Path(args.manifest or f"{args.output}.manifest.json")
    topk = int(args.topk)
    model = str(args.model)

    sentences_map = _index_by_input_id(args.sentences)
    mes_map = _index_by_input_id(args.mes)
    candidate_rows = list(read_jsonl(args.candidates))
    tech_index = _load_tech_index(args.tech_index)

    input_hashes = {
        "sentences": _sha256_file(args.sentences),
        "candidates": _sha256_file(args.candidates),
        "tech_index": _sha256_file(args.tech_index),
        "mes": _sha256_file(args.mes),
    }
    configuration = _configuration(args, model, topk)
    run_signature = _run_signature(configuration, input_hashes)
    done_ids = _prepare_output(
        output_path=output_path,
        manifest_path=manifest_path,
        overwrite=args.overwrite,
        resume=args.resume,
        run_signature=run_signature,
    )

    try:
        import openai  # type: ignore

        openai_version = getattr(openai, "__version__", "unknown")
    except Exception:
        openai_version = "unavailable"

    counters: Dict[str, int] = {
        "candidate_rows": len(candidate_rows),
        "processed": 0,
        "skipped_resume": 0,
        "api_attempts": 0,
        "api_retries": 0,
        "api_failures": 0,
        "response_validation_failures": 0,
        "partial_mes": 0,
        "complete_mes": 0,
        "empty_mes": 0,
        "invalid_evidence_ids_removed": 0,
        "generic_evidence_ids_ignored": 0,
        "citation_caps": 0,
        "mes_overlap_caps": 0,
        "reason_truncations": 0,
        "keyed_response_records": 0,
        "legacy_array_response_records": 0,
        "retrieval_score_clamps": 0,
    }

    manifest: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "script_sha256": _sha256_file(__file__),
        "run_signature": run_signature,
        "started_utc": _utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_package_version": openai_version,
        "openai_runtime": get_openai_runtime_config(),
        "configuration": configuration,
        "input_sha256": input_hashes,
        "resume_existing_rows": len(done_ids),
        "counters": counters,
    }
    _write_json(manifest_path, manifest)

    client = get_openai_client()

    for candidate_row in tqdm(candidate_rows, desc=f"rerank[{args.mode}]"):
        input_id = candidate_row.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("Candidate row without input_id.")
        if input_id in done_ids:
            counters["skipped_resume"] += 1
            continue

        sentence_row = sentences_map.get(input_id)
        if sentence_row is None:
            raise ValueError(f"{input_id}: no matching sentence record.")
        sentences = _ordered_sentences(sentence_row)
        raw_text = _raw_text(sentence_row, sentences)

        if args.mode == "generic" and not raw_text:
            raise ValueError(f"{input_id}: raw CVE text is empty.")
        if args.mode in {"evidence", "full"} and not sentences:
            raise ValueError(f"{input_id}: evidence units are empty for mode={args.mode}.")

        mes_row = mes_map.get(input_id, {})
        _validate_mes_for_mode(
            input_id,
            mes_row,
            mode=args.mode,
            allow_empty_mes=args.allow_empty_mes,
            require_complete_mes=args.require_complete_mes,
        )
        if args.mode in {"structure", "full"}:
            status = str(mes_row.get("status") or "")
            if status == "complete":
                counters["complete_mes"] += 1
            elif status == "partial":
                counters["partial_mes"] += 1
            else:
                counters["empty_mes"] += 1

        candidates = _select_candidates(
            candidate_row,
            topk=topk,
            require_exact_topk=args.require_exact_topk,
        )
        candidate_ids = [str(candidate["technique_id"]) for candidate in candidates]
        candidate_set_sha256 = _sha256_text(_canonical_json(candidate_ids))
        schema = _rerank_schema(candidate_ids)
        response_schema_sha256 = _sha256_text(_canonical_json(schema))

        messages, prompt_hash = _build_messages(
            input_id=input_id,
            mode=args.mode,
            raw_text=raw_text,
            sentences=sentences,
            mes_row=mes_row,
            candidates=candidates,
            tech_index=tech_index,
            max_technique_chars=args.max_technique_chars,
        )
        valid_eids, mes_eids = _context_evidence_ids(
            mode=args.mode,
            sentences=sentences,
            mes_row=mes_row,
        )

        llm_map: Optional[Dict[str, Dict[str, Any]]] = None
        parse_stats: Dict[str, int] = {}
        response_meta: Dict[str, Any] = {}
        last_error: Optional[Exception] = None
        attempts_used = 0
        attempt_seeds: List[int] = []

        for attempt_index in range(args.attempts):
            attempts_used += 1
            attempt_seed = int(args.seed) + attempt_index
            attempt_seeds.append(attempt_seed)
            counters["api_attempts"] += 1
            if attempt_index > 0:
                counters["api_retries"] += 1
            try:
                parsed_response, response_meta = _call_reranker(
                    client=client,
                    model=model,
                    messages=messages,
                    schema=schema,
                    temperature=args.temperature,
                    seed=attempt_seed,
                    max_tokens=args.max_tokens,
                )
                response_items = parsed_response.get("scores")
                if response_items is None and "ranking" in parsed_response:
                    response_items = parsed_response.get("ranking")
                llm_map, parse_stats = _parse_complete_ranking(
                    response_items,
                    candidate_ids=candidate_ids,
                    valid_evidence_ids=valid_eids,
                    mes_evidence_ids=mes_eids,
                    mode=args.mode,
                    citation_threshold=args.citation_threshold,
                    enforce_evidence_citations=args.enforce_evidence_citations,
                    require_mes_citation_for_strong_score=(
                        args.require_mes_citation_for_strong_score
                    ),
                    strong_score_threshold=args.strong_score_threshold,
                    max_reason_chars=args.max_reason_chars,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                counters["response_validation_failures"] += 1
                if attempt_index + 1 < args.attempts:
                    time.sleep(args.retry_backoff * (attempt_index + 1))

        if llm_map is None:
            counters["api_failures"] += 1
            manifest["last_error"] = {
                "input_id": input_id,
                "type": type(last_error).__name__ if last_error else "UnknownError",
                "message": str(last_error) if last_error else "Unknown reranking failure.",
                "utc": _utc_now(),
            }
            _write_json(manifest_path, manifest)
            raise RuntimeError(
                f"{input_id}: reranking failed after {args.attempts} attempts: {last_error}"
            ) from last_error

        for key, value in parse_stats.items():
            counters[key] += int(value)

        merged, score_clamps = _merge_scores(
            candidates=candidates,
            llm_map=llm_map,
            beta=args.beta,
            strict_scores=args.strict_scores,
        )
        counters["retrieval_score_clamps"] += score_clamps

        output_row = {
            "input_id": input_id,
            "candidates": merged,
            "rerank_metadata": {
                "version": SCRIPT_VERSION,
                "prompt_version": PROMPT_VERSION,
                "mode": args.mode,
                "model_requested": model,
                "temperature": args.temperature,
                "seed": args.seed,
                "max_completion_tokens": args.max_tokens,
                "topk": topk,
                "beta": args.beta,
                "attempts_used": attempts_used,
                "attempt_seeds": attempt_seeds,
                "candidate_set_sha256": candidate_set_sha256,
                "response_contract": "candidate_keyed_required_object-v1",
                "response_schema_sha256": response_schema_sha256,
                "prompt_sha256": prompt_hash,
                "mes_status": mes_row.get("status") if mes_row else None,
                "complete_core_chain": (
                    bool(mes_row.get("complete_core_chain")) if mes_row else None
                ),
                "mes_sha256": mes_row.get("mes_sha256") if mes_row else None,
                "parse_stats": parse_stats,
                **response_meta,
            },
        }
        _append_jsonl(output_path, output_row)
        done_ids.add(input_id)
        counters["processed"] += 1
        manifest["last_completed_input_id"] = input_id
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)

    manifest["finished_utc"] = _utc_now()
    manifest["output_sha256"] = _sha256_file(str(output_path))
    manifest.pop("last_error", None)
    _write_json(manifest_path, manifest)
    print(json.dumps(counters, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
