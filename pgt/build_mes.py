"""Deterministic Minimal Explainable Subgraph (MES) construction.

The algorithm consumes sentence-level evidence and typed extraction records and
writes one compact, evidence-linked subgraph per CVE.  It is deliberately
conservative: a structural element is eligible only when it has at least one
valid evidence identifier, either explicitly supplied by the extractor or
recovered by a unique exact substring match.

Input files
-----------
sentences.jsonl
    {"input_id": str, "sentences": {"E1": "...", ...}}
extraction.jsonl
    {"input_id": str, "preconditions": [...], "entry": [...],
     "vuln_type": [...], "behaviors": [...], "impacts": [...]}

Output
------
mes.jsonl
    Deterministic MES records including selected nodes, typed chain edges,
    supported_by edges, a minimum evidence cover, and a selection trace.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .io import read_jsonl

ALGORITHM_VERSION = "mes-v1.0.0"
TYPE_ORDER: Tuple[str, ...] = (
    "precondition",
    "entry",
    "vuln_type",
    "behavior",
    "impact",
)
CORE_TYPES: Tuple[str, ...] = ("entry", "behavior", "impact")
FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "precondition": ("preconditions", "precondition"),
    "entry": ("entry", "entries"),
    "vuln_type": ("vuln_type", "vulnerability_type", "vuln_types"),
    "behavior": ("behaviors", "behavior"),
    "impact": ("impacts", "impact"),
}
TEXT_KEYS: Tuple[str, ...] = (
    "text",
    "value",
    "span",
    "content",
    "name",
    "description",
    "label",
)
EVIDENCE_KEYS: Tuple[str, ...] = (
    "evidence_ids",
    "evidence_id",
    "evidence",
    "support",
    "supports",
    "source_ids",
    "sources",
    "refs",
)
CONFIDENCE_KEYS: Tuple[str, ...] = ("confidence", "score", "probability", "prob")


@dataclass(frozen=True)
class Element:
    node_id: str
    element_type: str
    text: str
    evidence_ids: Tuple[str, ...]
    confidence: Optional[float]
    evidence_link_method: str
    original_index: int

    @property
    def support_key(self) -> Tuple[int, int, float, int, str]:
        """Deterministic descending support key.

        Explicit evidence links outrank recovered substring links; more valid
        evidence identifiers outrank fewer; a supplied confidence is used only
        as a tertiary tie-break.  Original order and node id break remaining
        ties reproducibly.
        """
        explicit = 1 if self.evidence_link_method == "explicit" else 0
        conf = self.confidence if self.confidence is not None else -1.0
        return (explicit, len(self.evidence_ids), conf, -self.original_index, self.node_id)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _index_by_input_id(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        input_id = row.get("input_id")
        if isinstance(input_id, str) and input_id:
            out[input_id] = row
    return out


def _normalise_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9_.:/ -]+", "", value)
    return value.strip()


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _flatten_evidence_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        for token in re.split(r"[,;\s]+", value.strip()):
            if token:
                yield token
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and re.fullmatch(r"E\d+", key, flags=re.I):
                yield key
            yield from _flatten_evidence_values(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from _flatten_evidence_values(nested)


def _explicit_evidence_ids(item: Any, valid_ids: Sequence[str]) -> Tuple[str, ...]:
    if not isinstance(item, Mapping):
        return ()
    valid_lookup = {eid.upper(): eid for eid in valid_ids}
    found: List[str] = []
    for key in EVIDENCE_KEYS:
        if key not in item:
            continue
        for raw in _flatten_evidence_values(item.get(key)):
            canonical = valid_lookup.get(raw.upper())
            if canonical and canonical not in found:
                found.append(canonical)
    return tuple(found)


def _unique_substring_evidence(text: str, sentences: Mapping[str, str]) -> Tuple[str, ...]:
    """Recover one evidence id only when a unique exact normalised substring matches."""
    norm = _normalise_text(text)
    if len(norm) < 4:
        return ()
    matches: List[str] = []
    for eid, sentence in sentences.items():
        sent_norm = _normalise_text(str(sentence))
        if norm in sent_norm or (len(sent_norm) >= 4 and sent_norm in norm):
            matches.append(eid)
    return (matches[0],) if len(matches) == 1 else ()


def _extract_confidence(item: Any) -> Optional[float]:
    if not isinstance(item, Mapping):
        return None
    for key in CONFIDENCE_KEYS:
        value = item.get(key)
        try:
            if value is not None:
                return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return None


def _get_field_values(extraction: Mapping[str, Any], element_type: str) -> List[Any]:
    for field in FIELD_ALIASES[element_type]:
        value = extraction.get(field)
        if isinstance(value, list):
            return value
        if value is not None:
            return [value]
    return []


def _normalise_elements(
    extraction: Mapping[str, Any],
    sentences: Mapping[str, str],
    max_per_type: int,
) -> Tuple[Dict[str, List[Element]], Dict[str, Any]]:
    valid_ids = list(sentences.keys())
    by_type: Dict[str, List[Element]] = {t: [] for t in TYPE_ORDER}
    dropped: List[Dict[str, Any]] = []

    for element_type in TYPE_ORDER:
        raw_values = _get_field_values(extraction, element_type)
        candidates: List[Element] = []
        for idx, item in enumerate(raw_values):
            text = _extract_text(item)
            if not text:
                dropped.append({"type": element_type, "index": idx, "reason": "missing_text"})
                continue
            evidence_ids = _explicit_evidence_ids(item, valid_ids)
            method = "explicit"
            if not evidence_ids:
                evidence_ids = _unique_substring_evidence(text, sentences)
                method = "unique_substring"
            if not evidence_ids:
                dropped.append(
                    {
                        "type": element_type,
                        "index": idx,
                        "text": text,
                        "reason": "no_valid_evidence_link",
                    }
                )
                continue
            node_id = f"{element_type.upper()}::{idx}"
            candidates.append(
                Element(
                    node_id=node_id,
                    element_type=element_type,
                    text=text,
                    evidence_ids=tuple(evidence_ids),
                    confidence=_extract_confidence(item),
                    evidence_link_method=method,
                    original_index=idx,
                )
            )

        candidates.sort(key=lambda x: x.support_key, reverse=True)
        by_type[element_type] = candidates[: max(1, max_per_type)]
        for pruned in candidates[max(1, max_per_type) :]:
            dropped.append(
                {
                    "type": element_type,
                    "index": pruned.original_index,
                    "text": pruned.text,
                    "reason": "outside_max_per_type",
                }
            )

    trace = {
        "eligible_counts": {t: len(by_type[t]) for t in TYPE_ORDER},
        "dropped": dropped,
    }
    return by_type, trace


def _evidence_sort_key(eid: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", eid)
    return (int(match.group(1)) if match else 10**9, eid)


def _minimum_evidence_cover(elements: Sequence[Element], exact_limit: int) -> Tuple[str, ...]:
    """Select the smallest evidence-id set touching every structural node.

    Exact enumeration is used when the evidence union is at most ``exact_limit``;
    otherwise a deterministic greedy set cover is used.  Ties prefer earlier
    evidence identifiers (E1 before E2, etc.).
    """
    if not elements:
        return ()
    universe = sorted(
        {eid for element in elements for eid in element.evidence_ids},
        key=_evidence_sort_key,
    )
    node_sets = [set(element.evidence_ids) for element in elements]

    if len(universe) <= exact_limit:
        for size in range(1, len(universe) + 1):
            for combo in itertools.combinations(universe, size):
                selected = set(combo)
                if all(selected.intersection(support) for support in node_sets):
                    return tuple(combo)

    uncovered = set(range(len(elements)))
    selected: List[str] = []
    while uncovered:
        ranked: List[Tuple[int, Tuple[int, str], str, set[int]]] = []
        for eid in universe:
            if eid in selected:
                continue
            covered = {idx for idx in uncovered if eid in node_sets[idx]}
            ranked.append((len(covered), _evidence_sort_key(eid), eid, covered))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        if not ranked or ranked[0][0] == 0:
            break
        _, _, eid, covered = ranked[0]
        selected.append(eid)
        uncovered -= covered
    return tuple(selected)


def _chain_quality(elements: Sequence[Element], evidence_cover: Sequence[str]) -> Tuple[Any, ...]:
    explicit_count = sum(e.evidence_link_method == "explicit" for e in elements)
    evidence_links = sum(len(e.evidence_ids) for e in elements)
    conf_sum = sum(e.confidence for e in elements if e.confidence is not None)
    adjacent_overlap = 0
    for left, right in zip(elements, elements[1:]):
        adjacent_overlap += len(set(left.evidence_ids).intersection(right.evidence_ids))
    # Larger is better for the first four terms; smaller evidence cover and
    # lexical ids are preferred in the final tie-breaks.
    return (
        explicit_count,
        evidence_links,
        adjacent_overlap,
        round(conf_sum, 8),
        -len(evidence_cover),
        tuple(e.node_id for e in elements),
    )


def _select_chain(
    by_type: Mapping[str, Sequence[Element]],
    exact_limit: int,
) -> Tuple[List[Element], Tuple[str, ...], Dict[str, Any]]:
    required_types = [t for t in CORE_TYPES if by_type.get(t)]
    if not required_types:
        # Fallback to any evidence-linked structural type. This makes failure
        # explicit without fabricating an Entry/Behavior/Impact chain.
        required_types = [t for t in TYPE_ORDER if by_type.get(t)]

    if not required_types:
        return [], (), {"required_types": [], "complete_core_chain": False, "alternatives": 0}

    alternatives = 1
    for t in required_types:
        alternatives *= len(by_type[t])

    best_elements: List[Element] = []
    best_cover: Tuple[str, ...] = ()
    best_quality: Optional[Tuple[Any, ...]] = None
    for combo in itertools.product(*(by_type[t] for t in required_types)):
        elements = list(combo)
        cover = _minimum_evidence_cover(elements, exact_limit=exact_limit)
        quality = _chain_quality(elements, cover)
        if best_quality is None or quality > best_quality:
            best_elements = elements
            best_cover = cover
            best_quality = quality

    return best_elements, best_cover, {
        "required_types": required_types,
        "complete_core_chain": all(bool(by_type.get(t)) for t in CORE_TYPES),
        "alternatives": alternatives,
        "selected_quality": list(best_quality or ()),
    }


def _build_record(
    input_id: str,
    sentences: Mapping[str, str],
    extraction: Mapping[str, Any],
    max_per_type: int,
    exact_cover_limit: int,
) -> Dict[str, Any]:
    by_type, normalisation_trace = _normalise_elements(
        extraction=extraction,
        sentences=sentences,
        max_per_type=max_per_type,
    )
    chain, evidence_cover, chain_trace = _select_chain(by_type, exact_limit=exact_cover_limit)

    nodes: List[Dict[str, Any]] = [
        {"id": f"CVE::{input_id}", "type": "cve", "text": input_id}
    ]
    for element in chain:
        nodes.append(
            {
                "id": element.node_id,
                "type": element.element_type,
                "text": element.text,
                "evidence_ids": list(element.evidence_ids),
                "confidence": element.confidence,
                "evidence_link_method": element.evidence_link_method,
            }
        )
    for eid in evidence_cover:
        nodes.append(
            {
                "id": f"EVIDENCE::{eid}",
                "type": "evidence",
                "evidence_id": eid,
                "text": str(sentences.get(eid, "")),
            }
        )

    edges: List[Dict[str, str]] = []
    cve_id = f"CVE::{input_id}"
    if chain:
        edges.append({"source": cve_id, "target": chain[0].node_id, "type": "contains"})
    for left, right in zip(chain, chain[1:]):
        edges.append({"source": left.node_id, "target": right.node_id, "type": "typed_sequence"})
    selected_evidence = set(evidence_cover)
    for element in chain:
        for eid in element.evidence_ids:
            if eid in selected_evidence:
                edges.append(
                    {
                        "source": element.node_id,
                        "target": f"EVIDENCE::{eid}",
                        "type": "supported_by",
                    }
                )

    compact_parts = [f"{e.element_type.upper()}[{e.text}]" for e in chain]
    compact_text = " -> ".join(compact_parts)
    if evidence_cover:
        compact_text += " | evidence=" + ",".join(evidence_cover)

    signature_payload = {
        "algorithm": ALGORITHM_VERSION,
        "input_id": input_id,
        "chain": [e.node_id for e in chain],
        "evidence_ids": list(evidence_cover),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return {
        "input_id": input_id,
        "algorithm": ALGORITHM_VERSION,
        "parameters": {
            "max_per_type": max_per_type,
            "exact_cover_limit": exact_cover_limit,
            "evidence_recovery": "explicit_ids_then_unique_exact_normalised_substring",
            "chain_types": list(CORE_TYPES),
        },
        "complete_core_chain": bool(chain_trace.get("complete_core_chain")),
        "chain": [e.node_id for e in chain],
        "evidence_ids": list(evidence_cover),
        "nodes": nodes,
        "edges": edges,
        "compact_text": compact_text,
        "selection_trace": {
            **normalisation_trace,
            **chain_trace,
        },
        "mes_sha256": signature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic evidence-linked MES records.")
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_per_type", type=int, default=3)
    parser.add_argument("--exact_cover_limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        output.unlink()

    sentences_map = _index_by_input_id(args.sentences)
    extraction_map = _index_by_input_id(args.extraction)
    all_ids = sorted(set(sentences_map).intersection(extraction_map))

    complete = 0
    empty = 0
    for input_id in all_ids:
        sentences = sentences_map[input_id].get("sentences") or {}
        if not isinstance(sentences, Mapping):
            sentences = {}
        record = _build_record(
            input_id=input_id,
            sentences=sentences,
            extraction=extraction_map[input_id],
            max_per_type=max(1, args.max_per_type),
            exact_cover_limit=max(1, args.exact_cover_limit),
        )
        complete += int(record["complete_core_chain"])
        empty += int(not record["chain"])
        _append_jsonl(str(output), record)

    summary = {
        "algorithm": ALGORITHM_VERSION,
        "records": len(all_ids),
        "complete_core_chain": complete,
        "empty_mes": empty,
        "missing_sentences": len(set(extraction_map) - set(sentences_map)),
        "missing_extraction": len(set(sentences_map) - set(extraction_map)),
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
