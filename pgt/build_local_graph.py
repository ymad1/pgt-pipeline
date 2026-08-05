# pgt/build_local_graph.py
"""Build deterministic, evidence-linked local attack graphs.

The graph is the source graph from which the Minimal Explainable Subgraph
(MES) must later be selected.  It therefore contains both traceability edges
(`mentions`, `supported_by`) and explicit candidate attack-chain edges such as
Entry -> Behavior -> Impact.

Input extraction schema
-----------------------
Each JSONL record is expected to contain ``input_id`` and zero or more of:
``preconditions``, ``entry``, ``vuln_type``, ``behaviors``, ``relations``, and
``impacts``.  Structural elements may contain ``evidence_ids`` and
``confidence``.

Output
------
One JSON graph per ``input_id`` with:
- CVE, Evidence, Precondition, Entry, VulnType, Behavior, and Impact nodes;
- ``mentions`` and ``supported_by`` traceability edges;
- deterministic candidate structural edges:
  Precondition -> Entry, Precondition -> Behavior, Entry -> VulnType,
  Entry -> Behavior, VulnType -> Behavior, and Behavior -> Impact;
- optional extractor-provided relations when their endpoints can be resolved;
- graph statistics and warnings for reproducibility.

Example
-------
python -m pgt.build_local_graph \
  --extraction runs/extraction.jsonl \
  --sentences data/processed/sentences.jsonl \
  --output_dir runs/local_graphs
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

GRAPH_VERSION = "local-attack-graph-v2.0.1"

# Candidate structural relations.  Every pair in an available adjacent layer
# is retained; the MES stage performs the actual evidence-constrained pruning.
STRUCTURAL_PAIR_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("Precondition", "Entry", "enables"),
    ("Precondition", "Behavior", "enables"),
    ("Entry", "VulnType", "characterized_by"),
    ("Entry", "Behavior", "enables"),
    ("VulnType", "Behavior", "enables"),
    ("Behavior", "Impact", "causes"),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EVIDENCE_ID_RE = re.compile(r"^E\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_no} in {path}")
            rows.append(row)
    return rows


def _write_json(path: Path, obj: Mapping[str, Any], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def load_sentences_lookup(sentences_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """Return ``input_id -> {evidence_id: text}`` from Step-2 JSONL output."""
    if not sentences_jsonl.exists():
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    for row in _read_jsonl(sentences_jsonl):
        input_id = str(row.get("input_id", "")).strip()
        sentences = row.get("sentences") or {}
        if not input_id or not isinstance(sentences, Mapping):
            continue
        lookup[input_id] = {
            str(eid).strip(): str(text)
            for eid, text in sentences.items()
            if str(eid).strip()
        }
    return lookup


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalise_evidence_ids(value: Any) -> List[str]:
    """Normalise evidence references while preserving their first occurrence."""
    raw_values: List[Any] = []
    if isinstance(value, str):
        raw_values.extend(re.split(r"[,;\s]+", value.strip()))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values.extend(value)
    elif value is not None:
        raw_values.append(value)

    out: List[str] = []
    seen: Set[str] = set()
    for raw in raw_values:
        eid = str(raw).strip().upper()
        if not eid or not _EVIDENCE_ID_RE.fullmatch(eid) or eid in seen:
            continue
        seen.add(eid)
        out.append(eid)
    return out


def _normalise_confidence(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return round(max(0.0, min(1.0, score)), 6)


def _join_nonempty(parts: Iterable[Any], separator: str = "; ") -> str:
    values: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text and text not in values:
            values.append(text)
    return separator.join(values)


def _canonical_node_text(node_type: str, item: Mapping[str, Any]) -> str:
    """Create a stable textual representation from the real extraction fields."""
    if node_type == "Precondition":
        return _join_nonempty((item.get("condition"), item.get("text")))
    if node_type == "Entry":
        return _join_nonempty((item.get("vector"), item.get("detail")))
    if node_type == "VulnType":
        return _join_nonempty((item.get("type"), item.get("subtype")), separator=": ")
    if node_type == "Behavior":
        parts = []
        if item.get("action"):
            parts.append(f"action={str(item.get('action')).strip()}")
        if item.get("target"):
            parts.append(f"target={str(item.get('target')).strip()}")
        if item.get("impact"):
            parts.append(f"impact={str(item.get('impact')).strip()}")
        return "; ".join(parts)
    if node_type == "Impact":
        return _join_nonempty((item.get("type"), item.get("detail")))
    return ""


def _token_set(text: Any) -> Set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or "")) if len(m.group(0)) > 1}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _is_alias_aka(text: str) -> bool:
    """Detect short alias-only evidence such as ``aka \"...\"``."""
    value = (text or "").strip().lower()
    if not value:
        return False
    return value.startswith("aka ") or (" aka " in value and len(value) <= 120)


def _collect_evidence_ids(extraction: Mapping[str, Any]) -> Set[str]:
    eids: Set[str] = set()
    for field in ("preconditions", "entry", "vuln_type", "behaviors", "impacts", "relations"):
        for item in _as_list(extraction.get(field)):
            if isinstance(item, Mapping):
                eids.update(_normalise_evidence_ids(item.get("evidence_ids")))
    return eids


def _evidence_sort_key(eid: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", eid)
    return (int(match.group(1)) if match else 10**9, eid)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _dedupe_nodes(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id not in seen:
            seen.add(node_id)
            out.append(node)
    return out


def _edge_priority(edge: Mapping[str, Any]) -> int:
    origin = str(edge.get("origin", ""))
    if origin == "extractor_relation":
        return 3
    if origin == "deterministic_layer_rule":
        return 2
    return 1


def _dedupe_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate edges, preferring explicit extractor relations."""
    chosen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        edge_type = edge.get("type")
        if not all(isinstance(x, str) and x for x in (src, dst, edge_type)):
            continue
        key = (src, dst, edge_type)
        if key not in chosen:
            chosen[key] = edge
            order.append(key)
        elif _edge_priority(edge) > _edge_priority(chosen[key]):
            chosen[key] = edge
    return [chosen[key] for key in order]


def _filtered_node_evidence_ids(
    node_type: str,
    evidence_ids: Sequence[str],
    evidence_text: Mapping[str, str],
    filter_aka_for_behavior_impact: bool,
) -> List[str]:
    normalised = list(dict.fromkeys(evidence_ids))
    if filter_aka_for_behavior_impact and node_type in {"Behavior", "Impact"}:
        non_alias = [eid for eid in normalised if not _is_alias_aka(evidence_text.get(eid, ""))]
        if non_alias:
            return non_alias
    return normalised


def _structural_edge(
    src_node: Mapping[str, Any],
    dst_node: Mapping[str, Any],
    relation_type: str,
) -> Dict[str, Any]:
    src_evidence = set(_normalise_evidence_ids(src_node.get("evidence_ids")))
    dst_evidence = set(_normalise_evidence_ids(dst_node.get("evidence_ids")))
    shared_evidence = sorted(src_evidence & dst_evidence, key=_evidence_sort_key)
    evidence_jaccard = _jaccard(src_evidence, dst_evidence)
    lexical_jaccard = _jaccard(_token_set(src_node.get("text")), _token_set(dst_node.get("text")))

    confidences = [
        confidence
        for confidence in (
            _normalise_confidence(src_node.get("confidence")),
            _normalise_confidence(dst_node.get("confidence")),
        )
        if confidence is not None
    ]
    confidence_mean = sum(confidences) / len(confidences) if confidences else 0.5

    # This score is only a deterministic tie-break for later MES selection; it
    # is not a learned probability or a factual-faithfulness score.
    score = round(
        0.55 * evidence_jaccard + 0.25 * lexical_jaccard + 0.20 * confidence_mean,
        6,
    )
    return {
        "src": str(src_node["id"]),
        "dst": str(dst_node["id"]),
        "type": relation_type,
        "origin": "deterministic_layer_rule",
        "shared_evidence_ids": shared_evidence,
        "evidence_jaccard": round(evidence_jaccard, 6),
        "lexical_jaccard": round(lexical_jaccard, 6),
        "confidence_mean": round(confidence_mean, 6),
        "structural_score": score,
    }


def _resolve_node_ref(
    ref: Any,
    node_ids: Set[str],
    aliases: Mapping[str, str],
) -> Optional[str]:
    if ref is None:
        return None
    if isinstance(ref, Mapping):
        node_type = str(ref.get("type") or ref.get("node_type") or "").strip()
        index = ref.get("index") or ref.get("id")
        if node_type and index is not None:
            ref = f"{node_type}:{index}"
        else:
            ref = ref.get("ref") or ref.get("node")
    value = str(ref or "").strip()
    if not value:
        return None
    if value in node_ids:
        return value
    return aliases.get(value.upper())


def _validate_graph(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[str]:
    warnings: List[str] = []
    node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in node_ids or dst not in node_ids:
            warnings.append(f"dangling_edge:{src}->{dst}:{edge.get('type')}")

    for node in nodes:
        if node.get("type") in {"Precondition", "Entry", "VulnType", "Behavior", "Impact"}:
            evidence_ids = _normalise_evidence_ids(node.get("evidence_ids"))
            if not evidence_ids:
                warnings.append(f"structural_node_without_evidence:{node.get('id')}")
    return sorted(set(warnings))


# ---------------------------------------------------------------------------
# Build one graph
# ---------------------------------------------------------------------------

def build_local_graph_for_one(
    extraction: Dict[str, Any],
    sentences_lookup: Mapping[str, Mapping[str, str]],
    filter_aka_for_behavior_impact: bool = True,
) -> Dict[str, Any]:
    input_id = str(extraction.get("input_id", "")).strip()
    if not input_id:
        raise ValueError("Extraction record missing input_id")

    sent_map = {
        str(eid).strip().upper(): str(text)
        for eid, text in (sentences_lookup.get(input_id, {}) or {}).items()
        if str(eid).strip()
    }

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    nodes_by_type: Dict[str, List[Dict[str, Any]]] = {
        node_type: []
        for node_type in ("Precondition", "Entry", "VulnType", "Behavior", "Impact")
    }

    cve_node_id = f"CVE::{input_id}"
    nodes.append({"id": cve_node_id, "type": "CVE", "text": input_id})

    all_eids = sorted(_collect_evidence_ids(extraction), key=_evidence_sort_key)
    evidence_node_ids: Dict[str, str] = {}
    for eid in all_eids:
        evidence_node_id = f"EVIDENCE::{eid}"
        evidence_node_ids[eid] = evidence_node_id
        evidence_node: Dict[str, Any] = {
            "id": evidence_node_id,
            "type": "Evidence",
            "evidence_id": eid,
            "text": sent_map.get(eid, ""),
        }
        nodes.append(evidence_node)

    def ensure_evidence_node(eid: str) -> str:
        if eid not in evidence_node_ids:
            evidence_node_ids[eid] = f"EVIDENCE::{eid}"
            nodes.append(
                {
                    "id": evidence_node_ids[eid],
                    "type": "Evidence",
                    "evidence_id": eid,
                    "text": sent_map.get(eid, ""),
                }
            )
        return evidence_node_ids[eid]

    aliases: Dict[str, str] = {}

    def register_aliases(node_id: str, node_type: str, index: int) -> None:
        short_prefix = {
            "Precondition": "P",
            "Entry": "EN",
            "VulnType": "VT",
            "Behavior": "B",
            "Impact": "I",
        }[node_type]
        for alias in (
            node_id,
            f"{short_prefix}{index}",
            f"{node_type}:{index}",
            f"{node_type}_{index}",
            f"{node_type}{index}",
        ):
            aliases[alias.upper()] = node_id

    def add_structural_node(
        node_type: str,
        index: int,
        item: Mapping[str, Any],
        node_id: str,
        fields: Sequence[str],
    ) -> None:
        raw_evidence_ids = _normalise_evidence_ids(item.get("evidence_ids"))
        evidence_ids = _filtered_node_evidence_ids(
            node_type=node_type,
            evidence_ids=raw_evidence_ids,
            evidence_text=sent_map,
            filter_aka_for_behavior_impact=filter_aka_for_behavior_impact,
        )
        node: Dict[str, Any] = {
            "id": node_id,
            "type": node_type,
            "text": _canonical_node_text(node_type, item),
            "evidence_ids": evidence_ids,
        }
        confidence = _normalise_confidence(item.get("confidence"))
        if confidence is not None:
            node["confidence"] = confidence
        for field in fields:
            if field in item:
                node[field] = item.get(field)

        nodes.append(node)
        nodes_by_type[node_type].append(node)
        register_aliases(node_id, node_type, index)
        edges.append({"src": cve_node_id, "dst": node_id, "type": "mentions", "origin": "traceability"})
        for eid in evidence_ids:
            edges.append(
                {
                    "src": node_id,
                    "dst": ensure_evidence_node(eid),
                    "type": "supported_by",
                    "origin": "traceability",
                }
            )

    for index, item in enumerate(_as_list(extraction.get("preconditions")), start=1):
        if isinstance(item, Mapping):
            add_structural_node(
                "Precondition", index, item, f"PRECONDITION::P{index}",
                ("condition", "confidence"),
            )

    for index, item in enumerate(_as_list(extraction.get("entry")), start=1):
        if isinstance(item, Mapping):
            add_structural_node(
                "Entry", index, item, f"ENTRY::EN{index}",
                ("vector", "detail", "confidence"),
            )

    for index, item in enumerate(_as_list(extraction.get("vuln_type")), start=1):
        if isinstance(item, Mapping):
            # ``type`` is reserved for the graph node category. Preserve the
            # extracted vulnerability label under ``vuln_type`` so it cannot
            # overwrite ``type=VulnType`` in the output node.
            vuln_item = dict(item)
            if "type" in vuln_item:
                vuln_item["vuln_type"] = vuln_item.get("type")
            add_structural_node(
                "VulnType", index, vuln_item, f"VULNTYPE::VT{index}",
                ("vuln_type", "subtype", "confidence"),
            )

    for index, item in enumerate(_as_list(extraction.get("behaviors")), start=1):
        if isinstance(item, Mapping):
            add_structural_node(
                "Behavior", index, item, f"BEHAVIOR::B{index}",
                ("action", "target", "impact", "confidence"),
            )

    for index, item in enumerate(_as_list(extraction.get("impacts")), start=1):
        if isinstance(item, Mapping):
            # Keep the graph node type as Impact and store the extraction label
            # under impact_type, avoiding the previous type-field collision.
            impact_item = dict(item)
            if "type" in impact_item:
                impact_item["impact_type"] = impact_item.get("type")
            add_structural_node(
                "Impact", index, impact_item, f"IMPACT::I{index}",
                ("impact_type", "detail", "confidence"),
            )

    # Preserve extractor-provided relations first so deduplication prefers them.
    node_ids_now = {str(node["id"]) for node in nodes}
    unresolved_relations: List[Dict[str, Any]] = []
    for relation_index, relation in enumerate(_as_list(extraction.get("relations")), start=1):
        if not isinstance(relation, Mapping):
            continue
        src_ref = relation.get("src", relation.get("source"))
        dst_ref = relation.get("dst", relation.get("target"))
        src_id = _resolve_node_ref(src_ref, node_ids_now, aliases)
        dst_id = _resolve_node_ref(dst_ref, node_ids_now, aliases)
        relation_type = str(relation.get("type") or relation.get("relation") or "related_to").strip()
        if src_id and dst_id and relation_type:
            edge: Dict[str, Any] = {
                "src": src_id,
                "dst": dst_id,
                "type": relation_type,
                "origin": "extractor_relation",
            }
            evidence_ids = _normalise_evidence_ids(relation.get("evidence_ids"))
            if evidence_ids:
                edge["evidence_ids"] = evidence_ids
            confidence = _normalise_confidence(relation.get("confidence"))
            if confidence is not None:
                edge["confidence"] = confidence
            edges.append(edge)
        else:
            unresolved_relations.append(
                {
                    "index": relation_index,
                    "src": src_ref,
                    "dst": dst_ref,
                    "type": relation_type,
                }
            )

    # Add deterministic candidate chain edges.  These edges are deliberately
    # not pruned here: pruning belongs to the MES algorithm.
    for src_type, dst_type, relation_type in STRUCTURAL_PAIR_RULES:
        for src_node in nodes_by_type[src_type]:
            for dst_node in nodes_by_type[dst_type]:
                edges.append(_structural_edge(src_node, dst_node, relation_type))

    nodes = _dedupe_nodes(nodes)
    edges = _dedupe_edges(edges)
    warnings = _validate_graph(nodes, edges)
    warnings.extend(
        f"unresolved_extractor_relation:{item['index']}:{item['src']}->{item['dst']}"
        for item in unresolved_relations
    )
    warnings = sorted(set(warnings))

    structural_edge_types = {rule[2] for rule in STRUCTURAL_PAIR_RULES}
    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": {
            node_type: sum(node.get("type") == node_type for node in nodes)
            for node_type in ("CVE", "Evidence", "Precondition", "Entry", "VulnType", "Behavior", "Impact")
        },
        "traceability_edge_count": sum(edge.get("type") in {"mentions", "supported_by"} for edge in edges),
        "structural_edge_count": sum(
            edge.get("origin") in {"deterministic_layer_rule", "extractor_relation"}
            and edge.get("type") not in {"mentions", "supported_by"}
            for edge in edges
        ),
        "complete_entry_behavior_impact_layers": all(nodes_by_type[t] for t in ("Entry", "Behavior", "Impact")),
        "unresolved_relation_count": len(unresolved_relations),
        "structural_relation_types": sorted(
            {str(edge.get("type")) for edge in edges if edge.get("type") in structural_edge_types}
        ),
    }

    return {
        "input_id": input_id,
        "graph_version": GRAPH_VERSION,
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic local attack graphs.")
    parser.add_argument("--extraction", required=True, help="Input extraction JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for one graph JSON per input_id")
    parser.add_argument(
        "--sentences",
        default="data/processed/sentences.jsonl",
        help="Optional sentences.jsonl used to attach evidence text",
    )
    parser.add_argument(
        "--no_filter_aka",
        action="store_true",
        help='Keep alias-only evidence such as `aka "..."` for Behavior/Impact nodes',
    )
    args = parser.parse_args()

    extraction_path = Path(args.extraction)
    output_dir = Path(args.output_dir)
    sentences_path = Path(args.sentences)

    if not extraction_path.exists():
        raise FileNotFoundError(f"Extraction file not found: {extraction_path}")

    sentences_lookup = load_sentences_lookup(sentences_path)
    rows = _read_jsonl(extraction_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    complete = 0
    warnings_total = 0
    for row in rows:
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            continue
        graph = build_local_graph_for_one(
            extraction=row,
            sentences_lookup=sentences_lookup,
            filter_aka_for_behavior_impact=not args.no_filter_aka,
        )
        _write_json(output_dir / f"{input_id}.json", graph)
        written += 1
        complete += int(graph["stats"]["complete_entry_behavior_impact_layers"])
        warnings_total += len(graph["warnings"])

    summary = {
        "graph_version": GRAPH_VERSION,
        "records_read": len(rows),
        "graphs_written": written,
        "complete_entry_behavior_impact_layers": complete,
        "warnings": warnings_total,
        "sentences_file_found": sentences_path.exists(),
    }
    _write_json(output_dir / "_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
