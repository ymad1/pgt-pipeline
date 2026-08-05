"""Build a deterministic Minimal Explainable Subgraph (MES) from local attack graphs.

This implementation operates on the graph emitted by ``pgt.build_local_graph``.
It never invents structural or evidence edges: every node and edge in the MES is
copied from the corresponding local graph.  The algorithm selects one compact
Entry--Behavior--Impact explanation path (or the best explicitly marked partial
path), then computes a minimum evidence-node cover so that every retained
structural node remains traceable through an existing ``supported_by`` edge.

Selection objective (lexicographic and deterministic)
------------------------------------------------------
1. Prefer a complete directed Entry -> ... -> Behavior -> ... -> Impact path.
2. Maximise the proportion of adjacent path edges with shared evidence.
3. Maximise the mean number of shared-evidence identifiers per path edge.
4. Prefer a larger proportion of explicitly extracted structural relations.
5. Minimise the number of structural nodes and evidence nodes.
6. Use mean structural-edge score and mean node confidence only as tie-breaks.
7. Break remaining ties by the stable node-id sequence.

If no complete path exists, the same objective is applied after first maximising
coverage of the core roles {Entry, Behavior, Impact}.  The output is then marked
``status=partial`` and ``complete_core_chain=false``.  If no evidence-linked
structural node exists, an explicit empty MES record is emitted.

Input
-----
A directory containing one local-graph JSON file per CVE.  ``_summary.json`` and
other underscore-prefixed files are ignored.

Output
------
A JSONL file with one MES record per graph and a sidecar
``<output>.summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ALGORITHM_VERSION = "mes-v2.0.0"
CORE_TYPES: Tuple[str, ...] = ("Entry", "Behavior", "Impact")
STRUCTURAL_TYPES: Tuple[str, ...] = (
    "Precondition",
    "Entry",
    "VulnType",
    "Behavior",
    "Impact",
)
TYPE_RANK: Mapping[str, int] = {
    "Precondition": 0,
    "Entry": 1,
    "VulnType": 2,
    "Behavior": 3,
    "Impact": 4,
}
TRACE_EDGE_TYPES: Set[str] = {"mentions", "supported_by"}


@dataclass(frozen=True)
class PathCandidate:
    node_ids: Tuple[str, ...]
    edge_indices: Tuple[int, ...]
    evidence_cover: Tuple[str, ...]
    core_coverage: int
    complete: bool
    continuity_edges: int
    continuity_ratio: float
    shared_evidence_count: int
    mean_shared_evidence: float
    explicit_edge_count: int
    explicit_edge_ratio: float
    mean_edge_score: float
    mean_node_confidence: float

    @property
    def sort_key(self) -> Tuple[Any, ...]:
        """Ascending sort key; the first row is the selected candidate."""
        return (
            -int(self.complete),
            -self.core_coverage,
            -round(self.continuity_ratio, 12),
            -round(self.mean_shared_evidence, 12),
            -round(self.explicit_edge_ratio, 12),
            len(self.node_ids),
            len(self.evidence_cover),
            -round(self.mean_edge_score, 12),
            -round(self.mean_node_confidence, 12),
            self.node_ids,
        )


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _edge_endpoints(edge: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    src = edge.get("src", edge.get("source"))
    dst = edge.get("dst", edge.get("target"))
    return (
        str(src).strip() if isinstance(src, str) and src.strip() else None,
        str(dst).strip() if isinstance(dst, str) and dst.strip() else None,
    )


def _normalise_evidence_ids(value: Any) -> List[str]:
    if isinstance(value, str):
        raw_values: Iterable[Any] = re.split(r"[,;\s]+", value.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_values = value
    else:
        raw_values = []

    result: List[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        eid = raw.strip()
        if eid and eid not in result:
            result.append(eid)
    return result


def _evidence_sort_key(eid: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", eid)
    return (int(match.group(1)) if match else 10**9, eid)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _node_confidence(node: Mapping[str, Any]) -> float:
    value = node.get("confidence")
    if value is None:
        return 0.5
    return max(0.0, min(1.0, _safe_float(value, 0.5)))


def _is_explicit_edge(edge: Mapping[str, Any]) -> bool:
    origin = str(edge.get("origin", "")).lower()
    return origin in {
        "llm_extracted_relation",
        "explicit_relation",
        "extracted_relation",
        "llm_relation",
    }


def _structural_edge_score(edge: Mapping[str, Any]) -> float:
    for key in ("structural_score", "confidence", "score"):
        if key in edge:
            return max(0.0, min(1.0, _safe_float(edge.get(key), 0.0)))
    return 0.5 if _is_explicit_edge(edge) else 0.0


def _validate_graph(
    graph: Mapping[str, Any],
) -> Tuple[
    str,
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
]:
    input_id = str(graph.get("input_id", "")).strip()
    if not input_id:
        raise ValueError("Local graph is missing input_id")

    warnings: List[str] = []
    node_map: Dict[str, Dict[str, Any]] = {}
    for raw_node in graph.get("nodes") or []:
        if not isinstance(raw_node, Mapping):
            warnings.append("ignored_non_object_node")
            continue
        node_id = raw_node.get("id")
        node_type = raw_node.get("type")
        if not isinstance(node_id, str) or not node_id.strip():
            warnings.append("ignored_node_without_id")
            continue
        if not isinstance(node_type, str) or not node_type.strip():
            warnings.append(f"ignored_node_without_type:{node_id}")
            continue
        if node_id in node_map:
            warnings.append(f"duplicate_node_id:{node_id}")
            continue
        node_map[node_id] = dict(raw_node)

    edges: List[Dict[str, Any]] = []
    for raw_edge in graph.get("edges") or []:
        if not isinstance(raw_edge, Mapping):
            warnings.append("ignored_non_object_edge")
            continue
        src, dst = _edge_endpoints(raw_edge)
        if not src or not dst:
            warnings.append("ignored_edge_without_endpoints")
            continue
        if src not in node_map or dst not in node_map:
            warnings.append(f"ignored_dangling_edge:{src}->{dst}")
            continue
        edge = dict(raw_edge)
        # Canonical endpoint fields are added for internal processing only.
        edge["_src"] = src
        edge["_dst"] = dst
        edges.append(edge)

    return input_id, node_map, edges, warnings


def _build_evidence_support(
    node_map: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], int]]:
    """Return structural-node evidence support and edge lookup.

    Only existing ``supported_by`` edges to actual Evidence nodes count.  Node
    attributes are not used as a substitute because the MES must be a subgraph
    of the local graph rather than a reconstructed graph.
    """
    support: Dict[str, Set[str]] = {}
    support_edge_index: Dict[Tuple[str, str], int] = {}
    for idx, edge in enumerate(edges):
        if edge.get("type") != "supported_by":
            continue
        src = str(edge["_src"])
        dst = str(edge["_dst"])
        src_type = node_map[src].get("type")
        dst_node = node_map[dst]
        if src_type not in STRUCTURAL_TYPES or dst_node.get("type") != "Evidence":
            continue
        eid = dst_node.get("evidence_id")
        if not isinstance(eid, str) or not eid.strip():
            if dst.startswith("EVIDENCE::"):
                eid = dst.split("::", 1)[1]
            else:
                continue
        support.setdefault(src, set()).add(eid)
        support_edge_index[(src, eid)] = idx
    return support, support_edge_index


def _minimum_evidence_cover(
    node_ids: Sequence[str],
    support: Mapping[str, Set[str]],
    exact_limit: int,
) -> Optional[Tuple[str, ...]]:
    """Find the smallest evidence-id set that touches every selected node.

    Returns ``None`` when at least one structural node has no existing evidence
    support.  Exact enumeration is used for small evidence universes; a stable
    greedy cover is used above ``exact_limit``.
    """
    if not node_ids:
        return ()
    node_support: List[Set[str]] = []
    for node_id in node_ids:
        values = set(support.get(node_id, set()))
        if not values:
            return None
        node_support.append(values)

    universe = sorted(set().union(*node_support), key=_evidence_sort_key)
    if len(universe) <= exact_limit:
        for size in range(1, len(universe) + 1):
            for combo in itertools.combinations(universe, size):
                selected = set(combo)
                if all(selected & values for values in node_support):
                    return tuple(combo)

    uncovered = set(range(len(node_ids)))
    selected: List[str] = []
    while uncovered:
        options: List[Tuple[int, Tuple[int, str], str, Set[int]]] = []
        for eid in universe:
            if eid in selected:
                continue
            covered = {index for index in uncovered if eid in node_support[index]}
            options.append((-len(covered), _evidence_sort_key(eid), eid, covered))
        options.sort(key=lambda row: (row[0], row[1], row[2]))
        if not options or not options[0][3]:
            return None
        _, _, eid, covered = options[0]
        selected.append(eid)
        uncovered -= covered
    return tuple(selected)


def _build_structural_adjacency(
    node_map: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[Tuple[str, str], List[int]]]:
    adjacency: Dict[str, List[Tuple[str, int]]] = {}
    pair_edges: Dict[Tuple[str, str], List[int]] = {}
    for idx, edge in enumerate(edges):
        edge_type = str(edge.get("type", ""))
        if edge_type in TRACE_EDGE_TYPES:
            continue
        src = str(edge["_src"])
        dst = str(edge["_dst"])
        if node_map[src].get("type") not in STRUCTURAL_TYPES:
            continue
        if node_map[dst].get("type") not in STRUCTURAL_TYPES:
            continue
        # Reject backwards layer-rule edges. Explicit relations are retained as
        # long as they are directed and acyclic within the selected path.
        if not _is_explicit_edge(edge):
            src_rank = TYPE_RANK.get(str(node_map[src].get("type")), -1)
            dst_rank = TYPE_RANK.get(str(node_map[dst].get("type")), -1)
            if dst_rank <= src_rank:
                continue
        adjacency.setdefault(src, []).append((dst, idx))
        pair_edges.setdefault((src, dst), []).append(idx)

    for src in adjacency:
        adjacency[src].sort(key=lambda item: (item[0], item[1]))
    return adjacency, pair_edges


def _best_edge_index(edge_indices: Sequence[int], edges: Sequence[Mapping[str, Any]]) -> int:
    return sorted(
        edge_indices,
        key=lambda index: (
            -int(_is_explicit_edge(edges[index])),
            -_structural_edge_score(edges[index]),
            str(edges[index].get("type", "")),
            index,
        ),
    )[0]


def _enumerate_simple_paths(
    starts: Sequence[str],
    adjacency: Mapping[str, Sequence[Tuple[str, int]]],
    pair_edges: Mapping[Tuple[str, str], Sequence[int]],
    edges: Sequence[Mapping[str, Any]],
    max_nodes: int,
) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    paths: Dict[Tuple[str, ...], Tuple[int, ...]] = {}

    def dfs(node_path: Tuple[str, ...]) -> None:
        if node_path not in paths:
            edge_path: List[int] = []
            for src, dst in zip(node_path, node_path[1:]):
                edge_path.append(_best_edge_index(pair_edges[(src, dst)], edges))
            paths[node_path] = tuple(edge_path)
        if len(node_path) >= max_nodes:
            return
        current = node_path[-1]
        for dst, _ in adjacency.get(current, []):
            if dst in node_path:
                continue
            dfs(node_path + (dst,))

    for start in sorted(starts):
        dfs((start,))
    return sorted(paths.items(), key=lambda row: row[0])


def _path_types(node_ids: Sequence[str], node_map: Mapping[str, Mapping[str, Any]]) -> Tuple[str, ...]:
    return tuple(str(node_map[node_id].get("type", "")) for node_id in node_ids)


def _is_complete_path(types: Sequence[str]) -> bool:
    if not types or types[0] != "Entry" or types[-1] != "Impact":
        return False
    try:
        entry_index = types.index("Entry")
        behavior_index = types.index("Behavior")
        impact_index = len(types) - 1 - list(reversed(types)).index("Impact")
    except ValueError:
        return False
    return entry_index < behavior_index < impact_index


def _ordered_core_coverage(types: Sequence[str]) -> int:
    """Count core roles occurring in Entry -> Behavior -> Impact order."""
    cursor = -1
    coverage = 0
    for core_type in CORE_TYPES:
        try:
            cursor = list(types).index(core_type, cursor + 1)
        except ValueError:
            continue
        coverage += 1
    return coverage


def _make_candidate(
    node_ids: Tuple[str, ...],
    edge_indices: Tuple[int, ...],
    node_map: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    support: Mapping[str, Set[str]],
    exact_cover_limit: int,
) -> Optional[PathCandidate]:
    evidence_cover = _minimum_evidence_cover(node_ids, support, exact_cover_limit)
    if evidence_cover is None:
        return None

    continuity_edges = 0
    shared_count = 0
    explicit_count = 0
    edge_scores: List[float] = []
    for src, dst, edge_index in zip(node_ids, node_ids[1:], edge_indices):
        shared = set(support.get(src, set())) & set(support.get(dst, set()))
        if shared:
            continuity_edges += 1
            shared_count += len(shared)
        edge = edges[edge_index]
        explicit_count += int(_is_explicit_edge(edge))
        edge_scores.append(_structural_edge_score(edge))

    types = _path_types(node_ids, node_map)
    complete = _is_complete_path(types)
    core_coverage = len(set(types) & set(CORE_TYPES))
    # For partial paths, role order matters before all other quality terms.
    if not complete:
        core_coverage = _ordered_core_coverage(types)

    confidences = [_node_confidence(node_map[node_id]) for node_id in node_ids]
    return PathCandidate(
        node_ids=node_ids,
        edge_indices=edge_indices,
        evidence_cover=evidence_cover,
        core_coverage=core_coverage,
        complete=complete,
        continuity_edges=continuity_edges,
        continuity_ratio=(continuity_edges / len(edge_indices)) if edge_indices else 0.0,
        shared_evidence_count=shared_count,
        mean_shared_evidence=(shared_count / len(edge_indices)) if edge_indices else 0.0,
        explicit_edge_count=explicit_count,
        explicit_edge_ratio=(explicit_count / len(edge_indices)) if edge_indices else 0.0,
        mean_edge_score=(sum(edge_scores) / len(edge_scores)) if edge_scores else 0.0,
        mean_node_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
    )


def _select_primary_path(
    node_map: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    support: Mapping[str, Set[str]],
    max_path_nodes: int,
    exact_cover_limit: int,
) -> Tuple[Optional[PathCandidate], Dict[str, Any]]:
    structural_ids = sorted(
        node_id
        for node_id, node in node_map.items()
        if node.get("type") in STRUCTURAL_TYPES and support.get(node_id)
    )
    adjacency, pair_edges = _build_structural_adjacency(node_map, edges)
    enumerated = _enumerate_simple_paths(
        starts=structural_ids,
        adjacency=adjacency,
        pair_edges=pair_edges,
        edges=edges,
        max_nodes=max_path_nodes,
    )

    candidates: List[PathCandidate] = []
    for node_ids, edge_indices in enumerated:
        candidate = _make_candidate(
            node_ids=node_ids,
            edge_indices=edge_indices,
            node_map=node_map,
            edges=edges,
            support=support,
            exact_cover_limit=exact_cover_limit,
        )
        if candidate is not None:
            candidates.append(candidate)

    complete_candidates = [candidate for candidate in candidates if candidate.complete]
    pool = complete_candidates
    if not pool:
        # A one-node MES is allowed only as an explicitly marked partial result.
        pool = [candidate for candidate in candidates if candidate.core_coverage > 0]

    if not pool:
        return None, {
            "enumerated_path_count": len(enumerated),
            "eligible_candidate_count": len(candidates),
            "complete_candidate_count": 0,
            "selected_objective": None,
        }

    pool.sort(key=lambda candidate: candidate.sort_key)
    selected = pool[0]
    return selected, {
        "enumerated_path_count": len(enumerated),
        "eligible_candidate_count": len(candidates),
        "complete_candidate_count": len(complete_candidates),
        "selected_objective": {
            "complete": selected.complete,
            "core_coverage": selected.core_coverage,
            "continuity_edges": selected.continuity_edges,
            "continuity_ratio": round(selected.continuity_ratio, 8),
            "shared_evidence_count": selected.shared_evidence_count,
            "mean_shared_evidence": round(selected.mean_shared_evidence, 8),
            "explicit_edge_count": selected.explicit_edge_count,
            "explicit_edge_ratio": round(selected.explicit_edge_ratio, 8),
            "mean_edge_score": round(selected.mean_edge_score, 8),
            "mean_node_confidence": round(selected.mean_node_confidence, 8),
            "structural_node_count": len(selected.node_ids),
            "evidence_node_count": len(selected.evidence_cover),
            "stable_node_id_tiebreak": list(selected.node_ids),
        },
    }


def _select_supported_precondition(
    selected: PathCandidate,
    node_map: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    support: Mapping[str, Set[str]],
    exact_cover_limit: int,
) -> Tuple[Optional[str], Optional[int], Tuple[str, ...]]:
    """Optionally prepend one evidence-linked Precondition using an existing edge."""
    options: List[Tuple[Any, ...]] = []
    selected_set = set(selected.node_ids)
    for index, edge in enumerate(edges):
        if edge.get("type") in TRACE_EDGE_TYPES:
            continue
        src = str(edge["_src"])
        dst = str(edge["_dst"])
        if node_map[src].get("type") != "Precondition" or dst not in selected_set:
            continue
        if not support.get(src):
            continue
        extended_nodes = (src,) + selected.node_ids
        cover = _minimum_evidence_cover(extended_nodes, support, exact_cover_limit)
        if cover is None:
            continue
        shared = len(set(support[src]) & set(support[dst]))
        options.append(
            (
                -int(shared > 0),
                -shared,
                -int(_is_explicit_edge(edge)),
                -_structural_edge_score(edge),
                -_node_confidence(node_map[src]),
                len(cover),
                src,
                index,
                cover,
            )
        )
    if not options:
        return None, None, selected.evidence_cover
    options.sort(key=lambda row: row[:-1])
    best = options[0]
    return str(best[6]), int(best[7]), tuple(best[8])


def _clean_edge(edge: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in edge.items() if not key.startswith("_")}


def _find_mentions_edges(
    cve_id: Optional[str],
    structural_ids: Set[str],
    edges: Sequence[Mapping[str, Any]],
) -> List[int]:
    if not cve_id:
        return []
    result: List[int] = []
    for index, edge in enumerate(edges):
        if edge.get("type") != "mentions":
            continue
        if edge["_src"] == cve_id and edge["_dst"] in structural_ids:
            result.append(index)
    return result


def _build_mes_record(
    graph: Mapping[str, Any],
    max_path_nodes: int,
    exact_cover_limit: int,
    include_precondition: bool,
) -> Dict[str, Any]:
    input_id, node_map, edges, warnings = _validate_graph(graph)
    support, support_edge_index = _build_evidence_support(node_map, edges)
    selected, selection_trace = _select_primary_path(
        node_map=node_map,
        edges=edges,
        support=support,
        max_path_nodes=max_path_nodes,
        exact_cover_limit=exact_cover_limit,
    )

    if selected is None:
        payload = {
            "algorithm": ALGORITHM_VERSION,
            "input_id": input_id,
            "source_graph_version": graph.get("graph_version"),
            "status": "empty",
            "chain": [],
            "evidence_ids": [],
        }
        signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return {
            "input_id": input_id,
            "algorithm": ALGORITHM_VERSION,
            "source_graph_version": graph.get("graph_version"),
            "parameters": {
                "max_path_nodes": max_path_nodes,
                "exact_cover_limit": exact_cover_limit,
                "include_precondition": include_precondition,
                "selection": "deterministic_lexicographic_constrained_path",
            },
            "status": "empty",
            "complete_core_chain": False,
            "chain": [],
            "structural_node_ids": [],
            "evidence_ids": [],
            "nodes": [],
            "edges": [],
            "compact_text": "",
            "selection_trace": selection_trace,
            "warnings": warnings + ["no_evidence_linked_core_structural_path"],
            "mes_sha256": signature,
        }

    structural_ids = list(selected.node_ids)
    structural_edge_indices = list(selected.edge_indices)
    evidence_cover = selected.evidence_cover
    precondition_id: Optional[str] = None
    if include_precondition:
        precondition_id, precondition_edge_index, evidence_cover = _select_supported_precondition(
            selected=selected,
            node_map=node_map,
            edges=edges,
            support=support,
            exact_cover_limit=exact_cover_limit,
        )
        if precondition_id is not None and precondition_edge_index is not None:
            structural_ids.insert(0, precondition_id)
            structural_edge_indices.insert(0, precondition_edge_index)

    selected_structural_set = set(structural_ids)

    # Preserve source-graph node attributes exactly.  Structural nodes carry
    # their complete evidence_ids arrays, so the MES includes the traceability
    # closure for every selected structural node rather than pruning attributes.
    # ``evidence_cover`` remains the minimum cover used by path selection and is
    # recorded separately in selection_trace.
    output_evidence_ids = tuple(
        sorted(
            set().union(*(support.get(node_id, set()) for node_id in structural_ids)),
            key=_evidence_sort_key,
        )
    )
    selected_evidence_set = set(output_evidence_ids)
    evidence_node_ids: Dict[str, str] = {}
    for node_id, node in node_map.items():
        if node.get("type") != "Evidence":
            continue
        eid = node.get("evidence_id")
        if isinstance(eid, str) and eid in selected_evidence_set:
            evidence_node_ids[eid] = node_id

    missing_evidence_nodes = sorted(
        selected_evidence_set - set(evidence_node_ids), key=_evidence_sort_key
    )
    if missing_evidence_nodes:
        warnings.append("missing_evidence_nodes:" + ",".join(missing_evidence_nodes))

    cve_ids = sorted(
        node_id for node_id, node in node_map.items() if node.get("type") == "CVE"
    )
    cve_id = cve_ids[0] if cve_ids else None

    selected_node_ids: List[str] = []
    if cve_id:
        selected_node_ids.append(cve_id)
    selected_node_ids.extend(structural_ids)
    selected_node_ids.extend(
        evidence_node_ids[eid]
        for eid in sorted(evidence_node_ids, key=_evidence_sort_key)
    )

    selected_edge_indices: Set[int] = set(structural_edge_indices)
    selected_edge_indices.update(_find_mentions_edges(cve_id, selected_structural_set, edges))
    for node_id in structural_ids:
        for eid in output_evidence_ids:
            index = support_edge_index.get((node_id, eid))
            if index is not None:
                selected_edge_indices.add(index)

    nodes = [dict(node_map[node_id]) for node_id in selected_node_ids]
    selected_edges = [_clean_edge(edges[index]) for index in sorted(selected_edge_indices)]

    chain_types = [str(node_map[node_id].get("type", "")) for node_id in structural_ids]
    compact_parts = [
        f"{node_map[node_id].get('type')}[{str(node_map[node_id].get('text', '')).strip()}]"
        for node_id in structural_ids
    ]
    compact_text = " -> ".join(compact_parts)
    if evidence_cover:
        compact_text += " | evidence=" + ",".join(evidence_cover)

    status = "complete" if selected.complete else "partial"
    signature_payload = {
        "algorithm": ALGORITHM_VERSION,
        "input_id": input_id,
        "source_graph_version": graph.get("graph_version"),
        "status": status,
        "chain": structural_ids,
        "structural_edges": [
            {
                "src": edges[index]["_src"],
                "dst": edges[index]["_dst"],
                "type": edges[index].get("type"),
            }
            for index in structural_edge_indices
        ],
        "evidence_ids": list(output_evidence_ids),
        "minimum_evidence_cover": list(evidence_cover),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return {
        "input_id": input_id,
        "algorithm": ALGORITHM_VERSION,
        "source_graph_version": graph.get("graph_version"),
        "parameters": {
            "max_path_nodes": max_path_nodes,
            "exact_cover_limit": exact_cover_limit,
            "include_precondition": include_precondition,
            "selection": "deterministic_lexicographic_constrained_path",
            "minimum_evidence_cover": (
                "exact_enumeration_then_deterministic_greedy_above_limit"
            ),
            "subgraph_constraint": "all_nodes_and_edges_copied_from_local_graph",
            "traceability_closure": "all supported_by edges for selected structural nodes",
        },
        "status": status,
        "complete_core_chain": selected.complete,
        "chain": structural_ids,
        "chain_types": chain_types,
        "structural_node_ids": structural_ids,
        "evidence_ids": list(output_evidence_ids),
        "nodes": nodes,
        "edges": selected_edges,
        "compact_text": compact_text,
        "selection_trace": {
            **selection_trace,
            "precondition_added": precondition_id,
            "selected_structural_edge_count": len(structural_edge_indices),
            "minimum_evidence_cover": list(evidence_cover),
            "traceability_closure_evidence_ids": list(output_evidence_ids),
            "selected_traceability_edge_count": len(selected_edges) - len(structural_edge_indices),
        },
        "warnings": warnings,
        "mes_sha256": signature,
    }


def _graph_paths(graph_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.is_file() and not path.name.startswith("_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic MES records from local attack graph JSON files."
    )
    parser.add_argument("--graph_dir", required=True, help="Directory of local graph JSON files")
    parser.add_argument("--output", required=True, help="Output MES JSONL file")
    parser.add_argument(
        "--max_path_nodes",
        type=int,
        default=4,
        help="Maximum structural nodes in an enumerated primary path (default: 4)",
    )
    parser.add_argument(
        "--exact_cover_limit",
        type=int,
        default=20,
        help="Use exact evidence-cover enumeration up to this union size (default: 20)",
    )
    parser.add_argument(
        "--include_precondition",
        action="store_true",
        help="Prepend one supported Precondition connected by an existing structural edge",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output = Path(args.output)
    if not graph_dir.is_dir():
        raise NotADirectoryError(f"Graph directory not found: {graph_dir}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        output.unlink()

    paths = _graph_paths(graph_dir)
    complete = 0
    partial = 0
    empty = 0
    warnings_total = 0
    failed: List[Dict[str, str]] = []

    for path in paths:
        try:
            graph = _read_json(path)
            record = _build_mes_record(
                graph=graph,
                max_path_nodes=max(1, args.max_path_nodes),
                exact_cover_limit=max(1, args.exact_cover_limit),
                include_precondition=bool(args.include_precondition),
            )
            status = record["status"]
            complete += int(status == "complete")
            partial += int(status == "partial")
            empty += int(status == "empty")
            warnings_total += len(record.get("warnings") or [])
            _append_jsonl(output, record)
        except Exception as exc:  # keep the batch auditable instead of silently stopping
            failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "algorithm": ALGORITHM_VERSION,
        "graph_dir": str(graph_dir),
        "graphs_discovered": len(paths),
        "records_written": complete + partial + empty,
        "complete_core_chain": complete,
        "partial_mes": partial,
        "empty_mes": empty,
        "warnings": warnings_total,
        "failed": failed,
        "parameters": {
            "max_path_nodes": max(1, args.max_path_nodes),
            "exact_cover_limit": max(1, args.exact_cover_limit),
            "include_precondition": bool(args.include_precondition),
        },
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

    if failed:
        raise RuntimeError(f"MES construction failed for {len(failed)} graph file(s)")


if __name__ == "__main__":
    main()
