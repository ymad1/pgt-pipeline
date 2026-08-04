# pgt/build_local_graph.py
"""
Step 4: Build a local attribution subgraph (one JSON graph per input_id).

This version:
- Includes nodes for: CVE, Behavior, Precondition, Entry, VulnType, Impact, Evidence
- Ensures ALL evidence_ids referenced anywhere in extraction are present as Evidence nodes
- Adds evidence_text onto Evidence nodes by looking up Step2 sentences.jsonl (optional, default on)
- Fixes Impact node schema:
    node["type"] == "Impact"
    node["impact_type"] carries the original extraction impacts[i]["type"]
- Filters low-signal alias evidence like: aka "..." for Behavior/Impact supported_by edges by default
  (will keep aka evidence if it is the ONLY evidence for that node to avoid breaking connectivity)

Run:
  python -m pgt.build_local_graph --extraction runs/extract/dev/extraction.jsonl --output_dir runs/graphs/dev/local_graphs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------
# JSONL helpers
# -----------------------

def _read_jsonl(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def _write_json(path: Path, obj: Dict[str, Any], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------
# Sentences lookup (Step2 output)
# -----------------------

def load_sentences_lookup(sentences_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns: input_id -> { "E1": "...", "E2": "...", ... }
    """
    if not sentences_jsonl.exists():
        return {}

    rows = _read_jsonl(sentences_jsonl)
    lookup: Dict[str, Dict[str, str]] = {}
    for r in rows:
        input_id = str(r.get("input_id", "")).strip()
        sents = r.get("sentences") or {}
        if input_id and isinstance(sents, dict):
            lookup[input_id] = {str(k): str(v) for k, v in sents.items()}
    return lookup


# -----------------------
# Graph builder helpers
# -----------------------

def _collect_evidence_ids(extraction: Dict[str, Any]) -> Set[str]:
    """
    Collect all evidence_ids referenced anywhere in the extraction schema.
    """
    eids: Set[str] = set()

    def add_from_items(items: Any) -> None:
        if not isinstance(items, list):
            return
        for it in items:
            if isinstance(it, dict):
                ids = it.get("evidence_ids")
                if isinstance(ids, list):
                    for x in ids:
                        if x is None:
                            continue
                        sx = str(x).strip()
                        if sx:
                            eids.add(sx)

    add_from_items(extraction.get("preconditions"))
    add_from_items(extraction.get("entry"))
    add_from_items(extraction.get("vuln_type"))
    add_from_items(extraction.get("behaviors"))
    add_from_items(extraction.get("impacts"))

    # relations may or may not carry evidence_ids
    rels = extraction.get("relations")
    if isinstance(rels, list):
        for rel in rels:
            if isinstance(rel, dict):
                ids = rel.get("evidence_ids")
                if isinstance(ids, list):
                    for x in ids:
                        if x is None:
                            continue
                        sx = str(x).strip()
                        if sx:
                            eids.add(sx)

    return eids


def _dedupe_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for n in nodes:
        nid = n.get("id")
        if not isinstance(nid, str):
            continue
        if nid in seen:
            continue
        seen.add(nid)
        out.append(n)
    return out


def _dedupe_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        et = e.get("type")
        if not (isinstance(src, str) and isinstance(dst, str) and isinstance(et, str)):
            continue
        key = (src, dst, et)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _is_alias_aka(text: str) -> bool:
    """
    Heuristic: sentences like `aka "Windows SMB Remote Code Execution Vulnerability."`
    typically only provide a name/alias, not supporting evidence.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    # starts with aka or contains aka early
    if t.startswith("aka "):
        return True
    # also allow patterns like: ', aka "..."'
    if " aka " in t and len(t) <= 120:
        return True
    return False


# -----------------------
# Build graph for one sample
# -----------------------

def build_local_graph_for_one(
    extraction: Dict[str, Any],
    sentences_lookup: Dict[str, Dict[str, str]],
    filter_aka_for_behavior_impact: bool = True,
) -> Dict[str, Any]:
    input_id = str(extraction.get("input_id", "")).strip()
    if not input_id:
        raise ValueError("Extraction record missing input_id")

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # CVE node
    cve_node_id = f"CVE::{input_id}"
    nodes.append({"id": cve_node_id, "type": "CVE"})

    # Evidence nodes (ensure complete)
    all_eids = sorted(_collect_evidence_ids(extraction))
    sent_map = sentences_lookup.get(input_id, {})

    evidence_node_ids: Dict[str, str] = {}
    evidence_text: Dict[str, str] = {}

    for eid in all_eids:
        nid = f"EVIDENCE::{eid}"
        evidence_node_ids[eid] = nid
        text = sent_map.get(eid, "")
        evidence_text[eid] = text

        ev_node = {
            "id": nid,
            "type": "Evidence",
            "evidence_id": eid,
        }
        if text:
            ev_node["text"] = text
        nodes.append(ev_node)

    # helper: add supported_by edges with optional aka filtering
    def add_supported_by(src_node_id: str, src_node_type: str, evidence_ids: Any) -> None:
        if not isinstance(evidence_ids, list) or not evidence_ids:
            return

        # Normalize list
        norm: List[str] = []
        for x in evidence_ids:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                norm.append(sx)
        if not norm:
            return

        # If filtering enabled, drop aka evidences for Behavior/Impact when there are other evidences.
        filtered = norm
        if filter_aka_for_behavior_impact and src_node_type in ("Behavior", "Impact"):
            non_aka = [eid for eid in norm if not _is_alias_aka(evidence_text.get(eid, ""))]
            # Only apply the drop if it won't make the evidence empty
            if non_aka:
                filtered = non_aka

        for eid in filtered:
            ev_nid = evidence_node_ids.get(eid)
            if not ev_nid:
                # Create on the fly if missing
                ev_nid = f"EVIDENCE::{eid}"
                evidence_node_ids[eid] = ev_nid
                txt = sent_map.get(eid, "")
                evidence_text[eid] = txt
                ev_node = {"id": ev_nid, "type": "Evidence", "evidence_id": eid}
                if txt:
                    ev_node["text"] = txt
                nodes.append(ev_node)

            edges.append({"src": src_node_id, "dst": ev_nid, "type": "supported_by"})

    # Preconditions
    preconditions = extraction.get("preconditions") or []
    if isinstance(preconditions, list):
        for i, pc in enumerate(preconditions, start=1):
            if not isinstance(pc, dict):
                continue
            nid = f"PRECONDITION::P{i}"
            node = {"id": nid, "type": "Precondition"}
            for k in ["condition", "text", "confidence"]:
                if k in pc:
                    node[k] = pc[k]
            nodes.append(node)
            edges.append({"src": cve_node_id, "dst": nid, "type": "mentions"})
            add_supported_by(nid, "Precondition", pc.get("evidence_ids"))

    # Entry (avoid collision with Evidence E1 by using EN prefix)
    entries = extraction.get("entry") or []
    if isinstance(entries, list):
        for i, en in enumerate(entries, start=1):
            if not isinstance(en, dict):
                continue
            nid = f"ENTRY::EN{i}"
            node = {"id": nid, "type": "Entry"}
            for k in ["vector", "detail", "confidence"]:
                if k in en:
                    node[k] = en[k]
            nodes.append(node)
            edges.append({"src": cve_node_id, "dst": nid, "type": "mentions"})
            add_supported_by(nid, "Entry", en.get("evidence_ids"))

    # Vuln types
    vtypes = extraction.get("vuln_type") or []
    if isinstance(vtypes, list):
        for i, vt in enumerate(vtypes, start=1):
            if not isinstance(vt, dict):
                continue
            nid = f"VULNTYPE::VT{i}"
            node = {"id": nid, "type": "VulnType"}
            for k in ["type", "subtype", "confidence"]:
                if k in vt:
                    node[k] = vt[k]
            nodes.append(node)
            edges.append({"src": cve_node_id, "dst": nid, "type": "mentions"})
            add_supported_by(nid, "VulnType", vt.get("evidence_ids"))

    # Behaviors
    behaviors = extraction.get("behaviors") or []
    if isinstance(behaviors, list):
        for i, b in enumerate(behaviors, start=1):
            if not isinstance(b, dict):
                continue
            nid = f"BEHAVIOR::B{i}"
            node = {"id": nid, "type": "Behavior"}
            for k in ["action", "target", "impact", "confidence"]:
                if k in b:
                    node[k] = b[k]
            nodes.append(node)
            edges.append({"src": cve_node_id, "dst": nid, "type": "mentions"})
            add_supported_by(nid, "Behavior", b.get("evidence_ids"))

    # Impacts (FIX: node.type must be "Impact"; store original extraction impact type in impact_type)
    imps = extraction.get("impacts") or []
    if isinstance(imps, list):
        for i, imp in enumerate(imps, start=1):
            if not isinstance(imp, dict):
                continue
            nid = f"IMPACT::I{i}"
            node = {"id": nid, "type": "Impact"}

            # extraction may have {"type": "..."} -> store as impact_type
            if "type" in imp:
                node["impact_type"] = imp.get("type")
            if "detail" in imp:
                node["detail"] = imp.get("detail")
            if "confidence" in imp:
                node["confidence"] = imp.get("confidence")

            nodes.append(node)
            edges.append({"src": cve_node_id, "dst": nid, "type": "mentions"})
            add_supported_by(nid, "Impact", imp.get("evidence_ids"))

    # Relations (optional passthrough for behavior-behavior edges)
    rels = extraction.get("relations") or []
    if isinstance(rels, list):
        for rel in rels:
            if not isinstance(rel, dict):
                continue

            rel_type = rel.get("type") or rel.get("relation") or "related_to"
            src = rel.get("src")
            dst = rel.get("dst")

            def map_behavior_ref(x: Any) -> Optional[str]:
                if x is None:
                    return None
                s = str(x).strip()
                if not s:
                    return None
                if s.startswith("BEHAVIOR::"):
                    return s
                if s.startswith("B") and s[1:].isdigit():
                    return f"BEHAVIOR::{s}"
                return None

            src_id = map_behavior_ref(src)
            dst_id = map_behavior_ref(dst)
            if src_id and dst_id:
                edges.append({"src": src_id, "dst": dst_id, "type": str(rel_type)})

    nodes = _dedupe_nodes(nodes)
    edges = _dedupe_edges(edges)
    return {"nodes": nodes, "edges": edges}


# -----------------------
# Main
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PGT Step4: build local attribution graphs.")
    parser.add_argument("--extraction", required=True, help="Input extraction JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for per-input graphs")
    parser.add_argument(
        "--sentences",
        default="data/processed/sentences.jsonl",
        help="Optional Step2 sentences.jsonl for evidence text lookup",
    )
    parser.add_argument(
        "--no_filter_aka",
        action="store_true",
        help='Disable filtering of alias evidence like `aka "..."` for Behavior/Impact supported_by',
    )

    args = parser.parse_args()
    extraction_path = Path(args.extraction)
    output_dir = Path(args.output_dir)
    sentences_path = Path(args.sentences)

    sentences_lookup = load_sentences_lookup(sentences_path)

    rows = _read_jsonl(extraction_path)
    for r in rows:
        input_id = str(r.get("input_id", "")).strip()
        if not input_id:
            continue
        g = build_local_graph_for_one(
            r,
            sentences_lookup,
            filter_aka_for_behavior_impact=not args.no_filter_aka,
        )
        out_path = output_dir / f"{input_id}.json"
        _write_json(out_path, g)


if __name__ == "__main__":
    main()
