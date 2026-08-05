"""Deterministically export an Enterprise ATT&CK STIX bundle.

The exporter creates two artifacts used by the CVE-to-ATT&CK pipeline:

1. ``attack_kg.json``
   A traceable knowledge graph containing active technique/tactic nodes,
   ``subtechnique_of`` edges, and ``belongs_to_tactic`` edges.
2. ``technique_text_index.jsonl``
   One canonical, retrieval-ready record per active ATT&CK technique.

The default export excludes revoked and deprecated techniques.  Every output is
sorted deterministically and accompanied by a manifest containing the source
collection version, processing rules, object counts, and SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXPORT_VERSION = "attack-stix-export-v2.0.0"
KG_SCHEMA_VERSION = "attack-knowledge-graph-v2.0.0"
INDEX_SCHEMA_VERSION = "attack-technique-index-v2.0.0"
TEXT_NORMALIZATION_VERSION = "attack-text-normalization-v2.0.0"

_TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_CITATION_RE = re.compile(r"\(Citation:\s*[^\)]*\)", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: Optional[int] = None) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value, indent=2))


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = bytearray()
    for row in rows:
        payload.extend(_json_bytes(dict(row), indent=None))
    _atomic_write_bytes(path, bytes(payload))


def _clean_text(value: Any) -> str:
    """Normalize ATT&CK markdown/HTML without inventing new content."""
    text = html.unescape(str(value or ""))
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _CITATION_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _external_reference(obj: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    matches: List[Tuple[str, Optional[str]]] = []
    for reference in obj.get("external_references", []) or []:
        if not isinstance(reference, Mapping):
            continue
        if reference.get("source_name") != "mitre-attack":
            continue
        external_id = str(reference.get("external_id") or "").strip()
        if external_id:
            matches.append((external_id, reference.get("url")))
    if not matches:
        return None, None
    unique_ids = sorted({item[0] for item in matches})
    if len(unique_ids) != 1:
        raise ValueError(
            f"STIX object {obj.get('id')!r} has multiple mitre-attack IDs: {unique_ids}"
        )
    urls = sorted({str(url) for ext_id, url in matches if ext_id == unique_ids[0] and url})
    return unique_ids[0], (urls[0] if urls else None)


def _sorted_strings(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()}, key=str.casefold)


def _tactics(obj: Mapping[str, Any]) -> List[str]:
    names = {
        str(phase.get("phase_name") or "").strip()
        for phase in obj.get("kill_chain_phases", []) or []
        if isinstance(phase, Mapping)
    }
    return sorted(name for name in names if name)


def _collection_metadata(objects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    collections = [
        obj
        for obj in objects
        if obj.get("type") == "x-mitre-collection"
        and str(obj.get("name") or "").strip().casefold() == "enterprise att&ck".casefold()
    ]
    if len(collections) != 1:
        raise ValueError(
            "Expected exactly one 'Enterprise ATT&CK' x-mitre-collection object, "
            f"found {len(collections)}"
        )
    collection = collections[0]
    version = str(collection.get("x_mitre_version") or "").strip()
    modified = str(collection.get("modified") or "").strip()
    if not version or not modified:
        raise ValueError("Enterprise ATT&CK collection is missing version or modified timestamp")
    return {
        "collection_stix_id": collection.get("id"),
        "collection_name": collection.get("name"),
        "collection_version": version,
        "collection_modified": modified,
        "collection_created": collection.get("created"),
    }


def _tactic_catalog(objects: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if obj.get("type") != "x-mitre-tactic":
            continue
        short_name = str(obj.get("x_mitre_shortname") or "").strip()
        if not short_name:
            continue
        if short_name in catalog:
            raise ValueError(f"Duplicate ATT&CK tactic short name: {short_name}")
        catalog[short_name] = {
            "id": f"TACTIC::{short_name}",
            "node_type": "tactic",
            "short_name": short_name,
            "name": str(obj.get("name") or short_name).strip(),
            "stix_id": obj.get("id"),
            "description": _clean_text(obj.get("description")),
            "x_mitre_version": obj.get("x_mitre_version"),
            "modified": obj.get("modified"),
        }
    return catalog


def _is_included(obj: Mapping[str, Any], *, include_revoked: bool, include_deprecated: bool) -> bool:
    if not include_revoked and bool(obj.get("revoked", False)):
        return False
    if not include_deprecated and bool(obj.get("x_mitre_deprecated", False)):
        return False
    return True


def _active_subtechnique_relationships(
    objects: Sequence[Mapping[str, Any]],
    included_stix_ids: set[str],
) -> Dict[str, str]:
    parent_by_child: Dict[str, str] = {}
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "subtechnique-of":
            continue
        if bool(rel.get("revoked", False)) or bool(rel.get("x_mitre_deprecated", False)):
            continue
        child = str(rel.get("source_ref") or "")
        parent = str(rel.get("target_ref") or "")
        if child not in included_stix_ids or parent not in included_stix_ids:
            continue
        previous = parent_by_child.get(child)
        if previous is not None and previous != parent:
            raise ValueError(
                f"Sub-technique {child} has multiple active parents: {previous}, {parent}"
            )
        parent_by_child[child] = parent
    return parent_by_child


def build_export(
    bundle: Mapping[str, Any],
    *,
    source_path: Path,
    include_revoked: bool = False,
    include_deprecated: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    if bundle.get("type") != "bundle" or not isinstance(bundle.get("objects"), list):
        raise ValueError("Input must be a STIX bundle object with an 'objects' list")

    objects: List[Mapping[str, Any]] = [
        obj for obj in bundle["objects"] if isinstance(obj, Mapping)
    ]
    collection = _collection_metadata(objects)
    tactics = _tactic_catalog(objects)

    all_attack_patterns = [obj for obj in objects if obj.get("type") == "attack-pattern"]
    included_objects: List[Mapping[str, Any]] = []
    excluded_no_id = 0
    excluded_non_technique_id = 0
    excluded_revoked = 0
    excluded_deprecated = 0

    for obj in all_attack_patterns:
        technique_id, _ = _external_reference(obj)
        if not technique_id:
            excluded_no_id += 1
            continue
        if not _TECHNIQUE_ID_RE.fullmatch(technique_id):
            excluded_non_technique_id += 1
            continue
        if not include_revoked and bool(obj.get("revoked", False)):
            excluded_revoked += 1
            continue
        if not include_deprecated and bool(obj.get("x_mitre_deprecated", False)):
            excluded_deprecated += 1
            continue
        included_objects.append(obj)

    by_stix: Dict[str, Mapping[str, Any]] = {}
    by_technique_id: Dict[str, Mapping[str, Any]] = {}
    for obj in included_objects:
        stix_id = str(obj.get("id") or "").strip()
        technique_id, _ = _external_reference(obj)
        if not stix_id or not technique_id:
            raise ValueError("Included ATT&CK technique is missing STIX or external ID")
        if stix_id in by_stix:
            raise ValueError(f"Duplicate included STIX ID: {stix_id}")
        if technique_id in by_technique_id:
            raise ValueError(f"Duplicate included ATT&CK technique ID: {technique_id}")
        by_stix[stix_id] = obj
        by_technique_id[technique_id] = obj

    parent_stix_by_child = _active_subtechnique_relationships(objects, set(by_stix))
    technique_rows: List[Dict[str, Any]] = []
    technique_nodes: List[Dict[str, Any]] = []
    subtechnique_edges: List[Dict[str, Any]] = []
    tactic_edges: List[Dict[str, Any]] = []

    for technique_id in sorted(by_technique_id):
        obj = by_technique_id[technique_id]
        stix_id = str(obj["id"])
        _, source_url = _external_reference(obj)
        is_subtechnique = bool(obj.get("x_mitre_is_subtechnique", False))
        parent_technique_id: Optional[str] = None
        parent_name: Optional[str] = None

        if is_subtechnique:
            parent_stix_id = parent_stix_by_child.get(stix_id)
            if not parent_stix_id:
                raise ValueError(
                    f"Active sub-technique {technique_id} ({stix_id}) has no active subtechnique-of relation"
                )
            parent_obj = by_stix[parent_stix_id]
            parent_technique_id, _ = _external_reference(parent_obj)
            parent_name = str(parent_obj.get("name") or "").strip() or None
            expected_parent = technique_id.split(".", 1)[0]
            if parent_technique_id != expected_parent:
                raise ValueError(
                    f"Parent mismatch for {technique_id}: relationship gives "
                    f"{parent_technique_id}, ID prefix gives {expected_parent}"
                )
            subtechnique_edges.append(
                {
                    "id": f"SUBTECHNIQUE::{technique_id}->{parent_technique_id}",
                    "edge_type": "subtechnique_of",
                    "src": technique_id,
                    "dst": parent_technique_id,
                }
            )
        elif stix_id in parent_stix_by_child:
            raise ValueError(f"Technique {technique_id} has a parent relation but is not marked as a sub-technique")

        name = _clean_text(obj.get("name"))
        description = _clean_text(obj.get("description"))
        if not name or not description:
            raise ValueError(f"Technique {technique_id} is missing a usable name or description")
        technique_tactics = _tactics(obj)
        for tactic in technique_tactics:
            if tactic not in tactics:
                raise ValueError(f"Technique {technique_id} references unknown tactic {tactic!r}")
            tactic_edges.append(
                {
                    "id": f"TACTIC-MEMBERSHIP::{technique_id}->{tactic}",
                    "edge_type": "belongs_to_tactic",
                    "src": technique_id,
                    "dst": f"TACTIC::{tactic}",
                }
            )

        platforms = _sorted_strings(obj.get("x_mitre_platforms", []))
        text = f"{name}\n{description}".strip()
        common = {
            "technique_id": technique_id,
            "stix_id": stix_id,
            "name": name,
            "description": description,
            "tactics": technique_tactics,
            "platforms": platforms,
            "is_subtechnique": is_subtechnique,
            "parent_technique_id": parent_technique_id,
            "parent_name": parent_name,
            "source_url": source_url,
            "x_mitre_version": obj.get("x_mitre_version"),
            "x_mitre_attack_spec_version": obj.get("x_mitre_attack_spec_version"),
            "created": obj.get("created"),
            "modified": obj.get("modified"),
            "revoked": bool(obj.get("revoked", False)),
            "deprecated": bool(obj.get("x_mitre_deprecated", False)),
        }
        technique_nodes.append({"id": technique_id, "node_type": "technique", **common})
        technique_rows.append(
            {
                **common,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "text_normalization_version": TEXT_NORMALIZATION_VERSION,
                "attack_collection_version": collection["collection_version"],
            }
        )

    tactic_nodes = [tactics[name] for name in sorted(tactics)]
    nodes = technique_nodes + tactic_nodes
    edges = sorted(
        subtechnique_edges + tactic_edges,
        key=lambda edge: (edge["edge_type"], edge["src"], edge["dst"]),
    )

    source_metadata = {
        "bundle_id": bundle.get("id"),
        "bundle_path": source_path.name,
        "bundle_sha256": sha256_file(source_path),
        **collection,
        "attack_spec_versions": sorted(
            {
                str(obj.get("x_mitre_attack_spec_version"))
                for obj in included_objects
                if obj.get("x_mitre_attack_spec_version")
            }
        ),
    }
    stats = {
        "bundle_objects": len(objects),
        "attack_pattern_objects": len(all_attack_patterns),
        "included_techniques": len(technique_rows),
        "included_parent_techniques": sum(not row["is_subtechnique"] for row in technique_rows),
        "included_subtechniques": sum(bool(row["is_subtechnique"]) for row in technique_rows),
        "included_tactics": len(tactic_nodes),
        "subtechnique_edges": len(subtechnique_edges),
        "tactic_membership_edges": len(tactic_edges),
        "excluded_missing_attack_id": excluded_no_id,
        "excluded_non_technique_id": excluded_non_technique_id,
        "excluded_revoked": excluded_revoked,
        "excluded_deprecated": excluded_deprecated,
    }
    rules = {
        "domain": "enterprise-attack",
        "include_revoked": include_revoked,
        "include_deprecated": include_deprecated,
        "external_reference_source": "mitre-attack",
        "technique_id_pattern": _TECHNIQUE_ID_RE.pattern,
        "description_normalization": [
            "HTML entity decoding",
            "Markdown link target removal with visible label retained",
            "ATT&CK citation marker removal",
            "HTML tag removal",
            "Unicode-preserving whitespace collapse",
        ],
        "retrieval_text_fields": ["name", "description"],
        "ordering": "technique_id ascending; graph edges by edge_type/src/dst",
        "dynamic_timestamp_in_output": False,
    }

    kg = {
        "schema_version": KG_SCHEMA_VERSION,
        "export_version": EXPORT_VERSION,
        "source": source_metadata,
        "rules": rules,
        "stats": stats,
        "nodes": nodes,
        "edges": edges,
    }
    manifest = {
        "export_version": EXPORT_VERSION,
        "kg_schema_version": KG_SCHEMA_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "text_normalization_version": TEXT_NORMALIZATION_VERSION,
        "source": source_metadata,
        "rules": rules,
        "stats": stats,
    }
    return kg, technique_rows, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stix_bundle",
        required=True,
        help="Enterprise ATT&CK STIX bundle JSON (for example enterprise-attack.json)",
    )
    parser.add_argument("--attack_kg", required=True, help="Output attack_kg.json")
    parser.add_argument("--tech_index", required=True, help="Output technique_text_index.jsonl")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest JSON; defaults to <tech_index>.manifest.json",
    )
    parser.add_argument(
        "--include_revoked",
        action="store_true",
        help="Include revoked techniques (excluded by default)",
    )
    parser.add_argument(
        "--include_deprecated",
        action="store_true",
        help="Include deprecated techniques (excluded by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = Path(args.stix_bundle).resolve()
    attack_kg_path = Path(args.attack_kg).resolve()
    tech_index_path = Path(args.tech_index).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else Path(str(tech_index_path) + ".manifest.json")
    )

    output_paths = [attack_kg_path, tech_index_path, manifest_path]
    duplicate_outputs = [str(path) for path in output_paths if output_paths.count(path) > 1]
    if duplicate_outputs:
        raise ValueError(f"Output paths must be distinct: {sorted(set(duplicate_outputs))}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output(s); use --overwrite: " + ", ".join(existing)
            )

    with source_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    kg, technique_rows, manifest = build_export(
        bundle,
        source_path=source_path,
        include_revoked=bool(args.include_revoked),
        include_deprecated=bool(args.include_deprecated),
    )

    # Write primary artifacts first. Their hashes are then embedded in the manifest.
    _atomic_write_json(attack_kg_path, kg)
    _atomic_write_jsonl(tech_index_path, technique_rows)
    manifest = {
        **manifest,
        "outputs": {
            "attack_kg": {
                "path": attack_kg_path.name,
                "sha256": sha256_file(attack_kg_path),
                "bytes": attack_kg_path.stat().st_size,
            },
            "technique_text_index": {
                "path": tech_index_path.name,
                "sha256": sha256_file(tech_index_path),
                "bytes": tech_index_path.stat().st_size,
                "records": len(technique_rows),
            },
        },
    }
    _atomic_write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "ok",
                "attack_collection_version": manifest["source"]["collection_version"],
                "techniques": manifest["stats"]["included_techniques"],
                "parent_techniques": manifest["stats"]["included_parent_techniques"],
                "subtechniques": manifest["stats"]["included_subtechniques"],
                "revoked_excluded": manifest["stats"]["excluded_revoked"],
                "deprecated_excluded": manifest["stats"]["excluded_deprecated"],
                "attack_kg": str(attack_kg_path),
                "tech_index": str(tech_index_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
