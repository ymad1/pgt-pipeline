
import argparse, json
from typing import Dict, Any, List
from .io import write_jsonl

def _get_attack_id(obj: Dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in ("mitre-attack", "mitre-attack-ics", "mitre-mobile-attack"):
            return ref.get("external_id")
    return None

def _tactics(obj: Dict[str, Any]) -> List[str]:
    # In ATT&CK STIX, kill_chain_phases.phase_name is often used for tactic shortnames
    t = []
    for ph in obj.get("kill_chain_phases", []):
        phase = ph.get("phase_name")
        if phase:
            t.append(phase)
    return sorted(set(t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stix_bundle", required=True, help="STIX 2.x bundle JSON (e.g., enterprise-attack.json)")
    ap.add_argument("--attack_kg", required=True, help="output attack_kg.json")
    ap.add_argument("--tech_index", required=True, help="output technique_text_index.jsonl")
    args = ap.parse_args()

    with open(args.stix_bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objs = bundle.get("objects", [])
    techniques: List[Dict[str, Any]] = []
    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        tid = _get_attack_id(obj)
        if not tid or not tid.startswith("T"):
            continue
        techniques.append({
            "technique_id": tid,
            "stix_id": obj.get("id"),
            "name": obj.get("name"),
            "description": (obj.get("description") or "").strip(),
            "tactics": _tactics(obj),
            "platforms": obj.get("x_mitre_platforms", []) or [],
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
        })

    # write attack_kg.json (simple adjacency-free node list for MVP)
    kg = {
        "nodes": [{"id": t["technique_id"], **t} for t in techniques],
        "edges": []  # extend later (belongs_to tactic, has_keyword, etc.)
    }
    with open(args.attack_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    # write text index jsonl
    rows = []
    for t in techniques:
        rows.append({
            "technique_id": t["technique_id"],
            "text": f"{t['name']}\n{t['description']}".strip()
        })
    write_jsonl(args.tech_index, rows)

    print(f"Exported {len(techniques)} techniques")

if __name__ == "__main__":
    main()
