
import argparse, json, os
from typing import Dict, Any, List
from tqdm import tqdm
from .io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--local_graph_dir", required=True)
    ap.add_argument("--reasoning", required=True, help="reasoning.jsonl (can be stub)")
    ap.add_argument("--evidence_pack", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sent_map = {r["input_id"]: r for r in read_jsonl(args.sentences)}
    ep_map = {r["input_id"]: r for r in read_jsonl(args.evidence_pack)}
    # reasoning is optional; if absent, we still validate evidence_pack itself
    reasoning_map = {r.get("input_id"): r for r in read_jsonl(args.reasoning)} if os.path.exists(args.reasoning) else {}

    out: List[Dict[str, Any]] = []
    for input_id, srow in tqdm(sent_map.items(), desc="verify"):
        valid_eids = set(srow["sentences"].keys())

        # load graph
        gpath = os.path.join(args.local_graph_dir, f"{input_id}.json")
        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)
        node_ids = {n["id"] for n in g.get("nodes", [])}

        ep = ep_map.get(input_id, {})
        picked_eids = set((ep.get("picked_evidence") or {}).keys())

        evr = len([e for e in picked_eids if e in valid_eids]) / max(1, len(picked_eids))
        # PCS: for each path, check nodes exist (except TECHNIQUE::* which may not be in local graph)
        pcs_checks = []
        for tid, plist in (ep.get("paths") or {}).items():
            for p in plist:
                nodes = p.get("nodes", [])
                ok = True
                for n in nodes:
                    if n.startswith("TECHNIQUE::"):
                        continue
                    if n not in node_ids:
                        ok = False
                        break
                pcs_checks.append(1.0 if ok else 0.0)
        pcs = sum(pcs_checks)/max(1,len(pcs_checks))

        issues = []
        if evr < 1.0:
            issues.append("invalid_evidence_id_in_pack")
        if pcs < 1.0:
            issues.append("path_not_closed")

        out.append({
            "input_id": input_id,
            "EVR": evr,
            "PCS": pcs,
            "issues": issues
        })

    write_jsonl(args.output, out)

if __name__ == "__main__":
    main()
