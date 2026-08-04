
import argparse, json, os
from typing import Dict, Any, List
from tqdm import tqdm
from .io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--local_graph_dir", required=True)
    ap.add_argument("--output", required=True, help="paths.jsonl")
    ap.add_argument("--top_paths_per_candidate", type=int, default=1)
    args = ap.parse_args()

    out = []
    for row in tqdm(list(read_jsonl(args.candidates)), desc="paths"):
        input_id = row["input_id"]
        gpath = os.path.join(args.local_graph_dir, f"{input_id}.json")
        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        # MVP: create a trivial "path" that links CVE->Behavior->Evidence for each candidate
        # Later: map Behavior->Technique via keyword edges or learned linker.
        # Here we just reuse the best behavior nodes as explanation anchors.
        behavior_nodes = [n["id"] for n in g.get("nodes", []) if n.get("type")=="Behavior"][:3]
        evidence_nodes = [n["id"] for n in g.get("nodes", []) if n.get("type")=="Evidence"][:3]
        cve_node = f"CVE::{input_id}"

        cand_paths = {}
        for c in row["candidates"]:
            tid = c["technique_id"]
            pid = f"{tid}::P1"
            cand_paths[tid] = [{
                "path_id": pid,
                "nodes": [cve_node] + behavior_nodes + evidence_nodes + [f"TECHNIQUE::{tid}"],
                "edges": ["mentions"]*max(0,len(behavior_nodes)) + ["supported_by"]*max(0,len(evidence_nodes)) + ["linked_to"],
                "score": c["score_fused"],
            }][:args.top_paths_per_candidate]

        out.append({"input_id": input_id, "paths": cand_paths})

    write_jsonl(args.output, out)

if __name__ == "__main__":
    main()
