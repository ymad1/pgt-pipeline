
import argparse
from typing import Dict, Any, List
from .io import read_jsonl, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    out: List[Dict[str, Any]] = []
    for row in read_jsonl(args.candidates):
        input_id = row["input_id"]
        picks = []
        for c in row.get("candidates", [])[:args.topk]:
            picks.append({
                "technique_id": c["technique_id"],
                "stage": "primary_impact",
                "confidence": float(c["score_fused"]),
                "evidence_ids": [],       # to be filled by constrained reasoning w/ evidence_pack
                "evidence_paths": [],     # to be filled
                "rationale": "stub"
            })
        out.append({"input_id": input_id, "predictions": picks})
    write_jsonl(args.output, out)

if __name__ == "__main__":
    main()
