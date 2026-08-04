
import argparse
from typing import Dict, Any, List
from tqdm import tqdm
from .io import read_jsonl, write_jsonl

def approx_tokens(text: str) -> int:
    # very rough: 1 token ~ 0.75 words for English; for mixed text, word count is still ok as a heuristic
    words = text.split()
    return int(len(words) / 0.75) if words else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--max_candidates", type=int, default=20)
    args = ap.parse_args()

    sent_map = {r["input_id"]: r for r in read_jsonl(args.sentences)}
    cand_map = {r["input_id"]: r for r in read_jsonl(args.candidates)}
    path_map = {r["input_id"]: r for r in read_jsonl(args.paths)}

    out_rows: List[Dict[str, Any]] = []
    for input_id in tqdm(list(sent_map.keys()), desc="evidence_pack"):
        srow = sent_map[input_id]
        crow = cand_map.get(input_id, {"candidates":[]})
        prow = path_map.get(input_id, {"paths":{}})

        # pick evidence sentences in order until budget
        picked = {}
        total = 0
        for eid, sent in srow["sentences"].items():
            t = approx_tokens(sent)
            if total + t > args.budget:
                break
            picked[eid] = sent
            total += t

        # keep top candidates and their first path
        candidates = crow["candidates"][:args.max_candidates]
        paths = {}
        for c in candidates:
            tid = c["technique_id"]
            paths[tid] = (prow.get("paths", {}) or {}).get(tid, [])[:1]

        out_rows.append({
            "input_id": input_id,
            "budget": args.budget,
            "picked_evidence": picked,
            "paths": paths,
            "candidates": candidates,
            "approx_used_tokens": total
        })

    write_jsonl(args.output, out_rows)

if __name__ == "__main__":
    main()
