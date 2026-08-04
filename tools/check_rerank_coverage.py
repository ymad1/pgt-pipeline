import json

p = "runs/rerank/dev/reranked.jsonl"
with open(p, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        cid = row.get("input_id")
        cands = row.get("candidates", [])
        got = [c for c in cands if c.get("llm_score") is not None]
        print(f"{cid}: llm_scored={len(got)}/{len(cands)}")
        if row.get("_rerank_error"):
            print("  _rerank_error:", row["_rerank_error"])
