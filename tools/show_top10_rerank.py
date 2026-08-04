# tools/show_top10_rerank.py
import json

path = "runs/rerank/dev/reranked.jsonl"
with open(path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        print("=" * 80)
        print("input_id:", row.get("input_id"))
        cands = row.get("candidates", [])[:10]
        for i, c in enumerate(cands, 1):
            print(
                f"{i:02d}. {c['technique_id']}"
                f"  final={c.get('final_score', 0):.4f}"
                f"  fused={c.get('score_fused', 0):.4f}"
                f"  llm={c.get('llm_score') if c.get('llm_score') is not None else 'NA'}"
            )
            r = c.get("reason")
            if r:
                print("    reason:", r)
            e = c.get("evidence_ids")
            if e:
                print("    evidence_ids:", e)
