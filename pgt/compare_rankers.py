# -*- coding: utf-8 -*-
import json
import argparse
from collections import defaultdict

def to_parent(tid: str) -> str:
    if not tid:
        return tid
    return tid.split(".", 1)[0]

def load_labels(labels_jsonl: str, parent: bool) -> dict[str, set[str]]:
    mp: dict[str, set[str]] = {}
    with open(labels_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("input_id")
            labs = obj.get("labels", []) or []
            if not iid:
                continue
            s = set()
            for t in labs:
                if not isinstance(t, str):
                    continue
                t2 = to_parent(t) if parent else t
                if t2:
                    s.add(t2)
            mp[iid] = s
    return mp

def rank_candidates(cands: list[dict], key: str) -> list[str]:
    """
    key in {"score_text","score_graph","score_fused","llm_score","final_score"}
    NOTE: llm_score can be None for candidates outside rerank topk in your code.
    We treat None as -inf so they go to bottom.
    """
    def getv(c):
        v = c.get(key)
        if v is None:
            return float("-inf")
        try:
            return float(v)
        except Exception:
            return float("-inf")

    # stable sort: score desc, tie-breaker by score_fused desc
    return [
        c.get("technique_id")
        for c in sorted(
            cands,
            key=lambda x: (getv(x), float(x.get("score_fused") or 0.0)),
            reverse=True,
        )
        if isinstance(c.get("technique_id"), str) and c.get("technique_id")
    ]

def dedup_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def eval_one(reranked_jsonl: str, labels_map: dict[str, set[str]], parent: bool, ks: list[int]):
    # Methods to compare
    methods = {
        "text(score_text)": "score_text",
        "graph(score_graph)": "score_graph",
        "fused(score_fused)": "score_fused",
        "llm(llm_score)": "llm_score",
        "final(final_score)": "final_score",
    }

    # accumulators
    tot = 0
    stats = {m: {k: {"hit": 0.0, "p": 0.0, "r": 0.0} for k in ks} for m in methods}
    oracle = {k: 0.0 for k in ks}  # whether any gold appears in topk candidate set (upper bound given candidate universe)

    with open(reranked_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("input_id")
            if not iid:
                continue
            gold = labels_map.get(iid, set())
            if not gold:
                continue

            cands = row.get("candidates", []) or []
            if not isinstance(cands, list) or not cands:
                continue

            tot += 1

            # Precompute a “universe” set of candidate technique IDs (parented if needed)
            cand_univ = []
            for c in cands:
                tid = c.get("technique_id")
                if isinstance(tid, str) and tid:
                    tid2 = to_parent(tid) if parent else tid
                    if tid2:
                        cand_univ.append(tid2)
            cand_univ = set(cand_univ)

            for k in ks:
                oracle[k] += 1.0 if (len(gold.intersection(cand_univ)) > 0) else 0.0

            # Evaluate each ranking method
            for mname, key in methods.items():
                ranked = rank_candidates(cands, key=key)
                if parent:
                    ranked = [to_parent(t) for t in ranked]
                ranked = [t for t in ranked if t]  # drop empty
                ranked = dedup_keep_order(ranked)

                for k in ks:
                    topk = ranked[:k]
                    inter = len(gold.intersection(topk))
                    stats[mname][k]["hit"] += 1.0 if inter > 0 else 0.0
                    stats[mname][k]["p"] += inter / k
                    stats[mname][k]["r"] += inter / len(gold)

    return tot, stats, oracle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", required=True, help="e.g. runs\\...\\reranked_top20.jsonl")
    ap.add_argument("--labels", required=True, help="e.g. data\\...\\labels.jsonl")
    ap.add_argument("--parent", action="store_true", help="map sub-techniques to parent IDs (Txxxx)")
    ap.add_argument("--ks", default="1,3,5,10,20", help="comma-separated, default 1,3,5,10,20")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ks = sorted(set(ks))

    labels_map = load_labels(args.labels, parent=args.parent)
    tot, stats, oracle = eval_one(args.reranked, labels_map, parent=args.parent, ks=ks)

    print(f"N = {tot}  (parent={args.parent})")
    print("Oracle upper-bound (gold appears somewhere in candidate universe):")
    for k in ks:
        # oracle is actually "in universe", same for all k here, but keep output simple
        print(f"  Oracle={oracle[k]/tot:.4f} (candidate-universe contains any gold)")

    print("\n---- Rankings ----")
    for mname in stats:
        print(f"\n== {mname} ==")
        for k in ks:
            P = stats[mname][k]["p"] / tot
            R = stats[mname][k]["r"] / tot
            H = stats[mname][k]["hit"] / tot
            print(f"P@{k}={P:.4f}  R@{k}={R:.4f}  Hit@{k}={H:.4f}")

if __name__ == "__main__":
    main()
