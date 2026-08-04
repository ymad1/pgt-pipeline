# compare_on_oracle_subset.py
# -*- coding: utf-8 -*-
import json
import argparse
from collections import Counter

def to_parent(tid: str) -> str:
    if not tid:
        return tid
    return tid.split(".", 1)[0]

def load_preds(path: str, parent: bool):
    """
    Supports:
      {"input_id":..., "pred":[...], "gold":[...]}
      {"input_id":..., "pred":[...], "labels":[...]}
      {"input_id":..., "predictions":[...], "gold"/"labels":[...]}
    """
    m = {}
    missing_pred = 0
    missing_gold = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("input_id")
            if not pid:
                continue

            pred = obj.get("pred")
            if pred is None:
                pred = obj.get("predictions")
            if pred is None:
                pred = []
                missing_pred += 1

            gold = obj.get("gold")
            if gold is None:
                gold = obj.get("labels")
            if gold is None:
                gold = []
                missing_gold += 1

            if parent:
                pred = [to_parent(x) for x in pred]
                gold = [to_parent(x) for x in gold]

            m[pid] = {"pred": pred, "gold": gold}

    return m, {"missing_pred_field": missing_pred, "missing_gold_field": missing_gold}

def load_oracle_ids_from_reranked(reranked_jsonl: str, topk: int, parent: bool):
    """
    Oracle subset: gold intersects candidates[:topk]
    reranked_top20.jsonl typically has:
      {"input_id":..., "candidates":[{"technique_id":...}, ...]}
    It may NOT contain gold/labels; if so, we cannot define subset here.
    """
    oracle_ids = set()
    stats = Counter()
    with open(reranked_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("input_id")
            if not pid:
                continue

            cand_rows = obj.get("candidates") or []
            cand_ids = [c.get("technique_id") for c in cand_rows[:topk] if isinstance(c.get("technique_id"), str)]

            gold = obj.get("gold") or obj.get("labels")  # might be absent
            if gold is None:
                stats["reranked_missing_gold_field"] += 1
                continue

            if parent:
                cand_ids = [to_parent(x) for x in cand_ids]
                gold = [to_parent(x) for x in gold]

            gold_set = set(gold)
            if not gold_set:
                stats["reranked_gold_empty"] += 1
                continue

            stats["reranked_with_gold"] += 1
            if gold_set.intersection(cand_ids):
                oracle_ids.add(pid)
                stats["oracle_hits"] += 1
            else:
                stats["oracle_miss"] += 1

    return oracle_ids, stats

def eval_on_subset(pred_map, subset_ids, ks):
    tot = 0
    p = {k: 0.0 for k in ks}
    r = {k: 0.0 for k in ks}
    hit = {k: 0.0 for k in ks}

    for pid in subset_ids:
        row = pred_map.get(pid)
        if not row:
            continue
        pred = row["pred"]
        gold = set(row["gold"])
        if not gold:
            continue
        tot += 1
        for k in ks:
            topk = pred[:k]
            inter = len(gold.intersection(topk))
            p[k] += inter / k
            r[k] += inter / len(gold)
            hit[k] += 1.0 if inter > 0 else 0.0

    if tot == 0:
        return 0, None, None, None
    return tot, {k: p[k]/tot for k in ks}, {k: r[k]/tot for k in ks}, {k: hit[k]/tot for k in ks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", required=True, help="reranked_top20.jsonl (with candidates, ideally also gold/labels)")
    ap.add_argument("--before", required=True, help="predictions_top20.jsonl (baseline)")
    ap.add_argument("--after", required=True, help="predictions_reranked_top20_parented.jsonl (reranked preds)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--parent", action="store_true")
    ap.add_argument("--labels", default=None,
                    help="OPTIONAL: labels.jsonl with {input_id, labels:[...]} to define oracle subset if reranked file has no gold")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    # Load predictions first (also gives us gold availability)
    before, before_diag = load_preds(args.before, parent=args.parent)
    after,  after_diag  = load_preds(args.after,  parent=args.parent)

    # Define oracle subset
    subset_ids, oracle_stats = load_oracle_ids_from_reranked(args.reranked, args.topk, args.parent)

    # If reranked has no gold field, we can define oracle subset using labels + reranked candidates
    if len(subset_ids) == 0 and oracle_stats.get("reranked_missing_gold_field", 0) > 0 and args.labels:
        # Build label map
        label_map = {}
        with open(args.labels, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("input_id")
                labs = obj.get("labels") or obj.get("gold") or []
                if args.parent:
                    labs = [to_parent(x) for x in labs]
                label_map[pid] = set(labs)

        # Re-scan reranked candidates to build oracle subset with external labels
        subset_ids = set()
        stats2 = Counter()
        with open(args.reranked, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("input_id")
                if not pid:
                    continue
                gold_set = label_map.get(pid)
                if not gold_set:
                    continue
                cand_rows = obj.get("candidates") or []
                cand_ids = [c.get("technique_id") for c in cand_rows[:args.topk] if isinstance(c.get("technique_id"), str)]
                if args.parent:
                    cand_ids = [to_parent(x) for x in cand_ids]
                if gold_set.intersection(cand_ids):
                    subset_ids.add(pid)
                    stats2["oracle_hits"] += 1
                else:
                    stats2["oracle_miss"] += 1
        oracle_stats = stats2

    # Diagnostics: id overlap
    subset_size = len(subset_ids)
    before_overlap = sum(1 for pid in subset_ids if pid in before)
    after_overlap  = sum(1 for pid in subset_ids if pid in after)

    print(f"Oracle subset size = {subset_size} (parent={args.parent}, topk={args.topk})")
    print(f"Reranked oracle stats: {dict(oracle_stats)}")
    print(f"Before preds loaded: {len(before)}  diag={before_diag}")
    print(f"After  preds loaded: {len(after)}   diag={after_diag}")
    print(f"Subset overlap: before={before_overlap}/{subset_size}  after={after_overlap}/{subset_size}")
    print()

    # Evaluate
    n_b, p_b, r_b, h_b = eval_on_subset(before, subset_ids, ks)
    n_a, p_a, r_a, h_a = eval_on_subset(after,  subset_ids, ks)

    if n_b == 0 or n_a == 0:
        print("ERROR: tot == 0, so nothing was evaluated.")
        print("Most likely causes:")
        print("- gold field missing/empty in predictions files (need gold/labels per line), or")
        print("- input_id mismatch between files, or")
        print("- oracle subset could not be constructed (reranked has no gold and you did not pass --labels).")
        print()
        print("Fix:")
        print('  If reranked_top20.jsonl has no gold/labels, rerun with: --labels "data\\cve2attck_derived_20260107\\labels.jsonl"')
        return

    print(f"=== BEFORE ({args.before}) on oracle subset ===")
    print("N =", n_b)
    for k in ks:
        print(f"P@{k}={p_b[k]:.4f}  R@{k}={r_b[k]:.4f}  Hit@{k}={h_b[k]:.4f}")
    print()

    print(f"=== AFTER  ({args.after}) on oracle subset ===")
    print("N =", n_a)
    for k in ks:
        print(f"P@{k}={p_a[k]:.4f}  R@{k}={r_a[k]:.4f}  Hit@{k}={h_a[k]:.4f}")
    print()

    print("=== DELTA (AFTER - BEFORE) ===")
    for k in ks:
        print(f"Hit@{k}: {h_a[k]-h_b[k]:+.4f}   P@{k}: {p_a[k]-p_b[k]:+.4f}   R@{k}: {r_a[k]-r_b[k]:+.4f}")

if __name__ == "__main__":
    main()
