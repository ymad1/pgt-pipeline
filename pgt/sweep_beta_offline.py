# -*- coding: utf-8 -*-
import json
import argparse

def to_parent(tid: str) -> str:
    return tid.split(".", 1)[0] if tid else tid

def load_labels(labels_jsonl: str, parent: bool) -> dict[str, set[str]]:
    mp = {}
    with open(labels_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("input_id")
            labs = obj.get("labels", []) or obj.get("gold", []) or []
            if not iid:
                continue
            s = set()
            for t in labs:
                if isinstance(t, str) and t:
                    s.add(to_parent(t) if parent else t)
            mp[iid] = s
    return mp

def dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def eval_predictions(pred_map: dict[str, list[str]], labels_map: dict[str, set[str]], ks):
    tot = 0
    hit = {k: 0.0 for k in ks}
    p = {k: 0.0 for k in ks}
    r = {k: 0.0 for k in ks}

    for iid, pred in pred_map.items():
        gold = labels_map.get(iid, set())
        if not gold:
            continue
        tot += 1
        for k in ks:
            topk = pred[:k]
            inter = len(gold.intersection(topk))
            hit[k] += 1.0 if inter > 0 else 0.0
            p[k] += inter / k
            r[k] += inter / len(gold)

    return tot, {k: (p[k]/tot, r[k]/tot, hit[k]/tot) for k in ks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--parent", action="store_true")
    ap.add_argument("--ks", default="1,3,5,10,20")
    ap.add_argument("--metric_k", type=int, default=20, help="optimize for Hit@K")
    args = ap.parse_args()

    ks = sorted(set(int(x) for x in args.ks.split(",") if x.strip()))
    labels_map = load_labels(args.labels, parent=args.parent)

    # load reranked candidates once
    rows = []
    with open(args.reranked, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    best = None  # (score, beta, metrics)
    for beta in [i/100 for i in range(0, 101, 5)]:
        pred_map = {}
        for row in rows:
            iid = row.get("input_id")
            cands = row.get("candidates", []) or []
            scored = []
            for c in cands:
                tid = c.get("technique_id")
                if not isinstance(tid, str) or not tid:
                    continue
                tid2 = to_parent(tid) if args.parent else tid
                sf = float(c.get("score_fused") or 0.0)
                ls = c.get("llm_score", None)
                if ls is None:
                    final = sf
                else:
                    final = beta * sf + (1.0 - beta) * float(ls)
                scored.append((final, sf, tid2))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            pred = dedup_keep_order([t for _, __, t in scored])[:max(ks)]
            pred_map[iid] = pred

        tot, metrics = eval_predictions(pred_map, labels_map, ks)
        hitK = metrics[args.metric_k][2]
        if best is None or hitK > best[0]:
            best = (hitK, beta, tot, metrics)

    hitK, beta, tot, metrics = best
    print(f"BEST by Hit@{args.metric_k}: beta={beta:.2f}  N={tot}  (parent={args.parent})")
    for k in ks:
        P, R, H = metrics[k]
        print(f"P@{k}={P:.4f}  R@{k}={R:.4f}  Hit@{k}={H:.4f}")

if __name__ == "__main__":
    main()
