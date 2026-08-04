# -*- coding: utf-8 -*-
import json
from pathlib import Path

RUN = Path(r"runs\cve2attck_3598_20260107")
RERANKED = RUN / "reranked_top20.jsonl"

# 这里用你已有的 predictions 文件来拿 gold（最省事）
# 如果你想用 labels.jsonl，也可以改成读 labels.jsonl 建 labelMap
GOLD_SRC = RUN / "predictions_reranked_top20.jsonl"

TOPK = 20
KS = [1, 3, 5, 10, 20]

BETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def to_parent(tid: str) -> str:
    return tid.split(".", 1)[0] if tid else tid


def dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_gold_map(pred_path: Path):
    gold = {}
    with pred_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            obj = json.loads(line)
            gold[obj["input_id"]] = obj.get("gold", [])
    return gold


gold_map = load_gold_map(GOLD_SRC)


def eval_predictions(rows):
    tot = 0
    p = {k: 0.0 for k in KS}
    r = {k: 0.0 for k in KS}
    hit = {k: 0.0 for k in KS}

    for input_id, pred_list, gold_list in rows:
        gold = [to_parent(x) for x in gold_list]
        gold = dedup_keep_order(gold)
        gold_set = set(gold)
        if not gold_set:
            continue

        pred = [to_parent(x) for x in pred_list]
        pred = dedup_keep_order(pred)

        tot += 1
        for k in KS:
            topk = pred[:k]
            denom = max(1, len(topk))  # 关键：别用 k 去除超过长度的情况
            inter = len(gold_set.intersection(topk))
            p[k] += inter / denom
            r[k] += inter / len(gold_set)
            hit[k] += 1.0 if inter > 0 else 0.0

    return tot, {k: p[k] / tot for k in KS}, {k: r[k] / tot for k in KS}, {k: hit[k] / tot for k in KS}


def build_preds_for_beta(beta: float):
    rows = []
    with RERANKED.open("r", encoding="utf-8-sig") as f:
        for line in f:
            obj = json.loads(line)
            input_id = obj["input_id"]
            cands = obj.get("candidates", [])

            scored = []
            for c in cands:
                tid = c.get("technique_id")
                if not tid:
                    continue
                sf = float(c.get("score_fused") or 0.0)

                llm = c.get("llm_score", None)
                if llm is None:
                    final = sf
                else:
                    llm = float(llm or 0.0)
                    final = beta * sf + (1.0 - beta) * llm

                scored.append((final, sf, tid))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            pred = [tid for _, _, tid in scored[:TOPK]]

            gold = gold_map.get(input_id, [])
            rows.append((input_id, pred, gold))
    return rows


best = None
for beta in BETAS:
    rows = build_preds_for_beta(beta)
    tot, P, R, H = eval_predictions(rows)

    # 你可以按你最关心的指标挑“最好”，比如 Hit@10 或 P@1
    key_metric = H[10]  # 改这里：比如 H[1], H[10], R[20], etc.
    rec = (key_metric, beta, tot, P, R, H)

    if best is None or rec[0] > best[0]:
        best = rec

    print(f"\n=== beta={beta:.2f}  N={tot} ===")
    for k in KS:
        print(f"P@{k}={P[k]:.4f}  R@{k}={R[k]:.4f}  Hit@{k}={H[k]:.4f}")

print("\nBEST (by Hit@10):")
_, beta, tot, P, R, H = best
print(f"beta={beta:.2f}  N={tot}")
for k in KS:
    print(f"P@{k}={P[k]:.4f}  R@{k}={R[k]:.4f}  Hit@{k}={H[k]:.4f}")
