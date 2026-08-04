import json

RUN = r"runs\cve2attck_3598_20260107"
K = 20
path = fr"{RUN}\predictions_reranked_top{K}_parented.jsonl"

# 只评测到 top20
ks = [1, 3, 5, 10, 20]

def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

tot = 0
p = {k: 0.0 for k in ks}
r = {k: 0.0 for k in ks}
hit = {k: 0.0 for k in ks}

with open(path, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        pred = dedup_keep_order(obj.get("pred", []))
        gold = set(obj.get("gold", []))

        # gold 为空：通常跳过（跟你原逻辑一致）
        if not gold:
            continue

        tot += 1

        for k in ks:
            topk = pred[:k]
            topk_set = set(topk)

            inter = len(gold & topk_set)

            denom = max(1, min(k, len(topk)))  # 关键：不用固定 k，避免 pred 不足 k 时被稀释
            p[k] += inter / denom
            r[k] += inter / len(gold)
            hit[k] += 1.0 if inter > 0 else 0.0

print("N =", tot)
for k in ks:
    print(f"P@{k}={p[k]/tot:.4f}  R@{k}={r[k]/tot:.4f}  Hit@{k}={hit[k]/tot:.4f}")
