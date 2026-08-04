import json
import re
from collections import Counter

reranked_file = "../runs/cve2attck_3598_20260107/predictions_reranked_top20_parented_1513.jsonl"
src_file      = "../runs/cve2attck_3598_20260107/predictions_top20_1513.jsonl"

CVE_RE = re.compile(r"(CVE_\d{4}_\d+)", re.IGNORECASE)

def base_cve(input_id: str) -> str:
    if not isinstance(input_id, str):
        return ""
    m = CVE_RE.search(input_id)
    return m.group(1).upper() if m else input_id

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return list(x)

def read_jsonl(path: str):
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad

def build_key2row(rows):
    d = {}
    dup = 0
    for obj in rows:
        k = base_cve(obj.get("input_id", ""))
        if not k:
            continue
        if k in d:
            dup += 1
            continue
        d[k] = obj
    return d, dup

def overlap_count(pred, gold, k=None):
    pred_k = pred if k is None else pred[:k]
    s = set(pred_k)
    return sum(1 for g in gold if g in s)

def first_hit_rank(pred, gold):
    # 返回最早命中 gold 的位置(1-based)，没命中返回 None
    gold_set = set(gold)
    for i, p in enumerate(pred, start=1):
        if p in gold_set:
            return i
    return None

def summarize(file_name, key2row, keys, ks=(1,3,5,10,20)):
    n = len(keys)
    out = {"n": n}

    # 指标累计
    hits = {k: 0 for k in ks}
    rsum = {k: 0.0 for k in ks}
    psum = {k: 0.0 for k in ks}

    # 分布：first hit rank
    bucket = Counter()

    # 统计 gold 是否为空、pred 是否为空
    empty_gold = 0
    empty_pred = 0

    for key in keys:
        obj = key2row[key]
        pred = to_list(obj.get("pred", []))
        gold = to_list(obj.get("gold", []))

        if len(pred) == 0:
            empty_pred += 1
        if len(gold) == 0:
            empty_gold += 1
            # gold 为空的样本：这里跳过，避免除0；如你想当0贡献，可改
            continue

        # first hit rank bucket
        r = first_hit_rank(pred, gold)
        if r is None:
            bucket["miss"] += 1
        elif r == 1:
            bucket["1"] += 1
        elif 2 <= r <= 3:
            bucket["2-3"] += 1
        elif 4 <= r <= 5:
            bucket["4-5"] += 1
        elif 6 <= r <= 10:
            bucket["6-10"] += 1
        elif 11 <= r <= 20:
            bucket["11-20"] += 1
        else:
            bucket[">20"] += 1  # 理论上top20不会出现

        for k in ks:
            kk = min(k, len(pred))
            c = overlap_count(pred, gold, k=k)

            # hits@k: any gold in topk
            if c > 0:
                hits[k] += 1

            # R@k
            rsum[k] += c / len(gold)

            # P@k
            psum[k] += c / kk if kk > 0 else 0.0

    denom = n - empty_gold  # 参与计算的样本数（gold非空）
    out["empty_gold"] = empty_gold
    out["empty_pred"] = empty_pred
    out["effective_n"] = denom

    def safe_div(a, b):
        return a / b if b else 0.0

    out["hits"] = {f"hits@{k}": safe_div(hits[k], denom) for k in ks}
    out["recall"] = {f"R@{k}": safe_div(rsum[k], denom) for k in ks}
    out["precision"] = {f"P@{k}": safe_div(psum[k], denom) for k in ks}

    # 分布也转成比例
    total_bucket = sum(bucket.values())
    out["first_hit_rank_dist"] = {k: safe_div(v, total_bucket) for k, v in bucket.items()}

    return out

def print_report(title, rep, ks=(1,3,5,10,20)):
    print(f"\n====== {title} ======")
    print(f"样本数 n={rep['n']}（gold为空 {rep['empty_gold']}，参与计算 effective_n={rep['effective_n']}；pred为空 {rep['empty_pred']}）")
    print("\n-- hits@K（gold 命中率）--")
    for k in ks:
        print(f"hits@{k}: {rep['hits'][f'hits@{k}']:.4f}")
    print("\n-- R@K（recall）--")
    for k in ks:
        print(f"R@{k}: {rep['recall'][f'R@{k}']:.4f}")
    print("\n-- P@K（precision）--")
    for k in ks:
        print(f"P@{k}: {rep['precision'][f'P@{k}']:.4f}")

    print("\n-- first hit rank 分布（最早命中位置）--")
    # 固定顺序打印
    for b in ["1", "2-3", "4-5", "6-10", "11-20", "miss", ">20"]:
        if b in rep["first_hit_rank_dist"]:
            print(f"{b}: {rep['first_hit_rank_dist'][b]:.4f}")

# 1) 读取
rer_rows, rer_bad = read_jsonl(reranked_file)
src_rows, src_bad = read_jsonl(src_file)

rer_map, rer_dup = build_key2row(rer_rows)
src_map, src_dup = build_key2row(src_rows)

keys = sorted(set(rer_map.keys()) & set(src_map.keys()))

print("====== 文件读取/对齐信息 ======")
print(f"reranked: rows={len(rer_rows)}, bad_json={rer_bad}, dup_key_skipped={rer_dup}, unique_keys={len(rer_map)}")
print(f"src:      rows={len(src_rows)}, bad_json={src_bad}, dup_key_skipped={src_dup}, unique_keys={len(src_map)}")
print(f"common aligned keys: {len(keys)}")

# 2) 计算
KS = (1,3,5,10,20)
rep_src = summarize("src", src_map, keys, ks=KS)
rep_rer = summarize("reranked", rer_map, keys, ks=KS)

# 3) 打印
print_report("RERANK 前（src_file）", rep_src, ks=KS)
print_report("RERANK 后（reranked_file）", rep_rer, ks=KS)

# 4) 提升（后-前）
print("\n====== 提升（reranked - src） ======")
for k in KS:
    dh = rep_rer["hits"][f"hits@{k}"] - rep_src["hits"][f"hits@{k}"]
    dr = rep_rer["recall"][f"R@{k}"] - rep_src["recall"][f"R@{k}"]
    dp = rep_rer["precision"][f"P@{k}"] - rep_src["precision"][f"P@{k}"]
    print(f"K={k}: Δhits@{k}={dh:+.4f}  ΔR@{k}={dr:+.4f}  ΔP@{k}={dp:+.4f}")
