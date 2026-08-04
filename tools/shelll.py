import json
import os
import re

reranked_file = "../runs/cve2attck_3598_20260107/predictions_reranked_top20_parented_1513.jsonl"
src_file = "../runs/cve2attck_3598_20260107/predictions_top20_1513.jsonl"

CVE_RE = re.compile(r"(CVE_\d{4}_\d+)", re.IGNORECASE)

def base_cve(input_id: str) -> str:
    """
    从 input_id 中抽取 CVE 主键，如：
      CVE_2021_35296_augumented_9 -> CVE_2021_35296
      CVE_2022_22111 -> CVE_2022_22111
    若抽不到，就退化为原 input_id。
    """
    if not isinstance(input_id, str):
        return ""
    m = CVE_RE.search(input_id)
    return m.group(1).upper() if m else input_id

def read_jsonl(path: str):
    rows = []
    bad = 0
    empty = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                empty += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows.append(obj)
    return rows, bad, empty

def build_key2rows(rows):
    key2rows = {}
    for obj in rows:
        iid = obj.get("input_id", "")
        k = base_cve(iid)
        if not k:
            continue
        key2rows.setdefault(k, []).append(obj)
    return key2rows

def choose_src_row(src_rows_for_key, reranked_iid: str):
    """
    src 同一个 CVE 可能对应多个不同 input_id（如多个 augmented）。
    优先选：input_id 完全等于 reranked_iid 的那条；否则选第一条。
    """
    if reranked_iid:
        for r in src_rows_for_key:
            if r.get("input_id") == reranked_iid:
                return r
    return src_rows_for_key[0]

# 1) 读两个文件
reranked_rows, rer_bad, rer_empty = read_jsonl(reranked_file)
src_rows, src_bad, src_empty = read_jsonl(src_file)

# 2) 建索引（按 base CVE）
rer_key2rows = build_key2rows(reranked_rows)
src_key2rows = build_key2rows(src_rows)

# 3) 以 reranked 的顺序为准，构建“去重后的 ordered keys”
ordered_keys = []
seen = set()
dup_rer_key = 0

# 同时记录每个 key 在 reranked 中“首条”的 input_id（用于 src 选行）
rer_key2first_iid = {}

for obj in reranked_rows:
    iid = obj.get("input_id", "")
    k = base_cve(iid)
    if not k:
        continue
    if k in seen:
        dup_rer_key += 1
        continue
    seen.add(k)
    ordered_keys.append(k)
    rer_key2first_iid[k] = iid

# 4) 只保留两边都有的 key，保证数量/CVE 一致
missing_in_src = [k for k in ordered_keys if k not in src_key2rows]
final_keys = [k for k in ordered_keys if k in src_key2rows]

missing_in_rer = [k for k in src_key2rows.keys() if k not in rer_key2rows]

# 5) 写回：两个文件都按 final_keys 对齐输出（文件名不变）
def safe_rewrite(path, rows_iter):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        n = 0
        for obj in rows_iter:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n

# reranked：每个 key 取 reranked 中第一条（按 final_keys 顺序）
rer_out_rows = (rer_key2rows[k][0] for k in final_keys)

# src：每个 key 选一条最匹配的“CVE信息记录”
src_out_rows = (
    choose_src_row(src_key2rows[k], rer_key2first_iid.get(k, "")) for k in final_keys
)

rer_written = safe_rewrite(reranked_file, rer_out_rows)
src_written = safe_rewrite(src_file, src_out_rows)

print("====== 读取统计 ======")
print(f"reranked rows: {len(reranked_rows)}, bad: {rer_bad}, empty: {rer_empty}")
print(f"src rows:      {len(src_rows)}, bad: {src_bad}, empty: {src_empty}")

print("\n====== key 统计（按 base CVE）======")
print(f"reranked unique CVE keys: {len(ordered_keys)} (reranked 内重复 key 跳过数: {dup_rer_key})")
print(f"src unique CVE keys:      {len(src_key2rows)}")

print("\n====== 对齐结果 ======")
print(f"final aligned keys: {len(final_keys)}")
print(f"rewritten reranked_file lines: {rer_written}")
print(f"rewritten src_file lines:      {src_written}")

if missing_in_src:
    print(f"\n⚠️ reranked 里有 {len(missing_in_src)} 个 CVE 在 src 里找不到，示例: {missing_in_src[:10]}")
if missing_in_rer:
    print(f"\n⚠️ src 里有 {len(missing_in_rer)} 个 CVE 不在 reranked 里（不会写入），示例: {missing_in_rer[:10]}")

print("\n已完成：两个文件已按 CVE 对齐且数量一致，并原地重写。")
