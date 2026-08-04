import json
import re
import sys
from pathlib import Path

import pandas as pd

def pick_id_col(df: pd.DataFrame) -> str:
    # 常见 id 列名
    candidates = [c for c in df.columns if str(c).lower() in ["input_id", "cve_id", "cve", "id"]]
    if candidates:
        return candidates[0]
    # 兜底：找最像 CVE 的列
    cve_re = re.compile(r"^CVE[-_]\d{4}[-_]\d{4,}$", re.I)
    best = None
    best_cnt = -1
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        cnt = s.map(lambda x: bool(cve_re.match(x))).sum()
        if cnt > best_cnt:
            best_cnt = cnt
            best = c
    if best is None:
        raise ValueError("找不到 input_id/cve_id 列，也没法从内容推断 CVE 列")
    return best

def pick_text_col(df: pd.DataFrame) -> str:
    # 常见文本列名
    name_candidates = [c for c in df.columns if str(c).lower() in ["raw_text", "text", "description", "sentence", "summary"]]
    if name_candidates:
        return name_candidates[0]
    # 兜底：选“平均长度最长”的字符串列
    str_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            str_cols.append(c)
    if not str_cols:
        raise ValueError("找不到文本列（没有 object/string 列）")
    best = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return best

def parse_labels(y: pd.DataFrame, id_col: str):
    """
    支持两种常见 y 格式：
    1) 每行一个 technique：包含 technique_id/label/attack_technique 等列
    2) 多标签 one-hot：除了 id_col 之外其他列是 technique_id，值为 0/1
    """
    cols_lower = {c: str(c).lower() for c in y.columns}
    # case 1: one row per label
    tech_col = None
    for c, lc in cols_lower.items():
        if lc in ["technique_id", "technique", "attack_technique", "tactic_technique", "label"]:
            tech_col = c
            break

    labels = {}

    if tech_col is not None and tech_col != id_col:
        for _, row in y.iterrows():
            i = str(row[id_col]).strip()
            t = str(row[tech_col]).strip()
            if not i or not t or t.lower() == "nan":
                continue
            labels.setdefault(i, set()).add(t)
        return {k: sorted(v) for k, v in labels.items()}

    # case 2: one-hot columns
    tech_cols = [c for c in y.columns if c != id_col]
    for _, row in y.iterrows():
        i = str(row[id_col]).strip()
        if not i or i.lower() == "nan":
            continue
        ts = []
        for c in tech_cols:
            v = row[c]
            try:
                if float(v) == 1.0:
                    ts.append(str(c))
            except Exception:
                pass
        labels[i] = sorted(set(ts))
    return labels

def main(x_csv: str, y_csv: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)

    x_id = pick_id_col(X)
    x_text = pick_text_col(X)

    y_id = pick_id_col(y)

    # 统一 input_id（用 X 的 id 字段为准）
    X["input_id"] = X[x_id].astype(str).str.strip().str.replace("-", "_")
    y_ids = y[y_id].astype(str).str.strip().str.replace("-", "_")
    y = y.copy()
    y["input_id"] = y_ids

    labels_map = parse_labels(y.drop(columns=[c for c in y.columns if c == y_id]), "input_id")

    # 写 sentences.jsonl（默认每条只有 E1 = raw_text）
    sent_path = out / "sentences.jsonl"
    with sent_path.open("w", encoding="utf-8") as f:
        for _, r in X.iterrows():
            iid = str(r["input_id"]).strip()
            txt = str(r[x_text]).strip()
            if not iid or iid.lower() == "nan" or not txt or txt.lower() == "nan":
                continue
            obj = {
                "input_id": iid,
                "raw_text": txt,
                "sentences": {"E1": txt}
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 写 labels.jsonl
    lab_path = out / "labels.jsonl"
    with lab_path.open("w", encoding="utf-8") as f:
        for iid in X["input_id"].astype(str).str.strip():
            if not iid or iid.lower() == "nan":
                continue
            labs = labels_map.get(iid, [])
            obj = {"input_id": iid, "labels": labs}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"OK: {sent_path} / {lab_path}")
    print(f"sentences lines: {sum(1 for _ in open(sent_path, 'r', encoding='utf-8'))}")
    print(f"labels    lines: {sum(1 for _ in open(lab_path,  'r', encoding='utf-8'))}")

if __name__ == "__main__":
    # 用法：python tools/make_cve2attck_jsonl.py <X.csv> <y.csv> <out_dir>
    if len(sys.argv) != 4:
        print("Usage: python tools/make_cve2attck_jsonl.py <X.csv> <y.csv> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
