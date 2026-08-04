import json
import pandas as pd
from pathlib import Path

def normalize_id(x: str) -> str:
    return str(x).strip().replace("-", "_")

def build_name2id(enterprise_attack_json: str) -> dict:
    data = json.loads(Path(enterprise_attack_json).read_text(encoding="utf-8"))
    name2id = {}
    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        name = obj.get("name")
        if not name:
            continue
        tid = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                tid = ref["external_id"].upper()
                break
        if tid:
            name2id[name.strip().lower()] = tid
    return name2id

def pick_id_col(df: pd.DataFrame) -> str:
    # 常见列名
    for c in df.columns:
        if str(c).lower() in ["input_id", "cve_id", "cve", "id"]:
            return c
    # 兜底：第一列
    return df.columns[0]

def main(x_csv, y_csv, enterprise_json, out_labels):
    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)

    if len(X) != len(y):
        raise ValueError(f"X/y 行数不一致: X={len(X)}, y={len(y)} (必须用配套的 X/y)")

    x_id_col = pick_id_col(X)
    X_ids = X[x_id_col].astype(str).map(normalize_id).tolist()

    name2id = build_name2id(enterprise_json)

    # y 的列名是 technique name；如果 y 有无关列（比如 index），这里简单过滤掉全 0/1 的列
    # 更稳：只保留数值列
    y_num = y.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"])
    if y_num.shape[1] == 0:
        raise ValueError("y 中没找到数值列（0/1 one-hot 列）")

    out_path = Path(out_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing_names = set()
    non_empty = 0

    with out_path.open("w", encoding="utf-8") as f:
        for iid, row in zip(X_ids, y_num.itertuples(index=False, name=None)):
            # 找出为 1 的列
            labels_name = []
            for col, val in zip(y_num.columns, row):
                try:
                    v = float(val)
                except Exception:
                    continue
                if v == 1.0:
                    labels_name.append(str(col))

            # 名称 -> technique id
            labels_id = []
            for n in labels_name:
                tid = name2id.get(n.strip().lower())
                if tid:
                    labels_id.append(tid)
                else:
                    missing_names.add(n)

            if labels_id:
                non_empty += 1

            f.write(json.dumps({"input_id": iid, "labels": sorted(set(labels_id))}, ensure_ascii=False) + "\n")

    print(f"OK: {out_path}")
    print(f"lines: {sum(1 for _ in open(out_path, 'r', encoding='utf-8'))}, non_empty: {non_empty}")
    if missing_names:
        ms = sorted(list(missing_names))[:30]
        print(f"WARNING: {len(missing_names)} 个 technique name 没映射到 id（前30个）: {ms}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python tools/make_labels_from_onehot.py <X.csv> <y.csv> <enterprise-attack.json> <out_labels.jsonl>")
        sys.exit(1)
    main(*sys.argv[1:])
