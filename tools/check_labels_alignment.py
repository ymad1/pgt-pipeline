import json
import pandas as pd
from pathlib import Path

def norm_id(x: str) -> str:
    return str(x).strip().replace("-", "_")

def build_maps(enterprise_attack_json: str):
    data = json.loads(Path(enterprise_attack_json).read_text(encoding="utf-8"))
    name2id, id2name = {}, {}
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
            id2name[tid] = name.strip()
    return name2id, id2name

def load_labels_jsonl(labels_jsonl: str):
    mp = {}
    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            mp[norm_id(j["input_id"])] = sorted(set(j.get("labels", [])))
    return mp

def pick_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() in ["input_id", "cve_id", "cve", "id"]:
            return c
    return df.columns[0]

def main(x_csv, y_csv, labels_jsonl, enterprise_json, cves):
    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_csv)
    if len(X) != len(y):
        raise ValueError(f"X/y 行数不一致: X={len(X)} y={len(y)}")

    x_id_col = pick_id_col(X)
    X_ids = X[x_id_col].astype(str).map(norm_id)

    y_num = y.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"])
    if y_num.shape[1] == 0:
        raise ValueError("y 中没找到数值列(0/1)")

    name2id, id2name = build_maps(enterprise_json)
    labels_map = load_labels_jsonl(labels_jsonl)

    for cve in cves:
        cve_n = norm_id(cve)
        matches = X_ids[X_ids == cve_n].index.tolist()
        if not matches:
            print(f"\n[{cve}] ❌ 不在 X 里")
            continue

        idx = matches[0]
        row = y_num.iloc[idx]
        ones = [col for col, v in row.items() if float(v) == 1.0]

        mapped_ids = []
        missing_names = []
        for n in ones:
            tid = name2id.get(str(n).strip().lower())
            if tid:
                mapped_ids.append(tid)
            else:
                missing_names.append(str(n))

        mapped_ids = sorted(set(mapped_ids))
        jsonl_ids = labels_map.get(cve_n, [])

        print(f"\n[{cve}] idx={idx}")
        print(f"y=1 的列数: {len(ones)}")
        print(f"能映射到 Txxxx 的列数: {len(mapped_ids)}  |  映射不到的列数: {len(missing_names)}")
        if missing_names:
            print("映射不到的列(前10):", missing_names[:10])

        print("labels.jsonl:", jsonl_ids)
        print("从 y 映射 :", mapped_ids)

        ok = (jsonl_ids == mapped_ids)
        print("一致性:", "✅ OK" if ok else "❌ MISMATCH")

        # 额外：把 id 反查回 technique name，方便人工看
        back_names = [id2name.get(t, "?") for t in jsonl_ids]
        print("id -> name:", list(zip(jsonl_ids, back_names)))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python tools/check_labels_alignment.py <X.csv> <y.csv> <labels.jsonl> <enterprise-attack.json> <CVE1> [CVE2 ...]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
