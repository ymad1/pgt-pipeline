import argparse, json, os, re
from datasets import load_dataset

def read_tech_index(path):
    name2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = obj.get("technique_id")
            ttext = obj.get("technique_text", "") or ""
            # technique_text 第一行通常是 technique 名称
            tname = (ttext.splitlines()[0].strip() if ttext else "").strip()
            if tid and tname and tname not in name2id:
                name2id[tname] = tid
    return name2id

def sentencize(text):
    text = (text or "").strip()
    if not text:
        return []
    # 简单通用句切：先按 . ! ?；再兜底按分号/换行
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        parts = re.split(r'\s*;\s*', text)
        parts = [p.strip() for p in parts if p and p.strip()]
    return parts[:20]  # 避免证据段过长

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=r"data\cve2attck")
    ap.add_argument("--tech_index", default=r"data\attack\technique_text_index.jsonl")
    ap.add_argument("--dataset", default="readerbench/cve-2-att-ck")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name2id = read_tech_index(args.tech_index)

    ds = load_dataset(args.dataset, split=args.split)  # public HF dataset
    cols = ds.column_names

    # 尽量自动适配字段名
    id_col = "ID" if "ID" in cols else ("cve_id" if "cve_id" in cols else cols[0])
    desc_col = "Description" if "Description" in cols else ("description" if "description" in cols else cols[1])

    label_cols = [c for c in cols if c not in (id_col, desc_col)]
    sent_path = os.path.join(args.out_dir, "sentences.jsonl")
    gold_path = os.path.join(args.out_dir, "gold_labels.jsonl")

    miss_names = set()
    n = 0

    with open(sent_path, "w", encoding="utf-8") as fs, open(gold_path, "w", encoding="utf-8") as fg:
        for row in ds:
            cve = str(row.get(id_col, "")).strip()
            desc = str(row.get(desc_col, "")).strip()
            if not cve or not desc:
                continue

            sents = sentencize(desc)
            ev = {f"E{i+1}": s for i, s in enumerate(sents)} if sents else {"E1": desc}

            # 标签列形如 "Initial Access - Exploit Public-Facing Application"
            tids = []
            for c in label_cols:
                v = row.get(c, 0)
                try:
                    is_pos = float(v) > 0
                except Exception:
                    is_pos = bool(v)
                if not is_pos:
                    continue

                tech_name = c.split(" - ", 1)[-1].strip()
                tid = name2id.get(tech_name)
                if tid:
                    tids.append(tid)
                else:
                    miss_names.add(tech_name)

            fs.write(json.dumps({"input_id": cve, "raw_text": desc, "sentences": ev}, ensure_ascii=False) + "\n")
            fg.write(json.dumps({"input_id": cve, "technique_ids": sorted(set(tids))}, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] rows_written={n}")
    print(f"[OK] sentences={sent_path}")
    print(f"[OK] gold_labels={gold_path}")
    if miss_names:
        print(f"[WARN] technique names not found in tech_index: {len(miss_names)} (showing up to 30)")
        for x in list(sorted(miss_names))[:30]:
            print("  -", x)

if __name__ == "__main__":
    main()
