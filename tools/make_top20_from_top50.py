# make_top20_from_top50.py
# -*- coding: utf-8 -*-
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="predictions_top50.jsonl")
    ap.add_argument("--out", dest="out", required=True, help="predictions_top20.jsonl")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    n = 0
    with open(args.inp, "r", encoding="utf-8-sig") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # 兼容字段：pred / predictions
            pred = obj.get("pred") or obj.get("predictions") or []
            obj["pred"] = pred[: args.k]

            # gold 原样保留（如果你文件里叫 labels 也兼容）
            if "gold" not in obj and "labels" in obj:
                obj["gold"] = obj["labels"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"OK: wrote {n} lines -> {args.out}")

if __name__ == "__main__":
    main()
