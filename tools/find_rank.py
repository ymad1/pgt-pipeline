import json

PATH = r"runs/retrieval/dev/candidates.jsonl"
TARGET_INPUT = "CVE-2021-44228"
NEEDLE = "T1190"

for line in open(PATH, encoding="utf-8"):
    r = json.loads(line)
    if r.get("input_id") != TARGET_INPUT:
        continue
    cands = r.get("candidates") or []
    for i, c in enumerate(cands, 1):
        if c.get("technique_id") == NEEDLE:
            print("input_id:", TARGET_INPUT)
            print("technique:", NEEDLE)
            print("rank:", i)
            print("scores:", c)
            raise SystemExit(0)
    print("input_id:", TARGET_INPUT)
    print("technique:", NEEDLE)
    print("NOT FOUND in candidates list")
    raise SystemExit(0)

print("No row found for input_id:", TARGET_INPUT)
