import json
from collections import Counter
from pathlib import Path

path = Path("runs/cve2attck/verifier_report.jsonl")  # 按你的全量输出路径改一下
evr, pcs = [], []
issue_counter = Counter()
n = 0

for line in path.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    n += 1
    evr.append(r.get("EVR", 0.0))
    pcs.append(r.get("PCS", 0.0))
    for it in (r.get("issues") or []):
        # it 可能是字符串，也可能是对象；都兼容一下
        if isinstance(it, str):
            issue_counter[it] += 1
        elif isinstance(it, dict):
            issue_counter[it.get("type","unknown")] += 1
        else:
            issue_counter["unknown"] += 1

def pct(x): return f"{100.0*x:.2f}%"

print("N =", n)
print("EVR avg/min/p10/p50 =", sum(evr)/n, min(evr), sorted(evr)[int(0.1*n)], sorted(evr)[int(0.5*n)])
print("PCS avg/min/p10/p50 =", sum(pcs)/n, min(pcs), sorted(pcs)[int(0.1*n)], sorted(pcs)[int(0.5*n)])
print("issues top10 =", issue_counter.most_common(10))
