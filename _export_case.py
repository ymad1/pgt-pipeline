import json, os

cve = "CVE_2022_22965"  # <<< 改这里

files = {
  "sentences": r"runs/cve2attck_3598_20260107/sentences.head200.jsonl",
  "extraction": r"runs/cve2attck_3598_20260107/extraction.head200.jsonl",
  "rerank": r"runs/cve2attck_3598_20260107/predictions_reranked_top20.head200.jsonl",
  "evidence_pack": r"runs/cve2attck_3598_20260107/evidence_pack.head200.jsonl",
  "verify": r"runs/cve2attck_3598_20260107/verify.head200.jsonl",
}

def get_row(path):
  with open(path, "r", encoding="utf-8-sig") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      if obj.get("input_id") == cve:
        return obj
  return None

bundle = {"input_id": cve}
missing = []
for name, path in files.items():
  row = get_row(path)
  if row is None:
    missing.append(name)
  bundle[name] = row

bundle["_missing_sections"] = missing

out_path = f"case_{cve}.json"
with open(out_path, "w", encoding="utf-8") as w:
  json.dump(bundle, w, ensure_ascii=False, indent=2)

print("WROTE:", os.path.abspath(out_path))
if missing:
  print("MISSING:", ", ".join(missing))
