import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_PATH = Path("data/processed/cve2attack_labeled.jsonl")
ATTACK_STIX_PATH = Path("data/raw/attack_stix/enterprise-attack.json")  # 你下载好的 STIX

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def load_attack_name_to_tid(stix_path: Path):
    """
    Build mapping:
      - technique name -> Txxxx(.xxx)
      - "tactic - technique name" -> Txxxx(.xxx)  (更稳一点)
    """
    obj = json.loads(stix_path.read_text(encoding="utf-8"))
    items = obj.get("objects", [])

    name_to_tid = {}
    tactic_name_to_tid = {}

    for it in items:
        if it.get("type") != "attack-pattern":
            continue

        # external_id like T1210 or T1210.001
        tid = None
        for ref in it.get("external_references", []):
            if ref.get("source_name") in ("mitre-attack", "mitre-ics-attack", "mitre-mobile-attack"):
                ext = ref.get("external_id")
                if ext and ext.startswith("T"):
                    tid = ext
                    break
        if not tid:
            continue

        tech_name = it.get("name")
        if not tech_name:
            continue

        name_to_tid[norm(tech_name)] = tid

        # tactic info (kill_chain_phases phase_name like "initial-access")
        for kp in it.get("kill_chain_phases", []):
            if kp.get("kill_chain_name") == "mitre-attack":
                phase = kp.get("phase_name")
                if phase:
                    # store as "Initial Access - Exploit Public-Facing Application" style key
                    # normalize: "initial access - exploit public-facing application"
                    tactic_display = phase.replace("-", " ")
                    key = norm(f"{tactic_display} - {tech_name}")
                    tactic_name_to_tid[key] = tid

    return name_to_tid, tactic_name_to_tid

def main():
    if not ATTACK_STIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing ATT&CK STIX at {ATTACK_STIX_PATH}. "
            f"Download enterprise STIX bundle (enterprise-attack.json) and place it there."
        )

    name_to_tid, tactic_name_to_tid = load_attack_name_to_tid(ATTACK_STIX_PATH)

    ds = load_dataset("kenta-ikumi/CVE2ATTACK")["train"]  # HF 上目前是 train split :contentReference[oaicite:5]{index=5}
    df = ds.to_pandas()

    # HF 的列里：一列是 CVE 编号 + 一列是描述 + 其余很多列是 label(0/1)
    # 这里做一个“尽量鲁棒”的猜测：找出最像 CVE 和 description 的列
    cve_col = None
    desc_col = None
    for c in df.columns:
        lc = c.lower()
        if cve_col is None and "cve" in lc:
            cve_col = c
        if desc_col is None and ("description" in lc or "desc" == lc):
            desc_col = c

    if cve_col is None:
        # 兜底：第一列
        cve_col = df.columns[0]
    if desc_col is None:
        # 兜底：第二列
        desc_col = df.columns[1]

    label_cols = [c for c in df.columns if c not in (cve_col, desc_col)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_mapped = 0
    n_unmapped_labels = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            input_id = str(row[cve_col]).strip()
            raw_text = str(row[desc_col]).strip()

            labels = []
            for col in label_cols:
                v = row[col]
                try:
                    is_pos = int(v) == 1
                except Exception:
                    is_pos = False
                if not is_pos:
                    continue

                # col 形如 "Initial Access - Exploit Public-Facing Application"
                key1 = norm(col)
                tid = tactic_name_to_tid.get(key1)
                if tid is None:
                    # 再试一次：只用 technique 名称（去掉 tactic 前缀）
                    if " - " in col:
                        tech_name = col.split(" - ", 1)[1]
                    else:
                        tech_name = col
                    tid = name_to_tid.get(norm(tech_name))

                if tid is None:
                    n_unmapped_labels += 1
                    continue
                labels.append(tid)

            n_total += 1
            if labels:
                n_mapped += 1

            out = {
                "input_id": input_id,
                "raw_text": raw_text,
                "labels": sorted(set(labels)),
                "source": "cve2attack",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote: {OUT_PATH}")
    print(f"Rows: {n_total}, rows_with>=1_label: {n_mapped}")
    print(f"Unmapped positive labels (needs mapping fix): {n_unmapped_labels}")

if __name__ == "__main__":
    main()
