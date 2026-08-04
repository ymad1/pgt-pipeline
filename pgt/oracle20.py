import json, re

RUN = r"runs\cve2attck_3598_20260107"
K = 20
cand_path = RUN + r"\candidates.jsonl"
pred_path = RUN + rf"\predictions_reranked_top{K}_parented.jsonl"

def parent(t):
    return re.sub(r"\..*$", "", t)

# gold from predictions file (already aligned)
gold_map = {}
pred_map = {}
with open(pred_path, "r", encoding="utf-8-sig") as f:
    for line in f:
        o = json.loads(line)
        gid = o["input_id"]
        gold_map[gid] = [parent(x) for x in o.get("gold", [])]
        pred_map[gid] = [parent(x) for x in o.get("pred", [])]

tot = 0
oracle_full = 0
oracle_at20 = 0
hit_at20 = 0

# breakdown
missing_in_candidates = 0
only_beyond20 = 0
in_top20_but_missed = 0
hit = 0

with open(cand_path, "r", encoding="utf-8-sig") as f:
    for line in f:
        c = json.loads(line)
        iid = c["input_id"]
        gold = set(gold_map.get(iid, []))
        if not gold:
            continue
        tot += 1

        cand = [parent(x.get("technique_id","")) for x in c.get("candidates", [])]
        cand_set = set(cand)
        top20 = cand[:K]

        has_any_full = len(gold & cand_set) > 0
        has_any_20 = len(gold & set(top20)) > 0
        has_any_pred = len(gold & set(pred_map.get(iid, [])[:K])) > 0

        oracle_full += 1 if has_any_full else 0
        oracle_at20 += 1 if has_any_20 else 0
        hit_at20 += 1 if has_any_pred else 0

        if not has_any_full:
            missing_in_candidates += 1
        elif not has_any_20:
            only_beyond20 += 1
        elif not has_any_pred:
            in_top20_but_missed += 1
        else:
            hit += 1

print("N =", tot)
print("Oracle(full candidates) =", oracle_full / tot)
print("Oracle@20 (gold in candidates[:20]) =", oracle_at20 / tot)
print("Hit@20 (your pred[:20]) =", hit_at20 / tot)
print("--- breakdown counts ---")
print("A) gold missing from candidates:", missing_in_candidates)
print("B) gold only beyond top20:", only_beyond20)
print("C) gold in top20 candidates but missed by ranking:", in_top20_but_missed)
print("D) hit:", hit)
