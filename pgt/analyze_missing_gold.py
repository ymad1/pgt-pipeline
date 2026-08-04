# pgt/analyze_missing_gold.py
# Usage:
#   python pgt/analyze_missing_gold.py --run runs/cve2attck_3598_20260107 --labels data/cve2attck_derived_20260107/labels.jsonl --tech_index data/attack_cache/technique_text_index.jsonl --topk 20 --parent
#
import argparse, json, os
from collections import Counter, defaultdict

def to_parent(tid: str) -> str:
    if not tid:
        return tid
    return tid.split(".", 1)[0]

def read_jsonl(path: str):
    # 兼容 UTF-8 BOM
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/... folder")
    ap.add_argument("--labels", required=True, help="labels.jsonl")
    ap.add_argument("--tech_index", required=True, help="technique_text_index.jsonl")
    ap.add_argument("--topk", type=int, default=20, help="topk cutoff for Oracle@K analysis")
    ap.add_argument("--parent", action="store_true", help="normalize sub-techniques to parent")
    args = ap.parse_args()

    run_dir = args.run
    cand_path = os.path.join(run_dir, "candidates.jsonl")

    # 1) load labels: input_id -> set(gold)
    gold_map = {}
    n_gold_empty = 0
    for row in read_jsonl(args.labels):
        iid = row.get("input_id")
        labs = row.get("labels") or []
        if not iid:
            continue
        if args.parent:
            labs = [to_parent(x) for x in labs]
        labs = [x for x in labs if isinstance(x, str) and x.strip()]
        if not labs:
            n_gold_empty += 1
            continue
        gold_map[iid] = set(labs)

    # 2) scan candidates and classify A/B/C/D like你 oracle20.py
    A_ids = []  # gold missing from candidate universe
    B_ids = []  # gold only beyond topK
    C_ids = []  # gold in topK candidates but missed by pred (这里只做 oracle 不需要 pred，先留空)
    D_ids = []  # hit within topK candidates

    missing_gold_counter = Counter()  # gold tids for A cases
    missing_gold_by_input = {}        # input_id -> list(gold)

    # also track: gold tids overall (helpful)
    all_gold_counter = Counter()

    N = 0
    for row in read_jsonl(cand_path):
        iid = row.get("input_id")
        if iid not in gold_map:
            continue
        gold = gold_map[iid]
        if not gold:
            continue
        N += 1
        for g in gold:
            all_gold_counter[g] += 1

        cands = row.get("candidates") or []
        tids = []
        for c in cands:
            tid = c.get("technique_id")
            if isinstance(tid, str) and tid.strip():
                tids.append(to_parent(tid) if args.parent else tid)

        cand_set = set(tids)
        inter_all = gold.intersection(cand_set)

        if not inter_all:
            A_ids.append(iid)
            # 统计这个 input 的 gold
            glist = sorted(gold)
            missing_gold_by_input[iid] = glist
            for g in glist:
                missing_gold_counter[g] += 1
            continue

        # gold 在 universe 里，但看 topK
        topk = set(tids[: args.topk])
        inter_k = gold.intersection(topk)
        if not inter_k:
            B_ids.append(iid)
        else:
            D_ids.append(iid)

    # 3) load technique index ids
    tech_ids = set()
    for row in read_jsonl(args.tech_index):
        tid = row.get("technique_id") or row.get("id") or row.get("technique")
        if not tid:
            continue
        tid = str(tid).strip()
        if not tid:
            continue
        tid = to_parent(tid) if args.parent else tid
        tech_ids.add(tid)

    # 4) compare missing gold tids vs tech index
    missing_gold_tids = set(missing_gold_counter.keys())
    missing_from_index = sorted([t for t in missing_gold_tids if t not in tech_ids])
    present_in_index  = sorted([t for t in missing_gold_tids if t in tech_ids])

    # 5) write reports
    out_dir = os.path.join(run_dir, f"diagnostics_missing_gold_parent{str(args.parent).lower()}_top{args.topk}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "N_with_gold": N,
            "labels_gold_empty_skipped": n_gold_empty,
            "A_gold_missing_from_candidates": len(A_ids),
            "B_gold_only_beyond_topk": len(B_ids),
            "D_gold_in_topk_candidates": len(D_ids),
            "topk": args.topk,
            "parent": args.parent,
            "unique_missing_gold_tids": len(missing_gold_tids),
            "missing_gold_tids_missing_from_tech_index": len(missing_from_index),
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "A_input_ids.jsonl"), "w", encoding="utf-8") as f:
        for iid in A_ids:
            f.write(json.dumps({"input_id": iid, "gold": missing_gold_by_input[iid]}, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "missing_gold_tids_top100.txt"), "w", encoding="utf-8") as f:
        for tid, cnt in missing_gold_counter.most_common(100):
            f.write(f"{tid}\t{cnt}\n")

    with open(os.path.join(out_dir, "missing_gold_tids_missing_from_tech_index.txt"), "w", encoding="utf-8") as f:
        for tid in missing_from_index:
            f.write(tid + "\n")

    with open(os.path.join(out_dir, "missing_gold_tids_present_in_tech_index.txt"), "w", encoding="utf-8") as f:
        for tid in present_in_index:
            f.write(tid + "\n")

    print(f"[OK] wrote: {out_dir}")
    print(f"N={N}  A={len(A_ids)}  B={len(B_ids)}  D={len(D_ids)}  (topk={args.topk}, parent={args.parent})")
    print(f"unique missing gold tids (A): {len(missing_gold_tids)}")
    print(f"missing gold tids NOT in tech_index: {len(missing_from_index)}")
    if missing_from_index[:10]:
        print("examples missing from tech_index:", missing_from_index[:10])

if __name__ == "__main__":
    main()
