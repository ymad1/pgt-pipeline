# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

SUBTECH_RE = re.compile(r"^(T\d{4})\.\d{3}$", re.I)

BAD_WORDS = ("fail", "error", "invalid", "mismatch", "missing")


def to_parent_tid(tid: str) -> str:
    if not tid:
        return tid
    m = SUBTECH_RE.match(tid)
    return m.group(1) if m else tid


def read_jsonl(path: str, encoding="utf-8-sig"):
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_paths(paths_jsonl: str, parent: bool):
    """
    paths.jsonl line schema (expected):
      {"input_id": "...", "paths": {"Txxxx": [{"nodes":[...], "edges":[...], "score":...}, ...], ...}}
    """
    out = {}
    for row in read_jsonl(paths_jsonl):
        iid = row.get("input_id")
        if not iid:
            continue
        p = row.get("paths") or {}
        if not isinstance(p, dict):
            p = {}
        if parent:
            # also index by parent id
            p2 = defaultdict(list)
            for k, v in p.items():
                pk = to_parent_tid(str(k))
                if isinstance(v, list):
                    p2[pk].extend(v)
            # keep original keys too (in case you want to debug)
            out[iid] = {"orig": p, "parent": dict(p2)}
        else:
            out[iid] = {"orig": p}
    return out


def extract_path_evidence_ids(path_obj) -> list[str]:
    """From best path nodes: 'EVIDENCE::E1' -> 'E1'."""
    if not isinstance(path_obj, dict):
        return []
    nodes = path_obj.get("nodes") or []
    eids = []
    for n in nodes:
        if isinstance(n, str) and n.startswith("EVIDENCE::"):
            eids.append(n.split("::", 1)[1])
    # unique preserve order
    seen = set()
    out = []
    for x in eids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pick_best_path(paths_for_tid):
    """Pick first path as best (pipeline usually sorts). Fallback to max score if present."""
    if not isinstance(paths_for_tid, list) or not paths_for_tid:
        return None
    # if score exists, choose max score
    scored = []
    for p in paths_for_tid:
        if isinstance(p, dict):
            s = p.get("score")
            try:
                s = float(s)
            except Exception:
                s = None
            scored.append((s, p))
    if any(s is not None for s, _ in scored):
        scored.sort(key=lambda x: (x[0] is not None, x[0] if x[0] is not None else -1e9), reverse=True)
        return scored[0][1]
    return paths_for_tid[0]


def parse_verifier_pass(ver_row: dict) -> bool:
    """
    verifier_report.jsonl schema may vary.
    Heuristic:
      - if row has explicit boolean like ok/pass/valid -> use it
      - else if any string field contains bad words -> fail
      - else pass
    """
    for k in ("ok", "pass", "passed", "valid", "is_valid"):
        v = ver_row.get(k)
        if isinstance(v, bool):
            return bool(v)

    # common patterns
    if isinstance(ver_row.get("errors"), list) and len(ver_row["errors"]) > 0:
        return False
    if isinstance(ver_row.get("failures"), list) and len(ver_row["failures"]) > 0:
        return False

    # scan strings
    def scan(obj) -> bool:
        if obj is None:
            return True
        if isinstance(obj, str):
            low = obj.lower()
            return not any(w in low for w in BAD_WORDS)
        if isinstance(obj, (int, float, bool)):
            return True
        if isinstance(obj, list):
            return all(scan(x) for x in obj)
        if isinstance(obj, dict):
            return all(scan(v) for v in obj.values())
        return True

    return scan(ver_row)


def load_verifier(verifier_jsonl: str):
    out = {}
    for row in read_jsonl(verifier_jsonl):
        iid = row.get("input_id")
        if not iid:
            continue
        out[iid] = row
    return out


def load_sentences(sentences_jsonl: str):
    out = {}
    for row in read_jsonl(sentences_jsonl):
        iid = row.get("input_id")
        if not iid:
            continue
        sents = row.get("sentences") or {}
        if not isinstance(sents, dict):
            sents = {}
        out[iid] = sents
    return out


def load_reranked_candidates(reranked_jsonl: str, parent: bool):
    """
    reranked_top20.jsonl schema (expected):
      {"input_id": "...", "candidates": [{"technique_id":..., "final_score":..., "llm_score":..., "evidence_ids":[...], ...}, ...]}
    """
    out = {}
    for row in read_jsonl(reranked_jsonl):
        iid = row.get("input_id")
        if not iid:
            continue
        cands = row.get("candidates") or []
        if not isinstance(cands, list):
            cands = []
        # normalize tids if requested
        normed = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            tid = c.get("technique_id")
            if isinstance(tid, str) and parent:
                c = dict(c)
                c["technique_id"] = to_parent_tid(tid)
            normed.append(c)
        out[iid] = normed
    return out


def topk_list(cands: list[dict], k: int, key="final_score"):
    # assume already sorted; if not, sort by key desc
    if not cands:
        return []
    # check if sorted: skip; safer: sort once
    def score(x):
        try:
            return float(x.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    cands_sorted = sorted(cands, key=score, reverse=True)
    return cands_sorted[:k]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"run folder, e.g. runs\cve2attck_3598_20260107")
    ap.add_argument("--reranked", required=True, help="reranked jsonl filename or path")
    ap.add_argument("--paths", default=None, help="paths.jsonl filename or path (optional)")
    ap.add_argument("--verifier", default=None, help="verifier_report.jsonl filename or path (optional)")
    ap.add_argument("--sentences", default=None, help="sentences.jsonl filename or path (optional, for examples.md)")
    ap.add_argument("--out", default=None, help="output folder (default: <run>\\explainability)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--ks", default="1,3,5,10", help="k list, e.g. 1,3,5,10")
    ap.add_argument("--parent", action="store_true", help="normalize sub-techniques to parent (Txxxx)")
    ap.add_argument("--example_n", type=int, default=30, help="number of examples to dump")
    args = ap.parse_args()

    run = args.run
    reranked = args.reranked
    paths = args.paths
    verifier = args.verifier
    sentences = args.sentences

    if not os.path.isabs(reranked):
        reranked = os.path.join(run, reranked)

    if paths and not os.path.isabs(paths):
        paths = os.path.join(run, paths)
    if verifier and not os.path.isabs(verifier):
        verifier = os.path.join(run, verifier)
    if sentences and not os.path.isabs(sentences):
        sentences = os.path.join(run, sentences)

    out_dir = args.out or os.path.join(run, "explainability")
    ensure_dir(out_dir)

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    topk = int(args.topk)

    # load inputs
    reranked_map = load_reranked_candidates(reranked, parent=args.parent)
    paths_map = load_paths(paths, parent=args.parent) if paths else {}
    verifier_map = load_verifier(verifier) if verifier else {}
    sentences_map = load_sentences(sentences) if sentences else {}

    # per-sample rows
    rows = []
    bucket_counts = Counter()
    bucket_examples = defaultdict(list)

    # aggregated metrics by k
    agg = {
        "N": 0,
        "top1_has_llm_evidence": 0,
        "top1_has_path": 0,
        "top1_has_path_evidence": 0,
        "top1_verifier_pass": 0,
    }
    # per-k coverage: at least one candidate in top-k has path/evidence
    cov = {k: {"any_path": 0, "any_llm_evidence": 0, "any_path_evidence": 0} for k in ks}

    # extra: which tids most often have no path (top1)
    top1_missing_path_tid = Counter()

    for iid, cands in reranked_map.items():
        if not cands:
            continue
        agg["N"] += 1

        # pick topK cands by final_score
        top_all = topk_list(cands, topk, key="final_score")

        # helper: lookup path by tid
        def get_paths_for_tid(tid: str):
            if not paths_map:
                return []
            pr = paths_map.get(iid)
            if not pr:
                return []
            if args.parent:
                # prefer parent-indexed
                p = pr.get("parent") or {}
                got = p.get(tid) or []
                if got:
                    return got
                # fallback to original keys
                o = pr.get("orig") or {}
                return o.get(tid) or []
            else:
                o = pr.get("orig") or {}
                return o.get(tid) or []

        # verifier pass?
        vpass = None
        if verifier_map:
            vrow = verifier_map.get(iid)
            if vrow is not None:
                vpass = parse_verifier_pass(vrow)

        # top1
        top1 = top_all[0]
        tid1 = str(top1.get("technique_id") or "")
        llm_eids_1 = top1.get("evidence_ids") or []
        if not isinstance(llm_eids_1, list):
            llm_eids_1 = []
        has_llm_evidence_1 = len(llm_eids_1) > 0

        # path evidence
        pbest = None
        peids_1 = []
        has_path_1 = False
        has_path_evidence_1 = False
        if paths_map:
            plist = get_paths_for_tid(tid1)
            pbest = pick_best_path(plist)
            if pbest:
                has_path_1 = True
                peids_1 = extract_path_evidence_ids(pbest)
                has_path_evidence_1 = len(peids_1) > 0

        # aggregate top1 metrics
        agg["top1_has_llm_evidence"] += 1 if has_llm_evidence_1 else 0
        agg["top1_has_path"] += 1 if has_path_1 else 0
        agg["top1_has_path_evidence"] += 1 if has_path_evidence_1 else 0
        if vpass is True:
            agg["top1_verifier_pass"] += 1

        # coverage by k: any within top-k has stuff
        for k in ks:
            sub = top_all[:k]
            any_llm = False
            any_path = False
            any_path_ev = False
            for c in sub:
                tid = str(c.get("technique_id") or "")
                le = c.get("evidence_ids") or []
                if isinstance(le, list) and len(le) > 0:
                    any_llm = True
                if paths_map:
                    plist = get_paths_for_tid(tid)
                    p = pick_best_path(plist)
                    if p:
                        any_path = True
                        if len(extract_path_evidence_ids(p)) > 0:
                            any_path_ev = True
            cov[k]["any_llm_evidence"] += 1 if any_llm else 0
            cov[k]["any_path"] += 1 if any_path else 0
            cov[k]["any_path_evidence"] += 1 if any_path_ev else 0

        # bucket for diagnosis (top1 only)
        if paths_map and not has_path_1:
            b = "top1_no_path"
            top1_missing_path_tid[tid1] += 1
        elif paths_map and has_path_1 and not has_path_evidence_1:
            b = "top1_path_no_evidence_nodes"
        elif verifier_map and vpass is False:
            b = "verifier_fail"
        elif has_llm_evidence_1 is False:
            b = "top1_llm_no_evidence_ids"
        else:
            b = "ok_explained"
        bucket_counts[b] += 1
        if len(bucket_examples[b]) < 30:
            bucket_examples[b].append(iid)

        # row
        rows.append(
            {
                "input_id": iid,
                "top1_tid": tid1,
                "top1_final_score": top1.get("final_score"),
                "top1_llm_score": top1.get("llm_score"),
                "top1_has_llm_evidence": int(has_llm_evidence_1),
                "top1_llm_evidence_n": len(llm_eids_1),
                "top1_has_path": int(has_path_1),
                "top1_path_evidence_n": len(peids_1),
                "verifier_pass": "" if vpass is None else int(bool(vpass)),
                "bucket": b,
            }
        )

    # write summary csv
    csv_path = os.path.join(out_dir, "explainability_summary.csv")
    fieldnames = [
        "input_id",
        "top1_tid",
        "top1_final_score",
        "top1_llm_score",
        "top1_has_llm_evidence",
        "top1_llm_evidence_n",
        "top1_has_path",
        "top1_path_evidence_n",
        "verifier_pass",
        "bucket",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # write metrics json
    N = max(1, agg["N"])
    metrics = {
        "N": agg["N"],
        "parent": bool(args.parent),
        "topk": topk,
        "top1": {
            "has_llm_evidence_rate": agg["top1_has_llm_evidence"] / N,
            "has_path_rate": agg["top1_has_path"] / N if paths_map else None,
            "has_path_evidence_rate": agg["top1_has_path_evidence"] / N if paths_map else None,
            "verifier_pass_rate": agg["top1_verifier_pass"] / N if verifier_map else None,
        },
        "coverage_by_k": {
            str(k): {
                "any_llm_evidence_rate": cov[k]["any_llm_evidence"] / N,
                "any_path_rate": (cov[k]["any_path"] / N) if paths_map else None,
                "any_path_evidence_rate": (cov[k]["any_path_evidence"] / N) if paths_map else None,
            }
            for k in ks
        },
        "buckets": dict(bucket_counts),
        "top1_missing_path_tid_top20": top1_missing_path_tid.most_common(20),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "error_buckets.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"buckets": dict(bucket_counts), "examples": dict(bucket_examples)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # examples.md (requires sentences.jsonl to be useful)
    ex_path = os.path.join(out_dir, "examples.md")
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(f"# Explainability examples\n\n")
        f.write(f"- parent={bool(args.parent)} topk={topk}\n\n")

        # pick examples: some ok, some no_path, some verifier_fail
        pick_order = ["ok_explained", "top1_no_path", "top1_path_no_evidence_nodes", "verifier_fail"]
        picked = []
        for b in pick_order:
            for iid in bucket_examples.get(b, []):
                picked.append((b, iid))
                if len(picked) >= args.example_n:
                    break
            if len(picked) >= args.example_n:
                break

        for (b, iid) in picked:
            cands = reranked_map.get(iid) or []
            top_all = topk_list(cands, topk, key="final_score")
            if not top_all:
                continue
            top1 = top_all[0]
            tid1 = str(top1.get("technique_id") or "")
            llm_eids = top1.get("evidence_ids") or []
            if not isinstance(llm_eids, list):
                llm_eids = []

            # path evidence
            peids = []
            pbest = None
            if paths_map:
                pr = paths_map.get(iid)
                if pr:
                    if args.parent:
                        plist = (pr.get("parent") or {}).get(tid1) or (pr.get("orig") or {}).get(tid1) or []
                    else:
                        plist = (pr.get("orig") or {}).get(tid1) or []
                    pbest = pick_best_path(plist)
                    if pbest:
                        peids = extract_path_evidence_ids(pbest)

            f.write(f"## {iid}  ({b})\n\n")
            f.write(f"- Top1: `{tid1}`  final={top1.get('final_score')}  llm={top1.get('llm_score')}\n")
            f.write(f"- LLM evidence_ids: {llm_eids}\n")
            f.write(f"- Path evidence_ids: {peids}\n\n")

            sents = sentences_map.get(iid) or {}
            if sents:
                # print only referenced evidence ids if possible
                show_ids = list(dict.fromkeys([*llm_eids, *peids]))[:5]
                if not show_ids:
                    show_ids = list(sents.keys())[:2]
                f.write("**Evidence snippets**\n\n")
                for eid in show_ids:
                    txt = sents.get(eid)
                    if isinstance(txt, str):
                        f.write(f"- {eid}: {txt}\n")
                f.write("\n")

            if pbest and isinstance(pbest, dict):
                nodes = pbest.get("nodes") or []
                edges = pbest.get("edges") or []
                f.write("**Best path (truncated)**\n\n")
                f.write(f"- nodes: {nodes[:12]}\n")
                f.write(f"- edges: {edges[:12]}\n\n")

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {os.path.join(out_dir, 'metrics.json')}")
    print(f"[OK] wrote: {os.path.join(out_dir, 'error_buckets.json')}")
    print(f"[OK] wrote: {ex_path}")


if __name__ == "__main__":
    main()
