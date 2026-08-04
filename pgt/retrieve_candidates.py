# pgt/retrieve_candidates.py
"""
Step 6: Retrieve candidate ATT&CK techniques (TEXT + GRAPH).

This "improved baseline" version computes:
- score_text  : TF-IDF cosine between (raw_text + all split sentences) and technique doc
- score_graph : TF-IDF cosine between (graph-derived structured query) and technique doc
and fuses:
- score_fused = alpha*score_text + (1-alpha)*score_graph

Key improvements vs previous baseline:
1) Graph query uses ONLY structured node fields (Behavior/VulnType/Entry/Precondition/Impact)
   -> Evidence text is NOT appended by default (reduces noise).
2) Simple keyword boosting for domain terms (e.g., JNDI/LDAP/SMB) to increase signal.
3) Output directory is auto-created to avoid FileNotFoundError.

Inputs:
  --sentences        data/processed/sentences.jsonl        (Step2 output)
  --local_graph_dir  runs/graphs/<run_id>/local_graphs     (Step4 output)
  --tech_index       data/attack/technique_text_index.jsonl (Step5 output)
Output:
  --output           runs/retrieval/<run_id>/candidates.jsonl

Run example:
  python -m pgt.retrieve_candidates `
    --sentences data/processed/sentences.jsonl `
    --local_graph_dir runs/graphs/dev/local_graphs `
    --tech_index data/attack/technique_text_index.jsonl `
    --output runs/retrieval/dev/candidates.jsonl `
    --topn 50 `
    --alpha 0.55

Note:
- Pure-python TF-IDF cosine (no sklearn dependency).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------
# IO helpers
# -----------------------

def read_jsonl(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------
# Tokenization / TF-IDF
# -----------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")  # keep alnum tokens


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = _TOKEN_RE.findall(text)
    out: List[str] = []
    for t in toks:
        if len(t) <= 2:
            continue
        # drop tiny pure numbers
        if t.isdigit() and len(t) <= 3:
            continue
        out.append(t)
    return out


def build_idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    """
    idf(t) = log((N+1)/(df+1)) + 1
    """
    N = len(docs_tokens)
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1.0) / (d + 1.0)) + 1.0
    return idf


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    vec: Dict[str, float] = {}
    for t, c in tf.items():
        w = idf.get(t)
        if w is None:
            continue
        vec[t] = (1.0 + math.log(c)) * w  # log-tf
    return vec


def dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, v in a.items():
        bv = b.get(k)
        if bv is not None:
            s += v * bv
    return s


def norm(a: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in a.values()))


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


# -----------------------
# Load technique docs
# -----------------------

def load_technique_index(path: Path) -> Tuple[List[str], List[str]]:
    """
    Accepts jsonl with fields like:
      {"technique_id":"Txxxx", "text":"..."}
    Or:
      {"technique_id":"Txxxx", "name":"...", "description":"...", ...}
    Builds one text document per technique.
    """
    rows = read_jsonl(path)
    tech_ids: List[str] = []
    docs: List[str] = []

    for r in rows:
        tid = str(r.get("technique_id", "")).strip()
        if not tid:
            continue

        if isinstance(r.get("text"), str):
            txt = r["text"]
        else:
            parts: List[str] = []
            for k in ["name", "description", "tactics", "platforms"]:
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
                elif isinstance(v, list) and v:
                    parts.append(" ".join(str(x) for x in v))
            txt = " ".join(parts)

        tech_ids.append(tid)
        docs.append(txt)

    return tech_ids, docs


# -----------------------
# Graph -> query text (structured only) + boosting
# -----------------------

def load_local_graph(graph_path: Path) -> Dict[str, Any]:
    with graph_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def graph_to_query_text_structured_only(g: Dict[str, Any]) -> str:
    """
    Build a graph-derived query text from structured nodes ONLY:
      Behavior, VulnType, Entry, Precondition, Impact
    Evidence text is intentionally excluded to reduce noise.

    Then apply lightweight keyword boosting for high-signal domain terms.
    """
    nodes = g.get("nodes") or []

    behavior_parts: List[str] = []
    vtype_parts: List[str] = []
    entry_parts: List[str] = []
    precond_parts: List[str] = []
    impact_parts: List[str] = []

    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = n.get("type")
        if t == "Behavior":
            for k in ["action", "target", "impact"]:
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    behavior_parts.append(v.strip())
        elif t == "VulnType":
            for k in ["type", "subtype"]:
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    vtype_parts.append(v.strip())
        elif t == "Entry":
            for k in ["vector", "detail"]:
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    entry_parts.append(v.strip())
        elif t == "Precondition":
            v = n.get("condition") or n.get("text")
            if isinstance(v, str) and v.strip():
                precond_parts.append(v.strip())
        elif t == "Impact":
            # our fixed schema: impact_type + detail (or legacy "type")
            for k in ["impact_type", "detail", "type"]:
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    impact_parts.append(v.strip())

    base_parts = behavior_parts + vtype_parts + entry_parts + precond_parts + impact_parts
    q = " ".join(base_parts).strip()

    # ---- keyword boosting ----
    q_low = q.lower()
    boost: List[str] = []

    # Log4Shell-ish signals
    if "jndi" in q_low:
        boost += ["jndi"] * 4
    if "ldap" in q_low:
        boost += ["ldap"] * 4
    if "log4j" in q_low:
        boost += ["log4j"] * 4
    # If we see JNDI/LDAP/log4j signals, bias toward exploitation vocabulary used in ATT&CK docs
    if ("jndi" in q_low) or ("ldap" in q_low) or ("log4j" in q_low):
        boost += ["exploit"] * 5
        boost += ["public"] * 3 + ["facing"] * 3
        boost += ["application"] * 3
        boost += ["remote"] * 2 + ["execution"] * 2
        boost += ["server"] * 2
    if "endpoint" in q_low:
        boost += ["endpoint"] * 1
    if "remote fetch" in q_low:
        boost += ["remote"] * 2 + ["fetch"] * 2
    if "code execution" in q_low or "rce" in q_low:
        boost += ["execute"] * 2 + ["code"] * 2 + ["execution"] * 2

    # EternalBlue-ish signals
    if "smb" in q_low or "smbv1" in q_low:
        boost += ["smb"] * 4
    if "crafted" in q_low and "packet" in q_low:
        boost += ["crafted"] * 2 + ["packets"] * 2
    if "remote attackers" in q_low:
        boost += ["remote"] * 2 + ["attackers"] * 2

    # General exploitation signals
    if "exploit" in q_low or "exploitation" in q_low:
        boost += ["exploit"] * 2
    if "injection" in q_low:
        boost += ["injection"] * 2

    return (q + " " + " ".join(boost)).strip()


# -----------------------
# Sentences -> text query
# -----------------------

def sentences_to_query_text(row: Dict[str, Any]) -> str:
    raw = str(row.get("raw_text", "") or "")
    sents = row.get("sentences") or {}
    sent_text = ""
    if isinstance(sents, dict):
        # stable order E1,E2...
        sent_text = " ".join(str(v) for _, v in sorted(sents.items()))
    return (raw + " " + sent_text).strip()


# -----------------------
# Main
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PGT Step6: retrieve candidate techniques (text + graph).")
    parser.add_argument("--sentences", required=True, help="Step2 sentences.jsonl")
    parser.add_argument("--local_graph_dir", required=True, help="Step4 local graph directory")
    parser.add_argument("--tech_index", required=True, help="Step5 technique_text_index.jsonl")
    parser.add_argument("--output", required=True, help="Output candidates.jsonl")
    parser.add_argument("--topn", type=int, default=50, help="Top-N candidates per input")
    parser.add_argument("--alpha", type=float, default=0.55, help="Fusion weight for text score (0..1)")
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be within [0,1]")

    sentences_rows = read_jsonl(Path(args.sentences))
    local_graph_dir = Path(args.local_graph_dir)
    tech_ids, tech_docs = load_technique_index(Path(args.tech_index))

    # Build TF-IDF model over technique docs
    tech_tokens = [tokenize(d) for d in tech_docs]
    idf = build_idf(tech_tokens)
    tech_vecs = [tfidf_vector(toks, idf) for toks in tech_tokens]

    out_rows: List[Dict[str, Any]] = []

    for row in sentences_rows:
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            continue

        # Text query from Step2 (raw + sentences)
        q_text = sentences_to_query_text(row)
        q_text_vec = tfidf_vector(tokenize(q_text), idf)

        # Graph query from Step4 (structured only)
        g_path = local_graph_dir / f"{input_id}.json"
        if g_path.exists():
            g = load_local_graph(g_path)
            q_graph = graph_to_query_text_structured_only(g)
        else:
            q_graph = ""
        q_graph_vec = tfidf_vector(tokenize(q_graph), idf)

        scored: List[Tuple[str, float, float, float]] = []
        for tid, dvec in zip(tech_ids, tech_vecs):
            s_text = cosine(q_text_vec, dvec)
            s_graph = cosine(q_graph_vec, dvec)
            s_fused = alpha * s_text + (1.0 - alpha) * s_graph
            scored.append((tid, s_fused, s_text, s_graph))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: int(args.topn)]

        out_rows.append(
            {
                "input_id": input_id,
                "candidates": [
                    {
                        "technique_id": tid,
                        "score_fused": sf,
                        "score_text": st,
                        "score_graph": sg,
                    }
                    for (tid, sf, st, sg) in top
                ],
            }
        )

    write_jsonl(Path(args.output), out_rows)


if __name__ == "__main__":
    main()
