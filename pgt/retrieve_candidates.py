"""Retrieve ATT&CK technique candidates from evidence and MES text views.

The retriever implements the candidate-generation stage used by the revised
pipeline.  It deliberately contains no CVE-, product-, protocol-, or
technique-specific boosting rules.  Both views are represented in the same
TF--IDF space built from the ATT&CK technique corpus:

* ``score_text``: cosine similarity between the concatenated evidence units of
  one CVE and an ATT&CK technique document;
* ``score_structure``: cosine similarity between a deterministic textual
  rendering of the selected Minimal Explainable Subgraph (MES) and the same
  technique document;
* ``score_fused``: ``alpha * score_text + (1-alpha) * score_structure``.

Raw cosine scores are already bounded in [0, 1].  The default normalization
therefore performs no per-query rescaling (``none_cosine_shared_space``), which
avoids changing score meaning according to the other candidates present in a
query.  A deterministic per-query min--max option is provided only for an
explicit calibration ablation.

Inputs
------
``--sentences``
    JSONL produced by ``pgt.split_sentences``.  Evidence units are used once;
    ``raw_text`` is only a fallback when no evidence dictionary is available.
``--mes``
    JSONL produced by ``pgt.build_mes``.
``--tech_index``
    ATT&CK technique text index, one JSON object per line.

Output
------
A candidate JSONL file and a ``.summary.json`` sidecar containing the exact
configuration, input hashes, corpus counts, and MES coverage statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

RETRIEVAL_VERSION = "candidate-retrieval-v2.0.0"
DEFAULT_ALPHA = 0.60
DEFAULT_TOPN = 20
STRUCTURAL_TYPES: Tuple[str, ...] = (
    "Precondition",
    "Entry",
    "VulnType",
    "Behavior",
    "Impact",
)
_TYPE_LABEL: Mapping[str, str] = {
    "Precondition": "precondition",
    "Entry": "entry",
    "VulnType": "vulnerability type",
    "Behavior": "behavior",
    "Impact": "impact",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_EVIDENCE_ID_RE = re.compile(r"^(?:E|e)(\d+)$")


# ---------------------------------------------------------------------------
# IO and provenance
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_no}")
            yield row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic TF--IDF
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Tokenize using the documented alphanumeric unigram rule."""
    tokens: List[str] = []
    for token in _TOKEN_RE.findall((text or "").lower()):
        if len(token) <= 2:
            continue
        if token.isdigit() and len(token) <= 3:
            continue
        tokens.append(token)
    return tokens


def build_idf(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Smoothed IDF: log((N+1)/(df+1)) + 1."""
    n_docs = len(documents)
    if n_docs == 0:
        raise ValueError("Technique corpus is empty")
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    return {
        token: math.log((n_docs + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in sorted(document_frequency.items())
    }


def tfidf_vector(tokens: Sequence[str], idf: Mapping[str, float]) -> Dict[str, float]:
    term_frequency = Counter(tokens)
    return {
        token: (1.0 + math.log(count)) * idf[token]
        for token, count in term_frequency.items()
        if token in idf
    }


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = numerator / (left_norm * right_norm)
    # Guard against tiny floating-point excursions.
    return max(0.0, min(1.0, score))


def minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return [0.0 for _ in values]
    scale = high - low
    return [(value - low) / scale for value in values]


# ---------------------------------------------------------------------------
# Technique corpus
# ---------------------------------------------------------------------------


def _technique_text(row: Mapping[str, Any]) -> str:
    direct = row.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: List[str] = []
    for key in ("name", "description", "tactics", "platforms"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            pieces = [str(item).strip() for item in value if str(item).strip()]
            if pieces:
                parts.append(" ".join(pieces))
    fields = row.get("fields")
    if isinstance(fields, Mapping):
        for key in ("name", "description", "text"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n".join(parts).strip()


def load_technique_index(path: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Load and deterministically merge duplicate technique IDs."""
    by_id: MutableMapping[str, List[str]] = defaultdict(list)
    source_rows = 0
    skipped_rows = 0
    for row in read_jsonl(path):
        source_rows += 1
        raw_id = row.get("technique_id") or row.get("id") or row.get("technique")
        technique_id = str(raw_id or "").strip()
        text = _technique_text(row)
        if not technique_id or not text:
            skipped_rows += 1
            continue
        if text not in by_id[technique_id]:
            by_id[technique_id].append(text)

    technique_ids = sorted(by_id)
    technique_docs = ["\n".join(by_id[technique_id]) for technique_id in technique_ids]
    if not technique_ids:
        raise ValueError(f"No usable techniques found in {path}")

    metadata = {
        "source_rows": source_rows,
        "usable_techniques": len(technique_ids),
        "duplicate_id_rows_merged": sum(max(0, len(texts) - 1) for texts in by_id.values()),
        "skipped_rows": skipped_rows,
    }
    return technique_ids, technique_docs, metadata


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def _evidence_sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
    evidence_id = str(item[0])
    match = _EVIDENCE_ID_RE.match(evidence_id)
    return (int(match.group(1)) if match else 10**9, evidence_id)


def evidence_query(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Return evidence text once, plus a provenance label for its source."""
    sentences = row.get("sentences")
    if isinstance(sentences, Mapping):
        fragments = [
            str(text).strip()
            for _, text in sorted(sentences.items(), key=_evidence_sort_key)
            if isinstance(text, str) and text.strip()
        ]
        if fragments:
            return " ".join(fragments), "evidence_units"

    raw_text = row.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text.strip(), "raw_text_fallback"
    return "", "empty"


def load_mes(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    by_input_id: Dict[str, Dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for line_no, row in enumerate(read_jsonl(path), start=1):
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            raise ValueError(f"MES record {line_no} is missing input_id")
        if input_id in by_input_id:
            raise ValueError(f"Duplicate MES input_id: {input_id}")
        by_input_id[input_id] = row
        status_counts[str(row.get("status", "unknown"))] += 1
    return by_input_id, dict(status_counts)


def _node_text(node: Mapping[str, Any]) -> str:
    value = node.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()

    parts: List[str] = []
    for key in (
        "condition",
        "vector",
        "detail",
        "vuln_type",
        "subtype",
        "action",
        "target",
        "impact",
        "impact_type",
    ):
        value = node.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "; ".join(parts)


def mes_query(mes: Mapping[str, Any]) -> str:
    """Render only the retained structural MES path; evidence text is excluded."""
    raw_nodes = mes.get("nodes") or []
    node_map: Dict[str, Mapping[str, Any]] = {
        str(node.get("id")): node
        for node in raw_nodes
        if isinstance(node, Mapping) and isinstance(node.get("id"), str)
    }

    chain = [node_id for node_id in (mes.get("chain") or []) if node_id in node_map]
    structural_ids = [
        node_id
        for node_id in (mes.get("structural_node_ids") or [])
        if node_id in node_map
    ]

    ordered_ids: List[str] = []
    # Optional precondition nodes are retained before the selected main chain.
    for node_id in structural_ids:
        if node_map[node_id].get("type") == "Precondition" and node_id not in ordered_ids:
            ordered_ids.append(node_id)
    for node_id in chain:
        if node_id not in ordered_ids:
            ordered_ids.append(node_id)
    for node_id in structural_ids:
        if node_id not in ordered_ids:
            ordered_ids.append(node_id)

    edge_types: Dict[Tuple[str, str], str] = {}
    for edge in mes.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        src = edge.get("src", edge.get("source"))
        dst = edge.get("dst", edge.get("target"))
        edge_type = edge.get("type")
        if isinstance(src, str) and isinstance(dst, str) and isinstance(edge_type, str):
            if edge_type not in {"mentions", "supported_by"}:
                edge_types[(src, dst)] = edge_type

    parts: List[str] = []
    previous_id: Optional[str] = None
    for node_id in ordered_ids:
        node = node_map[node_id]
        node_type = str(node.get("type", ""))
        if node_type not in STRUCTURAL_TYPES:
            continue
        if previous_id is not None:
            relation = edge_types.get((previous_id, node_id))
            if relation:
                parts.append(relation.replace("_", " "))
        text = _node_text(node)
        label = _TYPE_LABEL[node_type]
        parts.append(f"{label} {text}".strip())
        previous_id = node_id

    if parts:
        return " ".join(parts).strip()

    # Compatibility fallback for valid older MES records.
    compact = mes.get("compact_text")
    if isinstance(compact, str) and compact.strip():
        return compact.split("| evidence=", 1)[0].strip()
    return ""


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def to_parent(technique_id: str) -> str:
    return technique_id.split(".", 1)[0]


def _collapse_to_parent(scored: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse sub-techniques after scoring, keeping the best representative."""
    grouped: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in scored:
        grouped[to_parent(str(item["technique_id"]))].append(item)

    collapsed: List[Dict[str, Any]] = []
    for parent_id, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: (
                -float(item["score_fused"]),
                -float(item["score_text"]),
                -float(item["score_structure"]),
                str(item["technique_id"]),
            ),
        )
        representative = ordered[0]
        collapsed.append(
            {
                "technique_id": parent_id,
                "score_fused": representative["score_fused"],
                "score_text": representative["score_text"],
                "score_structure": representative["score_structure"],
                # Legacy alias retained temporarily for downstream compatibility.
                "score_graph": representative["score_structure"],
                "representative_technique_id": representative["technique_id"],
                "source_technique_ids": sorted(str(item["technique_id"]) for item in members),
            }
        )
    return collapsed


def rank_one(
    *,
    query_text: str,
    query_structure: str,
    technique_ids: Sequence[str],
    technique_vectors: Sequence[Mapping[str, float]],
    idf: Mapping[str, float],
    alpha: float,
    normalization: str,
    parent_normalization: bool,
    topn: int,
) -> List[Dict[str, Any]]:
    text_vector = tfidf_vector(tokenize(query_text), idf)
    structure_vector = tfidf_vector(tokenize(query_structure), idf)

    raw_text_scores = [cosine(text_vector, vector) for vector in technique_vectors]
    raw_structure_scores = [cosine(structure_vector, vector) for vector in technique_vectors]

    if normalization == "none":
        text_scores = raw_text_scores
        structure_scores = raw_structure_scores
    elif normalization == "minmax":
        text_scores = minmax_normalize(raw_text_scores)
        structure_scores = minmax_normalize(raw_structure_scores)
    else:  # defensive; argparse also constrains this value
        raise ValueError(f"Unsupported score normalization: {normalization}")

    scored: List[Dict[str, Any]] = []
    for technique_id, score_text, score_structure in zip(
        technique_ids, text_scores, structure_scores
    ):
        score_fused = alpha * score_text + (1.0 - alpha) * score_structure
        scored.append(
            {
                "technique_id": technique_id,
                "score_fused": round(score_fused, 12),
                "score_text": round(score_text, 12),
                "score_structure": round(score_structure, 12),
                # Temporary alias for old evaluation scripts.  New code should
                # use score_structure because the view is derived from MES text.
                "score_graph": round(score_structure, 12),
            }
        )

    if parent_normalization:
        scored = _collapse_to_parent(scored)

    scored.sort(
        key=lambda item: (
            -float(item["score_fused"]),
            -float(item["score_text"]),
            -float(item["score_structure"]),
            str(item["technique_id"]),
        )
    )
    selected = scored[:topn]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve ATT&CK candidates using evidence and MES TF-IDF views."
    )
    parser.add_argument("--sentences", required=True, help="sentences.jsonl")
    parser.add_argument("--mes", required=True, help="MES JSONL from pgt.build_mes")
    parser.add_argument("--tech_index", required=True, help="technique_text_index.jsonl")
    parser.add_argument("--output", required=True, help="output candidates.jsonl")
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--score_normalization",
        choices=("none", "minmax"),
        default="none",
        help="none = shared cosine scale; minmax = explicit calibration ablation",
    )
    parser.add_argument(
        "--normalize_to_parent",
        action="store_true",
        help="collapse sub-techniques to parent IDs after scoring",
    )
    parser.add_argument(
        "--allow_missing_mes",
        action="store_true",
        help="allow a sentence record without a matching MES and use a zero structure view",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    sentences_path = Path(args.sentences)
    mes_path = Path(args.mes)
    technique_path = Path(args.tech_index)
    output_path = Path(args.output)
    summary_path = Path(str(output_path) + ".summary.json")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")
    if args.topn <= 0:
        raise ValueError("--topn must be greater than zero")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be within [0, 1]")

    technique_ids, technique_docs, technique_metadata = load_technique_index(technique_path)
    technique_tokens = [tokenize(document) for document in technique_docs]
    idf = build_idf(technique_tokens)
    technique_vectors = [tfidf_vector(tokens, idf) for tokens in technique_tokens]

    mes_by_id, mes_file_status_counts = load_mes(mes_path)
    output_rows: List[Dict[str, Any]] = []
    runtime_status_counts: Counter[str] = Counter()
    text_source_counts: Counter[str] = Counter()
    missing_mes_ids: List[str] = []

    seen_input_ids: set[str] = set()
    for row in read_jsonl(sentences_path):
        input_id = str(row.get("input_id", "")).strip()
        if not input_id:
            raise ValueError("Sentence record is missing input_id")
        if input_id in seen_input_ids:
            raise ValueError(f"Duplicate sentence input_id: {input_id}")
        seen_input_ids.add(input_id)

        text_query, text_source = evidence_query(row)
        text_source_counts[text_source] += 1

        mes = mes_by_id.get(input_id)
        if mes is None:
            if not args.allow_missing_mes:
                raise KeyError(
                    f"No MES record for {input_id}. Use --allow_missing_mes only for a declared ablation."
                )
            structure_query = ""
            mes_status = "missing"
            complete_core_chain = False
            mes_sha256 = None
            missing_mes_ids.append(input_id)
        else:
            structure_query = mes_query(mes)
            mes_status = str(mes.get("status", "unknown"))
            complete_core_chain = bool(mes.get("complete_core_chain", False))
            raw_mes_hash = mes.get("mes_sha256")
            mes_sha256 = str(raw_mes_hash) if raw_mes_hash else None
        runtime_status_counts[mes_status] += 1

        candidates = rank_one(
            query_text=text_query,
            query_structure=structure_query,
            technique_ids=technique_ids,
            technique_vectors=technique_vectors,
            idf=idf,
            alpha=float(args.alpha),
            normalization=str(args.score_normalization),
            parent_normalization=bool(args.normalize_to_parent),
            topn=int(args.topn),
        )

        output_rows.append(
            {
                "input_id": input_id,
                "candidates": candidates,
                "retrieval_metadata": {
                    "version": RETRIEVAL_VERSION,
                    "alpha": float(args.alpha),
                    "score_normalization": (
                        "none_cosine_shared_space"
                        if args.score_normalization == "none"
                        else "per_query_minmax"
                    ),
                    "text_query_source": text_source,
                    "text_query_sha256": sha256_text(text_query),
                    "structure_query_sha256": sha256_text(structure_query),
                    "structure_query_empty": not bool(structure_query),
                    "mes_status": mes_status,
                    "complete_core_chain": complete_core_chain,
                    "mes_sha256": mes_sha256,
                    "parent_normalization": bool(args.normalize_to_parent),
                },
            }
        )

    write_jsonl(output_path, output_rows)

    summary = {
        "version": RETRIEVAL_VERSION,
        "configuration": {
            "alpha": float(args.alpha),
            "topn": int(args.topn),
            "score_normalization": (
                "none_cosine_shared_space"
                if args.score_normalization == "none"
                else "per_query_minmax"
            ),
            "parent_normalization": bool(args.normalize_to_parent),
            "allow_missing_mes": bool(args.allow_missing_mes),
            "tokenization": "lowercase_alphanumeric_unigrams_len_gt_2",
            "tf": "1_plus_log_count",
            "idf": "log((N+1)/(df+1))+1_over_technique_corpus",
            "text_view": "ordered_evidence_units_once_raw_text_only_as_fallback",
            "structure_view": "deterministic_text_rendering_of_selected_mes_structural_path",
            "heuristic_boosting": False,
        },
        "inputs": {
            "sentences": str(sentences_path),
            "sentences_sha256": sha256_file(sentences_path),
            "mes": str(mes_path),
            "mes_sha256": sha256_file(mes_path),
            "tech_index": str(technique_path),
            "tech_index_sha256": sha256_file(technique_path),
        },
        "counts": {
            "sentence_records": len(output_rows),
            "technique_corpus": technique_metadata,
            "mes_status_in_file": mes_file_status_counts,
            "mes_status_used": dict(sorted(runtime_status_counts.items())),
            "text_query_sources": dict(sorted(text_source_counts.items())),
            "missing_mes_count": len(missing_mes_ids),
            "empty_structure_query_count": sum(
                1
                for row in output_rows
                if row["retrieval_metadata"]["structure_query_empty"]
            ),
        },
        "missing_mes_ids": missing_mes_ids,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "records": len(output_rows),
                "techniques": len(technique_ids),
                "mes_status": dict(sorted(runtime_status_counts.items())),
                "missing_mes": len(missing_mes_ids),
                "output": str(output_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
