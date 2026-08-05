# pgt/split_sentences.py
"""Deterministic evidence segmentation for CVE descriptions.

This module converts one canonical CVE description into ordered, citable evidence
units ``E1..En``.  It is intentionally independent of labels and model outputs.
The output remains backward compatible with the rest of the pipeline through the
``sentences`` mapping, while adding exact character spans and a reproducibility
record.

Accepted input JSONL fields
---------------------------
Each non-empty line must contain an identifier (``input_id`` by default) and a
text field.  With ``--text_field auto`` (the default), text is resolved in this
order: ``raw_text``, ``text``, ``description``, ``cve_description``.

Output JSONL fields
-------------------
{
  "input_id": "CVE_...",
  "raw_text": "canonical whitespace-normalized text",
  "sentences": {"E1": "...", "E2": "..."},
  "evidence_spans": {
    "E1": {"start": 0, "end": 42, "text_sha256": "..."}
  },
  "segmentation": {...},
  "provenance": {...}                 # preserved when present in input
}

The concatenation of evidence units with one space is required to reconstruct
``raw_text`` exactly.  If that invariant fails, execution stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SEGMENTATION_VERSION = "evidence-segmentation-v2.0.0"
DEFAULT_TEXT_FIELDS: Tuple[str, ...] = (
    "raw_text",
    "text",
    "description",
    "cve_description",
)

# Common abbreviations for which a terminal period is not a sentence boundary.
# The list is deliberately small and fixed so that segmentation is reproducible.
_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "fig.",
    "no.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "u.s.",
    "u.k.",
}

# Clause markers kept at the beginning of the right-hand evidence unit.
_CLAUSE_MARKER_RE = re.compile(
    r"\s+(?=(?:"
    r"which\s+(?:allows?|could\s+allow|may\s+allow|can\s+allow)\b|"
    r"allows?\b|allowing\b|"
    r"resulting\s+in\b|leading\s+to\b|thereby\b|"
    r"due\s+to\b|because\b|"
    r"via\b|through\b|"
    r"when\b|if\b|"
    r"by\s+[A-Za-z][A-Za-z-]*ing\b"
    r"))",
    re.IGNORECASE,
)

# A sentence boundary is considered only when punctuation is followed by
# whitespace/end-of-text.  Further checks reject decimals, versions and
# abbreviations.
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s+|$)")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize_text(value: Any) -> str:
    """Return a deterministic one-line representation of a source description."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _natural_evidence_key(eid: str) -> Tuple[int, str]:
    m = re.fullmatch(r"E(\d+)", str(eid))
    return (int(m.group(1)), str(eid)) if m else (10**9, str(eid))


def _resolve_text(row: Mapping[str, Any], text_field: str) -> Tuple[str, str]:
    fields = DEFAULT_TEXT_FIELDS if text_field == "auto" else (text_field,)
    for field in fields:
        if field in row and row[field] is not None:
            text = _canonicalize_text(row[field])
            if text:
                return field, text
    raise KeyError(
        "No non-empty text field found. Tried: " + ", ".join(repr(x) for x in fields)
    )


def _prefix_ends_with_abbreviation(prefix: str) -> bool:
    lower = prefix.lower().rstrip()
    return any(lower.endswith(abbr) for abbr in _ABBREVIATIONS)


def _is_sentence_boundary(text: str, punct_start: int, punct_end: int) -> bool:
    """Return whether one punctuation match closes a sentence."""
    punct = text[punct_start:punct_end]
    if not punct:
        return False

    # Decimal/version components such as 5.0 or 2.12.3 are not boundaries.
    if punct[0] == ".":
        prev_char = text[punct_start - 1] if punct_start > 0 else ""
        next_nonspace = punct_end
        while next_nonspace < len(text) and text[next_nonspace].isspace():
            next_nonspace += 1
        next_char = text[next_nonspace] if next_nonspace < len(text) else ""
        if prev_char.isdigit() and next_char.isdigit():
            return False
        if _prefix_ends_with_abbreviation(text[:punct_end]):
            return False

    # If there is following text, require a plausible sentence start.  CVE
    # descriptions commonly begin a new sentence with NOTE, HOWEVER, a product
    # name, a number, a quote, or a parenthesis.
    cursor = punct_end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    return text[cursor].isupper() or text[cursor].isdigit() or text[cursor] in "\"'([*"


def _primary_split(text: str) -> List[str]:
    """Split at validated sentence boundaries while preserving punctuation."""
    if not text:
        return []
    boundaries: List[int] = []
    for match in _TERMINAL_PUNCT_RE.finditer(text):
        if _is_sentence_boundary(text, match.start(), match.end()):
            boundaries.append(match.end())

    out: List[str] = []
    start = 0
    for end in boundaries:
        fragment = text[start:end].strip()
        if fragment:
            out.append(fragment)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out or [text]


def _split_at_positions(text: str, positions: Sequence[int], min_chars: int) -> List[str]:
    """Split before selected positions, retaining all source characters."""
    accepted: List[int] = []
    last = 0
    for pos in sorted(set(int(x) for x in positions if 0 < int(x) < len(text))):
        left = text[last:pos].strip()
        right = text[pos:].strip()
        if len(left) >= min_chars and len(right) >= min_chars:
            accepted.append(pos)
            last = pos

    if not accepted:
        return [text]

    out: List[str] = []
    start = 0
    for pos in accepted:
        fragment = text[start:pos].strip()
        if fragment:
            out.append(fragment)
        start = pos
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _looks_like_semicolon_enumeration(text: str) -> bool:
    parts = [x.strip() for x in text.split(";") if x.strip()]
    if len(parts) < 4:
        return False
    avg_len = sum(len(x) for x in parts) / len(parts)
    # Version/product enumerations tend to contain many short fragments.
    return avg_len <= 55


def _split_semicolons(text: str, min_chars: int) -> List[str]:
    if ";" not in text or _looks_like_semicolon_enumeration(text):
        return [text]
    positions = [m.end() for m in re.finditer(r";(?=\s+)", text)]
    return _split_at_positions(text, positions, min_chars=min_chars)


def _split_clause_markers(text: str, min_chars: int) -> List[str]:
    positions = [m.end() for m in _CLAUSE_MARKER_RE.finditer(text)]
    return _split_at_positions(text, positions, min_chars=min_chars)


def _split_long_fragment(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Deterministically wrap an unusually long fragment at safe whitespace.

    Commas are preferred.  Plain whitespace is a last resort.  The operation
    preserves source order and exact text reconstruction under one-space join.
    """
    if len(text) <= max_chars:
        return [text]

    out: List[str] = []
    remaining = text
    while len(remaining) > max_chars:
        lower = max(min_chars, int(max_chars * 0.55))
        window = remaining[: max_chars + 1]
        candidates = [m.end() for m in re.finditer(r",(?=\s+)", window) if m.end() >= lower]
        if candidates:
            cut = candidates[-1]
        else:
            whitespace = [m.start() for m in re.finditer(r"\s+", window) if m.start() >= lower]
            cut = whitespace[-1] if whitespace else max_chars
        left = remaining[:cut].strip()
        right = remaining[cut:].strip()
        if not left or not right:
            break
        out.append(left)
        remaining = right
    if remaining:
        out.append(remaining)
    return out


def _merge_short_fragments(fragments: Sequence[str], min_chars: int) -> List[str]:
    """Merge very short fragments with an adjacent unit without reordering text."""
    out: List[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if out and len(fragment) < min_chars:
            out[-1] = f"{out[-1]} {fragment}".strip()
        else:
            out.append(fragment)

    if len(out) >= 2 and len(out[0]) < min_chars:
        out[1] = f"{out[0]} {out[1]}".strip()
        out = out[1:]
    return out


def _cap_fragment_count(fragments: Sequence[str], max_evidence: int) -> List[str]:
    """Merge adjacent fragments deterministically until the cap is satisfied."""
    out = [x.strip() for x in fragments if x and x.strip()]
    if max_evidence < 1:
        raise ValueError("max_evidence must be >= 1")
    while len(out) > max_evidence:
        # Merge the adjacent pair with the smallest combined length.  Ties are
        # resolved by the earliest position, making the result deterministic.
        idx = min(range(len(out) - 1), key=lambda i: (len(out[i]) + len(out[i + 1]), i))
        out[idx : idx + 2] = [f"{out[idx]} {out[idx + 1]}".strip()]
    return out


def segment_text(
    text: str,
    *,
    aggressive: bool,
    max_chars: int,
    min_chars: int,
    max_evidence: int,
) -> List[str]:
    """Return deterministic, ordered evidence fragments for canonical ``text``."""
    if not text:
        return []
    if min_chars < 1:
        raise ValueError("min_chars must be >= 1")
    if max_chars < max(2 * min_chars, 40):
        raise ValueError("max_chars must be at least max(2*min_chars, 40)")

    primary = _primary_split(text)
    fragments: List[str] = []
    for sentence in primary:
        current = [sentence]
        if aggressive:
            expanded: List[str] = []
            for item in current:
                expanded.extend(_split_semicolons(item, min_chars=min_chars))
            current = expanded

            expanded = []
            for item in current:
                expanded.extend(_split_clause_markers(item, min_chars=min_chars))
            current = expanded

        expanded = []
        for item in current:
            expanded.extend(_split_long_fragment(item, max_chars=max_chars, min_chars=min_chars))
        fragments.extend(expanded)

    fragments = _merge_short_fragments(fragments, min_chars=min_chars)
    fragments = _cap_fragment_count(fragments, max_evidence=max_evidence)

    reconstructed = " ".join(fragments)
    if reconstructed != text:
        raise RuntimeError(
            "Evidence segmentation changed the canonical source text. "
            f"source_sha256={_sha256_text(text)} reconstructed_sha256={_sha256_text(reconstructed)}"
        )
    return fragments


def _evidence_spans(text: str, fragments: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    spans: Dict[str, Dict[str, Any]] = {}
    cursor = 0
    for index, fragment in enumerate(fragments, start=1):
        eid = f"E{index}"
        start = text.find(fragment, cursor)
        if start < 0:
            raise RuntimeError(f"Could not locate {eid} in canonical text after offset {cursor}")
        end = start + len(fragment)
        spans[eid] = {
            "start": start,
            "end": end,
            "text_sha256": _sha256_text(fragment),
        }
        cursor = end
    return spans


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Line {line_no} in {path} is not a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_requested_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    duplicates = [iid for iid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:10]}")
    return ids


def build_evidence_sentences(
    input_rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "input_id",
    text_field: str = "auto",
    aggressive_split: bool = False,
    max_chars: int = 420,
    min_chars: int = 24,
    max_evidence: int = 12,
    requested_ids: Optional[Sequence[str]] = None,
    allow_empty: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build evidence records and a deterministic summary."""
    by_id: Dict[str, Mapping[str, Any]] = {}
    input_order: List[str] = []
    for row_index, row in enumerate(input_rows, start=1):
        if id_field not in row:
            raise KeyError(f"Input row {row_index} is missing id field {id_field!r}")
        input_id = str(row[id_field]).strip()
        if not input_id:
            raise ValueError(f"Input row {row_index} has an empty {id_field!r}")
        if input_id in by_id:
            raise ValueError(f"Duplicate input_id: {input_id}")
        by_id[input_id] = row
        input_order.append(input_id)

    selected_ids = list(requested_ids) if requested_ids is not None else input_order
    missing = [iid for iid in selected_ids if iid not in by_id]
    if missing:
        raise KeyError(f"Requested IDs not found in input: {missing[:20]}")

    out_rows: List[Dict[str, Any]] = []
    field_counts: Counter[str] = Counter()
    evidence_counts: Counter[int] = Counter()
    empty_ids: List[str] = []
    long_evidence_count = 0

    for input_id in selected_ids:
        row = by_id[input_id]
        try:
            resolved_field, raw_text = _resolve_text(row, text_field=text_field)
        except KeyError:
            if not allow_empty:
                raise
            resolved_field, raw_text = text_field, ""

        if not raw_text:
            empty_ids.append(input_id)
            if not allow_empty:
                raise ValueError(f"Empty canonical text for {input_id}")
            fragments: List[str] = []
        else:
            fragments = segment_text(
                raw_text,
                aggressive=aggressive_split,
                max_chars=max_chars,
                min_chars=min_chars,
                max_evidence=max_evidence,
            )

        sentences = {f"E{i}": fragment for i, fragment in enumerate(fragments, start=1)}
        spans = _evidence_spans(raw_text, fragments)
        if set(sentences) != set(spans):
            raise RuntimeError(f"Evidence/spans ID mismatch for {input_id}")

        field_counts[resolved_field] += 1
        evidence_counts[len(fragments)] += 1
        long_evidence_count += sum(len(x) > max_chars for x in fragments)

        output: Dict[str, Any] = {
            "input_id": input_id,
            "raw_text": raw_text,
            "sentences": sentences,
            "evidence_spans": spans,
            "segmentation": {
                "version": SEGMENTATION_VERSION,
                "mode": "aggressive_clause" if aggressive_split else "sentence",
                "source_text_field": resolved_field,
                "source_text_sha256": _sha256_text(raw_text),
                "evidence_count": len(fragments),
                "reconstruction_sha256": _sha256_text(" ".join(fragments)),
                "parameters": {
                    "max_chars": max_chars,
                    "min_chars": min_chars,
                    "max_evidence": max_evidence,
                },
            },
        }
        if isinstance(row.get("provenance"), Mapping):
            output["provenance"] = dict(row["provenance"])
        out_rows.append(output)

    summary = {
        "segmentation_version": SEGMENTATION_VERSION,
        "records": len(out_rows),
        "requested_id_filter": requested_ids is not None,
        "aggressive_split": aggressive_split,
        "parameters": {
            "max_chars": max_chars,
            "min_chars": min_chars,
            "max_evidence": max_evidence,
            "allow_empty": allow_empty,
        },
        "source_text_field_counts": dict(sorted(field_counts.items())),
        "evidence_count_distribution": {
            str(k): evidence_counts[k] for k in sorted(evidence_counts)
        },
        "total_evidence_units": sum(k * v for k, v in evidence_counts.items()),
        "empty_record_count": len(empty_ids),
        "empty_record_ids": empty_ids,
        "evidence_over_max_chars_after_capping": long_evidence_count,
        "labels_or_predictions_used": False,
    }
    return out_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stable, citable evidence units from canonical CVE text."
    )
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output sentences JSONL")
    parser.add_argument("--id_field", default="input_id")
    parser.add_argument(
        "--text_field",
        default="auto",
        help="Text field name, or 'auto' for raw_text/text/description/cve_description",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Optional newline-delimited ID file; output follows this exact order",
    )
    parser.add_argument(
        "--aggressive_split",
        action="store_true",
        help="Additionally split evidence-linked clauses at fixed markers",
    )
    parser.add_argument("--max_chars", type=int, default=420)
    parser.add_argument("--min_chars", type=int, default=24)
    parser.add_argument("--max_evidence", type=int, default=12)
    parser.add_argument("--allow_empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ids_path = Path(args.ids) if args.ids else None
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if ids_path is not None and not ids_path.is_file():
        raise FileNotFoundError(ids_path)
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} or {manifest_path}. Use --overwrite."
        )

    input_rows = _read_jsonl(input_path)
    requested_ids = _load_requested_ids(ids_path)
    out_rows, summary = build_evidence_sentences(
        input_rows,
        id_field=args.id_field,
        text_field=args.text_field,
        aggressive_split=args.aggressive_split,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        max_evidence=args.max_evidence,
        requested_ids=requested_ids,
        allow_empty=args.allow_empty,
    )

    _write_jsonl(output_path, out_rows)
    manifest: Dict[str, Any] = {
        **summary,
        "input": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
        },
        "id_file": (
            {"path": str(ids_path), "sha256": _sha256_file(ids_path)}
            if ids_path is not None
            else None
        ),
        "output": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
