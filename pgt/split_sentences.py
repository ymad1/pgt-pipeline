# pgt/split_sentences.py
"""
Step 2: Split raw text into evidence sentences E1..En.

Input (JSONL): each line at least contains:
  - id_field (default: "input_id")
  - text_field (default: "text")

Output (JSONL): each line:
{
  "input_id": "...",
  "raw_text": "...",
  "sentences": {"E1": "...", "E2": "...", ...}
}

Run:
  python -m pgt.split_sentences --input data/raw/triage_samples.jsonl --output data/processed/sentences.jsonl --aggressive_split
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any


# -----------------------
# Sentence splitting rules
# -----------------------

# Primary split:
# - split on newlines
# - split on . ! ? followed by whitespace
# - avoid splitting around numbers like "2.12.3" (not perfect, but helps)
_SENT_END_RE = re.compile(
    r"""
    (?:\r?\n)+
    |
    (?<!\d)[.!?](?!\d)\s+                 # 非数字结尾句子
    |
    (?<=\d)[.!?]\s+(?=[A-Z])              # 数字结尾句子：如 "5.0.3. Due ..."
    """,
    re.VERBOSE,
)


# Secondary split keywords: keep keyword with the right-hand segment.
_KEYWORD_SPLIT_RE = re.compile(r"\s+(aka|allows|via|through)\s+", re.IGNORECASE)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _primary_split(text: str) -> List[str]:
    text = _normalize_ws(text)
    if not text:
        return []
    parts = _SENT_END_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _looks_like_semicolon_list(s: str) -> bool:
    """
    Heuristic: if a sentence contains many semicolons and each segment is short,
    it's probably a product/version enumeration list. Don't split by ';' in that case.
    """
    segs = [x.strip() for x in s.split(";") if x.strip()]
    if len(segs) >= 5:
        avg_len = sum(len(x) for x in segs) / max(len(segs), 1)
        if avg_len <= 45:
            return True
    return False


def _secondary_split_one(sentence: str) -> List[str]:
    """
    Aggressive split of a single sentence:
    1) split by keywords (aka/allows/via/through) and keep keyword in RHS
    2) optionally split by ';' if it's not an enum list
    3) optionally split by ', ' only when very long and comma-heavy, and avoid too-short fragments
    """
    s = sentence.strip()
    if not s:
        return []

    # 1) keyword split (rebuild keeping the keyword)
    pieces = _KEYWORD_SPLIT_RE.split(s)
    # pieces: [pre, kw1, post1, kw2, post2, ...] OR [s] if no keyword
    if len(pieces) == 1:
        chunks = [s]
    else:
        chunks: List[str] = []
        pre = pieces[0].strip()
        if pre:
            chunks.append(pre)
        i = 1
        while i + 1 < len(pieces):
            kw = pieces[i].strip()
            post = pieces[i + 1].strip()
            if post:
                chunks.append(f"{kw} {post}".strip())
            i += 2

    # 2) semicolon split (only if not enum list)
    tmp: List[str] = []
    for c in chunks:
        if ";" in c and not _looks_like_semicolon_list(c):
            tmp.extend([x.strip() for x in c.split(";") if x.strip()])
        else:
            tmp.append(c)
    chunks = tmp

    # 3) comma split (very conservative)
    tmp = []
    for c in chunks:
        if len(c) >= 320 and c.count(",") >= 4:
            segs = [x.strip() for x in c.split(", ") if x.strip()]

            merged: List[str] = []
            buf = ""
            for seg in segs:
                if not buf:
                    buf = seg
                elif len(buf) < 80:
                    buf = f"{buf}, {seg}"
                else:
                    merged.append(buf)
                    buf = seg
            if buf:
                merged.append(buf)

            tmp.extend(merged)
        else:
            tmp.append(c)
    chunks = tmp

    return [x.strip() for x in chunks if x and x.strip()]


def aggressive_secondary_split(
    sentences: List[str],
    max_len: int = 220,
    max_evidence: int = 10,
) -> List[str]:
    """
    Apply secondary splitting if:
      - only 1 primary sentence, OR
      - any sentence is longer than max_len

    Also cap the number of evidences to max_evidence.
    """
    if not sentences:
        return []

    need = (len(sentences) <= 1) or any(len(s) > max_len for s in sentences)
    if not need:
        return sentences

    out: List[str] = []
    for s in sentences:
        if len(s) > max_len or len(sentences) <= 1:
            out.extend(_secondary_split_one(s))
        else:
            out.append(s)

    out = [x for x in out if x]

    # Cap evidence count: if too many, merge the tail into the last slot
    if len(out) > max_evidence:
        head = out[: max_evidence - 1]
        tail = " ".join(out[max_evidence - 1 :]).strip()
        if tail:
            head.append(tail)
        out = head

    return out


# -----------------------
# IO helpers
# -----------------------

def _read_jsonl(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: List[Dict[str, Any]], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------
# Main
# -----------------------

def build_evidence_sentences(
    input_rows: List[Dict[str, Any]],
    id_field: str,
    text_field: str,
    aggressive_split: bool,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(input_rows):
        if id_field not in row:
            raise KeyError(f"Row {idx} missing id_field='{id_field}'. Keys={list(row.keys())}")
        if text_field not in row:
            raise KeyError(f"Row {idx} missing text_field='{text_field}'. Keys={list(row.keys())}")

        input_id = str(row[id_field]).strip()
        raw_text = str(row[text_field])

        sent_list = _primary_split(raw_text)
        if aggressive_split:
            sent_list = aggressive_secondary_split(sent_list)

        sentences = {f"E{i+1}": s for i, s in enumerate(sent_list) if s.strip()}

        out_rows.append(
            {
                "input_id": input_id,
                "raw_text": _normalize_ws(raw_text),
                "sentences": sentences,
            }
        )

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PGT Step2: split into evidence sentences.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--id_field", default="input_id", help="ID field in input JSON")
    parser.add_argument("--text_field", default="text", help="Text field in input JSON")
    parser.add_argument(
        "--aggressive_split",
        action="store_true",
        help="Enable secondary splitting for long/1-sentence samples",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = _read_jsonl(in_path)
    out_rows = build_evidence_sentences(
        rows,
        id_field=args.id_field,
        text_field=args.text_field,
        aggressive_split=args.aggressive_split,
    )
    _write_jsonl(out_path, out_rows)


if __name__ == "__main__":
    main()
