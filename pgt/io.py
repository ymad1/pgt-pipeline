# pgt/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Dict, Any

def read_jsonl(path: str):
    p = Path(path)
    # utf-8-sig: automatically strips BOM if present
    with p.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (file={p}, line={i})", e.doc, e.pos
                )

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
