import argparse
from typing import Dict, Any, List
from tqdm import tqdm

from .io import read_jsonl, write_jsonl
from .llm import call_llm_extract
from .schema import validate_evidence_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True, help="sentences.jsonl")
    ap.add_argument("--output", required=True, help="extraction.jsonl")
    args = ap.parse_args()

    out: List[Dict[str, Any]] = []

    for row in tqdm(list(read_jsonl(args.sentences)), desc="extract"):
        input_id = row["input_id"]
        sentences: Dict[str, str] = row["sentences"]

        extraction = call_llm_extract(input_id, sentences)
        extraction["input_id"] = input_id

        valid = set(sentences.keys())
        errors = validate_evidence_ids(extraction, valid)

        prev = extraction.get("_validation_errors") or []
        if not isinstance(prev, list):
            prev = [str(prev)]
        extraction["_validation_errors"] = prev + errors

        out.append(extraction)

    write_jsonl(args.output, out)


if __name__ == "__main__":
    main()
