# PGT (Proof-Grounded TTP) Pipeline Starter

This starter kit follows the checklist in `pgt步骤.txt`:

- Step 2: sentence splitting + evidence ids (E1..En)
- Step 3: structured extraction (LLM stub + strict schema with evidence_ids)
- Step 4: build local attribution graph per input
- Step 5: export ATT&CK techniques from STIX bundle
- Step 6-11: candidate retrieval, path generation, budget-aware evidence pack, verifier, evaluation

## Quickstart

### 1) Create evidence-numbered sentences
```bash
python -m pgt.split_sentences \
  --input data/raw/triage_samples.jsonl \
  --text_field text \
  --id_field input_id \
  --output data/processed/sentences.jsonl
```

### 2) Create extractions (LLM stub)
This repo contains an **LLM call placeholder**. You can:
- call your own model/provider inside `pgt/llm.py`
- or import your TRIAGE repo inference code and return JSON that matches `pgt/schema.py`

```bash
python -m pgt.extract \
  --sentences data/processed/sentences.jsonl \
  --output runs/extract/dev/extraction.jsonl
```

### 3) Build local graphs
```bash
python -m pgt.build_local_graph \
  --extraction runs/extract/dev/extraction.jsonl \
  --output_dir runs/graphs/dev/local_graphs
```

### 4) Export ATT&CK techniques from STIX (offline)
Download `enterprise-attack.json` from `mitre-attack/attack-stix-data` (or use your local copy),
then:
```bash
python -m pgt.export_attack_stix \
  --stix_bundle path/to/enterprise-attack.json \
  --attack_kg data/attack/attack_kg.json \
  --tech_index data/attack/technique_text_index.jsonl
```

### 5) Retrieve candidates (text+graph fusion baseline)
```bash
python -m pgt.retrieve_candidates \
  --sentences data/processed/sentences.jsonl \
  --local_graph_dir runs/graphs/dev/local_graphs \
  --attack_kg data/attack/attack_kg.json \
  --tech_index data/attack/technique_text_index.jsonl \
  --output runs/retrieval/dev/candidates.jsonl
```

### 6) Generate paths + evidence packs + verify + evaluate
```bash
python -m pgt.generate_paths --candidates runs/retrieval/dev/candidates.jsonl --local_graph_dir runs/graphs/dev/local_graphs --output runs/paths/dev/paths.jsonl
python -m pgt.build_evidence_pack --sentences data/processed/sentences.jsonl --candidates runs/retrieval/dev/candidates.jsonl --paths runs/paths/dev/paths.jsonl --output runs/evidence_pack/dev/evidence_pack.jsonl
python -m pgt.verify --sentences data/processed/sentences.jsonl --local_graph_dir runs/graphs/dev/local_graphs --reasoning runs/reason/dev/reasoning.jsonl --evidence_pack runs/evidence_pack/dev/evidence_pack.jsonl --output runs/verify/dev/verifier_report.jsonl

python -m pgt.eval_rank --gold data/raw/triage_samples.jsonl --pred runs/retrieval/dev/candidates.jsonl --gold_field techniques --k 1 5 10
```

## Dependencies
- Python 3.10+
- `pip install -r requirements.txt`

If you prefer minimal deps, remove `networkx` and store graphs as edge lists only.
