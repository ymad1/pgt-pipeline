# Proof-Grounded CVE-to-ATT&CK Pipeline

This repository implements the reproducible pipeline used to map CVE descriptions to MITRE ATT&CK techniques with evidence-grounded extraction, local attack graphs, Minimal Explainable Subgraphs (MES), candidate retrieval, controlled LLM reranking, and held-out evaluation.

The formal experiment is executed through one fail-fast entry point:

```bash
python tools/run_pipeline.py --config configs/reviewer2_experiment.json
```

The orchestrator is the source of truth for the paper experiment. Older exploratory scripts may remain in the repository for historical reference, but they are not part of the formal pipeline unless they are called by `tools/run_pipeline.py`.

## Formal pipeline

```text
Raw X/y CSV files
  -> canonical CVE-level records and label unions
  -> fixed, leakage-free development/test split
  -> versioned active Enterprise ATT&CK index
  -> deterministic evidence segmentation (E1...En)
  -> evidence-linked structured extraction
  -> local attack graph
  -> Minimal Explainable Subgraph (MES)
  -> development-only retrieval parameter selection
  -> evidence/MES candidate retrieval
  -> development-only beta selection
  -> four controlled test rerankers
  -> repeated-run evaluation and statistical tests
  -> coverage audits and final run manifest
```

The four held-out reranking conditions use the same CVEs, candidate set, model settings, candidate budget, and random seeds:

| Mode | Information supplied to the reranker |
|---|---|
| `generic` | CVE text and candidate technique definitions |
| `evidence` | Generic inputs plus numbered evidence units |
| `structure` | Generic inputs plus MES structure |
| `full` | Generic inputs plus evidence and MES |

## Reproducibility safeguards

The formal workflow enforces the following controls:

- Base-CVE deduplication and union of labels from duplicate or augmented records.
- A fixed multilabel development/test split with explicit overlap checks.
- Development-only selection of retrieval `alpha`, candidate budget `Top-N`, and reranking `beta`.
- Fixed model snapshots, temperatures, seeds, token budgets, retry settings, and prompt hashes.
- Active ATT&CK technique filtering with ATT&CK version and source-file hashes.
- Deterministic evidence IDs and source-text integrity checks.
- Schema validation at every JSONL boundary.
- MES nodes and edges must be a true subgraph of the corresponding local graph.
- Identical candidate sets across reranking conditions.
- Atomic writes, resume protection, file hashes, stage manifests, and a final run manifest.
- Repeated runs, bootstrap confidence intervals, McNemar tests, paired sign-flip tests, and Holm correction in the formal evaluation.

## Repository layout

```text
configs/
  reviewer2_experiment.json       Version-controlled formal experiment settings

data/
  attack/enterprise-attack.json   Local Enterprise ATT&CK STIX bundle
  cve2attck_src_20260107/         Source X/y CSV files

pgt/
  split_sentences.py              Deterministic evidence segmentation
  llm.py                          Evidence-linked structured extraction
  extract.py                      Extraction -> graph -> MES orchestrator
  build_local_graph.py            Local attack graph construction
  build_mes.py                    Minimal Explainable Subgraph construction
  export_attack_stix.py           Versioned active ATT&CK export
  retrieve_candidates.py          Evidence/MES retrieval
  sweep_retrieval.py              Development-only alpha and Top-N selection
  rerank.py                       Four controlled reranking modes
  sweep_beta_offline.py           Development-only beta selection
  compare_rankers.py              Formal repeated-run statistical evaluation
  analyze_missing_gold.py         Gold/index/candidate coverage audit
  schema.py                       Shared record contracts
  io.py                           Validated atomic JSONL I/O

tools/
  run_pipeline.py                 Formal end-to-end entry point
  make_cve2attck_jsonl.py         Canonical dataset construction
  make_fixed_splits.py            Fixed leakage-free split creation
  check_labels_alignment.py       Dataset and split audit
  check_hits.py                   Quick metric check using formal definitions
```

Generated experiment artifacts are written under `runs/` and are intentionally ignored by Git.

## Environment setup

Supported target: Python 3.10-3.12.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before the final paper run, freeze the exact installed environment:

```bash
python -m pip freeze > requirements-lock.txt
```

Commit the lock file together with the final experiment configuration and code revision used for the reported results.

## OpenAI configuration

LLM stages require an API key. The key itself is never written to experiment manifests.

Environment-variable configuration:

```bash
# Linux/macOS
export OPENAI_API_KEY="..."

# Windows PowerShell
$env:OPENAI_API_KEY="..."
```

An optional local `secrets.json` may also be used:

```json
{
  "OPENAI_API_KEY": "..."
}
```

`secrets.json`, `.env`, virtual environments, and `runs/` are ignored by Git. Optional runtime variables include `OPENAI_BASE_URL`, `OPENAI_PROXY`, `OPENAI_TIMEOUT`, `OPENAI_CONNECT_TIMEOUT`, `OPENAI_MAX_RETRIES`, `OPENAI_HTTP2`, and `OPENAI_TRUST_ENV`.

## Input requirements

The version-controlled configuration expects:

```text
data/cve2attck_src_20260107/X_train.csv
data/cve2attck_src_20260107/y_train.csv
data/cve2attck_src_20260107/X_test.csv
data/cve2attck_src_20260107/y_test.csv
data/attack/enterprise-attack.json
```

The canonical builder validates row alignment, technique-name mapping, duplicate CVEs, label types, empty labels, and source provenance. The fixed-split stage then merges the source partitions at base-CVE level before creating the development and held-out test sets.

## Recommended execution sequence

### 1. Inspect the configuration

```bash
cat configs/reviewer2_experiment.json
```

A fresh template can be generated without running the pipeline:

```bash
python tools/run_pipeline.py \
  --write_default_config configs/local_experiment.json
```

### 2. Print the full execution plan

This performs no API calls and does not create experiment outputs:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json \
  --plan
```

### 3. Run a small smoke experiment

Start with a small number of development and test CVEs to verify credentials, model access, schemas, paths, and stage interfaces before paying for a full run:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json \
  --smoke 2 \
  --overwrite
```

Smoke-run parameter selections and metrics are interface checks only. They must not be reported as paper results.

### 4. Run the formal experiment

Use a new or empty workspace and the committed configuration:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json
```

Do not use `--overwrite` on a completed formal run unless intentionally discarding it. For an interrupted run with the same configuration, use:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json \
  --resume
```

Resume is rejected when the configuration hash differs from the existing run state.

## Stage-level execution

Run all dependencies through a named stage:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json \
  --through extract
```

Run only one stage after its prerequisites already exist:

```bash
python tools/run_pipeline.py \
  --config configs/reviewer2_experiment.json \
  --stage evaluate
```

Formal stage order:

```text
data -> attack -> segment -> extract -> select_retrieval -> retrieve
-> rerank_dev -> select_beta -> rerank_test -> evaluate -> audit
```

## Output structure

The default workspace is:

```text
runs/reviewer2_v2/
```

Important outputs include:

```text
fixed_split/
  development/                    Fixed development records, labels, and IDs
  test/                           Held-out test records, labels, and IDs
  split_manifest.json

attack_cache/
  technique_text_index.jsonl
  attack_kg.json
  attack_manifest.json

pipeline/
  sentences.jsonl
  extraction.jsonl
  local_graphs/
  mes.jsonl
  candidates.jsonl

retrieval_selection/
  retrieval_sweep.csv
  selected_retrieval.json
  retrieval_selection_manifest.json

beta_selection/
  beta_sweep.csv
  selected_beta.json

reranking/test/
  generic/seed_*.jsonl
  evidence/seed_*.jsonl
  structure/seed_*.jsonl
  full/seed_*.jsonl

evaluation/test/
  summary and per-method/per-technique statistical outputs
  evaluation_manifest.json

audit/test_candidate_coverage/
  gold/index/candidate diagnostics
  audit_manifest.json

pipeline_state.json
final_run_manifest.json
```

`final_run_manifest.json` is created only after all required stages and audit artifacts exist. It records the configuration hash, selected retrieval settings, selected beta, and hashes of the principal artifacts.

## Evaluation definitions

The formal evaluator reports sample-level and label-level ranking metrics, including:

- Hit@K, Precision@K, Recall@K, and AP@K.
- Mean Reciprocal Rank.
- CVE-macro and label-micro summaries.
- Technique-macro recall.
- Head/long-tail technique performance.
- Bootstrap confidence intervals across CVEs.
- Paired significance tests between methods with multiple-comparison correction.

The quick checker in `tools/check_hits.py` imports the same metric implementation, but it is not a substitute for the formal repeated-run evaluation in `pgt.compare_rankers`.

## Formal-run checklist

Before reporting results, verify all of the following:

1. The Git commit hash and `configs/reviewer2_experiment.json` are archived.
2. `requirements-lock.txt` reflects the environment used for the run.
3. The ATT&CK STIX bundle and its hash match `attack_manifest.json`.
4. Development/test overlap is zero.
5. No fallback extraction is present in the formal run.
6. Retrieval parameters were selected only from the development split.
7. Beta was selected only from the development split.
8. All four test methods use the same candidates and seeds.
9. All expected reranking runs completed without missing CVEs.
10. `final_run_manifest.json` and evaluation manifests exist.
11. Smoke outputs are kept separate from formal outputs.
12. Reported paper tables are generated only from the held-out test evaluation directory.

## Scope and limitations

This repository makes the computational procedure auditable; it does not guarantee identical LLM outputs across different providers, model revisions, or unavailable model snapshots. The manifests therefore record model names, prompts, seeds, temperatures, token budgets, retry settings, prompt hashes, and provider fingerprints when available.

No formal paper result is bundled or claimed merely because the pipeline compiles or a smoke run succeeds. Full results require real API execution on the fixed development and held-out test sets, followed by the generated statistical evaluation and audit checks.
