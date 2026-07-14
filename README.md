# SLMs for email triage and intelligent sorting — experimental pipeline

This repository contains the **source code of the experimental pipeline** for the
Bachelor's thesis *"SLMs for email triage and intelligent sorting"* (SLMs para la
organización inteligente de correos). It lets you inspect and reproduce the
experiments that compare five email-classification techniques on a controlled,
synthetic, 35-class energy-sector corpus, plus a binary spam/ham baseline on a
real public dataset.

- **Repository:** <https://github.com/GuilleSuarezC/slms-for-email-triage-and-intelligent-sorting>
- **Scope:** research code + the synthetic corpus + reproduction instructions.
  It is **not** a production system: there is no deployed API, no email-client
  integration and no triage service (see [Integration](#integration-in-a-triage-flow)).

## What the pipeline does

Two experimental tracks:

- **Track A — binary spam/ham** on a real Kaggle dataset (~194k emails).
- **Track B — 35-class classification** on a synthetic energy-sector corpus
  (20,000 emails), comparing five techniques:
  1. **Classical ML** — TF-IDF / BoW / One-Hot / averaged Word2Vec × LogReg / SVM / RF.
  2. **Zero-shot SLM** — `qwen/qwen2.5-vl-7b` via a local LM Studio server.
  3. **Few-shot SLM** — same model, 1 example per class.
  4. **Embedding k-NN** — Sentence-BERT `all-MiniLM-L6-v2` + ChromaDB, k=5.
  5. **LoRA fine-tuning** — `Qwen/Qwen2.5-7B-Instruct`, QLoRA 4-bit.

Plus a **synthetic-corpus generator** (two local models via LM Studio).

All numerical/ML randomness is fixed with **seed 42**; SLM and LoRA inference are
**deterministic (temperature 0 / greedy)**.

## Project status

| Component | Status |
|-----------|--------|
| Classical ML (Tracks A & B) | ✅ Implemented · reproducible on CPU (Track A needs Kaggle download) |
| Embedding k-NN baseline | ✅ Implemented · reproducible on CPU (downloads the embedding model) |
| Zero/few-shot SLM classification | ✅ Implemented · **requires a local LM Studio server + the SLM** (long run) |
| LoRA fine-tuning | ✅ Implemented · **requires an NVIDIA GPU** (QLoRA 4-bit) |
| Synthetic corpus generator | ✅ Implemented · **requires local LM Studio models** |
| Synthetic 35-class corpus | ✅ Included in `data/` |
| Local inference demo (Streamlit) | ✅ Implemented (research demo only) |
| Published reference metrics | ✅ Included in `results/` |
| Real spam/ham dataset · SLM weights · LoRA adapter · merged model | ⛔ Not distributed — see [`resources/README.md`](resources/README.md) |
| HTTP/REST API · email-client integration · triage queue/service · deployment/monitoring | ⛔ Not implemented — **out of scope** of the thesis |

## Repository structure

```
.
├── README.md                 # this file
├── LICENSE                   # MIT (set the copyright holder before publishing)
├── requirements.txt          # core CPU dependencies
├── requirements-lora.txt     # GPU fine-tuning stack (LoRA only)
├── Dockerfile / .dockerignore
├── .gitignore / .env.example
├── configs/README.md         # reproducibility parameter map (seeds, splits, hyper-params)
├── data/
│   ├── email_dataset.csv     # 35-class synthetic corpus (20,000 rows)
│   ├── metadata.csv          # per-email generation metadata
│   └── DATA_CARD.md          # synthetic-data disclaimer + provenance
├── src/
│   ├── generation/           # synthetic-corpus generator (LM Studio)
│   ├── classical_ml/
│   │   ├── binary_spam/      # Track A
│   │   └── multiclass_35/    # Track B — classical ML
│   ├── slm_prompting/        # Track B — zero/few-shot SLM (hold-out + CV)
│   ├── embeddings/           # Track B — Sentence-BERT + ChromaDB k-NN
│   ├── lora/                 # Track B — LoRA fine-tuning (self-contained project)
│   └── monitoring/           # CPU/RAM/GPU monitoring harness (classical ML)
├── scripts/                  # run_*.sh reproduction wrappers + comparison plots
├── tests/                    # offline smoke tests (config, schema, byte-compile)
├── examples/                 # 12-row corpus excerpt + SCHEMA.md
├── resources/README.md       # how to obtain the non-distributed resources
└── results/                  # published reference metrics + comparison figures
```

## Requirements

- **Python 3.11+** (developed against 3.11; also runs on 3.13).
- OS: Linux/macOS/Windows. The classical-ML and embedding tracks are CPU-only.
- **NVIDIA GPU** — only for LoRA fine-tuning (`requirements-lora.txt`, QLoRA 4-bit
  via `bitsandbytes`). Do not assume a CUDA version; install the matching PyTorch
  build from <https://pytorch.org/get-started/locally/>.
- **LM Studio** (local, OpenAI-compatible server) — only for the synthetic-corpus
  generator and the zero/few-shot SLM track. No cloud API is used.
- **Kaggle credentials** — only for Track A (see `.env.example`).

## Local installation

```bash
# 1. Clone
git clone https://github.com/GuilleSuarezC/slms-for-email-triage-and-intelligent-sorting.git
cd slms-for-email-triage-and-intelligent-sorting

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Core dependencies (classical ML + embeddings + SLM clients + plots)
pip install -r requirements.txt

# 4. (Optional) LoRA fine-tuning stack — needs an NVIDIA GPU
#    Install the CUDA build of PyTorch first (see pytorch.org), then:
pip install -r requirements-lora.txt

# 5. (Optional) credentials for Track A
cp .env.example .env                 # then fill in KAGGLE_USERNAME / KAGGLE_KEY
```

Verify the environment with the offline smoke tests (no data/GPU/LM Studio needed):

```bash
python tests/run_smoke.py
```

## Running with Docker

The image covers the **CPU-reproducible** parts (classical ML, embeddings, tests).
SLM (needs host LM Studio) and LoRA (needs host GPU) are not run inside it.

```bash
# Build
docker build -t slms-email-triage .

# Verify the image (runs the offline smoke tests by default)
docker run --rm slms-email-triage

# Run a stage, mounting the dataset as a volume
docker run --rm -v "$(pwd)/data:/app/data" slms-email-triage \
    python src/embeddings/task1_knn_chromadb.py --csv_path data/email_dataset.csv --k 5
```

## Data and resources

- **Included:** the synthetic 35-class corpus (`data/email_dataset.csv`) and its
  metadata. It is **100% synthetic** — no real personal data. Read
  [`data/DATA_CARD.md`](data/DATA_CARD.md) before use (it explains the realistic
  but fabricated PII and the use of real company domains in fake addresses).
- **Not included** (large / third-party / regenerated): the real Kaggle spam
  dataset, the SLM weights, the LoRA adapter (~161 MB), the merged model (~15 GB),
  and the ChromaDB store. Each is documented in
  [`resources/README.md`](resources/README.md) with its expected location and how
  to obtain or regenerate it.

## Reproducing the pipeline

Run from the repository root, with `.venv` activated. Each wrapper in `scripts/`
encodes the correct paths and the canonical settings.

| Stage | Command | Needs | Output |
|-------|---------|-------|--------|
| 0. Generate corpus *(optional; corpus already in `data/`)* | `python src/generation/energy35C_email_dataset_generator_v11.py` | LM Studio + 2 local models | a new `email_dataset.csv` |
| A. Binary spam/ham ML | `bash scripts/run_binary_ml.sh` | Kaggle creds (auto-download) | metrics per classifier |
| B1. 35-class classical ML | `bash scripts/run_ml_35class.sh` | CPU only | metrics per model×vectorizer |
| B2. Embedding k-NN | `bash scripts/run_embeddings.sh` | downloads MiniLM | `metrics_embeddings.json` |
| B3. Zero/few-shot SLM | `bash scripts/run_slm_holdout.sh` | **LM Studio + Qwen2.5-VL-7B** | predictions + `metrics_summary.csv` |
| B4. LoRA fine-tuning | `bash scripts/run_lora_holdout.sh` | **NVIDIA GPU** | adapter + `test_metrics.json` |
| Plots | `python scripts/plots/plot_holdout_comparison.py` | CPU | comparison figures |

Notes:

- **B3 is very long** (~17 h, ~12,000 LLM calls for the full 6,000-sample test
  partition). Add `--max-rows N` to try a small subset.
- **B4 requires a GPU.** To reproduce the *published* LoRA result you must use the
  **70/30** split — see the discrepancy note below.

## Reproducing the published results

The canonical metrics are in `results/` (see [`results/README.md`](results/README.md)).
Hold-out (70/30, n_test = 6,000) macro-F1, matching the thesis "Cross-Method
Comparison" table:

| Technique | Accuracy | macro-F1 | Reference file |
|-----------|:-------:|:--------:|----------------|
| Classical ML (TF-IDF + SVM) | 0.8357 | 0.7905 | `results/ml_holdout/results.csv` |
| Embedding k-NN (MiniLM) | 0.4970 | 0.4264 | `results/embeddings/metrics_embeddings.json` |
| Zero-shot SLM | 0.6682 | 0.5972 | `results/slm_holdout/metrics_summary.csv` |
| Few-shot SLM (1-shot) | 0.6723 | 0.6202 | `results/slm_holdout/metrics_summary.csv` |
| **LoRA fine-tuning** | **0.8830** | **0.8542** | `results/lora_holdout/test_metrics.json` |

If a result cannot be fully reproduced, it is because a required resource is not
distributed (SLM weights, GPU, real dataset) — see `resources/README.md`.

## Reproducibility & known discrepancies

- **Seeds:** global seed **42** across all tracks (`RANDOM_STATE = 42` in each
  script; `seed: 42` in `src/lora/config.yaml`).
- **Determinism:** SLM classification uses **temperature 0**; LoRA inference is
  **greedy**. Data splits are stratified and serialized. Corpus *generation* uses
  stochastic sampling, but records the per-email temperature/top_p/top_k/seed so it
  remains auditable.
- **Non-determinism that remains:** LM Studio inference and GPU kernels are not
  guaranteed to be bit-exact reproducible even at temperature 0; exact SLM/LoRA
  numbers may vary slightly across hardware and runtime versions. The classical-ML
  and embedding tracks are fully deterministic given the fixed seed.

Discrepancies between the shipped code and the thesis, documented rather than
silently "fixed":

1. **LoRA hold-out split.** `src/lora/config.yaml` ships `data.test_size: 0.20`
   (an 80/20 split, from a superseded run). The **published** LoRA result
   (macro-F1 0.8542) uses a **70/30** split. To reproduce it, set
   `data.test_size: 0.30` before running the LoRA pipeline. The config is committed
   verbatim on purpose.
2. **Learning rate / batch size.** `config.yaml` records `lr = 2e-4` and an
   effective batch of `1 × 16`; the thesis text does not state these. The config is
   the source of truth here.
3. **Base-model mismatch.** Zero/few-shot uses the multimodal `Qwen2.5-VL-7B`,
   while LoRA fine-tunes the text-only `Qwen2.5-7B-Instruct`. The thesis
   acknowledges this affects the cross-technique comparison.
4. **Generator model alias.** The corpus metadata records `google/gemma-4-e4b`,
   which is a local-runtime alias, not a verified official model name.
5. **LoRA 5-fold CV was not completed** (only 3 of 5 folds ran; the cost of
   fine-tuning a 7B model five times was prohibitive). LoRA therefore has **no**
   cross-validated number — the thesis states this explicitly.

## Integration in a triage flow

**Implemented (research only).**
- Batch classifiers that map an email to one of 35 labels (the scripts above).
- A **local inference demo** (`src/lora/app/streamlit_app.py`) that classifies a
  single pasted email using the local LoRA model or an LM Studio server:
  ```bash
  streamlit run src/lora/app/streamlit_app.py   # needs the LoRA adapter or LM Studio
  ```
- Programmatic single-email inference in `src/lora/src/infer.py`.

**Intended interface (as described in the thesis).**
- Conceptually, the classifier is the component that would assign an incoming
  email to a category so it can be routed. The thesis discusses this as the target
  use case; it does not define an API contract for it.

**Not implemented (explicitly out of scope of the thesis — §1.7).**
- No HTTP/REST API, no IMAP/SMTP or email-client plugin, no message queue or
  microservice, no dashboard, and no deployment/monitoring. The thesis states:
  *"Production deployment infrastructure, long-term monitoring and operational
  integration are not covered."* This repository does not add any of these.

## Limitations and resources not included

- The **real spam/ham dataset** is not redistributed (download via Kaggle; license
  to be verified — see `resources/README.md`).
- The **SLM weights**, **LoRA adapter** and **merged model** are not shipped (size /
  external); regenerate them with the pipeline or load them via LM Studio / HF.
- Full reproduction of the **SLM** and **LoRA** results requires substantial
  resources (LM Studio + a 7B model and ~17 h for SLM; an NVIDIA GPU for LoRA).
- Results characterise a **synthetic benchmark**, not real-world performance.
- **License holder** and the **Kaggle dataset license** are open items (below).

## Relation to the thesis document

| Thesis section | Code | Command | Reference result |
|----------------|------|---------|------------------|
| Data generation (§4) | `src/generation/` | stage 0 | `data/` |
| Classical ML baselines (§4–5) | `src/classical_ml/` | `run_binary_ml.sh`, `run_ml_35class.sh` | `results/ml_holdout/` |
| Embedding baseline (§4–5) | `src/embeddings/` | `run_embeddings.sh` | `results/embeddings/` |
| Zero/few-shot SLM (§4–5) | `src/slm_prompting/` | `run_slm_holdout.sh` | `results/slm_holdout/` |
| LoRA fine-tuning (§4–5) | `src/lora/` | `run_lora_holdout.sh` | `results/lora_holdout/` |
| Reproducibility (§4) | `configs/README.md`, seeds in every script | — | — |
| Cross-method comparison (§5) | `scripts/plots/` | plot scripts | `results/comparison/` |

## Citation

Please cite the associated Bachelor's thesis (metadata taken from its title page):

> Suárez Corrales, Guillermo. *SLMs para la organización inteligente de correos*
> (SLMs for email triage and intelligent sorting). Bachelor's thesis, Escuela
> Politécnica de Ingeniería de Gijón, Universidad de Oviedo, July 2026.
> Tutors: Luciano Sánchez Ramos and Diego García Vega.

No DOI is available for the thesis.

## License

The **source code** is released under the **MIT License** (see `LICENSE`;
copyright holder: Guillermo Suárez Corrales, the thesis author — add the
University of Oviedo / EPI Gijón there too if the institution holds rights). The
MIT license does not automatically extend to the synthetic dataset
(`data/DATA_CARD.md`) or to the third-party Kaggle dataset (its own license applies).
