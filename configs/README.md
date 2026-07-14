# Configuration & reproducibility parameter map

There is **one YAML config file** in this project — the LoRA pipeline's
`src/lora/config.yaml` (its loader reads `config.yaml` from the working
directory, so it lives next to the LoRA scripts). The other tracks keep their
settings as **module-level constants** at the top of each script. This page maps
every reproducibility-critical value to its single source of truth, so nothing
here duplicates (and can drift from) the actual code.

## Global invariants (stated in the thesis)

| Parameter | Value | Source of truth |
|-----------|-------|-----------------|
| Random seed (all tracks) | **42** | each script (`RANDOM_STATE = 42`) and `src/lora/config.yaml` (`seed: 42`) |
| SLM / LoRA inference | **temperature 0 / greedy** | `TEMPERATURE = 0.0` in the SLM scripts; `infer.greedy: true` in `src/lora/config.yaml` |

## Per-track settings

| Track | File | Key constants |
|-------|------|---------------|
| Binary spam/ham ML (Track A) | `src/classical_ml/binary_spam/spam_classifier_pipeline.py` | `RANDOM_STATE=42`, `TEST_SIZE=0.30`, `CV_FOLDS=10`, `OPTUNA_TRIALS=30`, `MAX_FEATURES=20_000` |
| 35-class ML — CV variant | `src/classical_ml/multiclass_35/email_classifier_nuevo_dataset.py` | `RANDOM_STATE=42`, `TEST_SIZE=0.30`, `N_SPLITS_CV=5` |
| 35-class ML — hold-out (canonical) | `src/classical_ml/multiclass_35/email_classifier_nuevo_dataset_holdout.py` | `RANDOM_STATE=42`; split via `--test_size` (use `0.30` for the 70/30 result) |
| Zero/few-shot SLM — hold-out (canonical) | `src/slm_prompting/llm_email_classifier_holdout_v4.py` | `MODEL_NAME="qwen/qwen2.5-vl-7b"`, `TEMPERATURE=0.0`, `N_FEW_SHOT_PER_CLASS=1`; split via `--test-size` |
| Zero/few-shot SLM — 5-fold CV | `src/slm_prompting/llm_email_classifier_cv_v4.py` | `MODEL_NAME="qwen/qwen2.5-vl-7b"`, `TEMPERATURE=0.0`, `N_SPLITS=5`, `N_FEW_SHOT_PER_CLASS=1` |
| Embedding k-NN | `src/embeddings/task1_knn_chromadb.py` | `DEFAULT_CONFIG`: `k_neighbors=5`, `random_state=42`, `test_size`, `cv_folds` (all overridable via CLI) |
| LoRA fine-tuning | `src/lora/config.yaml` | `lora.r=16`, `lora.alpha=32`, `lora.dropout=0.05`, `train.epochs=3`, `train.lr=0.0002`, effective batch `1×16`, `model.id="Qwen/Qwen2.5-7B-Instruct"`, `load_in_4bit: true` |

## ⚠️ Documented discrepancy — LoRA hold-out split

`src/lora/config.yaml` ships `data.test_size: 0.20` (an 80/20 split, matching an
earlier superseded run). The **published canonical LoRA result** in the thesis
(accuracy 0.883, macro-F1 0.8542) uses a **70/30** split. To reproduce that
result, set `data.test_size: 0.30` before running the LoRA pipeline. The config
is committed **verbatim** (unchanged) on purpose; see the "Reproducibility &
known discrepancies" section of the top-level `README.md`.
