# Reference results

These are the **published metric files** produced by the canonical runs, copied
here so you can compare your own re-runs against the thesis values. They are
small (JSON/CSV + two summary figures); large prediction dumps and per-fold
checkpoints are intentionally excluded.

All hold-out numbers use the **70/30** split (n_test = 6,000) on the 35-class
synthetic corpus.

## Unified hold-out comparison (thesis Chapter 5, "Cross-Method Comparison")

| Technique | Accuracy | macro-F1 | Files |
|-----------|:-------:|:--------:|-------|
| Classical ML (TF-IDF + SVM) | 0.8357 | 0.7905 | `ml_holdout/results.csv` |
| Embedding k-NN (MiniLM, k=5) | 0.4970 | 0.4264 | `embeddings/metrics_embeddings.json` |
| Zero-shot SLM (Qwen2.5-VL-7B) | 0.6682 | 0.5972 | `slm_holdout/metrics_summary.csv` (+ reports) |
| Few-shot SLM (1-shot) | 0.6723 | 0.6202 | `slm_holdout/metrics_summary.csv` (+ reports) |
| **LoRA fine-tuning (Qwen2.5-7B-Instruct)** | **0.8830** | **0.8542** | `lora_holdout/test_metrics.json` (+ report, confusion) |

Backing data + figure: `comparison/holdout_comparison_5techniques.png` and
`comparison/holdout_comparison_5techniques_data.csv`.

## 5-fold cross-validation (thesis Chapter 5)

Reported in the thesis for the four techniques that were cross-validated
(macro-F1): Classical ML 0.7766 ± 0.0062 · Embedding k-NN 0.4451 ± 0.0108 ·
Zero-shot 0.5952 · Few-shot 0.5944. **LoRA was not cross-validated** (the cost of
fine-tuning a 7B model five times was prohibitive — see the thesis and the
top-level README). Backing data + figure:
`comparison/cv5_comparison_ml_embeddings_finetuning.png` and its `_data.csv`.

## Notes

- `slm_holdout/` contains only the small metric artefacts (`metrics_summary.csv`,
  `classification_report_{zero,few}_shot.csv`, `confusion_matrix_{zero,few}_shot.csv`).
  The multi-MB prediction and error logs are **not** included.
- `lora_holdout/` corresponds to run **R15** (the corrected 70/30 LoRA hold-out),
  the canonical fine-tuning result reported in the thesis.
- Re-running a stage will reproduce these values up to the non-determinism noted
  in the top-level README (LM Studio, GPU kernels).
