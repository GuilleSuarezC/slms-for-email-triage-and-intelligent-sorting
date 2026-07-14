#!/usr/bin/env python3
"""Reads: 35classPKL/MLcon35Clases/plots/results.csv (tfidf+svm row, cv_* columns),
Embeddings/outputs_task1/task1.out (verified CV-5 numbers hardcoded below, log is not
machine-parseable), FineTuning/email-classifier-lora/outputs/cv_fold_{0,1,2}/val_metrics.json.
Writes: evidence/plots/cv5_comparison_ml_embeddings_finetuning.png + _data.csv.

The LoRA bar is a DELIBERATELY abandoned/partial CV-5 attempt (3/5 folds, 1 epoch/fold)
and is rendered with a visually distinct style (critical-red, hatch texture, on-chart
annotation) so it cannot be misread as a real CV-5 estimate even out of context.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evidence" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Classical ML: genuine 5-fold CV, best combo (TF-IDF+SVM) ----
ml_df = pd.read_csv(ROOT / "35classPKL" / "MLcon35Clases" / "plots" / "results.csv")
ml_row = ml_df[(ml_df["vectorizer"] == "tfidf") & (ml_df["model"] == "svm")].iloc[0]
ml_f1_mean, ml_f1_std = ml_row["cv_f1_macro_mean"], ml_row["cv_f1_macro_std"]

# ---- Embeddings k-NN: genuine 5-fold CV (verified from outputs_task1/task1.out log) ----
emb_f1_mean, emb_f1_std = 0.4451, 0.0108

# ---- LoRA fine-tuning: abandoned CV, only 3 of 5 folds completed, 1 epoch/fold ----
fold_f1 = []
for fold in (0, 1, 2):
    with open(ROOT / "FineTuning" / "email-classifier-lora" / "outputs" / f"cv_fold_{fold}" / "val_metrics.json") as f:
        fold_f1.append(json.load(f)["f1_macro"])
lora_f1_mean, lora_f1_std = float(np.mean(fold_f1)), float(np.std(fold_f1))

data = pd.DataFrame([
    ("Classical ML\n(TF-IDF+SVM)", ml_f1_mean, ml_f1_std, "5/5", "genuine"),
    ("Embedding k-NN\n(MiniLM)", emb_f1_mean, emb_f1_std, "5/5", "genuine"),
    ("LoRA Fine-tuning\n(abandoned)", lora_f1_mean, lora_f1_std, "3/5", "abandoned — not a CV-5 estimate"),
], columns=["technique", "f1_macro_mean", "f1_macro_std", "folds_completed", "status"])
data.to_csv(OUT_DIR / "cv5_comparison_ml_embeddings_finetuning_data.csv", index=False)

# ---- Plot ----
COLOR_GENUINE = "#2a78d6"   # categorical slot 1, blue
COLOR_ABANDONED = "#d03b3b"  # status "critical" red — reserved, signals "not a normal series"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = [COLOR_GENUINE, COLOR_GENUINE, COLOR_ABANDONED]
hatches = [None, None, "///"]

bars = ax.bar(data["technique"], data["f1_macro_mean"], yerr=data["f1_macro_std"],
              color=colors, capsize=6, edgecolor=INK, linewidth=0.6)
for bar, hatch in zip(bars, hatches):
    if hatch:
        bar.set_hatch(hatch)

for i, row in data.iterrows():
    ax.annotate(f"{row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}\n({row['folds_completed']} folds)",
                (i, row["f1_macro_mean"] + row["f1_macro_std"]),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=INK)

ax.annotate("3/5 folds only, 1 epoch/fold, abandoned —\nNOT a CV-5 estimate (see final holdout result,\nF1-macro=0.876, in the unified holdout table)",
            xy=(2, lora_f1_mean), xytext=(1.05, 0.62),
            fontsize=8.5, color=COLOR_ABANDONED, ha="left",
            arrowprops=dict(arrowstyle="->", color=COLOR_ABANDONED, lw=1.2))

ax.set_ylabel("F1-macro (mean ± std across folds)", color=INK)
ax.set_ylim(0, 1.0)
ax.set_title("5-fold cross-validation: genuine (ML, Embeddings) vs.\nabandoned partial attempt (LoRA fine-tuning)", fontsize=11, color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)

fig.tight_layout()
out_path = OUT_DIR / "cv5_comparison_ml_embeddings_finetuning.png"
fig.savefig(out_path, dpi=200)
print(f"Generado: {out_path}")
print(f"Generado: {OUT_DIR / 'cv5_comparison_ml_embeddings_finetuning_data.csv'}")
