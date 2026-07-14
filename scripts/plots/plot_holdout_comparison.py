#!/usr/bin/env python3
"""Reads: Holdout/ML/results.csv, Embeddings/outputs_task1/task1.out (verified holdout
numbers hardcoded below, log is not machine-parseable), Holdout/Zero&Fewshot/outputs/
metrics_summary.csv, archivos_de_outputs/test_metrics.json (70/30 holdout rerun, see R15
in evidence/result_register.md; supersedes the old FineTuning/.../test_metrics.json 80/20 source).
Writes: evidence/plots/holdout_comparison_5techniques.png +
_data.csv (companion, exact plotted numbers for spot-checking).
"""
import json
import csv
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evidence" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Classical ML (best combo: TF-IDF + SVM), canonical holdout source ----
ml_df = pd.read_csv(ROOT / "Holdout" / "ML" / "results.csv")
ml_best = ml_df.loc[ml_df["f1_macro"].idxmax()]

# ---- Embeddings k-NN, holdout 70/30 (verified from outputs_task1/task1.out log) ----
emb_accuracy = 0.4970
emb_f1_macro = 0.4264

# ---- SLM zero-shot / few-shot, full-scale holdout rerun ----
slm_df = pd.read_csv(ROOT / "Holdout" / "Zero&Fewshot" / "outputs" / "metrics_summary.csv")
zero = slm_df[slm_df["strategy"] == "zero_shot"].iloc[0]
few = slm_df[slm_df["strategy"] == "few_shot"].iloc[0]

# ---- LoRA fine-tuning, final holdout model (70/30 corrected split, R15) ----
with open(ROOT / "archivos_de_outputs" / "test_metrics.json") as f:
    lora = json.load(f)

rows = [
    ("Classical ML\n(TF-IDF+SVM)", ml_best["accuracy"], ml_best["f1_macro"], 6000, "70/30"),
    ("Embedding k-NN\n(MiniLM)", emb_accuracy, emb_f1_macro, 6000, "70/30"),
    ("Zero-shot\nSLM", zero["accuracy"], zero["f1_macro"], 6000, "70/30"),
    ("Few-shot\nSLM (1-shot)", few["accuracy"], few["f1_macro"], 6000, "70/30"),
    ("LoRA\nFine-tuning", lora["accuracy"], lora["f1_macro"], 6000, "70/30"),
]

data = pd.DataFrame(rows, columns=["technique", "accuracy", "f1_macro", "n_test", "split_ratio"])
data.to_csv(OUT_DIR / "holdout_comparison_5techniques_data.csv", index=False)

# ---- Plot: grouped bar chart, categorical palette slots 1 (blue) and 3 (yellow) ----
# per dataviz reference palette (light mode, printed-page surface).
COLOR_F1 = "#2a78d6"       # slot 1 blue
COLOR_ACC = "#eda100"      # slot 3 yellow
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(data))
width = 0.36

bars_f1 = ax.bar([i - width / 2 for i in x], data["f1_macro"], width, label="F1-macro", color=COLOR_F1)
bars_acc = ax.bar([i + width / 2 for i in x], data["accuracy"], width, label="Accuracy", color=COLOR_ACC)

for bars in (bars_f1, bars_acc):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8, color=INK)

xtick_labels = [f"{t}\n(n={n}, {s})" for t, n, s in zip(data["technique"], data["n_test"], data["split_ratio"])]
ax.set_xticks(list(x))
ax.set_xticklabels(xtick_labels, fontsize=8.5, color=INK)
ax.set_ylabel("Score", color=INK)
ax.set_ylim(0, 1.05)
ax.set_title("Holdout comparison across all five 35-class techniques\n"
             "(each on its own held-out partition — see split ratio/n below)", fontsize=11, color=INK)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)

fig.tight_layout()
out_path = OUT_DIR / "holdout_comparison_5techniques.png"
fig.savefig(out_path, dpi=200)
print(f"Generado: {out_path}")
print(f"Generado: {OUT_DIR / 'holdout_comparison_5techniques_data.csv'}")
