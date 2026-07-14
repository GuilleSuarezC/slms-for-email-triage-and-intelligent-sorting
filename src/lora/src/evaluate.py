"""Evaluación: accuracy, F1 macro (obligatorio), F1 weighted y matriz de confusión."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import get_logger, save_json
from .infer import predict_many

log = get_logger()


def compute_metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "n": len(y_true),
    }


def evaluate_df(model, encoder, df, labels, cfg, out_prefix: str | None = None) -> dict:
    """Predice sobre df, calcula métricas y opcionalmente guarda artefactos."""
    y_true = df["label"].tolist()
    y_pred = predict_many(model, encoder, df["text"].tolist(), labels, cfg)

    metrics = compute_metrics(y_true, y_pred, labels)
    log.info("Métricas -> acc=%.4f | f1_macro=%.4f | f1_weighted=%.4f (n=%d)",
             metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"], metrics["n"])

    if out_prefix:
        _save_artifacts(y_true, y_pred, labels, metrics, out_prefix)
    return metrics


def _save_artifacts(y_true, y_pred, labels, metrics, out_prefix):
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    save_json(metrics, f"{out_prefix}_metrics.json")

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    save_json(report, f"{out_prefix}_classification_report.json")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(f"{out_prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")

    _plot_confusion(cm, labels, f"{out_prefix}_confusion_matrix.png")
    log.info("Artefactos de evaluación guardados con prefijo: %s", out_prefix)


def _plot_confusion(cm, labels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(6, n * 0.35)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("No se pudo dibujar la matriz de confusión: %s", e)


def aggregate_cv(fold_metrics: list[dict]) -> dict:
    """Media y desviación de las métricas a lo largo de los folds."""
    keys = ["accuracy", "f1_macro", "f1_weighted"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["folds"] = fold_metrics
    return agg
