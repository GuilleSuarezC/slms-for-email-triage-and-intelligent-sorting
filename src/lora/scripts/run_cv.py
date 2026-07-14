#!/usr/bin/env python
"""Paso 2 (opcional): Stratified K-Fold sobre el 80% para estimar el rendimiento.

OJO de coste: con k=5 entrenas el modelo de 7B CINCO veces. Usa cv.cv_epochs
reducido (config) y/o baja cv.folds, o desactiva con cv.enabled=false.
"""
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import get_logger, load_config, load_json, save_json, set_seed
from src.data import kfold_splits
from src.evaluate import aggregate_cv, evaluate_df
from src.train import train_lora

log = get_logger()


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    if not cfg.cv.enabled:
        log.info("CV desactivada en config (cv.enabled=false). Saltando.")
        return

    labels = load_json(cfg.paths.labels_file)
    train_df = pd.read_csv(Path(cfg.paths.output_dir) / "train.csv", dtype=str)

    fold_metrics = []
    for fold, (tr, va) in enumerate(kfold_splits(train_df, cfg)):
        log.info("===== FOLD %d/%d =====", fold + 1, cfg.cv.folds)
        fold_dir = Path(cfg.paths.output_dir) / f"cv_fold_{fold}"
        model, encoder = train_lora(tr, labels, cfg, str(fold_dir), epochs=cfg.cv.cv_epochs)

        m = evaluate_df(model, encoder, va, labels, cfg,
                        out_prefix=str(fold_dir / "val"))
        m["fold"] = fold
        fold_metrics.append(m)

        # liberar VRAM entre folds
        del model
        gc.collect()
        torch.cuda.empty_cache()

    agg = aggregate_cv(fold_metrics)
    save_json(agg, str(Path(cfg.paths.output_dir) / "cv_metrics.json"))
    log.info("CV terminada -> f1_macro %.4f ± %.4f | acc %.4f ± %.4f",
             agg["f1_macro_mean"], agg["f1_macro_std"],
             agg["accuracy_mean"], agg["accuracy_std"])


if __name__ == "__main__":
    main(*sys.argv[1:])
