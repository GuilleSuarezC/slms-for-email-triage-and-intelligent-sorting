#!/usr/bin/env python
"""Paso 3: entrenar el adaptador LoRA FINAL con todo el 80% (train+cv)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_logger, load_config, load_json, set_seed
from src.train import train_lora

log = get_logger()


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    labels = load_json(cfg.paths.labels_file)
    train_df = pd.read_csv(Path(cfg.paths.output_dir) / "train.csv", dtype=str)

    log.info("Entrenando modelo FINAL con %d ejemplos.", len(train_df))
    train_lora(train_df, labels, cfg, cfg.paths.final_adapter_dir, epochs=cfg.train.epochs)
    log.info("Listo. Adaptador final en %s", cfg.paths.final_adapter_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
