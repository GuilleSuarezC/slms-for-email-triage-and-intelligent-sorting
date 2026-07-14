#!/usr/bin/env python
"""Paso 1: limpieza, validación, generación de labels.json y splits 80/20."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_logger, load_config, set_seed
from src.data import build_labels, holdout_split, load_and_clean

log = get_logger()


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    df = load_and_clean(cfg)
    labels = build_labels(df, cfg)

    if len(labels) != 35:
        log.warning("Se esperaban 35 clases y se detectaron %d. Revisa el dataset.", len(labels))

    train_df, test_df = holdout_split(df, cfg)

    out = Path(cfg.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    log.info("Guardados outputs/train.csv y outputs/test.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
