#!/usr/bin/env python
"""Paso 4: evaluación FINAL y ÚNICA sobre el 20% test (sin tuning posterior)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import PeftModel

from src.config import get_logger, load_config, load_json, set_seed
from src.evaluate import evaluate_df
from src.modeling import load_base_model
from src.train import make_encoder

log = get_logger()


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    labels = load_json(cfg.paths.labels_file)
    test_df = pd.read_csv(Path(cfg.paths.output_dir) / "test.csv", dtype=str)

    encoder = make_encoder(cfg)
    base = load_base_model(cfg, for_training=False)
    model = PeftModel.from_pretrained(base, cfg.paths.final_adapter_dir)
    model.eval()

    log.info("Evaluación final sobre %d ejemplos de test.", len(test_df))
    evaluate_df(model, encoder, test_df, labels, cfg,
                out_prefix=str(Path(cfg.paths.output_dir) / "test"))


if __name__ == "__main__":
    main(*sys.argv[1:])
