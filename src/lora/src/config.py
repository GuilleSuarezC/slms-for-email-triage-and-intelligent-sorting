"""Configuración, semillas y utilidades transversales."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    """Lee el YAML y lo devuelve como objeto navegable por puntos (cfg.model.id)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def set_seed(seed: int) -> None:
    """Reproducibilidad: fija TODAS las fuentes de aleatoriedad relevantes."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
