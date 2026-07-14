"""Lectura, limpieza, validación y particionado del dataset.

Reglas duras del proyecto:
  - Solo se usan dos columnas: `email_corpus` y `class_label`.
  - Cualquier otra columna se ignora por completo.
  - 80% train+CV / 20% test final (estratificado).
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import get_logger, save_json

log = get_logger()


def load_and_clean(cfg) -> pd.DataFrame:
    """Lee el CSV y deja un DataFrame limpio con exactamente dos columnas."""
    df = pd.read_csv(cfg.data.csv_path, sep=cfg.data.sep, dtype=str, keep_default_na=True)

    # Validar columnas obligatorias
    for col in (cfg.data.text_col, cfg.data.label_col):
        if col not in df.columns:
            raise ValueError(
                f"Falta la columna obligatoria '{col}'. Columnas encontradas: {list(df.columns)}"
            )

    # Quedarnos SOLO con las dos columnas relevantes (ignorar el resto)
    df = df[[cfg.data.text_col, cfg.data.label_col]].copy()
    df.columns = ["text", "label"]

    n0 = len(df)

    # Eliminar nulos
    df = df.dropna(subset=["text", "label"])

    # Trim
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()
    if cfg.data.lowercase_labels:
        df["label"] = df["label"].str.lower()

    # Eliminar textos o etiquetas vacías
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates().reset_index(drop=True)

    log.info("Dataset limpio: %d filas (descartadas %d de %d).", len(df), n0 - len(df), n0)
    _report_distribution(df, cfg.data.min_per_class_warn)
    return df


def _report_distribution(df: pd.DataFrame, min_warn: int) -> None:
    counts = df["label"].value_counts()
    log.info("Clases detectadas: %d", counts.shape[0])
    lengths = df["text"].str.len()
    log.info("Longitud emails (chars): min=%d  mediana=%d  max=%d",
             lengths.min(), int(lengths.median()), lengths.max())
    rare = counts[counts < min_warn]
    if not rare.empty:
        log.warning("¡Clases con muy pocas muestras (<%d)! Pueden romper la estratificación: %s",
                    min_warn, dict(rare))


def build_labels(df: pd.DataFrame, cfg) -> list[str]:
    """Lista cerrada y ordenada de etiquetas. Se guarda en labels.json."""
    labels = sorted(df["label"].unique().tolist())
    save_json(labels, cfg.paths.labels_file)
    log.info("Guardadas %d etiquetas en %s", len(labels), cfg.paths.labels_file)
    return labels


def holdout_split(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 estratificado. El test NO se toca hasta la evaluación final."""
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
        stratify=df["label"],
    )
    log.info("Split holdout -> train+cv=%d | test=%d", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def kfold_splits(train_df: pd.DataFrame, cfg) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Genera (train_fold, val_fold) con StratifiedKFold sobre el 80%."""
    skf = StratifiedKFold(n_splits=cfg.cv.folds, shuffle=True, random_state=cfg.seed)
    X = train_df.index.values
    y = train_df["label"].values
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        yield (
            train_df.iloc[tr_idx].reset_index(drop=True),
            train_df.iloc[va_idx].reset_index(drop=True),
        )
