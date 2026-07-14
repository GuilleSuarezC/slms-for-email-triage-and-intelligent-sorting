"""
inference_embeddings.py
=========================
Script de inferencia para el clasificador basado en EMBEDDINGS + ChromaDB
(k-NN) entrenado/almacenado con task1_knn_chromadb.py.

Adaptado a partir de `inference_35_clases.py` (clasificadores clásicos
TF-IDF/BoW/Word2Vec + LogReg/SVM/RF). Se mantiene el mismo flujo general
—carga de "bundle", detección de columnas, benchmark de throughput con
bootstrap, métricas, plots, exportación de predicciones—, pero:

    - El "vectorizador" es la función de embeddings de ChromaDB
      (SentenceTransformerEmbeddingFunction), no TF-IDF/BoW/Word2Vec.
    - El "modelo" NO es un .pkl de sklearn: es la propia colección
      persistente de ChromaDB (creada por task1_knn_chromadb.py en
      `./chroma_db_emails`), que actúa como índice de vecinos más
      cercanos. La predicción es un voto mayoritario (con Laplace
      smoothing para obtener pseudo-probabilidades) entre los k vecinos
      más cercanos devueltos por Chroma.
    - El benchmark de throughput, igual que en el script original, hace
      BOOTSTRAP (muestreo con reemplazo) del conjunto de test disponible
      (~20.000 correos) para completar hasta `--benchmark_n_samples`
      (por defecto 100.000), y así medir throughput/latencia realistas a
      la escala de producción sin necesitar 100k correos reales
      etiquetados.

Uso
───
    python inference_embeddings.py --list
    python inference_embeddings.py
    python inference_embeddings.py --k 10
    python inference_embeddings.py --save_predictions
    python inference_embeddings.py --benchmark_n_samples 100000
    python inference_embeddings.py --no_benchmark
    python inference_embeddings.py --text-col "email_corpus" --label-col class_label

"Bundle" (colección ChromaDB persistente)
───────────────────────────────────────────
    Generada por task1_knn_chromadb.py -> store_embeddings_in_chroma():
        persist_dir      : "./chroma_db_emails"
        collection_name  : "email_corpus_embeddings"
        metadatos/doc    : {"class_label": <str>} por cada correo
    Este script se conecta a esa colección (PersistentClient) y la usa
    directamente como índice k-NN; no hace falta reentrenar ni recalcular
    embeddings de las muestras ya almacenadas.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import chromadb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
)

# Reutilizamos la generación de embeddings ya implementada para la Tarea 1
# (mismo modelo / mismo patrón que L1-student.ipynb).
from utils_embeddings import get_embedding_function

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

RANDOM_STATE        = 42
TEST_SIZE           = 0.9   # igual que en inference_35_clases.py: se usa un
                             # test_size grande para evaluar sobre la mayor
                             # parte posible del dataset (~20k correos)
THROUGHPUT_REPEATS   = 5
DEFAULT_BENCHMARK_N  = 100_000   # 20k reales -> bootstrap hasta 100k

DEFAULT_PERSIST_DIR      = "./chroma_db_emails"
DEFAULT_COLLECTION_NAME  = "email_corpus_embeddings"
DEFAULT_K                = 5
LAPLACE_ALPHA             = 1.0   # suavizado para las pseudo-probabilidades

TEXT_COL_CANDIDATES  = ["email_corpus", "prompt", "text", "situacion", "situación",
                         "mensaje", "description", "descripcion", "descripción", "input"]
LABEL_COL_CANDIDATES = ["class_label", "label", "categoria", "categoría", "category",
                         "clase", "class", "tipo", "intent", "intencion", "intención", "etiqueta"]

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    return logging.getLogger("InferenceEmbeddings")

logger = setup_logging()

# ==============================================================================
# PREPROCESAMIENTO
# ==============================================================================

def clean_text(text: str) -> str:
    """
    A diferencia de TF-IDF/BoW (donde conviene normalizar agresivamente:
    minúsculas, quitar puntuación, etc.), los modelos de sentence
    embeddings se benefician de conservar la puntuación y el casing
    original (aportan señal semántica). Aquí solo se recorta espacios
    sobrantes, igual que se hizo al generar los embeddings en Task 1.
    """
    return " ".join(str(text).split())


def _normalize_colname(c: str) -> str:
    return str(c).strip().lower().lstrip("#").strip()


def _detect_column(columns: list, candidates: list, exclude=None) -> str | None:
    cols = [c for c in columns if c != exclude]
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    return None


def _detect_label_column(df: pd.DataFrame, candidates: list, exclude,
                          expected_n_classes: int) -> str | None:
    """Igual que en inference_35_clases.py: si hay varias columnas candidatas,
    se elige la que tenga un nº de categorías más cercano a las clases
    conocidas por la colección ChromaDB (expected_n_classes)."""
    cols = [c for c in df.columns if c != exclude]
    matched = []
    for cand in candidates:
        if cand in cols and cand not in matched:
            matched.append(cand)
    for cand in candidates:
        for c in cols:
            if cand in c and c not in matched:
                matched.append(c)

    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]

    scored = []
    for c in matched:
        try:
            n_unique = df[c].nunique(dropna=True)
        except Exception:
            n_unique = -1
        scored.append((abs(n_unique - expected_n_classes), matched.index(c), c, n_unique))
    scored.sort(key=lambda t: (t[0], t[1]))
    best = scored[0]
    if best[2] != matched[0]:
        logger.info("Varias columnas candidatas a etiqueta %s. Se elige '%s' (%d categorías).",
                    [(c, n) for _, _, c, n in scored], best[2], best[3])
    return best[2]


def sniff_separator(csv_path: Path, default: str = ",") -> str:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.count(";") > first_line.count(","):
        return ";"
    if first_line.count(",") > 0:
        return ","
    return default


def load_and_prepare(data_dir: Path, text_col_arg, label_col_arg, sep_arg,
                      known_classes: set) -> pd.DataFrame:
    """Carga el CSV, autodetecta columnas texto/etiqueta y descarta filas
    con etiquetas no vistas en la colección ChromaDB (equivalente a
    'clases desconocidas para el label_encoder' del script original)."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No se encontró ningún CSV en '%s'.", data_dir)
        sys.exit(1)

    csv_path = csv_files[0]
    sep = sep_arg or sniff_separator(csv_path)
    df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding="utf-8", quotechar='"', engine="python")

    raw_cols = list(df.columns)
    norm_cols = [_normalize_colname(c) for c in raw_cols]
    df = df.rename(columns=dict(zip(raw_cols, norm_cols)))

    text_col = _normalize_colname(text_col_arg) if text_col_arg else None
    if text_col and text_col not in df.columns:
        logger.error("--text-col '%s' no existe. Columnas: %s", text_col_arg, list(df.columns))
        sys.exit(1)
    if not text_col:
        text_col = _detect_column(df.columns, TEXT_COL_CANDIDATES)

    label_col = _normalize_colname(label_col_arg) if label_col_arg else None
    if label_col and label_col not in df.columns:
        logger.error("--label-col '%s' no existe. Columnas: %s", label_col_arg, list(df.columns))
        sys.exit(1)
    if not label_col:
        label_col = _detect_label_column(df, LABEL_COL_CANDIDATES, exclude=text_col,
                                          expected_n_classes=len(known_classes))

    if text_col is None or label_col is None:
        logger.error("No se detectaron columnas texto/etiqueta. Columnas: %s.", list(df.columns))
        sys.exit(1)

    logger.info("Columna de texto: '%s' | Columna de etiqueta: '%s'", text_col, label_col)

    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    before = len(df)
    unknown_mask = ~df["label"].isin(known_classes)
    if unknown_mask.any():
        logger.warning(
            "%d filas tienen etiquetas no presentes en la colección ChromaDB y se descartan: %s",
            unknown_mask.sum(), sorted(df.loc[unknown_mask, "label"].unique())[:10],
        )
        df = df[~unknown_mask]

    logger.info("Dataset cargado: %d muestras (de %d) | %d clases presentes (de %d posibles).",
                len(df), before, df["label"].nunique(), len(known_classes))
    return df.reset_index(drop=True)

# ==============================================================================
# REPRODUCCIÓN DEL SPLIT DE TEST
# ==============================================================================

def get_test_split(df: pd.DataFrame):
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    logger.info("Conjunto de test: %d muestras", len(X_test))
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)

# ==============================================================================
# BOOTSTRAP: 20k -> 100k (para el benchmark de throughput)
# ==============================================================================

def bootstrap_to_size(X_raw: pd.Series, n_samples: int, random_state: int = RANDOM_STATE) -> pd.Series:
    """
    Si el dataset de test disponible (~20.000 correos) es menor que
    `n_samples` (por defecto 100.000), se completa mediante BOOTSTRAP
    (muestreo CON reemplazo) hasta alcanzar el tamaño deseado. Esto NO se
    usa para calcular accuracy/F1 (esas métricas se calculan sobre las
    muestras reales únicamente); se usa solo para medir el rendimiento
    real (throughput/latencia) del pipeline de embeddings + consulta
    k-NN a la escala de 100k, sin necesitar 100k correos reales.
    """
    rng = np.random.RandomState(random_state)
    n_available = len(X_raw)

    if n_samples <= n_available:
        idx = rng.choice(n_available, size=n_samples, replace=False)
    else:
        logger.info(
            "Test set disponible (%d) < tamaño de benchmark pedido (%d). "
            "Aplicando BOOTSTRAP (muestreo con reemplazo) %d -> %d muestras.",
            n_available, n_samples, n_available, n_samples,
        )
        idx = rng.choice(n_available, size=n_samples, replace=True)

    return X_raw.iloc[idx].reset_index(drop=True)

# ==============================================================================
# CARGA DEL "BUNDLE": colección ChromaDB persistente (Task 1)
# ==============================================================================

def load_embeddings_bundle(persist_dir: Path, collection_name: str, embedding_function):
    """
    Se conecta a la colección ChromaDB persistente creada por
    task1_knn_chromadb.py y recupera:
        - la propia colección (usada como índice k-NN)
        - el listado de clases conocidas (a partir de los metadatos
          'class_label' de todos los documentos almacenados)
        - metadatos de "entrenamiento" (nº de muestras, nº de clases)
    """
    if not persist_dir.exists():
        logger.error(
            "No se encontró la base de datos ChromaDB en '%s'. "
            "Ejecuta primero task1_knn_chromadb.py para generarla.", persist_dir,
        )
        sys.exit(1)

    logger.info("Conectando a ChromaDB persistente en '%s' ...", persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error("No se pudo abrir la colección '%s': %s", collection_name, e)
        available = [c.name for c in client.list_collections()]
        logger.error("Colecciones disponibles en '%s': %s", persist_dir, available)
        sys.exit(1)

    n_stored = collection.count()
    logger.info("Recuperando metadatos de %d documentos para listar clases ...", n_stored)
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    classes = sorted({m["class_label"] for m in all_meta if "class_label" in m})

    meta = {
        "trained_on_n_samples": n_stored,
        "n_classes": len(classes),
        "classes": classes,
        "collection_name": collection_name,
        "persist_dir": str(persist_dir),
    }

    logger.info("Colección cargada     : %s", collection_name)
    logger.info("Documentos almacenados : %d", n_stored)
    logger.info("Nº de clases           : %d", len(classes))

    return collection, classes, meta

# ==============================================================================
# EMBEDDINGS DE LOS TEXTOS DE ENTRADA (equivalente a "vectorizar")
# ==============================================================================

def embed_texts(embedding_function, texts, batch_size: int = 128) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        embeddings.extend(embedding_function(batch))
    return np.asarray(embeddings, dtype=np.float32)

# ==============================================================================
# INFERENCIA: k-NN vía ChromaDB (voto mayoritario + pseudo-probabilidades)
# ==============================================================================

def run_inference(collection, classes: list, query_embeddings: np.ndarray, k: int,
                   batch_size: int = 200) -> tuple:
    """
    Ejecuta la consulta k-NN sobre la colección ChromaDB para cada
    embedding de entrada y devuelve:
        y_pred  : etiqueta predicha (voto mayoritario de los k vecinos)
        y_proba : matriz (n_muestras, n_clases) de pseudo-probabilidades,
                  calculadas como la fracción de votos de cada clase entre
                  los k vecinos, con suavizado de Laplace (alpha=1) para
                  evitar 0/1 exactos (necesarios para log_loss/roc_auc).
        elapsed : tiempo total de inferencia (segundos)
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    y_pred, y_proba = [], []

    t0 = time.perf_counter()
    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i:i + batch_size]
        results = collection.query(
            query_embeddings=batch.tolist(), n_results=k, include=["metadatas"],
        )
        for metadatas in results["metadatas"]:
            neighbor_labels = [m["class_label"] for m in metadatas]
            counts = Counter(neighbor_labels)

            most_common_label = counts.most_common(1)[0][0]
            y_pred.append(most_common_label)

            proba = np.full(n_classes, LAPLACE_ALPHA, dtype=np.float64)
            for lbl, cnt in counts.items():
                if lbl in class_to_idx:
                    proba[class_to_idx[lbl]] += cnt
            proba /= proba.sum()
            y_proba.append(proba)

    elapsed = time.perf_counter() - t0
    return np.array(y_pred), np.array(y_proba), elapsed

# ==============================================================================
# MÉTRICAS
# ==============================================================================

def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                     classes: list) -> dict:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test_idx = np.array([class_to_idx[y] for y in y_test])

    metrics = {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "log_loss":        round(log_loss(y_test_idx, y_proba, labels=list(range(len(classes)))), 4),
    }
    try:
        metrics["roc_auc_ovr_macro"] = round(
            roc_auc_score(y_test_idx, y_proba, multi_class="ovr", average="macro",
                          labels=list(range(len(classes)))), 4)
    except ValueError as e:
        logger.warning("No se pudo calcular roc_auc_ovr: %s", e)
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics

# ==============================================================================
# THROUGHPUT BENCHMARK (con bootstrap 20k -> 100k)
# ==============================================================================

def benchmark_throughput(collection, embedding_function, classes, k,
                          X_test_raw: pd.Series, n_repeats: int) -> dict:
    n = len(X_test_raw)
    logger.info("Benchmarking: %d muestras x %d repeticiones ...", n, n_repeats)

    # "Warm-up": primera pasada fuera de la medición (carga de modelo, JIT, etc.)
    _emb = embed_texts(embedding_function, X_test_raw)
    run_inference(collection, classes, _emb, k)
    del _emb

    embed_times, query_times = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        emb = embed_texts(embedding_function, X_test_raw)
        embed_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        run_inference(collection, classes, emb, k)
        query_times.append(time.perf_counter() - t1)

    embed_times = np.array(embed_times)
    query_times = np.array(query_times)
    total_times = embed_times + query_times
    avg_embed, avg_query, avg_total = embed_times.mean(), query_times.mean(), total_times.mean()
    latency_ms = (total_times / n) * 1_000

    return {
        "n_samples":            n,
        "n_repeats":            n_repeats,
        "total_time_s":         round(float(avg_total), 6),
        "throughput_eps":       round(n / avg_total, 2),
        "latency_ms":           round((avg_total / n) * 1_000, 4),
        "latency_p50_ms":       round(float(np.percentile(latency_ms, 50)), 4),
        "latency_p95_ms":       round(float(np.percentile(latency_ms, 95)), 4),
        "latency_p99_ms":       round(float(np.percentile(latency_ms, 99)), 4),
        "embed_time_s":         round(float(avg_embed), 6),
        "embed_throughput_eps": round(n / avg_embed, 2),
        "embed_latency_ms":     round((avg_embed / n) * 1_000, 4),
        "query_time_s":         round(float(avg_query), 6),
        "query_throughput_eps": round(n / avg_query, 2),
        "query_latency_ms":     round((avg_query / n) * 1_000, 4),
    }


def log_throughput_report(tp: dict, k: int):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  THROUGHPUT REPORT — EMBEDDINGS + ChromaDB k-NN (k=%d)", k)
    logger.info("=" * 60)
    logger.info("  Muestras (bootstrap incl.) : %d", tp["n_samples"])
    logger.info("  Repeticiones benchmark     : %d", tp["n_repeats"])
    logger.info("")
    logger.info("  -- Pipeline completo -------------------------------------")
    logger.info("  Tiempo total (media)   : %.4f s", tp["total_time_s"])
    logger.info("  Throughput             : %s muestras/s", f"{tp['throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/muestra", tp["latency_ms"])
    logger.info("  Latencia p50           : %.4f ms/muestra", tp["latency_p50_ms"])
    logger.info("  Latencia p95           : %.4f ms/muestra", tp["latency_p95_ms"])
    logger.info("  Latencia p99           : %.4f ms/muestra", tp["latency_p99_ms"])
    logger.info("")
    logger.info("  -- Solo generacion de embeddings ---------------------------")
    logger.info("  Tiempo (media)         : %.4f s", tp["embed_time_s"])
    logger.info("  Throughput             : %s muestras/s", f"{tp['embed_throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/muestra", tp["embed_latency_ms"])
    logger.info("")
    logger.info("  -- Solo consulta k-NN (ChromaDB) ---------------------------")
    logger.info("  Tiempo (media)         : %.4f s", tp["query_time_s"])
    logger.info("  Throughput             : %s muestras/s", f"{tp['query_throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/muestra", tp["query_latency_ms"])
    logger.info("=" * 60)


def plot_throughput(tp: dict, k: int, output_dir: Path):
    title_suffix = f"EMBEDDINGS + ChromaDB k-NN (k={k})"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Throughput Benchmark — {title_suffix}", fontsize=13, fontweight="bold")
    colors = ["#4C9BE8", "#E87B4C", "#6DBE6D"]

    ax = axes[0]
    stages = ["Embeddings", "Consulta k-NN", "Total (pipeline)"]
    times_s = [tp["embed_time_s"], tp["query_time_s"], tp["total_time_s"]]
    bars = ax.barh(stages, times_s, color=colors, edgecolor="grey", linewidth=0.5)
    for bar, val in zip(bars, times_s):
        ax.text(val + max(times_s) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val * 1_000:.2f} ms", va="center", fontsize=10)
    ax.set_xlabel("Tiempo promedio (s)", fontsize=11)
    ax.set_title("Tiempo medio por etapa", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(times_s) * 1.30)

    ax2 = axes[1]
    thr_vals = [tp["embed_throughput_eps"], tp["query_throughput_eps"], tp["throughput_eps"]]
    bars2 = ax2.bar(["Embeddings", "Consulta k-NN", "Total\n(pipeline)"],
                    thr_vals, color=colors, edgecolor="grey", linewidth=0.5, width=0.5)
    for bar, val in zip(bars2, thr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(thr_vals) * 0.015,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Muestras por segundo", fontsize=11)
    ax2.set_title("Throughput (muestras/s)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(thr_vals) * 1.20)

    fig.text(0.5, -0.05,
             f"Latencia pipeline — media: {tp['latency_ms']:.4f} ms  |  "
             f"p50: {tp['latency_p50_ms']:.4f} ms  |  "
             f"p95: {tp['latency_p95_ms']:.4f} ms  |  "
             f"p99: {tp['latency_p99_ms']:.4f} ms",
             ha="center", fontsize=9, color="dimgrey")

    plt.tight_layout()
    path = output_dir / "throughput_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# VISUALIZACIONES DE CLASIFICACIÓN
# ==============================================================================

def plot_confusion_matrix(y_test, y_pred, classes, output_dir: Path):
    n = len(classes)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.32)))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True,
                xticklabels=classes, yticklabels=classes,
                annot=(n <= 15), fmt="d" if n <= 15 else "")
    ax.set_title("Confusion Matrix — Embeddings + ChromaDB k-NN", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    path = output_dir / "confusion_matrix_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_per_class_metrics(y_test, y_pred, classes, output_dir: Path):
    report = classification_report(y_test, y_pred, labels=classes, target_names=classes,
                                    output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T.loc[list(classes)]
    df = df.sort_values("f1-score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.32)))
    y_pos = np.arange(len(df))
    bar_h = 0.25
    ax.barh(y_pos - bar_h, df["precision"], height=bar_h, label="Precision", color="#4C9BE8")
    ax.barh(y_pos,         df["recall"],    height=bar_h, label="Recall",    color="#E87B4C")
    ax.barh(y_pos + bar_h, df["f1-score"],  height=bar_h, label="F1-score",  color="#6DBE6D")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title("Métricas por clase — Embeddings + ChromaDB k-NN", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "per_class_metrics_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_confidence_distribution(y_test, y_pred, y_proba, output_dir: Path):
    """Distribución de la pseudo-probabilidad (fracción de votos, con
    suavizado de Laplace) asignada a la clase predicha, separando
    aciertos de errores."""
    confidence = y_proba.max(axis=1)
    correct = (np.asarray(y_test) == np.asarray(y_pred))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidence[correct],  bins=40, alpha=0.6, color="steelblue",
            label=f"Correctas (n={correct.sum()})", density=True)
    ax.hist(confidence[~correct], bins=40, alpha=0.6, color="tomato",
            label=f"Incorrectas (n={(~correct).sum()})", density=True)
    ax.axvline(1 / y_proba.shape[1], color="black", linestyle="--", linewidth=1,
               label=f"Azar (1/{y_proba.shape[1]} clases)")
    ax.set_xlabel("Confianza (fraccion de votos k-NN, suavizada)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución de confianza — Embeddings + ChromaDB k-NN", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "confidence_distribution_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# UTILIDADES
# ==============================================================================

def list_collections(persist_dir: Path):
    print("\nColecciones ChromaDB disponibles:")
    print("-" * 40)
    if not persist_dir.exists():
        print(f"  (!) No existe el directorio '{persist_dir}'. "
              f"Ejecuta primero task1_knn_chromadb.py.")
        return
    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = client.list_collections()
    if not collections:
        print("  (!) No hay colecciones almacenadas todavia.")
    for c in collections:
        col = client.get_collection(c.name)
        print(f"  --collection {c.name}  (documentos: {col.count()})")
    print()

# ==============================================================================
# ARGPARSE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia (embeddings + ChromaDB k-NN) sobre el conjunto de test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python inference_embeddings.py --list
  python inference_embeddings.py
  python inference_embeddings.py --k 10 --save_predictions
  python inference_embeddings.py --benchmark_n_samples 100000
  python inference_embeddings.py --no_benchmark
  python inference_embeddings.py --text-col email_corpus --label-col class_label
        """,
    )
    parser.add_argument("--persist_dir",   type=str, default=DEFAULT_PERSIST_DIR,
                        help="Directorio de la base de datos ChromaDB persistente (Task 1).")
    parser.add_argument("--collection",    type=str, default=DEFAULT_COLLECTION_NAME,
                        help="Nombre de la colección ChromaDB a usar como índice k-NN.")
    parser.add_argument("--k",             type=int, default=DEFAULT_K,
                        help="Número de vecinos (k) para la clasificación k-NN.")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="Nombre del modelo de sentence-transformers (por defecto, el de Chroma).")
    parser.add_argument("--data_dir",      type=str, default="data")
    parser.add_argument("--output_dir",    type=str, default="inference_results_embeddings")
    parser.add_argument("--text-col",      type=str, default=None, metavar="COL")
    parser.add_argument("--label-col",     type=str, default=None, metavar="COL")
    parser.add_argument("--sep",           type=str, default=None, metavar="CHAR")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Guarda CSV con predicción y confianza por muestra.")
    parser.add_argument("--save_full_proba", action="store_true",
                        help="Añade la pseudo-probabilidad de las 35 clases al CSV de predicciones.")
    parser.add_argument("--list", action="store_true",
                        help="Lista las colecciones ChromaDB disponibles y sale.")
    parser.add_argument("--benchmark_repeats", type=int, default=THROUGHPUT_REPEATS, metavar="N")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="Omite el benchmark de throughput (y por tanto el bootstrap a 100k).")
    parser.add_argument("--benchmark_n_samples", type=int, default=DEFAULT_BENCHMARK_N, metavar="N",
                        help="Nº de muestras objetivo para el benchmark de throughput. Si el dataset "
                             "de test (~20.000) es menor, se hace BOOTSTRAP (con reemplazo) para "
                             "completar hasta este tamaño. Por defecto: 100000.")
    return parser.parse_args()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    persist_dir = Path(args.persist_dir)
    data_dir    = Path(args.data_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_collections(persist_dir)
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("  INFERENCIA — EMBEDDINGS + ChromaDB k-NN (k=%d)", args.k)
    logger.info("=" * 60)

    # -- Función de embeddings (mismo patrón que Task 1 / L1-student.ipynb) --
    embedding_function = get_embedding_function(args.embedding_model)

    # -- "Bundle": colección ChromaDB persistente ----------------------------
    collection, classes, meta = load_embeddings_bundle(persist_dir, args.collection, embedding_function)

    # -- Datos: subset de test con etiquetas reales ---------------------------
    df = load_and_prepare(data_dir, args.text_col, args.label_col, args.sep, set(classes))
    X_test_raw, y_test = get_test_split(df)
    y_test_arr = y_test.to_numpy()

    # -- Embeddings del conjunto de test ---------------------------------------
    logger.info("Generando embeddings del conjunto de test ...")
    X_test_emb = embed_texts(embedding_function, X_test_raw)
    logger.info("Test embebido: shape=%s", X_test_emb.shape)

    # -- Inferencia (k-NN via ChromaDB) -----------------------------------------
    logger.info("Ejecutando inferencia (k-NN, k=%d) ...", args.k)
    y_pred, y_proba, pred_elapsed = run_inference(collection, classes, X_test_emb, args.k)
    logger.info("Predicción: %.4f s | %d muestras | %.2f muestras/s | %.4f ms/muestra",
                pred_elapsed, len(y_test_arr), len(y_test_arr) / pred_elapsed,
                (pred_elapsed / len(y_test_arr)) * 1_000)

    # -- Métricas -------------------------------------------------------------------
    metrics = compute_metrics(y_test_arr, y_pred, y_proba, classes)

    logger.info("\n%s", "=" * 60)
    logger.info("MÉTRICAS SOBRE EL CONJUNTO DE TEST (embeddings + k-NN, %d clases)", len(classes))
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info("  %-18s %.4f", name.upper(), value)
    logger.info("=" * 60)
    logger.info("\nClassification Report:\n%s",
                classification_report(y_test_arr, y_pred, labels=classes, target_names=classes, digits=4))

    # -- Benchmark de throughput (con bootstrap 20k -> 100k) ---------------------
    throughput_data = {}
    if not args.no_benchmark:
        X_bench_raw = bootstrap_to_size(X_test_raw, args.benchmark_n_samples)
        throughput_data = benchmark_throughput(
            collection, embedding_function, classes, args.k, X_bench_raw, args.benchmark_repeats)
        log_throughput_report(throughput_data, args.k)
        plot_throughput(throughput_data, args.k, output_dir)

    # -- Guardar métricas en JSON ------------------------------------------------
    full_report = {
        "k": args.k, "n_classes": len(classes),
        **metrics,
        "meta": meta,
        **({"throughput": throughput_data} if throughput_data else {}),
    }
    metrics_path = output_dir / "metrics_embeddings.json"
    with open(metrics_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info("Métricas guardadas en '%s'.", metrics_path)

    # -- Visualizaciones -----------------------------------------------------------
    logger.info("Generando visualizaciones ...")
    plot_confusion_matrix(y_test_arr, y_pred, classes, output_dir)
    plot_per_class_metrics(y_test_arr, y_pred, classes, output_dir)
    plot_confidence_distribution(y_test_arr, y_pred, y_proba, output_dir)

    # -- CSV de predicciones (opcional) ---------------------------------------------
    if args.save_predictions:
        proba_pred = y_proba.max(axis=1)
        preds_df = pd.DataFrame({
            "text":       X_test_raw.values,
            "label_real": y_test_arr,
            "label_pred": y_pred,
            "confidence": np.round(proba_pred, 4),
            "correct":    (y_test_arr == y_pred).astype(int),
        })

        if args.save_full_proba:
            proba_cols = pd.DataFrame(np.round(y_proba, 4), columns=[f"proba_{c}" for c in classes])
            preds_df = pd.concat([preds_df.reset_index(drop=True),
                                  proba_cols.reset_index(drop=True)], axis=1)

        preds_path = output_dir / "predictions_embeddings.csv"
        preds_df.to_csv(preds_path, index=False)
        errors = preds_df[preds_df["correct"] == 0]
        logger.info("Predicciones guardadas en '%s'. Errores: %d / %d (%.2f%%)",
                    preds_path, len(errors), len(preds_df), 100 * len(errors) / len(preds_df))

        if len(errors) > 0:
            top_confused = (errors.groupby(["label_real", "label_pred"]).size()
                            .sort_values(ascending=False).head(10))
            logger.info("Top confusiones (real -> predicho):\n%s", top_confused.to_string())

    # -- Resumen ------------------------------------------------------------------------
    logger.info("\nArtefactos generados en '%s':", output_dir)
    for f in sorted(output_dir.iterdir()):
        if f.name.endswith((".png", ".json", ".csv")):
            logger.info("  %s", f.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
