"""
inference.py
============
Script de inferencia para los modelos de clasificación Spam vs Ham entrenados
con spam_classifier_experiment.py (v4+).

Carga el bundle completo (.pkl) que contiene modelo + vectorizador ya ajustados
sobre el 100% del dataset. No necesita re-ajustar nada ni depender del split.

Uso
───
    python inference.py --list
    python inference.py --model logreg
    python inference.py --model svm
    python inference.py --model rf
    python inference.py --model logreg --save_predictions
    python inference.py --model logreg --benchmark_repeats 10
    python inference.py --model logreg --no_benchmark

Estructura del bundle (.pkl)
─────────────────────────────
    {
        "model"      : estimador sklearn ajustado,
        "vectorizer" : vectorizador ajustado (sklearn o Word2Vec),
        "vec_name"   : "one_hot" | "bow" | "tfidf" | "word2vec",
        "meta"       : { trained_on_n_samples, experiment_test_f1, exported_at, ... }
    }
"""

import argparse
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, roc_curve, classification_report,
)

from gensim.models import Word2Vec

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

RANDOM_STATE       = 42
TEST_SIZE          = 0.9   # debe coincidir con el experimento
THROUGHPUT_REPEATS = 5

MODEL_FILES = {
    "logreg": "best_logreg.pkl",
    "svm":    "best_svm.pkl",
    "rf":     "best_rf.pkl",
}

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
    return logging.getLogger("Inference")

logger = setup_logging()

# ==============================================================================
# PREPROCESAMIENTO (idéntico al del entrenamiento)
# ==============================================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare(data_dir: Path) -> pd.DataFrame:
    """Carga el CSV, normaliza columnas y aplica la misma limpieza que en el entrenamiento."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No se encontró ningún CSV en '%s'.", data_dir)
        sys.exit(1)

    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip().lower() for c in df.columns]

    text_col = next(
        (c for c in df.columns if any(k in c for k in ("text", "message", "email"))), None)
    label_col = next(
        (c for c in df.columns if any(k in c for k in ("label", "class", "category"))), None)

    if text_col is None or label_col is None:
        logger.error("No se detectaron columnas text/label. Columnas: %s", list(df.columns))
        sys.exit(1)

    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)

    label_map = {"spam": 1, "1": 1, "yes": 1, "ham": 0, "0": 0, "no": 0}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("Dataset cargado: %d muestras (Ham=%d | Spam=%d)",
                len(df), (df["label"] == 0).sum(), (df["label"] == 1).sum())
    return df

# ==============================================================================
# REPRODUCCIÓN DEL SPLIT DE TEST
# ==============================================================================

def get_test_split(df: pd.DataFrame):
    """
    Reproduce el mismo split 70/30 estratificado del experimento.
    Solo se usa para obtener el conjunto de evaluación con etiquetas reales;
    el vectorizador NO se re-ajusta aquí (ya viene dentro del bundle).
    """
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    logger.info("Conjunto de test: %d muestras", len(X_test))
    return X_test, y_test

# ==============================================================================
# VECTORIZACIÓN — usa el vectorizador del bundle, sin re-ajustar
# ==============================================================================

def transform_with_bundle(vec_name: str, vectorizer, texts: pd.Series) -> np.ndarray:
    """
    Transforma textos usando el vectorizador ya ajustado que viene en el bundle.
    No llama a fit() en ningún momento.

    Parameters
    ----------
    vec_name   : str   nombre del vectorizador
    vectorizer : objeto ajustado (sklearn vectorizer o gensim Word2Vec)
    texts      : pd.Series  textos a transformar

    Returns
    -------
    sparse matrix o np.ndarray
    """
    if vec_name in ("one_hot", "bow", "tfidf"):
        return vectorizer.transform(texts)

    # Word2Vec: promedio de embeddings
    def _doc_vec(tokens):
        vecs = [vectorizer.wv[w] for w in tokens if w in vectorizer.wv]
        return np.mean(vecs, axis=0).astype(np.float32) if vecs \
               else np.zeros(vectorizer.vector_size, dtype=np.float32)

    return np.array([_doc_vec(t.split()) for t in texts], dtype=np.float32)

# ==============================================================================
# CARGA DEL BUNDLE
# ==============================================================================

def load_bundle(model_path: Path) -> tuple:
    """
    Carga el bundle guardado por el experimento y extrae sus componentes.

    Returns
    -------
    model, vectorizer, vec_name, meta
    """
    logger.info("Cargando bundle desde '%s' ...", model_path)
    bundle = joblib.load(model_path)

    # Compatibilidad: si el .pkl es un modelo suelto (formato antiguo, pre-v4)
    # no tiene las claves del bundle y se sale con un mensaje claro.
    if not isinstance(bundle, dict) or "model" not in bundle:
        logger.error(
            "El archivo '%s' no contiene un bundle válido. "
            "Parece ser un modelo exportado con una versión anterior (pre-v4). "
            "Vuelve a ejecutar spam_classifier_experiment.py para regenerarlo.",
            model_path,
        )
        sys.exit(1)

    model      = bundle["model"]
    vectorizer = bundle["vectorizer"]
    vec_name   = bundle["vec_name"]
    meta       = bundle.get("meta", {})

    logger.info("Modelo cargado  : %s", type(model).__name__)
    logger.info("Vectorizador    : %s", vec_name)
    logger.info("Entrenado con   : %d muestras", meta.get("trained_on_n_samples", "?"))
    logger.info("F1 en experimento: %.4f", meta.get("experiment_test_f1", float("nan")))
    logger.info("Exportado el    : %s", meta.get("exported_at", "?"))

    return model, vectorizer, vec_name, meta

# ==============================================================================
# INFERENCIA
# ==============================================================================

def run_inference(model, X_test) -> tuple:
    """
    Ejecuta la inferencia y devuelve predicciones, probabilidades
    y el tiempo de predicción (sin vectorización).
    """
    t0 = time.perf_counter()
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    return y_pred, y_proba, elapsed


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "log_loss":  round(log_loss(y_test, y_proba), 4),
    }

# ==============================================================================
# THROUGHPUT BENCHMARK
# ==============================================================================

def benchmark_throughput(model, vectorizer, vec_name, X_test_raw, n_repeats) -> dict:
    """
    Mide throughput del pipeline completo (vectorización + predicción)
    con warm-up y promedio de n_repeats repeticiones.
    """
    n = len(X_test_raw)
    logger.info("Benchmarking: %d correos × %d repeticiones ...", n, n_repeats)

    # Warm-up
    _Xw = transform_with_bundle(vec_name, vectorizer, X_test_raw)
    model.predict(_Xw); del _Xw

    vec_times, pred_times = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        X_vec = transform_with_bundle(vec_name, vectorizer, X_test_raw)
        vec_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        model.predict(X_vec)
        pred_times.append(time.perf_counter() - t1)

    vec_times   = np.array(vec_times)
    pred_times  = np.array(pred_times)
    total_times = vec_times + pred_times
    avg_vec, avg_pred, avg_total = vec_times.mean(), pred_times.mean(), total_times.mean()
    latency_ms = (total_times / n) * 1_000

    return {
        "n_emails":            n,
        "n_repeats":           n_repeats,
        "total_time_s":        round(float(avg_total), 6),
        "throughput_eps":      round(n / avg_total, 2),
        "latency_ms":          round((avg_total / n) * 1_000, 4),
        "latency_p50_ms":      round(float(np.percentile(latency_ms, 50)), 4),
        "latency_p95_ms":      round(float(np.percentile(latency_ms, 95)), 4),
        "latency_p99_ms":      round(float(np.percentile(latency_ms, 99)), 4),
        "vec_time_s":          round(float(avg_vec), 6),
        "vec_throughput_eps":  round(n / avg_vec, 2),
        "vec_latency_ms":      round((avg_vec / n) * 1_000, 4),
        "pred_time_s":         round(float(avg_pred), 6),
        "pred_throughput_eps": round(n / avg_pred, 2),
        "pred_latency_ms":     round((avg_pred / n) * 1_000, 4),
    }


def log_throughput_report(tp, model_name, vec_name):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  THROUGHPUT REPORT — %s + %s", model_name.upper(), vec_name.upper())
    logger.info("=" * 60)
    logger.info("  Correos en test set    : %d", tp["n_emails"])
    logger.info("  Repeticiones benchmark : %d", tp["n_repeats"])
    logger.info("")
    logger.info("  ── Pipeline completo ───────────────────────────────────")
    logger.info("  Tiempo total (media)   : %.4f s",   tp["total_time_s"])
    logger.info("  Throughput             : %s emails/s", f"{tp['throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/email", tp["latency_ms"])
    logger.info("  Latencia p50           : %.4f ms/email", tp["latency_p50_ms"])
    logger.info("  Latencia p95           : %.4f ms/email", tp["latency_p95_ms"])
    logger.info("  Latencia p99           : %.4f ms/email", tp["latency_p99_ms"])
    logger.info("")
    logger.info("  ── Solo vectorización ──────────────────────────────────")
    logger.info("  Tiempo (media)         : %.4f s",   tp["vec_time_s"])
    logger.info("  Throughput             : %s emails/s", f"{tp['vec_throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/email", tp["vec_latency_ms"])
    logger.info("")
    logger.info("  ── Solo predicción ─────────────────────────────────────")
    logger.info("  Tiempo (media)         : %.4f s",   tp["pred_time_s"])
    logger.info("  Throughput             : %s emails/s", f"{tp['pred_throughput_eps']:,.2f}")
    logger.info("  Latencia media         : %.4f ms/email", tp["pred_latency_ms"])
    logger.info("=" * 60)


def plot_throughput(tp, model_name, vec_name, output_dir):
    title_suffix = f"{model_name.upper()} + {vec_name.upper()}"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Throughput Benchmark — {title_suffix}", fontsize=13, fontweight="bold")
    colors = ["#4C9BE8", "#E87B4C", "#6DBE6D"]

    ax = axes[0]
    stages  = ["Vectorización", "Predicción", "Total (pipeline)"]
    times_s = [tp["vec_time_s"], tp["pred_time_s"], tp["total_time_s"]]
    bars = ax.barh(stages, times_s, color=colors, edgecolor="grey", linewidth=0.5)
    for bar, val in zip(bars, times_s):
        ax.text(val + max(times_s) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val * 1_000:.2f} ms", va="center", fontsize=10)
    ax.set_xlabel("Tiempo promedio (s)", fontsize=11)
    ax.set_title("Tiempo medio por etapa", fontsize=11)
    ax.grid(axis="x", alpha=0.3); ax.set_xlim(0, max(times_s) * 1.30)

    ax2 = axes[1]
    thr_vals = [tp["vec_throughput_eps"], tp["pred_throughput_eps"], tp["throughput_eps"]]
    bars2 = ax2.bar(["Vectorización", "Predicción", "Total\n(pipeline)"],
                    thr_vals, color=colors, edgecolor="grey", linewidth=0.5, width=0.5)
    for bar, val in zip(bars2, thr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(thr_vals) * 0.015,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Emails por segundo", fontsize=11)
    ax2.set_title("Throughput (emails/s)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3); ax2.set_ylim(0, max(thr_vals) * 1.20)

    fig.text(0.5, -0.05,
             f"Latencia pipeline — media: {tp['latency_ms']:.4f} ms  |  "
             f"p50: {tp['latency_p50_ms']:.4f} ms  |  "
             f"p95: {tp['latency_p95_ms']:.4f} ms  |  "
             f"p99: {tp['latency_p99_ms']:.4f} ms",
             ha="center", fontsize=9, color="dimgrey")

    plt.tight_layout()
    path = output_dir / f"throughput_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# VISUALIZACIONES DE CLASIFICACIÓN
# ==============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_roc_curve(y_test, y_proba, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"{model_name}  (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    ax.set_xlabel("FPR", fontsize=12); ax.set_ylabel("TPR", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"roc_curve_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_score_distribution(y_test, y_proba, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0, "steelblue", "Ham"), (1, "tomato", "Spam")]:
        mask = y_test == label
        ax.hist(y_proba[mask], bins=50, alpha=0.6, color=color,
                label=f"{name} (n={mask.sum()})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Umbral 0.5")
    ax.set_xlabel("P[Spam]", fontsize=12); ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title(f"Distribución de scores — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"score_distribution_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# UTILIDADES
# ==============================================================================

def list_models(models_dir: Path):
    print("\nModelos disponibles:")
    print("-" * 40)
    found = False
    for key, fname in MODEL_FILES.items():
        path   = models_dir / fname
        status = "OK" if path.exists() else "NO ENCONTRADO"
        print(f"  --model {key:<8}  {path}  [{status}]")
        if path.exists():
            found = True
    print()
    if not found:
        print("  ⚠  No se encontró ningún modelo. Ejecuta primero el experimento.")
    print()

# ==============================================================================
# ARGPARSE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia sobre el conjunto de test usando modelos entrenados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python inference.py --list
  python inference.py --model logreg
  python inference.py --model svm --save_predictions
  python inference.py --model logreg --benchmark_repeats 10
  python inference.py --model logreg --no_benchmark
        """,
    )
    parser.add_argument("--model", type=str, choices=list(MODEL_FILES.keys()),
                        help="Modelo a usar: logreg | svm | rf")
    parser.add_argument("--data_dir",    type=str, default="data")
    parser.add_argument("--models_dir",  type=str, default="models")
    parser.add_argument("--output_dir",  type=str, default="inference_results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Guarda CSV con predicción y probabilidad por muestra.")
    parser.add_argument("--list", action="store_true",
                        help="Lista los modelos disponibles y sale.")
    parser.add_argument("--benchmark_repeats", type=int, default=THROUGHPUT_REPEATS, metavar="N")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="Omite el benchmark de throughput.")
    return parser.parse_args()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models(models_dir)
        sys.exit(0)

    if args.model is None:
        print("ERROR: especifica --model o usa --list.\n")
        print("  python inference.py --model logreg")
        sys.exit(1)

    model_name = args.model
    model_path = models_dir / MODEL_FILES[model_name]

    logger.info("=" * 60)
    logger.info("  INFERENCIA — modelo: %s", model_name)
    logger.info("=" * 60)

    if not model_path.exists():
        logger.error("No se encontró '%s'. Ejecuta primero el experimento.", model_path)
        sys.exit(1)

    # ── Cargar bundle (modelo + vectorizador + meta) ───────────────────
    model, vectorizer, vec_name, meta = load_bundle(model_path)

    # ── Datos: solo necesitamos el subset de test con etiquetas reales ─
    df = load_and_prepare(data_dir)
    X_test_raw, y_test = get_test_split(df)
    y_test_arr = y_test.values

    # ── Vectorizar con el vectorizador del bundle (sin re-ajustar) ─────
    logger.info("Vectorizando test con el vectorizador del bundle (%s) ...", vec_name)
    X_test = transform_with_bundle(vec_name, vectorizer, X_test_raw)
    logger.info("Test vectorizado: shape=%s | sparse=%s", X_test.shape, issparse(X_test))

    # ── Inferencia ─────────────────────────────────────────────────────
    logger.info("Ejecutando inferencia ...")
    y_pred, y_proba, pred_elapsed = run_inference(model, X_test)
    logger.info("Predicción: %.4f s | %d correos | %.2f emails/s | %.4f ms/email",
                pred_elapsed, len(y_test_arr),
                len(y_test_arr) / pred_elapsed,
                (pred_elapsed / len(y_test_arr)) * 1_000)

    # ── Métricas ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_test_arr, y_pred, y_proba)

    logger.info("\n%s", "=" * 60)
    logger.info("MÉTRICAS SOBRE EL CONJUNTO DE TEST")
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info("  %-12s %.4f", name.upper(), value)
    logger.info("=" * 60)
    logger.info("\nClassification Report:\n%s",
                classification_report(y_test_arr, y_pred,
                                      target_names=["Ham", "Spam"], digits=4))

    # ── Benchmark de throughput ────────────────────────────────────────
    throughput_data = {}
    if not args.no_benchmark:
        throughput_data = benchmark_throughput(
            model, vectorizer, vec_name, X_test_raw, args.benchmark_repeats)
        log_throughput_report(throughput_data, model_name, vec_name)
        plot_throughput(throughput_data, model_name, vec_name, output_dir)

    # ── Guardar métricas en JSON ───────────────────────────────────────
    full_report = {
        "model": model_name, "vectorizer": vec_name,
        **metrics,
        "meta": meta,
        **({"throughput": throughput_data} if throughput_data else {}),
    }
    metrics_path = output_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Métricas guardadas en '%s'.", metrics_path)

    # ── Visualizaciones ────────────────────────────────────────────────
    logger.info("Generando visualizaciones ...")
    plot_confusion_matrix(y_test_arr, y_pred,   model_name, output_dir)
    plot_roc_curve(y_test_arr, y_proba,          model_name, output_dir)
    plot_score_distribution(y_test_arr, y_proba, model_name, output_dir)

    # ── CSV de predicciones (opcional) ────────────────────────────────
    if args.save_predictions:
        preds_df = pd.DataFrame({
            "text":       X_test_raw.values,
            "label_real": y_test_arr,
            "label_pred": y_pred,
            "proba_spam": np.round(y_proba, 4),
            "correct":    (y_test_arr == y_pred).astype(int),
        })
        preds_path = output_dir / f"predictions_{model_name}.csv"
        preds_df.to_csv(preds_path, index=False)
        errors = preds_df[preds_df["correct"] == 0]
        logger.info("Predicciones guardadas en '%s'. Errores: %d / %d (%.2f%%)",
                    preds_path, len(errors), len(preds_df),
                    100 * len(errors) / len(preds_df))
        fn = errors[errors["label_real"] == 1]
        fp = errors[errors["label_real"] == 0]
        logger.info("  Falsos Negativos (Spam→Ham): %d", len(fn))
        logger.info("  Falsos Positivos (Ham→Spam): %d", len(fp))

    # ── Resumen ────────────────────────────────────────────────────────
    logger.info("\nArtefactos generados en '%s':", output_dir)
    for f in sorted(output_dir.iterdir()):
        if f.name.endswith((".png", ".json", ".csv")):
            logger.info("  %s", f.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()