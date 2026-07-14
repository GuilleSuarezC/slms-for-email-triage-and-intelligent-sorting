"""
inference.py
============
Script de inferencia para los modelos de clasificación Spam vs Ham entrenados
con spam_classifier_experiment.py.

Carga el modelo guardado (.pkl), vectoriza el conjunto de test y produce
predicciones con métricas completas y un CSV de resultados.

Uso
───
    # Listar modelos disponibles
    python inference.py --list

    # Inferencia con un modelo concreto
    python inference.py --model logreg
    python inference.py --model svm
    python inference.py --model rf

    # Opciones adicionales
    python inference.py --model logreg --data_dir data --models_dir models --output_dir inference_results
    python inference.py --model logreg --save_predictions   # guarda CSV con predicciones por muestra

Modelos disponibles
───────────────────
    logreg  →  models/best_logreg.pkl
    svm     →  models/best_svm.pkl
    rf      →  models/best_rf.pkl
"""

import argparse
import json
import logging
import re
import sys
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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
TEST_SIZE          = 0.30
MAX_FEATURES       = 20_000   # debe coincidir con el valor usado en el entrenamiento
WORD2VEC_DIM       = 100
WORD2VEC_WINDOW    = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_EPOCHS    = 10

# Mapeo modelo → nombre del fichero .pkl
MODEL_FILES = {
    "logreg": "best_logreg.pkl",
    "svm":    "best_svm.pkl",
    "rf":     "best_rf.pkl",
}

# Mapeo modelo → vectorizador que usó durante el entrenamiento.
# Se infiere del nombre del fichero leído desde results.csv si existe;
# si no, se usa el vectorizador con mejor F1 por defecto (tfidf).
DEFAULT_VECTORIZER = "tfidf"

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
    """Lowercase + elimina no-alfanuméricos + normaliza espacios."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare(data_dir: Path) -> pd.DataFrame:
    """
    Carga el primer CSV encontrado en data_dir, normaliza columnas
    y aplica la misma limpieza que en el entrenamiento.
    """
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No se encontró ningún CSV en '%s'.", data_dir)
        sys.exit(1)

    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip().lower() for c in df.columns]

    text_col = next(
        (c for c in df.columns if any(k in c for k in ("text", "message", "email"))),
        None,
    )
    label_col = next(
        (c for c in df.columns if any(k in c for k in ("label", "class", "category"))),
        None,
    )
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
    Reproduce exactamente el mismo split 70/30 estratificado que se usó
    durante el entrenamiento (mismo random_state y test_size).

    Returns
    -------
    X_test : pd.Series
    y_test : pd.Series
    """
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    logger.info("Conjunto de test reproducido: %d muestras", len(X_test))
    return X_test, y_test

# ==============================================================================
# VECTORIZACIÓN
# ==============================================================================

def fit_vectorizer(vec_name: str, X_train: pd.Series):
    """
    Ajusta el vectorizador indicado sobre X_train.

    Necesario porque los vectorizadores no se guardaron en el .pkl;
    se re-ajustan aquí sobre el split de train con los mismos parámetros
    que se usaron durante el entrenamiento.

    Parameters
    ----------
    vec_name : str   "one_hot" | "bow" | "tfidf" | "word2vec"
    X_train  : pd.Series   textos de entrenamiento

    Returns
    -------
    vectorizer o Word2Vec model
    """
    logger.info("Ajustando vectorizador '%s' sobre train ...", vec_name)

    if vec_name == "one_hot":
        vec = CountVectorizer(binary=True,  max_features=MAX_FEATURES, dtype=np.float32)
        vec.fit(X_train)
        return vec

    if vec_name == "bow":
        vec = CountVectorizer(binary=False, max_features=MAX_FEATURES, dtype=np.float32)
        vec.fit(X_train)
        return vec

    if vec_name == "tfidf":
        vec = TfidfVectorizer(max_features=MAX_FEATURES, dtype=np.float32)
        vec.fit(X_train)
        return vec

    if vec_name == "word2vec":
        sentences = [t.split() for t in X_train]
        model = Word2Vec(
            sentences=sentences,
            vector_size=WORD2VEC_DIM,
            window=WORD2VEC_WINDOW,
            min_count=WORD2VEC_MIN_COUNT,
            workers=4,
            seed=RANDOM_STATE,
            epochs=WORD2VEC_EPOCHS,
        )
        logger.info("Word2Vec ajustado. Vocab: %d tokens.", len(model.wv))
        return model

    raise ValueError(f"Vectorizador desconocido: '{vec_name}'. "
                     f"Opciones: one_hot, bow, tfidf, word2vec")


def transform(vec_name: str, vectorizer, texts: pd.Series) -> np.ndarray:
    """Transforma textos usando el vectorizador ya ajustado."""
    if vec_name in ("one_hot", "bow", "tfidf"):
        return vectorizer.transform(texts)

    # Word2Vec: promedio de embeddings
    def _doc_vec(tokens):
        vecs = [vectorizer.wv[w] for w in tokens if w in vectorizer.wv]
        return np.mean(vecs, axis=0).astype(np.float32) if vecs \
               else np.zeros(vectorizer.vector_size, dtype=np.float32)

    return np.array([_doc_vec(t.split()) for t in texts], dtype=np.float32)

# ==============================================================================
# INFERENCIA
# ==============================================================================

def run_inference(model, X_test, y_test: np.ndarray):
    """
    Ejecuta la inferencia y devuelve predicciones y probabilidades.

    Parameters
    ----------
    model  : estimador sklearn ya ajustado
    X_test : sparse matrix o np.ndarray
    y_test : np.ndarray

    Returns
    -------
    y_pred  : np.ndarray
    y_proba : np.ndarray  (probabilidad de clase positiva)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> dict:
    """Calcula el conjunto completo de métricas de evaluación."""
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "log_loss":  round(log_loss(y_test, y_proba), 4),
    }

# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: Path):
    """Guarda la confusion matrix del modelo."""
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_roc_curve(y_test, y_proba, model_name: str, output_dir: Path):
    """Guarda la curva ROC del modelo."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"{model_name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"roc_curve_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_score_distribution(y_test, y_proba, model_name: str, output_dir: Path):
    """
    Guarda un histograma de la distribución de probabilidades predichas,
    separado por clase real (Ham vs Spam).
    Útil para evaluar la calibración del modelo.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0, "steelblue", "Ham"), (1, "tomato", "Spam")]:
        mask = y_test == label
        ax.hist(y_proba[mask], bins=50, alpha=0.6, color=color,
                label=f"{name} (n={mask.sum()})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Umbral 0.5")
    ax.set_xlabel("Probabilidad predicha (P[Spam])", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title(f"Distribución de scores — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"score_distribution_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# UTILIDADES
# ==============================================================================

def resolve_vectorizer_for_model(model_name: str, results_csv: Path) -> str:
    """
    Determina qué vectorizador usó el mejor modelo guardado.

    Si existe results.csv (generado por el experimento), lee de ahí el
    vectorizador con mayor F1 para ese modelo.
    Si no existe, devuelve DEFAULT_VECTORIZER.

    Parameters
    ----------
    model_name  : str
    results_csv : Path

    Returns
    -------
    str  nombre del vectorizador
    """
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        subset = df[df["model"] == model_name]
        if not subset.empty:
            vec = subset.loc[subset["f1"].idxmax(), "vectorizer"]
            logger.info(
                "Vectorizador detectado desde results.csv: '%s' (F1=%.4f)",
                vec, subset["f1"].max(),
            )
            return vec

    logger.warning(
        "No se encontró results.csv o no hay entradas para '%s'. "
        "Usando vectorizador por defecto: '%s'.",
        model_name, DEFAULT_VECTORIZER,
    )
    return DEFAULT_VECTORIZER


def list_models(models_dir: Path):
    """Imprime los modelos disponibles en models_dir."""
    print("\nModelos disponibles:")
    print("-" * 40)
    found = False
    for key, fname in MODEL_FILES.items():
        path = models_dir / fname
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
  python inference.py --model rf --data_dir data --models_dir models --output_dir out
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_FILES.keys()),
        help="Modelo a usar para la inferencia: logreg | svm | rf",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directorio con el CSV del dataset (default: data)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directorio con los modelos .pkl (default: models)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directorio donde guardar los resultados (default: inference_results)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Guarda un CSV con texto, label real, predicción y probabilidad por muestra.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista los modelos disponibles y sale.",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=["one_hot", "bow", "tfidf", "word2vec"],
        default=None,
        help=(
            "Fuerza el uso de un vectorizador concreto. "
            "Si no se especifica, se detecta automáticamente desde results.csv."
        ),
    )
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

    # ── Listar modelos y salir ─────────────────────────────────────────
    if args.list:
        list_models(models_dir)
        sys.exit(0)

    if args.model is None:
        print("ERROR: especifica --model o usa --list para ver los disponibles.\n")
        print("  python inference.py --model logreg")
        print("  python inference.py --list")
        sys.exit(1)

    model_name  = args.model
    model_path  = models_dir / MODEL_FILES[model_name]
    results_csv = Path("results.csv")

    logger.info("=" * 60)
    logger.info("  INFERENCIA — modelo: %s", model_name)
    logger.info("=" * 60)

    # ── Cargar modelo ──────────────────────────────────────────────────
    if not model_path.exists():
        logger.error(
            "No se encontró el modelo en '%s'. "
            "Ejecuta primero spam_classifier_experiment.py.", model_path
        )
        sys.exit(1)

    logger.info("Cargando modelo desde '%s' ...", model_path)
    model = joblib.load(model_path)
    logger.info("Modelo cargado: %s", type(model).__name__)

    # ── Datos ──────────────────────────────────────────────────────────
    df = load_and_prepare(data_dir)
    X_test_raw, y_test = get_test_split(df)

    # Necesitamos X_train solo para re-ajustar el vectorizador
    from sklearn.model_selection import train_test_split
    X_train_raw, _, _, _ = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # ── Vectorizador ───────────────────────────────────────────────────
    if args.vectorizer:
        vec_name = args.vectorizer
        logger.info("Vectorizador forzado por argumento: '%s'", vec_name)
    else:
        vec_name = resolve_vectorizer_for_model(model_name, results_csv)

    vectorizer = fit_vectorizer(vec_name, X_train_raw)
    X_test     = transform(vec_name, vectorizer, X_test_raw)

    logger.info(
        "Test vectorizado: shape=%s | sparse=%s",
        X_test.shape, issparse(X_test),
    )

    # ── Inferencia ─────────────────────────────────────────────────────
    logger.info("Ejecutando inferencia ...")
    y_test_arr      = y_test.values
    y_pred, y_proba = run_inference(model, X_test, y_test_arr)

    # ── Métricas ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_test_arr, y_pred, y_proba)

    logger.info("\n%s", "=" * 60)
    logger.info("MÉTRICAS SOBRE EL CONJUNTO DE TEST")
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info("  %-12s %.4f", name.upper(), value)
    logger.info("=" * 60)

    # Classification report detallado
    report = classification_report(
        y_test_arr, y_pred,
        target_names=["Ham", "Spam"],
        digits=4,
    )
    logger.info("\nClassification Report:\n%s", report)

    # Guardar métricas en JSON
    metrics_path = output_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, "vectorizer": vec_name, **metrics}, f, indent=2)
    logger.info("Métricas guardadas en '%s'.", metrics_path)

    # ── Visualizaciones ────────────────────────────────────────────────
    logger.info("Generando visualizaciones ...")
    plot_confusion_matrix(y_test_arr, y_pred,  model_name, output_dir)
    plot_roc_curve(y_test_arr, y_proba,         model_name, output_dir)
    plot_score_distribution(y_test_arr, y_proba, model_name, output_dir)

    # ── CSV de predicciones por muestra (opcional) ─────────────────────
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
        logger.info("Predicciones por muestra guardadas en '%s'.", preds_path)

        # Resumen rápido de errores
        errors = preds_df[preds_df["correct"] == 0]
        logger.info(
            "Errores totales: %d / %d  (%.2f%%)",
            len(errors), len(preds_df), 100 * len(errors) / len(preds_df),
        )
        fn = errors[errors["label_real"] == 1]   # Spam predicho como Ham
        fp = errors[errors["label_real"] == 0]   # Ham predicho como Spam
        logger.info("  Falsos Negativos (Spam→Ham): %d", len(fn))
        logger.info("  Falsos Positivos (Ham→Spam): %d", len(fp))

    # ── Resumen final ──────────────────────────────────────────────────
    logger.info("\nArtefactos generados en '%s':", output_dir)
    for f in sorted(output_dir.iterdir()):
        if f.name.endswith((".png", ".json", ".csv")):
            logger.info("  %s", f.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()