"""
email_classifier_nuevo_dataset.py
==================================
Pipeline completo de clasificación Spam vs Ham usando representaciones de texto clásicas
y modelos de ML con parámetros por defecto.

Dataset: Dataset propio de emails (CSV local con columnas email_corpus, subject, is_spam, ...)
Autor  : Senior Data Scientist — NLP & Classical ML

Uso
───
    # Entrenar todo (comportamiento por defecto)
    python email_classifier_nuevo_dataset.py

    # Apuntar a un CSV concreto
    python email_classifier_nuevo_dataset.py --data ruta/al/archivo.csv

    # Entrenar solo un modelo concreto
    python email_classifier_nuevo_dataset.py --models logreg
    python email_classifier_nuevo_dataset.py --models svm
    python email_classifier_nuevo_dataset.py --models rf

    # Entrenar una combinación de modelos
    python email_classifier_nuevo_dataset.py --models logreg svm

    # Restringir también los vectorizadores (útil para ahorrar RAM)
    python email_classifier_nuevo_dataset.py --models logreg --vecs tfidf bow
    python email_classifier_nuevo_dataset.py --models rf --vecs tfidf

    # Combinar subject + cuerpo del email (por defecto solo cuerpo)
    python email_classifier_nuevo_dataset.py --combine-subject

CAMBIOS v8 (adaptación al nuevo dataset)
─────────────────────────────────────────
• Eliminada la descarga automática de Kaggle. El CSV se lee directamente de disco.
  Se puede indicar la ruta con --data (por defecto busca en la carpeta "data/").
• Columna de texto: "email_corpus" (se elimina el prefijo "BODY:" si está presente).
• Columna de etiqueta: "is_spam" (True/False o 1/0 → se convierte a int binario).
• Opción --combine-subject para concatenar el asunto ("subject") al cuerpo del email.
• El limpiado de texto acepta caracteres acentuados y ñ (corpus mayormente en español).
• El resto del pipeline (vectorización, CV, evaluación, plots, exportación) es idéntico a v7.

MODELOS
───────
• "logreg" → LogisticRegression(n_jobs=-1)
• "svm"    → SGDClassifier(loss="modified_huber", n_jobs=-1)
• "rf"     → RandomForestClassifier(n_jobs=-1)
"""

# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import argparse
import os, sys, json, logging, warnings, random, re
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import issparse

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline          import Pipeline
from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model      import LogisticRegression, SGDClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
)

import joblib
from gensim.models import Word2Vec

# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL
# ==============================================================================

RANDOM_STATE       = 42
TEST_SIZE          = 0.30
N_SPLITS_CV        = 5
WORD2VEC_DIM       = 100
WORD2VEC_WINDOW    = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_EPOCHS    = 10
MAX_FEATURES       = 20_000

ALL_MODELS = ["logreg", "svm", "rf"]
ALL_VECS   = ["one_hot", "bow", "tfidf", "word2vec"]

OUTPUT_DIR = Path(".")
PLOTS_DIR  = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ==============================================================================
# 2. ARGPARSE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Email Spam vs Ham classifier — entrena y evalúa combinaciones de modelos y vectorizadores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python email_classifier_nuevo_dataset.py
  python email_classifier_nuevo_dataset.py --data mi_dataset.csv
  python email_classifier_nuevo_dataset.py --models logreg
  python email_classifier_nuevo_dataset.py --models svm rf
  python email_classifier_nuevo_dataset.py --models logreg --vecs tfidf bow
  python email_classifier_nuevo_dataset.py --models rf --vecs tfidf
  python email_classifier_nuevo_dataset.py --combine-subject
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help=(
            "Ruta al CSV (o carpeta que lo contenga). "
            "Por defecto busca el primer CSV en la carpeta 'data/'."
        ),
    )
    parser.add_argument(
        "--combine-subject",
        action="store_true",
        default=False,
        help=(
            "Si se activa, concatena la columna 'subject' al cuerpo del email "
            "antes de vectorizar. Por defecto solo se usa el cuerpo (email_corpus)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        metavar="MODEL",
        help=(
            "Modelos a entrenar. Opciones: logreg, svm, rf. "
            "Se pueden indicar varios separados por espacio. "
            "Por defecto: todos."
        ),
    )
    parser.add_argument(
        "--vecs",
        nargs="+",
        choices=ALL_VECS,
        default=ALL_VECS,
        metavar="VEC",
        help=(
            "Vectorizadores a usar. Opciones: one_hot, bow, tfidf, word2vec. "
            "Se pueden indicar varios separados por espacio. "
            "Por defecto: todos."
        ),
    )
    return parser.parse_args()

# ==============================================================================
# 3. LOGGING
# ==============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("experiment.log", mode="a"),  # append para ejecuciones parciales
        ],
    )
    return logging.getLogger("SpamClassifier")

logger = setup_logging()

# ==============================================================================
# 4. CARGA DEL DATASET
# ==============================================================================

# Columnas esperadas en el nuevo CSV
COL_BODY    = "email_corpus"   # cuerpo del email
COL_SUBJECT = "subject"        # asunto (opcional, ver --combine-subject)
COL_LABEL   = "is_spam"        # etiqueta: True/False o 1/0


def find_csv(data_arg: str | None) -> Path:
    """
    Devuelve la ruta al CSV a cargar.
      - Si data_arg es un fichero .csv existente → lo usa directamente.
      - Si data_arg es una carpeta → primer CSV dentro de ella.
      - Si data_arg es None → primer CSV en la carpeta 'data/'.
    """
    if data_arg is not None:
        p = Path(data_arg)
        if p.is_file():
            return p
        if p.is_dir():
            csvs = list(p.glob("*.csv"))
            if csvs:
                return csvs[0]
        raise FileNotFoundError(f"No se encontró ningún CSV en '{data_arg}'.")

    default_dir = Path("data")
    if default_dir.exists():
        csvs = list(default_dir.glob("*.csv"))
        if csvs:
            return csvs[0]

    # Último recurso: primer CSV en el directorio actual
    csvs = list(Path(".").glob("*.csv"))
    if csvs:
        return csvs[0]

    raise FileNotFoundError(
        "No se encontró ningún CSV. Usa --data para indicar la ruta al fichero."
    )


def load_dataset(data_arg: str | None = None, combine_subject: bool = False) -> pd.DataFrame:
    """
    Carga el CSV del nuevo dataset y devuelve un DataFrame con columnas
    'text' (cuerpo limpio, opcionalmente con asunto) y 'label' (0/1).

    Columnas esperadas en el CSV:
      - email_corpus : cuerpo del email (puede empezar por "BODY:")
      - subject      : asunto (solo si combine_subject=True)
      - is_spam      : etiqueta booleana o 0/1
    """
    csv_path = find_csv(data_arg)
    logger.info("Cargando dataset desde '%s' ...", csv_path)

    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8",
        dtype=str,
        quotechar='"',
        engine="python",
    )
    logger.info("Shape original: %s", df.shape)

    # ── Normalizar nombres de columna ──────────────────────────────────
    df.columns = [c.strip().lower() for c in df.columns]

    required = {COL_BODY, COL_LABEL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El CSV no contiene las columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # ── Construir columna de texto ─────────────────────────────────────
    # Eliminar el prefijo "BODY:" o "BODY: " que generan algunos registros
    body = df[COL_BODY].fillna("").str.replace(r"(?i)^body:\s*", "", regex=True)

    if combine_subject and COL_SUBJECT in df.columns:
        subject = df[COL_SUBJECT].fillna("")
        df["text"] = subject + " " + body
        logger.info("Columna de texto: subject + email_corpus (--combine-subject activado).")
    else:
        df["text"] = body
        logger.info("Columna de texto: email_corpus.")

    # ── Construir etiqueta binaria ─────────────────────────────────────
    # is_spam puede ser True/False (string) o 1/0
    label_raw = df[COL_LABEL].str.strip().str.lower()
    label_map = {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
    df["label"] = label_raw.map(label_map)

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    if len(df) < before:
        logger.warning("Eliminadas %d filas con valores nulos o etiquetas desconocidas.",
                       before - len(df))

    logger.info("Distribución — Ham (0): %d | Spam (1): %d",
                (df["label"] == 0).sum(), (df["label"] == 1).sum())

    return df[["text", "label"]]

# ==============================================================================
# 5. PREPROCESAMIENTO
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Lowercase + elimina caracteres no-alfanuméricos conservando acentos y ñ
    (el corpus puede estar en español) + normaliza espacios.
    """
    text = str(text).lower()
    # Mantiene letras (incluyendo acentuadas y ñ), dígitos y espacios
    text = re.sub(r"[^a-záéíóúüñàèìòùâêîôûäëïöüa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia texto y elimina nulos.
    La conversión de etiquetas a 0/1 ya se hace en load_dataset,
    por lo que aquí solo se limpia el texto y se descartan nulos residuales.
    """
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    # Descartar textos vacíos tras la limpieza
    df = df[df["text"].str.strip() != ""]
    if len(df) < before:
        logger.warning("Eliminadas %d filas con texto vacío tras la limpieza.", before - len(df))

    logger.info("Dataset listo — Ham: %d | Spam: %d",
                (df["label"] == 0).sum(), (df["label"] == 1).sum())
    return df


def split_data(df: pd.DataFrame):
    """Division estratificada 70/30."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    logger.info("Train: %d muestras | Test: %d muestras  (%%Spam train: %.1f%%)",
                len(X_train), len(X_test), 100 * y_train.mean())
    return X_train, X_test, y_train, y_test

# ==============================================================================
# 6. VECTORIZADORES
# ==============================================================================

def get_sparse_vectorizers() -> dict:
    """Devuelve instancias nuevas de los tres vectorizadores sparse."""
    return {
        "one_hot": CountVectorizer(binary=True,  max_features=MAX_FEATURES, dtype=np.float32),
        "bow":     CountVectorizer(binary=False, max_features=MAX_FEATURES, dtype=np.float32),
        "tfidf":   TfidfVectorizer(max_features=MAX_FEATURES, dtype=np.float32),
    }


def build_fresh_vectorizer(vec_name: str):
    """Devuelve una instancia nueva sin ajustar del vectorizador sparse indicado."""
    vecs = get_sparse_vectorizers()
    if vec_name not in vecs:
        raise ValueError(f"vec_name desconocido para vectorizador sparse: '{vec_name}'")
    return vecs[vec_name]


def tokenize_corpus(texts: pd.Series) -> list:
    return [t.split() for t in texts]


def train_word2vec(corpus: pd.Series) -> Word2Vec:
    """Entrena Word2Vec sobre el corpus dado."""
    logger.info("Entrenando Word2Vec (dim=%d, n=%d) ...", WORD2VEC_DIM, len(corpus))
    model = Word2Vec(
        sentences=tokenize_corpus(corpus),
        vector_size=WORD2VEC_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=4,
        seed=RANDOM_STATE,
        epochs=WORD2VEC_EPOCHS,
    )
    logger.info("Word2Vec listo. Vocab: %d tokens.", len(model.wv))
    return model


def document_vector(tokens: list, w2v_model: Word2Vec) -> np.ndarray:
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs \
           else np.zeros(w2v_model.vector_size, dtype=np.float32)


def vectorize_word2vec(texts: pd.Series, w2v_model: Word2Vec) -> np.ndarray:
    return np.array(
        [document_vector(tok, w2v_model) for tok in tokenize_corpus(texts)],
        dtype=np.float32,
    )

# ==============================================================================
# 7. MODELOS
# ==============================================================================

def get_models() -> dict:
    """Instancias frescas de los tres modelos con parámetros por defecto."""
    return {
        "logreg": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
        "svm": SGDClassifier(
            loss="modified_huber", random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
        "rf": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1),
    }

# ==============================================================================
# 8. CROSS-VALIDATION 10-FOLD (sin data leakage mediante Pipeline)
# ==============================================================================

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Wrapper sklearn-compatible para Word2Vec.
    Permite usarlo dentro de un Pipeline garantizando que en cada fold
    de CV el modelo se ajusta solo sobre los datos de train de ese fold.
    """
    def __init__(self, vector_size=100, window=5, min_count=2,
                 epochs=10, workers=4, seed=42):
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.epochs      = epochs
        self.workers     = workers
        self.seed        = seed

    def fit(self, X, y=None):
        sentences = [t.split() for t in X]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
        )
        return self

    def transform(self, X):
        def _doc_vec(tokens):
            vecs = [self.model_.wv[w] for w in tokens if w in self.model_.wv]
            return np.mean(vecs, axis=0).astype(np.float32) if vecs \
                   else np.zeros(self.vector_size, dtype=np.float32)
        return np.array([_doc_vec(t.split()) for t in X], dtype=np.float32)


def get_cv_pipeline(vec_name: str, model_name: str) -> Pipeline:
    """
    Construye un Pipeline sklearn (vectorizador + modelo) para usarlo en CV.
    En cada fold el vectorizador se ajusta solo sobre el subconjunto de train
    de ese fold, eliminando el data leakage.
    """
    vectorizers = {
        "one_hot":  CountVectorizer(binary=True,  max_features=MAX_FEATURES, dtype=np.float32),
        "bow":      CountVectorizer(binary=False, max_features=MAX_FEATURES, dtype=np.float32),
        "tfidf":    TfidfVectorizer(max_features=MAX_FEATURES, dtype=np.float32),
        "word2vec": Word2VecVectorizer(
            vector_size=WORD2VEC_DIM, window=WORD2VEC_WINDOW,
            min_count=WORD2VEC_MIN_COUNT, epochs=WORD2VEC_EPOCHS,
            seed=RANDOM_STATE,
        ),
    }
    return Pipeline([
        ("vectorizer", vectorizers[vec_name]),
        ("model",      get_models()[model_name]),
    ])


def cross_validate_model(
    model_name: str,
    vec_name: str,
    X_train_raw: pd.Series,
    y_train: np.ndarray,
) -> dict:
    """
    Evalúa un pipeline (vectorizador + modelo) con 10-Fold Stratified CV
    sobre los textos crudos del train, sin data leakage.
    """
    cv       = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    pipeline = get_cv_pipeline(vec_name, model_name)

    scores = cross_validate(
        pipeline, X_train_raw, y_train,
        cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "cv_f1_mean":  round(float(scores["test_f1"].mean()), 4),
        "cv_f1_std":   round(float(scores["test_f1"].std()),  4),
        "cv_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
        "cv_auc_std":  round(float(scores["test_roc_auc"].std()),  4),
        "cv_acc_mean": round(float(scores["test_accuracy"].mean()), 4),
        "cv_acc_std":  round(float(scores["test_accuracy"].std()),  4),
    }

# ==============================================================================
# 9. RE-ENTRENAMIENTO SOBRE EL DATASET COMPLETO Y EXPORTACIÓN
# ==============================================================================

def retrain_on_full_dataset(
    model_name: str,
    vec_name: str,
    df_full: pd.DataFrame,
    test_f1: float,
) -> None:
    """
    Re-entrena vectorizador y modelo sobre el 100% del dataset
    y guarda el bundle completo en disco.
    """
    X_all = df_full["text"]
    y_all = df_full["label"].values

    logger.info("Re-entrenando '%s + %s' sobre el dataset completo (%d muestras) ...",
                vec_name, model_name, len(X_all))

    if vec_name == "word2vec":
        vectorizer = train_word2vec(X_all)
        X_vec = vectorize_word2vec(X_all, vectorizer)
    else:
        vectorizer = build_fresh_vectorizer(vec_name)
        X_vec = vectorizer.fit_transform(X_all)

    logger.info("  Vectorización completa: shape=%s | sparse=%s",
                X_vec.shape, issparse(X_vec))

    model = get_models()[model_name]
    model.fit(X_vec, y_all)
    logger.info("  Modelo re-entrenado sobre %d muestras.", len(y_all))

    bundle = {
        "model":      model,
        "vectorizer": vectorizer,
        "vec_name":   vec_name,
        "meta": {
            "model_name":            model_name,
            "vec_name":              vec_name,
            "trained_on_n_samples":  int(len(y_all)),
            "max_features":          MAX_FEATURES,
            "word2vec_dim":          WORD2VEC_DIM,
            "random_state":          RANDOM_STATE,
            "experiment_test_f1":    round(float(test_f1), 4),
            "exported_at":           datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }

    out_path = MODELS_DIR / f"best_{model_name}.pkl"
    joblib.dump(bundle, out_path)
    logger.info("  Bundle guardado en '%s'  (model + vectorizer + meta).", out_path)


def save_best_models(trained_models, results_df, df_full) -> None:
    """
    Para cada tipo de modelo presente en results_df, identifica la mejor
    combinación (mayor F1 en test), re-entrena sobre el 100% del dataset
    y guarda el bundle completo.
    """
    logger.info("")
    logger.info("Exportando bundles (re-entrenados sobre el 100%% del dataset) ...")

    for model_name in results_df["model"].unique():
        subset = results_df[results_df["model"] == model_name]
        best_row = subset.loc[subset["f1"].idxmax()]
        logger.info("Mejor combo para '%s': vectorizador='%s'  F1=%.4f (test)",
                    model_name, best_row["vectorizer"], best_row["f1"])

        retrain_on_full_dataset(
            model_name=model_name,
            vec_name=best_row["vectorizer"],
            df_full=df_full,
            test_f1=best_row["f1"],
        )

# ==============================================================================
# 10. EVALUACIÓN SOBRE TEST
# ==============================================================================

def evaluate_model(model, X_test, y_test, vec_name, model_name) -> dict:
    """Calcula métricas sobre el conjunto de test."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "vectorizer": vec_name,
        "model":      model_name,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_proba), 4),
        "log_loss":   round(log_loss(y_test, y_proba), 4),
    }

# ==============================================================================
# 11. VISUALIZACIONES
# ==============================================================================

def plot_confusion_matrices(trained_models, X_tests_per_key, y_test):
    n = len(trained_models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for ax, ((vec, mdl), model) in zip(axes, trained_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_tests_per_key[(vec, mdl)]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_title(f"{vec.upper()} + {mdl}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_roc_curves(trained_models, X_tests_per_key, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20.colors
    for i, ((vec, mdl), model) in enumerate(trained_models.items()):
        y_proba = model.predict_proba(X_tests_per_key[(vec, mdl)])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr,
                label=f"{vec.upper()} + {mdl}  (AUC={roc_auc_score(y_test, y_proba):.3f})",
                color=colors[i % len(colors)], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    ax.set_xlabel("FPR", fontsize=12); ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("ROC Curves — Comparativa", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_precision_recall_curves(trained_models, X_tests_per_key, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20.colors
    for i, ((vec, mdl), model) in enumerate(trained_models.items()):
        y_proba = model.predict_proba(X_tests_per_key[(vec, mdl)])[:, 1]
        p, r, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(r, p, label=f"{vec.upper()} + {mdl}",
                color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Comparativa", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    path = PLOTS_DIR / "precision_recall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_model_comparison(results_df):
    df = results_df.copy()
    df["combo"] = df["vectorizer"].str.upper() + " + " + df["model"]
    df = df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.60)))
    bars = ax.barh(df["combo"], df["f1"],
                   color=plt.cm.RdYlGn(df["f1"]), edgecolor="grey",
                   linewidth=0.5, label="F1 test")

    if "cv_f1_mean" in df.columns:
        ax.errorbar(
            df["cv_f1_mean"], range(len(df)),
            xerr=df["cv_f1_std"],
            fmt="D", color="steelblue", markersize=5,
            capsize=4, linewidth=1.2, label="F1 CV (media ± std)",
        )

    for bar, val in zip(bars, df["f1"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Comparativa F1-Score — Vectorizador + Modelo\n"
                 "(barras = test | rombos = media CV ± std)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, min(1.08, df["f1"].max() + 0.10))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = PLOTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)

# ==============================================================================
# 12. EXPORTACIÓN DE RESULTADOS (incremental)
# ==============================================================================

def export_results(new_results: list) -> pd.DataFrame:
    """
    Guarda results.csv de forma incremental.

    Si ya existe un results.csv de una ejecución anterior, carga los resultados
    previos y añade los nuevos, evitando duplicados por (vectorizer, model).
    Esto permite ejecutar el script varias veces con distintos --models o --vecs
    y acumular todos los resultados en un único fichero.
    """
    csv_path = OUTPUT_DIR / "results.csv"
    new_df   = pd.DataFrame(new_results)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        # Eliminar entradas previas que se hayan re-ejecutado en esta sesión
        mask = old_df.set_index(["vectorizer", "model"]).index.isin(
            new_df.set_index(["vectorizer", "model"]).index
        )
        old_df = old_df[~mask]
        combined = pd.concat([old_df, new_df], ignore_index=True)
        logger.info(
            "results.csv actualizado: %d entradas previas + %d nuevas = %d total.",
            len(old_df), len(new_df), len(combined),
        )
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    logger.info("Resultados guardados en '%s'.", csv_path)
    return combined

# ==============================================================================
# 13. PIPELINE PRINCIPAL
# ==============================================================================

def run_experiment(selected_models: list, selected_vecs: list,
                   data_arg: str | None = None, combine_subject: bool = False):
    """
    Orquesta el pipeline completo end-to-end para las combinaciones
    de modelos y vectorizadores seleccionadas.

      1. Datos
      2. Vectorización (solo sobre train, solo vectorizadores seleccionados)
      3. 10-Fold CV sin data leakage
      4. Entrenamiento final sobre el 100% del train
      5. Evaluación sobre test
      6. Exportación incremental de results.csv
      7. Visualizaciones (sobre las combinaciones de esta ejecución)
      8. Re-entrenamiento sobre el 100% del dataset + exportación de bundles
    """
    logger.info("=" * 65)
    logger.info("  EMAIL CLASSIFIER EXPERIMENT  v8 -- Start")
    logger.info("  Modelos    : %s", selected_models)
    logger.info("  Vectorizers: %s", selected_vecs)
    logger.info("  Combine subject: %s", combine_subject)
    logger.info("=" * 65)

    # ── 1. Datos ───────────────────────────────────────────────────────
    df_raw  = load_dataset(data_arg=data_arg, combine_subject=combine_subject)
    df_full = prepare_data(df_raw)

    X_train_raw, X_test_raw, y_train, y_test = split_data(df_full)
    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    # ── 2. Vectorización (solo vectorizadores seleccionados) ───────────
    logger.info("Vectorizando textos (fit solo sobre train) ...")

    sparse_vecs = get_sparse_vectorizers()
    X_trains: dict = {}
    X_tests:  dict = {}

    for vec_name in selected_vecs:
        if vec_name == "word2vec":
            continue  # se gestiona aparte
        vectorizer = sparse_vecs[vec_name]
        logger.info("  Ajustando '%s' ...", vec_name)
        X_trains[vec_name] = vectorizer.fit_transform(X_train_raw)
        X_tests[vec_name]  = vectorizer.transform(X_test_raw)
        logger.info("  '%s' -> shape=%s | dtype=%s | sparse=%s",
                    vec_name, X_trains[vec_name].shape,
                    X_trains[vec_name].dtype, issparse(X_trains[vec_name]))

    if "word2vec" in selected_vecs:
        logger.info("  Entrenando y vectorizando 'word2vec' ...")
        w2v_model            = train_word2vec(X_train_raw)
        X_trains["word2vec"] = vectorize_word2vec(X_train_raw, w2v_model)
        X_tests["word2vec"]  = vectorize_word2vec(X_test_raw,  w2v_model)
        logger.info("  'word2vec' -> shape=%s | dtype=%s",
                    X_trains["word2vec"].shape, X_trains["word2vec"].dtype)

    total_combos = len(selected_vecs) * len(selected_models)
    logger.info("Combinaciones a entrenar: %d vec x %d modelos = %d",
                len(selected_vecs), len(selected_models), total_combos)

    # ── 3-5. CV + entrenamiento final + evaluación sobre test ──────────
    results:         list = []
    trained_models:  dict = {}
    X_tests_per_key: dict = {}

    pbar = tqdm(total=total_combos, desc="CV + entrenamiento", unit="combo")

    for vec_name in selected_vecs:
        X_tr = X_trains[vec_name]
        X_te = X_tests[vec_name]

        for model_name in selected_models:
            logger.info("  [%s + %s]", vec_name, model_name)

            # a) 10-Fold CV sin data leakage
            logger.info("    Ejecutando %d-Fold CV (sin data leakage) ...", N_SPLITS_CV)
            cv_metrics = cross_validate_model(
                model_name, vec_name, X_train_raw, y_train_arr)
            logger.info(
                "    CV — F1: %.4f ± %.4f | AUC: %.4f ± %.4f | Acc: %.4f ± %.4f",
                cv_metrics["cv_f1_mean"],  cv_metrics["cv_f1_std"],
                cv_metrics["cv_auc_mean"], cv_metrics["cv_auc_std"],
                cv_metrics["cv_acc_mean"], cv_metrics["cv_acc_std"],
            )

            # b) Entrenamiento final sobre el 100% del train
            logger.info("    Entrenando sobre el 100%% del train ...")
            model = get_models()[model_name]
            model.fit(X_tr, y_train_arr)

            # c) Evaluación sobre test
            test_metrics = evaluate_model(model, X_te, y_test_arr, vec_name, model_name)
            logger.info(
                "    TEST — F1: %.4f | AUC: %.4f | Acc: %.4f",
                test_metrics["f1"], test_metrics["roc_auc"], test_metrics["accuracy"],
            )

            results.append({**test_metrics, **cv_metrics})
            trained_models[(vec_name, model_name)]  = model
            X_tests_per_key[(vec_name, model_name)] = X_te
            pbar.update(1)

    pbar.close()

    # ── 6. Exportar results.csv (incremental) ─────────────────────────
    results_df = export_results(results)

    summary_cols = ["vectorizer", "model", "f1", "roc_auc", "accuracy",
                    "cv_f1_mean", "cv_f1_std"]
    logger.info("\n%s", "=" * 65)
    logger.info("RESULTADOS DE ESTA EJECUCIÓN")
    logger.info("(test = 30%% | cv = 10-Fold sobre 70%%)")
    logger.info("=" * 65)
    # Mostrar solo las filas de esta ejecución, no las acumuladas
    current_df = pd.DataFrame(results)
    logger.info("\n%s", current_df[summary_cols].to_string(index=False))

    # ── 7. Visualizaciones (combinaciones de esta ejecución) ───────────
    logger.info("Generando visualizaciones ...")
    plot_confusion_matrices(trained_models, X_tests_per_key, y_test_arr)
    plot_roc_curves(trained_models, X_tests_per_key, y_test_arr)
    plot_precision_recall_curves(trained_models, X_tests_per_key, y_test_arr)
    plot_model_comparison(results_df)   # usa todos los resultados acumulados

    # ── 8. Re-entrenamiento sobre el 100% + exportación de bundles ─────
    save_best_models(
        trained_models=trained_models,
        results_df=current_df,   # solo los modelos entrenados en esta ejecución
        df_full=df_full,
    )

    # ── 9. Resumen final ───────────────────────────────────────────────
    best = current_df.loc[current_df["f1"].idxmax()]
    logger.info(
        "\n MEJOR COMBO (esta ejecución): %s + %s"
        "\n   F1 test: %.4f | AUC test: %.4f"
        "\n   F1 CV:   %.4f ± %.4f",
        best["vectorizer"].upper(), best["model"],
        best["f1"], best["roc_auc"],
        best["cv_f1_mean"], best["cv_f1_std"],
    )
    logger.info("=" * 65)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        selected_models=args.models,
        selected_vecs=args.vecs,
        data_arg=args.data,
        combine_subject=args.combine_subject,
    )