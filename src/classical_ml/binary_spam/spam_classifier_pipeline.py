"""
=============================================================================
Pipeline End-to-End: Clasificación Binaria de Spam vs Ham (NLP)
=============================================================================
Dataset : 190K+ Spam | Ham Email Dataset (Kaggle)
Autor   : Data Scientist Senior - NLP & ML
Versión : 1.0

Estructura del pipeline:
  1. Descarga y preparación de datos
  2. Feature Engineering (One-Hot, BoW, TF-IDF, Word2Vec)
  3. Optimización de hiperparámetros con Optuna + CV=10
  4. Evaluación final en test set (30%)
  5. Visualizaciones comparativas
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. DEPENDENCIAS
# ─────────────────────────────────────────────────────────────────────────────
# Instalar antes de ejecutar:
#   pip install kaggle optuna lightgbm gensim scikit-learn
#               imbalanced-learn matplotlib seaborn pandas numpy

import os
import re
import warnings
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score,
    log_loss, classification_report, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models import Word2Vec

import lightgbm as lgb

# Suprimir warnings no críticos
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL  (modifica aquí para ajustar el experimento)
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.30          # 30 % para test
CV_FOLDS       = 10            # CV estrictamente 10-fold
OPTUNA_TRIALS  = 30            # Número de trials por método
MAX_FEATURES   = 20_000        # Vocabulario máximo para vectorizadores
W2V_DIM        = 100           # Dimensión de embeddings Word2Vec
W2V_WINDOW     = 5
W2V_MIN_COUNT  = 2
W2V_EPOCHS     = 10

# ── Métrica/objetivo de Optuna ───────────────────────────────────────────────
# Opciones disponibles: "f1_macro" | "accuracy" | "log_loss"
OPTIMIZE_METRIC = "f1_macro"    # <--- CAMBIA AQUÍ LA MÉTRICA A OPTIMIZAR

DATASET_DIR  = Path("./data")
FIGURES_DIR  = Path("./figures")
DATASET_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(dest_dir: Path) -> Path:
    """
    Descarga el dataset de Kaggle usando la API oficial.

    Requiere tener configurado ~/.kaggle/kaggle.json con tus credenciales.
    Si prefieres hacerlo manualmente:
      kaggle datasets download -d meruvulikith/190k-spam-ham-email-dataset-for-classification

    Returns
    -------
    Path al archivo CSV descomprimido.
    """
    import kaggle  # noqa: importación diferida para evitar error si no está instalado

    dataset_slug = "meruvulikith/190k-spam-ham-email-dataset-for-classification"
    zip_path = dest_dir / "spam_ham.zip"

    if not any(dest_dir.glob("*.csv")):
        logger.info("Descargando dataset de Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
        logger.info("Dataset descargado y descomprimido en %s", dest_dir)
    else:
        logger.info("Dataset ya presente en %s, omitiendo descarga.", dest_dir)

    # Buscar el CSV resultante (el nombre puede variar)
    csv_files = list(dest_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún CSV en {dest_dir}. "
            "Verifica la descarga manual o tus credenciales de Kaggle."
        )
    return csv_files[0]


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """
    Carga el CSV, normaliza nombres de columnas y codifica la etiqueta.

    El dataset de Kaggle contiene columnas 'text' y 'label'
    (spam=1, ham=0 tras la codificación).
    """
    logger.info("Cargando dataset desde %s ...", csv_path)
    df = pd.read_csv(csv_path)

    # ── Normalización de nombres de columna ──────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()

    # Mapeo flexible: admite variaciones comunes en el nombre de columnas
    col_map = {}
    for col in df.columns:
        if col in ("text", "message", "email", "sms", "content", "body"):
            col_map[col] = "text"
        elif col in ("label", "class", "category", "spam", "target"):
            col_map[col] = "label"
    df.rename(columns=col_map, inplace=True)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"No se encontraron columnas 'text' y 'label'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # ── Limpieza básica ───────────────────────────────────────────────────────
    df.dropna(subset=["text", "label"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str)

    # ── Codificación de etiquetas (spam=1, ham=0) ────────────────────────────
    label_vals = df["label"].unique()
    if set(label_vals) <= {0, 1}:
        df["label"] = df["label"].astype(int)
    else:
        # Valores como "spam"/"ham" o "1"/"0"
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"].astype(str).str.lower())
        # Nos aseguramos de que "spam" → 1 (mayor valor lexicográfico en sklearn)
        if "spam" in le.classes_:
            spam_idx = list(le.classes_).index("spam")
            if spam_idx == 0:            # ham=0 → spam debería ser 1
                df["label"] = 1 - df["label"]

    logger.info(
        "Dataset cargado: %d registros | Spam: %d | Ham: %d",
        len(df),
        df["label"].sum(),
        (df["label"] == 0).sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO DE TEXTO
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Limpieza básica del texto:
      - Minúsculas
      - Elimina URLs, números, puntuación
      - Múltiples espacios → uno solo
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"\d+", " ", text)                      # Números
    text = re.sub(r"[^a-z\s]", " ", text)                 # Puntuación
    text = re.sub(r"\s+", " ", text).strip()              # Espacios múltiples
    return text


def tokenize(text: str) -> list[str]:
    """Tokenización simple por espacios tras preprocesamiento."""
    return text.split()


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING – 4 MÉTODOS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. One-Hot Encoding ─────────────────────────────────────────────────────
def build_onehot_vectorizer(max_features: int = MAX_FEATURES) -> CountVectorizer:
    """
    One-Hot: presencia/ausencia binaria de palabras.
    binary=True hace que cada término sea 0 o 1 (sin contar frecuencia).
    """
    return CountVectorizer(
        preprocessor=preprocess_text,
        tokenizer=tokenize,
        binary=True,
        max_features=max_features,
        token_pattern=None,      # Usamos nuestro tokenizador
    )


# ── 3b. Bag of Words (frecuencia de conteo) ──────────────────────────────────
def build_bow_vectorizer(max_features: int = MAX_FEATURES) -> CountVectorizer:
    """
    Bag of Words: frecuencia absoluta de cada término.
    """
    return CountVectorizer(
        preprocessor=preprocess_text,
        tokenizer=tokenize,
        binary=False,
        max_features=max_features,
        token_pattern=None,
    )


# ── 3c. TF-IDF ────────────────────────────────────────────────────────────────
def build_tfidf_vectorizer(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """
    TF-IDF: pondera la relevancia de cada término en el corpus.
    """
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        tokenizer=tokenize,
        max_features=max_features,
        sublinear_tf=True,       # log(1 + tf) suaviza frecuencias altas
        token_pattern=None,
    )


# ── 3d. Word2Vec (vector promedio del documento) ─────────────────────────────
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer que:
      1. Entrena un modelo Word2Vec sobre el corpus de entrenamiento.
      2. Representa cada documento como la media de sus word-vectors.

    Si una palabra no está en el vocabulario se ignora.
    Si el documento queda vacío, se devuelve un vector de ceros.
    """

    def __init__(
        self,
        vector_size: int = W2V_DIM,
        window: int = W2V_WINDOW,
        min_count: int = W2V_MIN_COUNT,
        epochs: int = W2V_EPOCHS,
        workers: int = 4,
        sg: int = 0,             # 0 = CBOW, 1 = Skip-gram
    ):
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.epochs      = epochs
        self.workers     = workers
        self.sg          = sg
        self.model_      = None

    def _tokenize_corpus(self, X):
        return [tokenize(preprocess_text(doc)) for doc in X]

    def fit(self, X, y=None):
        sentences = self._tokenize_corpus(X)
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            sg=self.sg,
            seed=RANDOM_STATE,
        )
        return self

    def transform(self, X, y=None):
        sentences = self._tokenize_corpus(X)
        vectors = []
        wv = self.model_.wv
        for tokens in sentences:
            valid = [wv[t] for t in tokens if t in wv]
            if valid:
                vectors.append(np.mean(valid, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN OBJETIVO DE OPTUNA (métrica configurable)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(X_train_feat, y_train, metric: str = OPTIMIZE_METRIC):
    """
    Fábrica de funciones objetivo para Optuna.

    Parámetros
    ----------
    X_train_feat : array-like
        Features ya transformadas del set de entrenamiento.
    y_train : array-like
        Etiquetas del set de entrenamiento.
    metric : str
        Métrica a optimizar: "f1_macro" | "accuracy" | "log_loss"

    Returns
    -------
    Función objetivo compatible con optuna.Study.optimize().
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        # ── Elección del clasificador y espacio de búsqueda ─────────────────
        clf_name = trial.suggest_categorical(
            "classifier", ["logistic_regression", "lightgbm"]
        )

        if clf_name == "logistic_regression":
            C       = trial.suggest_float("lr_C", 1e-3, 100.0, log=True)
            solver  = trial.suggest_categorical(
                "lr_solver", ["lbfgs", "liblinear", "saga"]
            )
            max_iter = trial.suggest_int("lr_max_iter", 200, 1000, step=100)
            clf = LogisticRegression(
                C=C, solver=solver, max_iter=max_iter,
                class_weight="balanced", random_state=RANDOM_STATE,
            )

        elif clf_name == "lightgbm":
            clf = lgb.LGBMClassifier(
                n_estimators   = trial.suggest_int("lgbm_n_est", 50, 400, step=50),
                learning_rate  = trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
                num_leaves     = trial.suggest_int("lgbm_leaves", 20, 150),
                max_depth      = trial.suggest_int("lgbm_depth", 3, 12),
                subsample      = trial.suggest_float("lgbm_subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("lgbm_colsample", 0.5, 1.0),
                class_weight   = "balanced",
                random_state   = RANDOM_STATE,
                verbose        = -1,
            )

        # ── CV=10 sobre el training set ──────────────────────────────────────
        if metric == "f1_macro":
            scores = cross_val_score(
                clf, X_train_feat, y_train,
                cv=cv, scoring="f1_macro", n_jobs=-1,
            )
            return scores.mean()   # maximizar

        elif metric == "accuracy":
            scores = cross_val_score(
                clf, X_train_feat, y_train,
                cv=cv, scoring="accuracy", n_jobs=-1,
            )
            return scores.mean()   # maximizar

        elif metric == "log_loss":
            scores = cross_val_score(
                clf, X_train_feat, y_train,
                cv=cv, scoring="neg_log_loss", n_jobs=-1,
            )
            return scores.mean()   # neg_log_loss → maximizar (menos negativo = menor pérdida)

        else:
            raise ValueError(
                f"Métrica '{metric}' no soportada. "
                "Elige entre: 'f1_macro', 'accuracy', 'log_loss'."
            )

    return objective


def build_best_classifier(best_params: dict) -> object:
    """
    Reconstruye el clasificador con los hiperparámetros óptimos encontrados
    por Optuna para entrenarlo en el conjunto de entrenamiento completo.
    """
    clf_name = best_params["classifier"]

    if clf_name == "logistic_regression":
        return LogisticRegression(
            C          = best_params["lr_C"],
            solver     = best_params["lr_solver"],
            max_iter   = best_params["lr_max_iter"],
            class_weight = "balanced",
            random_state = RANDOM_STATE,
        )
    elif clf_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators     = best_params["lgbm_n_est"],
            learning_rate    = best_params["lgbm_lr"],
            num_leaves       = best_params["lgbm_leaves"],
            max_depth        = best_params["lgbm_depth"],
            subsample        = best_params["lgbm_subsample"],
            colsample_bytree = best_params["lgbm_colsample"],
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            verbose          = -1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL POR MÉTODO DE REPRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    method_name: str,
    vectorizer,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    n_trials: int = OPTUNA_TRIALS,
    metric: str = OPTIMIZE_METRIC,
) -> dict:
    """
    Ejecuta el pipeline completo para un método de representación de texto:

      1. Transforma texto → features (solo fit en train).
      2. Optimiza hiperparámetros con Optuna + CV=10 en train.
      3. Entrena el modelo final en todo el train.
      4. Evalúa en test.

    Returns
    -------
    dict con métricas, predicciones y objetos para plotting.
    """
    logger.info("=" * 60)
    logger.info("MÉTODO: %s", method_name)
    logger.info("=" * 60)

    # ── Transformación de texto ──────────────────────────────────────────────
    logger.info("  Vectorizando texto...")
    X_tr_feat = vectorizer.fit_transform(X_train)
    X_te_feat = vectorizer.transform(X_test)

    # Word2Vec devuelve ndarray, los demás scipy sparse → los dejamos como están
    # LightGBM y LogReg aceptan ambos formatos.

    # ── Optimización con Optuna ──────────────────────────────────────────────
    logger.info("  Buscando hiperparámetros con Optuna (%d trials, CV=%d)...",
                n_trials, CV_FOLDS)

    direction = "minimize" if metric == "log_loss" else "maximize"
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    objective = make_objective(X_tr_feat, y_train, metric=metric)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value
    logger.info("  Mejor %s en CV: %.4f", metric, best_score)
    logger.info("  Hiperparámetros: %s", best_params)

    # ── Entrenamiento final en todo el training set ──────────────────────────
    logger.info("  Entrenando modelo final en todo el training set...")
    clf = build_best_classifier(best_params)
    clf.fit(X_tr_feat, y_train)

    # ── Evaluación en test set (¡nunca antes tocado!) ────────────────────────
    y_pred      = clf.predict(X_te_feat)
    y_proba     = clf.predict_proba(X_te_feat)[:, 1]

    f1      = f1_score(y_test, y_pred, average="macro")
    acc     = accuracy_score(y_test, y_pred)
    ll      = log_loss(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    avg_prec    = average_precision_score(y_test, y_proba)

    logger.info(
        "  TEST → F1-macro: %.4f | Accuracy: %.4f | AUC: %.4f | LogLoss: %.4f",
        f1, acc, roc_auc, ll,
    )
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    return {
        "method"    : method_name,
        "best_params": best_params,
        "cv_score"  : best_score,
        "f1"        : f1,
        "accuracy"  : acc,
        "log_loss"  : ll,
        "cm"        : cm,
        "fpr"       : fpr,
        "tpr"       : tpr,
        "roc_auc"   : roc_auc,
        "precision" : prec,
        "recall"    : rec,
        "avg_prec"  : avg_prec,
        "y_pred"    : y_pred,
        "y_proba"   : y_proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

# Paleta consistente para los 4 métodos
METHOD_COLORS = {
    "One-Hot"  : "#4C72B0",
    "BoW"      : "#DD8452",
    "TF-IDF"   : "#55A868",
    "Word2Vec" : "#C44E52",
}


def plot_confusion_matrices(results: list[dict], save_dir: Path = FIGURES_DIR):
    """
    Muestra las 4 matrices de confusión en un grid 2×2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Matrices de Confusión – Comparativa de Métodos", fontsize=14, y=1.02)

    for ax, res in zip(axes.flat, results):
        method = res["method"]
        cm_norm = res["cm"].astype(float) / res["cm"].sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"size": 12},
        )
        # Sobreescribir anotaciones con conteos absolutos + porcentaje
        for i in range(2):
            for j in range(2):
                count = res["cm"][i, j]
                pct   = cm_norm[i, j] * 100
                ax.texts[i * 2 + j].set_text(f"{count}\n({pct:.1f}%)")
        ax.set_title(f"{method}\nF1={res['f1']:.3f}", color=METHOD_COLORS[method], fontweight="bold")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

    plt.tight_layout()
    path = save_dir / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Guardado: %s", path)


def plot_roc_curves(results: list[dict], save_dir: Path = FIGURES_DIR):
    """
    Curvas ROC con AUC para los 4 métodos en un mismo plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in results:
        ax.plot(
            res["fpr"], res["tpr"],
            label=f"{res['method']} (AUC = {res['roc_auc']:.3f})",
            color=METHOD_COLORS[res["method"]], linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title("Curvas ROC – Comparativa de Métodos", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Guardado: %s", path)


def plot_precision_recall_curves(results: list[dict], save_dir: Path = FIGURES_DIR):
    """
    Curvas Precision-Recall con Average Precision (AP) para los 4 métodos.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in results:
        ax.plot(
            res["recall"], res["precision"],
            label=f"{res['method']} (AP = {res['avg_prec']:.3f})",
            color=METHOD_COLORS[res["method"]], linewidth=2,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Curvas Precision-Recall – Comparativa de Métodos", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    path = save_dir / "precision_recall_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Guardado: %s", path)


def plot_comparison_bar(results: list[dict], save_dir: Path = FIGURES_DIR):
    """
    Gráfico de barras comparativo: F1-macro, Accuracy y AUC-ROC
    para los 4 métodos de representación.
    """
    methods  = [r["method"]   for r in results]
    f1s      = [r["f1"]       for r in results]
    accs     = [r["accuracy"] for r in results]
    aucs     = [r["roc_auc"]  for r in results]
    colors   = [METHOD_COLORS[m] for m in methods]

    x       = np.arange(len(methods))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_f1  = ax.bar(x - width,    f1s,  width, label="F1-macro",  color=colors, alpha=0.9)
    bars_acc = ax.bar(x,            accs, width, label="Accuracy",  color=colors, alpha=0.65)
    bars_auc = ax.bar(x + width,    aucs, width, label="AUC-ROC",   color=colors, alpha=0.4)

    # Etiquetas sobre barras
    for bars in (bars_f1, bars_acc, bars_auc):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Método de Representación", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparativa de Rendimiento: F1-macro | Accuracy | AUC-ROC", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Añadir leyenda de colores por método
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=m) for m, c in METHOD_COLORS.items()]
    ax.legend(
        handles=legend_patches + ax.get_legend_handles_labels()[0][len(legend_patches):],
        fontsize=9, loc="lower right",
    )
    plt.tight_layout()

    path = save_dir / "comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Guardado: %s", path)


def plot_all(results: list[dict]):
    """Genera todos los plots de evaluación."""
    logger.info("\n📊 Generando visualizaciones...")
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_precision_recall_curves(results)
    plot_comparison_bar(results)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("🚀 Iniciando pipeline de clasificación Spam vs Ham")
    logger.info("   Métrica de optimización: %s", OPTIMIZE_METRIC)

    # ── 1. Descarga y preparación ────────────────────────────────────────────
    csv_path = download_dataset(DATASET_DIR)
    df       = load_and_prepare(csv_path)

    # Opcional: muestra para pruebas rápidas
    # df = df.sample(5000, random_state=RANDOM_STATE).reset_index(drop=True)

    # ── 2. División estratificada train / test ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = df["label"],
    )
    logger.info(
        "Train: %d | Test: %d  (Spam en train: %.1f%%)",
        len(X_train), len(X_test),
        y_train.mean() * 100,
    )

    # ── 3. Definir métodos de representación ─────────────────────────────────
    methods = [
        ("One-Hot",  build_onehot_vectorizer()),
        ("BoW",      build_bow_vectorizer()),
        ("TF-IDF",   build_tfidf_vectorizer()),
        ("Word2Vec", Word2VecVectorizer()),
    ]

    # ── 4. Ejecutar experimentos ──────────────────────────────────────────────
    all_results = []
    for name, vectorizer in methods:
        result = run_experiment(
            method_name = name,
            vectorizer  = vectorizer,
            X_train     = X_train,
            X_test      = X_test,
            y_train     = y_train,
            y_test      = y_test,
            n_trials    = OPTUNA_TRIALS,
            metric      = OPTIMIZE_METRIC,
        )
        all_results.append(result)

    # ── 5. Resumen final ──────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("RESUMEN FINAL (Test Set – 30%%)")
    logger.info("=" * 60)
    summary = pd.DataFrame([
        {
            "Método"    : r["method"],
            "F1-macro"  : round(r["f1"], 4),
            "Accuracy"  : round(r["accuracy"], 4),
            "AUC-ROC"   : round(r["roc_auc"], 4),
            "Log-Loss"  : round(r["log_loss"], 4),
            "Avg-Prec"  : round(r["avg_prec"], 4),
            "Classifier": r["best_params"]["classifier"],
        }
        for r in all_results
    ]).set_index("Método")
    print("\n", summary.to_string(), "\n")

    # ── 6. Visualizaciones ────────────────────────────────────────────────────
    plot_all(all_results)

    logger.info("✅ Pipeline completado. Figuras guardadas en: %s", FIGURES_DIR)
    return all_results, summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, summary = main()
