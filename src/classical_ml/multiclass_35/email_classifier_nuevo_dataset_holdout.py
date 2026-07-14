"""
email_classifier_nuevo_dataset_holdout.py
==========================================
Pipeline completo de clasificación multiclase (35 clases del dominio energético)
usando representaciones de texto clásicas y modelos de ML con parámetros por defecto.

Dataset: Dataset propio de emails (CSV local separado por ';', columnas
email_corpus, subject, class_label, ... — 35 clases del sector energético).
Autor  : Senior Data Scientist — NLP & Classical ML

NOTA IMPORTANTE SOBRE ESTE FICHERO
───────────────────────────────────
La versión original de este script (email_classifier_nuevo_dataset.py) fue
escrita para una fase previa de pruebas con un dataset de Spam/Ham binario
(columna "is_spam"). En la práctica este pipeline se usa con el dataset de
35 clases del sector energético (columna "class_label"), igual que
llm_email_classifier_holdout_v4.py. Este fichero adapta el pipeline a
clasificación MULTICLASE sobre "class_label" y sustituye la validación
cruzada por una evaluación HOLDOUT (partición única train/test).

Uso
───
    # Entrenar todo (comportamiento por defecto)
    python email_classifier_nuevo_dataset_holdout.py

    # Apuntar a un CSV concreto
    python email_classifier_nuevo_dataset_holdout.py --data ruta/al/archivo.csv

    # Entrenar solo un modelo concreto
    python email_classifier_nuevo_dataset_holdout.py --models logreg
    python email_classifier_nuevo_dataset_holdout.py --models svm
    python email_classifier_nuevo_dataset_holdout.py --models rf

    # Entrenar una combinación de modelos
    python email_classifier_nuevo_dataset_holdout.py --models logreg svm

    # Restringir también los vectorizadores (útil para ahorrar RAM)
    python email_classifier_nuevo_dataset_holdout.py --models logreg --vecs tfidf bow
    python email_classifier_nuevo_dataset_holdout.py --models rf --vecs tfidf

    # Combinar subject + cuerpo del email (por defecto solo cuerpo)
    python email_classifier_nuevo_dataset_holdout.py --combine-subject

    # Cambiar la proporción del holdout (por defecto 20% test)
    python email_classifier_nuevo_dataset_holdout.py --test-size 0.25

CAMBIOS respecto a la versión con Validación Cruzada
─────────────────────────────────────────────────────
• Columna de texto: "email_corpus" (se elimina el prefijo "BODY:" si está presente).
• Columna de etiqueta: "class_label" (35 clases del dominio energético, en vez de
  "is_spam" binario). Las filas con etiquetas no reconocidas se descartan.
• Eliminada la Validación Cruzada (StratifiedKFold + cross_validate): cada
  combinación vectorizador+modelo se entrena UNA vez sobre el train del holdout
  y se evalúa UNA vez sobre el test del holdout.
• Métricas multiclase: accuracy, precision/recall/f1 (macro y weighted) y
  ROC AUC One-vs-Rest (macro y weighted), igual que en la versión LLM.
• Matrices de confusión, curvas ROC y F1 por clase adaptadas a 35 clases
  (en vez de las 2 clases Ham/Spam).

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
import matplotlib.ticker as mticker
from tqdm import tqdm
from scipy.sparse import issparse

from sklearn.model_selection   import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model      import LogisticRegression, SGDClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, roc_curve, auc, classification_report,
)

import joblib
from gensim.models import Word2Vec

# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL
# ==============================================================================

RANDOM_STATE       = 42
TEST_SIZE          = 0.30
WORD2VEC_DIM       = 100
WORD2VEC_WINDOW    = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_EPOCHS    = 10
MAX_FEATURES       = 20_000
MAX_ROC_CLASSES    = 10   # nº de clases (por soporte) mostradas en las curvas ROC

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
# 1bis. DEFINICIÓN DE CLASES (idéntico a llm_email_classifier_holdout_v4.py)
# ==============================================================================
CLASS_DEFINITIONS = {
    "factura_error":           "El cliente reporta un error en el importe, fechas o conceptos de su factura eléctrica o de gas",
    "corte_suministro":        "Aviso o queja sobre un corte de suministro eléctrico o de gas, ya sea programado o por impago",
    "alta_consumo_anomalo":    "El cliente detecta un consumo inusualmente alto que no corresponde a su uso habitual",
    "solicitud_cambio_tarifa": "Petición para cambiar a otra tarifa eléctrica (PVPC, tarifa fija, discriminación horaria, etc.)",
    "averia_contador":         "Reporte de contador roto, con lectura incorrecta o que no funciona correctamente",
    "solicitud_nuevo_contrato":"Solicitud para dar de alta un nuevo contrato de luz o gas en un domicilio o local",
    "baja_contrato":           "Solicitud de cancelación o baja del contrato de suministro energético",
    "cambio_titular":          "Petición para cambiar el titular del contrato (herencia, compraventa, divorcio, etc.)",
    "reclamacion_factura":     "Reclamación formal sobre el cobro incorrecto de conceptos en la factura",
    "consulta_autoconsumo":    "Preguntas sobre instalación de placas solares, autoconsumo compartido o vertido a red",
    "instalacion_solar":       "Consulta o solicitud de información sobre instalación de paneles fotovoltaicos",
    "consulta_tarifas_pvpc":   "Preguntas sobre el Precio Voluntario al Pequeño Consumidor y precios de mercado",
    "fraude_energetico":       "Denuncia o sospecha de fraude en el suministro, enganche ilegal o manipulación de contador",
    "emergencia_gas":          "Situación de emergencia relacionada con gas: olor a gas, posible fuga, explosión",
    "fuga_gas":                "Detección o sospecha de fuga de gas en instalación doméstica o industrial",
    "consulta_eficiencia":     "Preguntas sobre cómo reducir el consumo energético y mejorar la eficiencia del hogar",
    "programa_fidelizacion":   "Consulta o queja sobre puntos, descuentos o beneficios del programa de fidelización",
    "solicitud_aplazamiento":  "Petición para aplazar o fraccionar el pago de una factura pendiente",
    "incidencia_apagón":       "Reporte de apagón o microinterrupción en la zona o edificio del cliente",
    "cambio_domiciliacion":    "Solicitud de cambio de cuenta bancaria para el cargo de las facturas",
    "factura_duplicada":       "El cliente ha recibido dos facturas del mismo período o un cobro duplicado",
    "consulta_lectura_contador":"Preguntas sobre cómo leer el contador, enviar lectura o interpretar los datos",
    "bono_social":             "Consulta o solicitud relacionada con el bono social eléctrico para hogares vulnerables",
    "contrato_empresarial":    "Gestión de contratos de suministro para empresas, polígonos industriales o grandes consumidores",
    "vehiculo_electrico":      "Preguntas sobre tarifas especiales, puntos de recarga o instalación de cargador para vehículo eléctrico",
    "reclamacion_calidad":     "Queja sobre la calidad del suministro: fluctuaciones de tensión, micro-cortes frecuentes",
    "denuncia_proveedor":      "Denuncia formal contra la comercializadora por prácticas abusivas o incumplimiento contractual",
    "phishing_energia":        "El cliente ha recibido un correo o llamada fraudulenta suplantando a la empresa energética",
    "newsletter":              "Boletín informativo de la empresa enviado al cliente",
    "oferta_comercial":        "Comunicación comercial con ofertas, promociones o nuevas tarifas",
    "respuesta_automatica":    "Respuesta automática del sistema o de un cliente ausente de la oficina",
    "spam_energia":            "Correo no deseado relacionado con servicios energéticos de terceros",
    "solicitud_certificado":   "Petición de certificado de consumo, contrato o instalación para trámites administrativos",
    "consulta_juridica":       "Consulta sobre aspectos legales del contrato, LOPD, o reclamaciones ante organismos reguladores",
    "otro_irrelevante":        "Correo que no tiene relación con servicios energéticos o está fuera del dominio",
}
VALID_LABELS = list(CLASS_DEFINITIONS.keys())   # Orden fijo para métricas y matrices

# ==============================================================================
# 2. ARGPARSE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Email classifier (35 clases, dominio energético) — entrena y evalúa combinaciones de modelos y vectorizadores mediante holdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python email_classifier_nuevo_dataset_holdout.py
  python email_classifier_nuevo_dataset_holdout.py --data mi_dataset.csv
  python email_classifier_nuevo_dataset_holdout.py --models logreg
  python email_classifier_nuevo_dataset_holdout.py --models svm rf
  python email_classifier_nuevo_dataset_holdout.py --models logreg --vecs tfidf bow
  python email_classifier_nuevo_dataset_holdout.py --models rf --vecs tfidf
  python email_classifier_nuevo_dataset_holdout.py --combine-subject
  python email_classifier_nuevo_dataset_holdout.py --test-size 0.25
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
        "--test-size",
        type=float,
        default=TEST_SIZE,
        metavar="FRACTION",
        help=f"Proporción del holdout (test) estratificado (default: {TEST_SIZE}).",
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
    return logging.getLogger("EmailClassifier35")

logger = setup_logging()

# ==============================================================================
# 4. CARGA DEL DATASET
# ==============================================================================

# Columnas esperadas en el CSV de 35 clases (separado por ';')
COL_BODY    = "email_corpus"   # cuerpo del email
COL_SUBJECT = "subject"        # asunto (opcional, ver --combine-subject)
COL_LABEL   = "class_label"    # etiqueta: una de las 35 clases del dominio energético


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
    Carga el CSV de 35 clases (separado por ';') y devuelve un DataFrame con
    columnas 'text' (cuerpo limpio, opcionalmente con asunto) y 'label'
    (una de las 35 clases válidas en CLASS_DEFINITIONS/VALID_LABELS).

    Columnas esperadas en el CSV:
      - email_corpus : cuerpo del email (puede empezar por "BODY:")
      - subject      : asunto (solo si combine_subject=True)
      - class_label  : una de las 35 clases del dominio energético
    """
    csv_path = find_csv(data_arg)
    logger.info("Cargando dataset desde '%s' ...", csv_path)

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8", dtype=str, low_memory=False)
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

    # ── Construir etiqueta multiclase ──────────────────────────────────
    df["label"] = df[COL_LABEL].str.strip()

    before = len(df)
    df = df.dropna(subset=["text", "label"])

    # Descartar filas cuya etiqueta no sea una de las 35 clases válidas
    invalid_mask = ~df["label"].isin(VALID_LABELS)
    if invalid_mask.any():
        logger.warning(
            "Eliminadas %d filas con class_label no reconocido.",
            int(invalid_mask.sum()),
        )
        df = df[~invalid_mask]

    if len(df) < before:
        logger.warning("Eliminadas %d filas en total (nulos o etiquetas no reconocidas).",
                       before - len(df))

    logger.info("Distribución de clases:\n%s", df["label"].value_counts().to_string())

    return df[["text", "label"]].reset_index(drop=True)

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
    La validación de etiquetas ya se hace en load_dataset,
    por lo que aquí solo se limpia el texto y se descartan nulos residuales.
    """
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    # Descartar textos vacíos tras la limpieza
    df = df[df["text"].str.strip() != ""]
    if len(df) < before:
        logger.warning("Eliminadas %d filas con texto vacío tras la limpieza.", before - len(df))

    logger.info("Dataset listo — %d muestras, %d clases presentes.",
                len(df), df["label"].nunique())
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """División estratificada holdout train/test (multiclase)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=test_size, stratify=df["label"], random_state=RANDOM_STATE,
    )
    logger.info("Train: %d muestras | Test: %d muestras (%.0f%% / %.0f%%)",
                len(X_train), len(X_test), 100 * (1 - test_size), 100 * test_size)
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
# 9. RE-ENTRENAMIENTO SOBRE EL DATASET COMPLETO Y EXPORTACIÓN
# ==============================================================================

def retrain_on_full_dataset(
    model_name: str,
    vec_name: str,
    df_full: pd.DataFrame,
    test_f1_macro: float,
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
            "model_name":              model_name,
            "vec_name":                vec_name,
            "trained_on_n_samples":    int(len(y_all)),
            "max_features":            MAX_FEATURES,
            "word2vec_dim":            WORD2VEC_DIM,
            "random_state":            RANDOM_STATE,
            "valid_labels":            VALID_LABELS,
            "n_classes":               len(VALID_LABELS),
            "experiment_test_f1_macro": round(float(test_f1_macro), 4),
            "exported_at":             datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }

    out_path = MODELS_DIR / f"best_{model_name}.pkl"
    joblib.dump(bundle, out_path)
    logger.info("  Bundle guardado en '%s'  (model + vectorizer + meta).", out_path)


def save_best_models(trained_models, results_df, df_full) -> None:
    """
    Para cada tipo de modelo presente en results_df, identifica la mejor
    combinación (mayor F1 macro en test), re-entrena sobre el 100% del dataset
    y guarda el bundle completo.
    """
    logger.info("")
    logger.info("Exportando bundles (re-entrenados sobre el 100%% del dataset) ...")

    for model_name in results_df["model"].unique():
        subset = results_df[results_df["model"] == model_name]
        best_row = subset.loc[subset["f1_macro"].idxmax()]
        logger.info("Mejor combo para '%s': vectorizador='%s'  F1 macro=%.4f (test)",
                    model_name, best_row["vectorizer"], best_row["f1_macro"])

        retrain_on_full_dataset(
            model_name=model_name,
            vec_name=best_row["vectorizer"],
            df_full=df_full,
            test_f1_macro=best_row["f1_macro"],
        )

# ==============================================================================
# 10. EVALUACIÓN SOBRE TEST (multiclase)
# ==============================================================================

def _aligned_proba(model, X) -> np.ndarray:
    """
    Devuelve la matriz predict_proba del modelo alineada con el orden fijo de
    VALID_LABELS (rellenando con 0.0 las clases que el modelo no haya visto
    en su partición de entrenamiento).
    """
    y_proba = model.predict_proba(X)
    model_classes = list(model.classes_)
    proba_full = np.zeros((y_proba.shape[0], len(VALID_LABELS)), dtype=float)
    for j, lbl in enumerate(VALID_LABELS):
        if lbl in model_classes:
            proba_full[:, j] = y_proba[:, model_classes.index(lbl)]
    return proba_full


def evaluate_model(model, X_test, y_test, vec_name, model_name) -> dict:
    """Calcula métricas multiclase sobre el conjunto de test/holdout."""
    y_pred     = model.predict(X_test)
    proba_full = _aligned_proba(model, X_test)

    metrics = {
        "vectorizer": vec_name,
        "model":      model_name,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
    }

    for avg in ["macro", "weighted"]:
        metrics[f"precision_{avg}"] = round(
            precision_score(y_test, y_pred, average=avg, labels=VALID_LABELS, zero_division=0), 4
        )
        metrics[f"recall_{avg}"] = round(
            recall_score(y_test, y_pred, average=avg, labels=VALID_LABELS, zero_division=0), 4
        )
        metrics[f"f1_{avg}"] = round(
            f1_score(y_test, y_pred, average=avg, labels=VALID_LABELS, zero_division=0), 4
        )

    for avg in ["macro", "weighted"]:
        try:
            y_true_bin = label_binarize(y_test, classes=VALID_LABELS)
            if y_true_bin.shape[1] < 2 or y_true_bin.sum(axis=0).min() == 0:
                raise ValueError("Holdout sin suficiente diversidad de clases para ROC AUC")
            metrics[f"roc_auc_{avg}"] = round(
                roc_auc_score(y_true_bin, proba_full, average=avg, multi_class="ovr"), 4
            )
        except Exception as e:
            logger.warning("ROC AUC (%s) no calculable para %s+%s: %s", avg, vec_name, model_name, e)
            metrics[f"roc_auc_{avg}"] = float("nan")

    try:
        metrics["log_loss"] = round(
            log_loss(y_test, proba_full, labels=VALID_LABELS), 4
        )
    except Exception as e:
        logger.warning("log_loss no calculable para %s+%s: %s", vec_name, model_name, e)
        metrics["log_loss"] = float("nan")

    return metrics

# ==============================================================================
# 11. VISUALIZACIONES
# ==============================================================================

def plot_confusion_matrices(trained_models, X_tests_per_key, y_test):
    """Matrices de confusión (35x35) por cada combinación vectorizador+modelo."""
    n = len(trained_models)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 10 * nrows))
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])
    for ax, ((vec, mdl), model) in zip(axes, trained_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_tests_per_key[(vec, mdl)]),
                              labels=VALID_LABELS)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(np.arange(len(VALID_LABELS)))
        ax.set_yticks(np.arange(len(VALID_LABELS)))
        ax.set_xticklabels(VALID_LABELS, rotation=90, fontsize=5)
        ax.set_yticklabels(VALID_LABELS, fontsize=5)
        ax.set_title(f"{vec.upper()} + {mdl}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion Matrices (holdout)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_roc_curves(trained_models, X_tests_per_key, y_test, max_classes: int = MAX_ROC_CLASSES):
    """
    Curvas ROC One-vs-Rest para cada combinación vectorizador+modelo,
    mostrando únicamente las `max_classes` clases con más soporte en el test.
    Se genera una figura por combinación (35 clases en un único gráfico no es legible).
    """
    y_true_bin = label_binarize(y_test, classes=VALID_LABELS)
    support = y_true_bin.sum(axis=0)
    top_idx = np.argsort(support)[::-1][:max_classes]
    top_labels = [VALID_LABELS[i] for i in top_idx]

    n = len(trained_models)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7 * nrows))
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])
    cmap = plt.cm.get_cmap("tab10", len(top_labels))

    for ax, ((vec, mdl), model) in zip(axes, trained_models.items()):
        proba_full = _aligned_proba(model, X_tests_per_key[(vec, mdl)])
        for i, (cls_idx, cls_name) in enumerate(zip(top_idx, top_labels)):
            y_true_cls = y_true_bin[:, cls_idx]
            if y_true_cls.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_cls, proba_full[:, cls_idx])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=cmap(i), lw=1.3,
                    label=f"{cls_name} (AUC={roc_auc_val:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR", fontsize=10); ax.set_ylabel("TPR", fontsize=10)
        ax.set_title(f"{vec.upper()} + {mdl} (top {len(top_labels)} clases)", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=6)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Curvas ROC OvR — Comparativa (holdout)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_f1_by_class(trained_models, X_tests_per_key, y_test):
    """F1-score por clase para cada combinación vectorizador+modelo."""
    fig, ax = plt.subplots(figsize=(20, 7))
    n_combos = len(trained_models)
    width = 0.8 / max(n_combos, 1)
    x = np.arange(len(VALID_LABELS))
    colors = plt.cm.tab10.colors

    for i, ((vec, mdl), model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_tests_per_key[(vec, mdl)])
        report = classification_report(
            y_test, y_pred, labels=VALID_LABELS, output_dict=True, zero_division=0
        )
        f1_vals = [report.get(lbl, {}).get("f1-score", 0.0) for lbl in VALID_LABELS]
        ax.bar(x + i * width, f1_vals, width,
               label=f"{vec.upper()} + {mdl}", color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (n_combos - 1) / 2)
    ax.set_xticklabels(VALID_LABELS, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score por Clase — Comparativa (holdout)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    path = PLOTS_DIR / "f1_by_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Guardado: %s", path)


def plot_model_comparison(results_df):
    """Comparativa de F1 macro (test/holdout) por combinación vectorizador+modelo."""
    df = results_df.copy()
    df["combo"] = df["vectorizer"].str.upper() + " + " + df["model"]
    df = df.sort_values("f1_macro", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.60)))
    bars = ax.barh(df["combo"], df["f1_macro"],
                   color=plt.cm.RdYlGn(df["f1_macro"]), edgecolor="grey",
                   linewidth=0.5, label="F1 macro (test/holdout)")

    for bar, val in zip(bars, df["f1_macro"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("F1-Score (macro)", fontsize=12)
    ax.set_title("Comparativa F1-Score Macro — Vectorizador + Modelo (holdout)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, min(1.08, df["f1_macro"].max() + 0.10))
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
                   data_arg: str | None = None, combine_subject: bool = False,
                   test_size: float = TEST_SIZE):
    """
    Orquesta el pipeline completo end-to-end para las combinaciones
    de modelos y vectorizadores seleccionadas, usando evaluación HOLDOUT
    (una única partición train/test, sin validación cruzada).

      1. Datos
      2. Vectorización (solo sobre train, solo vectorizadores seleccionados)
      3. Entrenamiento sobre el train del holdout
      4. Evaluación sobre el test del holdout
      5. Exportación incremental de results.csv
      6. Visualizaciones (sobre las combinaciones de esta ejecución)
      7. Re-entrenamiento sobre el 100% del dataset + exportación de bundles
    """
    logger.info("=" * 65)
    logger.info("  EMAIL CLASSIFIER EXPERIMENT (HOLDOUT, 35 clases) -- Start")
    logger.info("  Modelos    : %s", selected_models)
    logger.info("  Vectorizers: %s", selected_vecs)
    logger.info("  Combine subject: %s", combine_subject)
    logger.info("  Test size (holdout): %s", test_size)
    logger.info("=" * 65)

    # ── 1. Datos ───────────────────────────────────────────────────────
    df_raw  = load_dataset(data_arg=data_arg, combine_subject=combine_subject)
    df_full = prepare_data(df_raw)

    X_train_raw, X_test_raw, y_train, y_test = split_data(df_full, test_size=test_size)
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

    # ── 3-4. Entrenamiento sobre train + evaluación sobre test (holdout) ──
    results:         list = []
    trained_models:  dict = {}
    X_tests_per_key: dict = {}

    pbar = tqdm(total=total_combos, desc="Holdout: entrenamiento + evaluación", unit="combo")

    for vec_name in selected_vecs:
        X_tr = X_trains[vec_name]
        X_te = X_tests[vec_name]

        for model_name in selected_models:
            logger.info("  [%s + %s]", vec_name, model_name)

            # a) Entrenamiento sobre el train del holdout
            logger.info("    Entrenando sobre el train del holdout (%d muestras) ...", X_tr.shape[0])
            model = get_models()[model_name]
            model.fit(X_tr, y_train_arr)

            # b) Evaluación sobre el test del holdout
            test_metrics = evaluate_model(model, X_te, y_test_arr, vec_name, model_name)
            logger.info(
                "    HOLDOUT — F1 macro: %.4f | ROC AUC macro: %.4f | Acc: %.4f",
                test_metrics["f1_macro"], test_metrics["roc_auc_macro"], test_metrics["accuracy"],
            )

            results.append(test_metrics)
            trained_models[(vec_name, model_name)]  = model
            X_tests_per_key[(vec_name, model_name)] = X_te
            pbar.update(1)

    pbar.close()

    # ── 5. Exportar results.csv (incremental) ─────────────────────────
    results_df = export_results(results)

    summary_cols = ["vectorizer", "model", "accuracy", "f1_macro", "f1_weighted",
                    "roc_auc_macro", "roc_auc_weighted"]
    logger.info("\n%s", "=" * 65)
    logger.info("RESULTADOS DE ESTA EJECUCIÓN (holdout)")
    logger.info("(test = %.0f%% | train = %.0f%%)", 100 * test_size, 100 * (1 - test_size))
    logger.info("=" * 65)
    # Mostrar solo las filas de esta ejecución, no las acumuladas
    current_df = pd.DataFrame(results)
    logger.info("\n%s", current_df[summary_cols].to_string(index=False))

    # ── 6. Visualizaciones (combinaciones de esta ejecución) ───────────
    logger.info("Generando visualizaciones ...")
    plot_confusion_matrices(trained_models, X_tests_per_key, y_test_arr)
    plot_roc_curves(trained_models, X_tests_per_key, y_test_arr)
    plot_f1_by_class(trained_models, X_tests_per_key, y_test_arr)
    plot_model_comparison(results_df)   # usa todos los resultados acumulados

    # ── 7. Re-entrenamiento sobre el 100% + exportación de bundles ─────
    save_best_models(
        trained_models=trained_models,
        results_df=current_df,   # solo los modelos entrenados en esta ejecución
        df_full=df_full,
    )

    # ── 8. Resumen final ───────────────────────────────────────────────
    best = current_df.loc[current_df["f1_macro"].idxmax()]
    logger.info(
        "\n MEJOR COMBO (esta ejecución, holdout): %s + %s"
        "\n   F1 macro: %.4f | ROC AUC macro: %.4f | Accuracy: %.4f",
        best["vectorizer"].upper(), best["model"],
        best["f1_macro"], best["roc_auc_macro"], best["accuracy"],
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
        test_size=args.test_size,
    )