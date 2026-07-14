"""
task1_knn_chromadb.py
======================
TAREA 1 - Clasificacion de correos mediante embeddings + ChromaDB + k-NN.

Pipeline:
    1. Cargar el dataset (columna de entrada: 'email_corpus',
       columna objetivo: 'class_label').
    2. Generar embeddings de 'email_corpus' con la funcion de embeddings
       de ChromaDB (SentenceTransformerEmbeddingFunction), reutilizando
       el patron visto en L1-student.ipynb.
    3. Almacenar los embeddings (+ metadatos) en una base de datos
       ChromaDB persistente en disco.
    4. Clasificar mediante k-NN, donde la busqueda de vecinos la resuelve
       ChromaDB (ver ChromaKNNClassifier en utils_embeddings.py) y la
       prediccion final es un voto mayoritario entre los k vecinos.
    5. Evaluar con:
         - Holdout 70/30 (estratificado por clase)
         - Validacion cruzada estratificada de 5 folds (CV=5)
    6. Reportar accuracy, precision, recall, f1 (macro y weighted) y
       matriz de confusion (figura PNG), ademas de un grafico comparativo
       Holdout vs CV.

Ejecucion:
    python task1_knn_chromadb.py --csv_path ruta/a/emails.csv --k 5

Requisitos (requirements.txt):
    pandas, numpy, tqdm, chromadb, sentence-transformers, scikit-learn,
    matplotlib, seaborn
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils_embeddings import (
    load_data,
    get_embedding_function,
    generate_embeddings,
    store_embeddings_in_chroma,
    ChromaKNNClassifier,
    compute_metrics,
    print_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_metric_bars,
)


# ---------------------------------------------------------------------------
# CONFIGURACION (ajustar segun el entorno / dataset real)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "csv_path": "email_dataset.csv",   # <-- ruta al CSV con ~20.000 correos
    "text_col": "email_corpus",
    "label_col": "class_label",
    "k_neighbors": 5,        # k del algoritmo k-NN
    "test_size": 0.30,       # Holdout 70/30
    "cv_folds": 5,           # validacion cruzada
    "random_state": 42,
    "persist_dir": "./chroma_db_emails",
    "collection_name": "email_corpus_embeddings",
    "output_dir": "./outputs_task1",
}


# ---------------------------------------------------------------------------
# EVALUACION: HOLDOUT
# ---------------------------------------------------------------------------
def evaluate_holdout(embeddings: np.ndarray, labels: np.ndarray, k: int,
                      test_size: float, random_state: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Divide el dataset en train/test (estratificado para conservar la
    proporcion de las 35 clases) y evalua el clasificador k-NN basado en
    ChromaDB.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size,
        random_state=random_state, stratify=labels,
    )

    clf = ChromaKNNClassifier(k=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf.close()

    metrics = compute_metrics(y_test, y_pred)
    return metrics, y_test, y_pred


# ---------------------------------------------------------------------------
# EVALUACION: VALIDACION CRUZADA (K=5)
# ---------------------------------------------------------------------------
def evaluate_cross_validation(embeddings: np.ndarray, labels: np.ndarray, k: int,
                               cv_folds: int, random_state: int) -> tuple[dict, list[dict]]:
    """
    Validacion cruzada estratificada de 'cv_folds' particiones. Se
    devuelve la media +/- desviacion estandar de cada metrica, junto con
    las metricas de cada fold individual (para inspeccion/depuracion).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels), start=1):
        clf = ChromaKNNClassifier(k=k)
        clf.fit(embeddings[train_idx], labels[train_idx])
        y_pred = clf.predict(embeddings[test_idx])
        clf.close()

        m = compute_metrics(labels[test_idx], y_pred)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx}/{cv_folds}: accuracy={m['accuracy']:.4f}  "
              f"f1_macro={m['f1_macro']:.4f}")

    # Media y desviacion estandar por metrica
    agg = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        agg[key] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    return agg, fold_metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(config: dict):
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1) Carga de datos --------------------------------------------------
    df = load_data(config["csv_path"], config["text_col"], config["label_col"])
    texts = df[config["text_col"]].tolist()
    labels = df[config["label_col"]].astype(str).to_numpy()
    ids = df["email_id"].astype(str).tolist() if "email_id" in df.columns \
        else [str(i) for i in range(len(df))]

    # 2) Generacion de embeddings (ChromaDB / Sentence-Transformers) ----
    embedding_function = get_embedding_function()
    embeddings = generate_embeddings(texts, embedding_function)

    # 3) Almacenamiento en ChromaDB (persistente) ------------------------
    store_embeddings_in_chroma(
        ids=ids, documents=texts, embeddings=embeddings, labels=labels,
        collection_name=config["collection_name"], persist_dir=config["persist_dir"],
    )

    # 4-5) Clasificacion k-NN + evaluacion --------------------------------
    print("\n>>> Evaluacion HOLDOUT (70/30) <<<")
    holdout_metrics, y_test, y_pred = evaluate_holdout(
        embeddings, labels, k=config["k_neighbors"],
        test_size=config["test_size"], random_state=config["random_state"],
    )
    print_metrics(holdout_metrics, title="Holdout 70/30")
    print_classification_report(y_test, y_pred, title="Holdout - classification report")
    plot_confusion_matrix(
        y_test, y_pred, title="Matriz de confusion - Holdout 70/30",
        out_path=os.path.join(config["output_dir"], "confusion_matrix_holdout.png"),
    )

    print(f"\n>>> Evaluacion CROSS-VALIDATION (CV={config['cv_folds']}) <<<")
    cv_metrics, fold_metrics = evaluate_cross_validation(
        embeddings, labels, k=config["k_neighbors"],
        cv_folds=config["cv_folds"], random_state=config["random_state"],
    )
    print_metrics(cv_metrics, title=f"Media {config['cv_folds']}-fold CV (+ _std = desviacion)")

    # Matriz de confusion agregada de CV: se concatenan las predicciones
    # de todos los folds (cada muestra aparece exactamente una vez como test)
    skf_all_pred, skf_all_true = [], []
    from sklearn.model_selection import StratifiedKFold as _SKF
    skf = _SKF(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"])
    for train_idx, test_idx in skf.split(embeddings, labels):
        clf = ChromaKNNClassifier(k=config["k_neighbors"])
        clf.fit(embeddings[train_idx], labels[train_idx])
        skf_all_pred.extend(clf.predict(embeddings[test_idx]))
        skf_all_true.extend(labels[test_idx])
        clf.close()
    plot_confusion_matrix(
        skf_all_true, skf_all_pred, title=f"Matriz de confusion - {config['cv_folds']}-fold CV",
        out_path=os.path.join(config["output_dir"], "confusion_matrix_cv.png"),
    )

    # Comparativa Holdout vs CV ------------------------------------------------
    plot_metric_bars(
        {"Holdout 70/30": holdout_metrics, f"{config['cv_folds']}-fold CV": cv_metrics},
        out_path=os.path.join(config["output_dir"], "metrics_comparison_holdout_vs_cv.png"),
    )

    print(f"\nResultados y figuras guardados en: {config['output_dir']}")


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Tarea 1 - kNN + ChromaDB")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CONFIG["csv_path"])
    parser.add_argument("--k", type=int, default=DEFAULT_CONFIG["k_neighbors"])
    parser.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--cv_folds", type=int, default=DEFAULT_CONFIG["cv_folds"])
    parser.add_argument("--random_state", type=int, default=DEFAULT_CONFIG["random_state"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["csv_path"] = args.csv_path
    config["k_neighbors"] = args.k
    config["test_size"] = args.test_size
    config["cv_folds"] = args.cv_folds
    config["random_state"] = args.random_state
    return config


if __name__ == "__main__":
    main(parse_args())
