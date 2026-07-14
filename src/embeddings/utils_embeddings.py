"""
utils_embeddings.py
====================
Modulo de utilidades COMPARTIDO por las Tareas 1 (kNN + ChromaDB) y 2
(Bootstrapping). Se reutiliza aqui el patron ya usado en los notebooks del
curso (L1-student.ipynb / L2-student.ipynb):

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_function = SentenceTransformerEmbeddingFunction()

y el propio cliente `chromadb` para crear colecciones, anadir documentos
y consultarlos.

Se centraliza aqui todo lo que ambas tareas tienen en comun para evitar
duplicar codigo:
    - carga y limpieza del dataset (CSV)
    - generacion de embeddings (por lotes, con cache en disco)
    - un clasificador k-NN que usa ChromaDB como motor de busqueda de
      vecinos mas cercanos (satisface el requisito de "clasificar
      usando k-NN sobre los embeddings almacenados en ChromaDB")
    - calculo de metricas (accuracy, precision, recall, f1, matriz de
      confusion) y utilidades de visualizacion

Nota de diseno:
    Aunque el enunciado pide "dos programas independientes", esto se
    refiere a que cada tarea es una pipeline/entregable separado y
    ejecutable de forma autonoma (cada uno tiene su propio `main()` y se
    puede correr con `python taskX_....py`). Compartir un modulo de
    utilidades es una buena practica de ingenieria (DRY) y no rompe esa
    independencia: cada script se ejecuta solo con este archivo de apoyo.
"""

from __future__ import annotations

import os
import uuid
import hashlib
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")  # backend no interactivo -> guarda figuras a fichero
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["email_id", "email_corpus", "class_label"]


def load_data(csv_path: str, text_col: str = "email_corpus",
              label_col: str = "class_label") -> pd.DataFrame:
    """
    Carga el dataset de correos desde un CSV.

    Se valida la presencia de las columnas minimas necesarias y se
    eliminan filas con texto o etiqueta vacios, ya que no aportan
    informacion util al modelo (y romperian la generacion de embeddings).

    Parameters
    ----------
    csv_path : str
        Ruta al fichero CSV con las columnas descritas en el enunciado
        (email_id, subject, email_corpus, class_label, ...).
    text_col : str
        Columna de entrada principal (por defecto 'email_corpus').
    label_col : str
        Columna objetivo (por defecto 'class_label').

    Returns
    -------
    pd.DataFrame
        Dataframe limpio, con indice reiniciado.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontro el dataset en '{csv_path}'. "
            "Actualiza la constante CSV_PATH / el argumento --csv_path "
            "con la ruta real de tu fichero de correos."
        )

    df = pd.read_csv(csv_path, sep=";")

    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    n_before = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    df = df.reset_index(drop=True)

    n_after = len(df)
    if n_after < n_before:
        print(f"[load_data] Se descartaron {n_before - n_after} filas con "
              f"texto/etiqueta vacios ({n_before} -> {n_after}).")

    print(f"[load_data] Dataset cargado: {n_after} correos, "
          f"{df[label_col].nunique()} clases distintas.")
    return df


# ---------------------------------------------------------------------------
# 2. GENERACION DE EMBEDDINGS (via ChromaDB / Sentence-Transformers)
# ---------------------------------------------------------------------------

def get_embedding_function(model_name: str | None = None):
    """
    Devuelve la funcion de embeddings de ChromaDB usada en L1/L2
    (SentenceTransformerEmbeddingFunction). Si no se especifica modelo,
    usa el modelo por defecto de Chroma ('all-MiniLM-L6-v2').
    """
    if model_name:
        return SentenceTransformerEmbeddingFunction(model_name=model_name)
    return SentenceTransformerEmbeddingFunction()


def _cache_path(texts: Sequence[str], cache_dir: str) -> str:
    """Genera un nombre de fichero de cache determinista a partir del
    contenido de los textos, para no recalcular embeddings entre
    ejecuciones/tareas (Task 1 y Task 2 usan el mismo dataset)."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(("||".join(texts)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"embeddings_{h}.npy")


def generate_embeddings(texts: Sequence[str], embedding_function=None,
                         batch_size: int = 128, cache_dir: str = ".embedding_cache",
                         use_cache: bool = True) -> np.ndarray:
    """
    Genera embeddings para una lista de textos usando la funcion de
    embeddings de ChromaDB, procesando por lotes (importante para
    datasets grandes, ~20.000 correos).

    Incluye una cache en disco (.npy) indexada por el hash del contenido,
    de forma que si Task 1 y Task 2 se ejecutan sobre el mismo dataset no
    hace falta recalcular los embeddings dos veces.

    Returns
    -------
    np.ndarray de forma (n_textos, dim_embedding)
    """
    if embedding_function is None:
        embedding_function = get_embedding_function()

    cache_file = _cache_path(texts, cache_dir)
    if use_cache and os.path.exists(cache_file):
        print(f"[generate_embeddings] Cargando embeddings desde cache: {cache_file}")
        return np.load(cache_file)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
        batch = list(texts[i:i + batch_size])
        batch_embeddings = embedding_function(batch)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings, dtype=np.float32)

    if use_cache:
        np.save(cache_file, embeddings)
        print(f"[generate_embeddings] Embeddings guardados en cache: {cache_file}")

    return embeddings


# ---------------------------------------------------------------------------
# 3. ALMACENAMIENTO EN CHROMADB
# ---------------------------------------------------------------------------

def store_embeddings_in_chroma(ids: Sequence[str], documents: Sequence[str],
                                embeddings: np.ndarray, labels: Sequence,
                                collection_name: str = "email_corpus_embeddings",
                                persist_dir: str = "./chroma_db_emails"):
    """
    Persiste los embeddings (y sus metadatos: class_label) en una base de
    datos ChromaDB en disco (PersistentClient), tal y como pide el
    requisito 2 de la Tarea 1.

    Se guarda el 'class_label' como metadato de cada documento para poder
    recuperarlo despues al hacer consultas de vecinos mas cercanos.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # Si la coleccion ya existe de una ejecucion anterior, se recrea para
    # evitar mezclar datos de distintas ejecuciones/datasets.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    batch_size = 500  # limite practico de insercion por lote en Chroma
    for i in tqdm(range(0, len(ids), batch_size), desc="Almacenando en ChromaDB"):
        sl = slice(i, i + batch_size)
        collection.add(
            ids=list(ids[sl]),
            documents=list(documents[sl]),
            embeddings=embeddings[sl].tolist(),
            metadatas=[{"class_label": str(l)} for l in labels[sl]],
        )

    print(f"[store_embeddings_in_chroma] {collection.count()} documentos "
          f"almacenados en la coleccion '{collection_name}' (dir='{persist_dir}').")
    return client, collection


# ---------------------------------------------------------------------------
# 4. CLASIFICADOR k-NN BASADO EN CHROMADB
# ---------------------------------------------------------------------------

class ChromaKNNClassifier:
    """
    Clasificador k-Nearest Neighbour que delega la busqueda de vecinos en
    ChromaDB (indice HNSW) en lugar de reimplementar la busqueda por
    fuerza bruta. La prediccion se obtiene por VOTO MAYORITARIO entre las
    'class_label' de los k documentos mas cercanos (distancia coseno).

    Se crea una coleccion EFIMERA (en memoria) por cada `fit`, ya que en
    evaluacion (holdout / K-fold / bootstrap) necesitamos poder construir
    y descartar rapidamente muchos "indices" distintos sobre distintos
    subconjuntos de entrenamiento.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._client = chromadb.EphemeralClient()
        self._collection_name = f"knn_{uuid.uuid4().hex}"
        self._collection = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: Sequence):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        ids = [str(i) for i in range(len(embeddings))]

        self._collection = self._client.create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            self._collection.add(
                ids=ids[sl],
                embeddings=embeddings[sl].tolist(),
                metadatas=[{"label": str(l)} for l in labels[sl]],
            )
        self._fitted = True
        return self

    def predict(self, query_embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a .fit() antes de .predict().")

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        preds = []

        # Se consulta por lotes para no exceder limites de la API de Chroma
        batch_size = 200
        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i:i + batch_size]
            results = self._collection.query(
                query_embeddings=batch.tolist(),
                n_results=self.k,
                include=["metadatas"],
            )
            for metadatas in results["metadatas"]:
                neighbor_labels = [m["label"] for m in metadatas]
                # voto mayoritario; empatas se resuelven por el vecino mas
                # cercano gracias al orden que devuelve Chroma
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                preds.append(most_common)

        return np.array(preds)

    def close(self):
        """Libera la coleccion efimera (buena practica entre folds/iteraciones)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5. METRICAS DE EVALUACION
# ---------------------------------------------------------------------------

def compute_metrics(y_true: Iterable, y_pred: Iterable) -> dict:
    """
    Calcula el conjunto de metricas requerido. Al ser un problema
    multiclase (35 clases, potencialmente desbalanceadas), se usa
    promedio 'macro' (cada clase pesa igual) como metrica principal y se
    incluye tambien 'weighted' (pondera por soporte) para contexto.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_metrics(metrics: dict, title: str = "Metricas"):
    print(f"\n=== {title} ===")
    for name, value in metrics.items():
        print(f"  {name:20s}: {value:.4f}")


def print_classification_report(y_true, y_pred, title: str = "Classification report"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Matriz de confusion",
                           out_path: str = "confusion_matrix.png", normalize: bool = True):
    """
    Dibuja y guarda la matriz de confusion como PNG (35 clases -> figura
    grande para que las etiquetas sean legibles).
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    n = len(labels)
    fig_size = max(10, n * 0.35)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=n <= 20, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=True, square=True)
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_confusion_matrix] Figura guardada en '{out_path}'")


def plot_metric_bars(metrics_dict_by_method: dict, out_path: str = "metrics_comparison.png"):
    """
    Compara visualmente varias evaluaciones (p.ej. Holdout vs 5-fold CV)
    mediante un grafico de barras agrupado.
    """
    methods = list(metrics_dict_by_method.keys())
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)

    plt.figure(figsize=(9, 5))
    for i, method in enumerate(methods):
        values = [metrics_dict_by_method[method][m] for m in metric_names]
        plt.bar(x + i * width, values, width=width, label=method)

    plt.xticks(x + width * (len(methods) - 1) / 2, metric_names, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Comparacion de metricas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_metric_bars] Figura guardada en '{out_path}'")
