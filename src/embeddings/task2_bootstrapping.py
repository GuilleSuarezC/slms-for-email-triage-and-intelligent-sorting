"""
task2_bootstrapping.py
========================
TAREA 2 - Clasificacion de correos evaluada mediante Bootstrapping.

Por que Bootstrapping en este problema
---------------------------------------
El dataset tiene ~20.000 correos repartidos en 35 clases, generadas
sinteticamente con distintos grados de dificultad (columnas como
'is_ambiguous', 'is_multi_intent', 'noise_level', 'is_spam' sugieren que
no todas las clases son igual de faciles ni estan igual de balanceadas).
En ese escenario:

  1. Una unica particion Holdout puede ser "afortunada" o "desafortunada"
     segun que correos (dificiles/ambiguos) caigan en train o en test.
     Bootstrapping repite el proceso de muestreo cientos de veces, dando
     una estimacion mucho mas ESTABLE del rendimiento real del modelo.
  2. Ademas de la media, Bootstrapping proporciona la VARIANZA/INTERVALO
     DE CONFIANZA de cada metrica (accuracy, precision, recall, f1), algo
     que un solo split no puede dar. Esto es clave para saber si una
     diferencia entre dos modelos es real o solo ruido de muestreo.
  3. Es especialmente util con clases minoritarias: al re-muestrear con
     reemplazo N veces sobre todo el dataset, cada clase pequena aparece
     en muchas configuraciones distintas de train/test a lo largo de las
     iteraciones, en vez de depender de una sola asignacion fija.
  4. Aprovecha mejor los datos que una validacion cruzada repetida: cada
     iteracion usa (en promedio) ~63.2% de las muestras para entrenar y
     el resto ("out-of-bag", OOB) para testear, sin necesidad de separar
     de antemano un conjunto de test fijo que reduzca el dato disponible
     para entrenar.

Pipeline:
    1. Cargar datos y generar embeddings (reutilizando utils_embeddings.py,
       con cache para no recalcular si ya se ejecuto Task 1 sobre el
       mismo CSV).
    2. Para cada iteracion de bootstrap:
         a. Muestrear con reemplazo N indices (N = tamano del dataset)
            -> conjunto de entrenamiento.
         b. Las muestras NO seleccionadas ("out-of-bag") forman el
            conjunto de test de esa iteracion.
         c. Entrenar el clasificador k-NN y evaluar sobre el OOB set.
    3. Agregar resultados: media y desviacion estandar de cada metrica
       tras todas las iteraciones.
    4. Visualizar la distribucion de cada metrica (boxplot / histograma).

Ejecucion:
    python task2_bootstrapping.py --csv_path ruta/a/emails.csv \
        --n_iterations 100 --k 5
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_embeddings import (
    load_data,
    get_embedding_function,
    generate_embeddings,
    ChromaKNNClassifier,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "csv_path": "emails_dataset.csv",
    "text_col": "email_corpus",
    "label_col": "class_label",
    "k_neighbors": 5,
    "n_iterations": 100,     # numero de remuestreos bootstrap
    "random_state": 42,
    "output_dir": "./outputs_task2",
}

METRIC_NAMES = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
]


# ---------------------------------------------------------------------------
# BOOTSTRAPPING
# ---------------------------------------------------------------------------
def bootstrap_evaluate(embeddings: np.ndarray, labels: np.ndarray, k: int,
                        n_iterations: int, random_state: int) -> pd.DataFrame:
    """
    Ejecuta 'n_iterations' rondas de bootstrapping y devuelve un
    DataFrame con las metricas obtenidas en cada iteracion (una fila por
    iteracion, una columna por metrica).

    En cada iteracion:
        - se muestrea con reemplazo un conjunto de entrenamiento del
          mismo tamano que el dataset original.
        - las muestras que NO fueron seleccionadas ('out-of-bag') se usan
          como conjunto de test -> evaluacion honesta, sin fuga de datos.
    """
    rng = np.random.RandomState(random_state)
    n = len(labels)
    all_idx = np.arange(n)

    records = []
    skipped = 0

    for it in tqdm(range(n_iterations), desc="Bootstrapping"):
        train_idx = rng.randint(0, n, size=n)          # muestreo con reemplazo
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(train_idx)] = False
        oob_idx = all_idx[oob_mask]                      # muestras no vistas

        if len(oob_idx) < 2:
            # Extremadamente improbable, pero se protege el caso borde
            skipped += 1
            continue

        clf = ChromaKNNClassifier(k=k)
        clf.fit(embeddings[train_idx], labels[train_idx])
        y_pred = clf.predict(embeddings[oob_idx])
        clf.close()

        m = compute_metrics(labels[oob_idx], y_pred)
        m["iteration"] = it
        m["n_train"] = len(np.unique(train_idx))
        m["n_oob_test"] = len(oob_idx)
        records.append(m)

    if skipped:
        print(f"[bootstrap_evaluate] {skipped} iteraciones omitidas (OOB set demasiado pequeno).")

    return pd.DataFrame.from_records(records)


def summarize_bootstrap(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula media, desviacion estandar e intervalo de confianza (percentil
    2.5% - 97.5%) de cada metrica a partir de todas las iteraciones de
    bootstrap.
    """
    rows = []
    for metric in METRIC_NAMES:
        values = results_df[metric].to_numpy()
        rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(),
            "ci_2.5%": np.percentile(values, 2.5),
            "ci_97.5%": np.percentile(values, 97.5),
        })
    return pd.DataFrame(rows)


def plot_bootstrap_distributions(results_df: pd.DataFrame, out_path: str):
    """Boxplot con la distribucion de cada metrica a lo largo de las
    iteraciones de bootstrap (visualiza la variabilidad estimada)."""
    core_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    data = [results_df[m].to_numpy() for m in core_metrics]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=core_metrics, showmeans=True)
    plt.ylabel("Score")
    plt.title(f"Distribucion de metricas - Bootstrapping ({len(results_df)} iteraciones)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_bootstrap_distributions] Figura guardada en '{out_path}'")


def plot_metric_history(results_df: pd.DataFrame, out_path: str):
    """Muestra como varia el accuracy/f1 iteracion a iteracion, util para
    comprobar visualmente que el proceso ha convergido a una zona estable."""
    plt.figure(figsize=(9, 5))
    plt.plot(results_df["iteration"], results_df["accuracy"], label="accuracy", alpha=0.7)
    plt.plot(results_df["iteration"], results_df["f1_macro"], label="f1_macro", alpha=0.7)
    plt.xlabel("Iteracion de bootstrap")
    plt.ylabel("Score")
    plt.title("Evolucion de metricas por iteracion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_metric_history] Figura guardada en '{out_path}'")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(config: dict):
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1) Carga de datos + embeddings (con cache compartida con Task 1) --
    df = load_data(config["csv_path"], config["text_col"], config["label_col"])
    texts = df[config["text_col"]].tolist()
    labels = df[config["label_col"]].astype(str).to_numpy()

    embedding_function = get_embedding_function()
    embeddings = generate_embeddings(texts, embedding_function)

    # 2) Evaluacion por bootstrapping -------------------------------------
    results_df = bootstrap_evaluate(
        embeddings, labels, k=config["k_neighbors"],
        n_iterations=config["n_iterations"], random_state=config["random_state"],
    )
    results_df.to_csv(os.path.join(config["output_dir"], "bootstrap_raw_results.csv"), index=False)

    # 3) Resumen: media y desviacion estandar ------------------------------
    summary_df = summarize_bootstrap(results_df)
    summary_df.to_csv(os.path.join(config["output_dir"], "bootstrap_summary.csv"), index=False)

    print(f"\n=== Resultados Bootstrapping ({len(results_df)} iteraciones validas, k={config['k_neighbors']}) ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 4) Visualizacion -------------------------------------------------------
    plot_bootstrap_distributions(
        results_df, out_path=os.path.join(config["output_dir"], "bootstrap_metric_boxplot.png")
    )
    plot_metric_history(
        results_df, out_path=os.path.join(config["output_dir"], "bootstrap_metric_history.png")
    )

    print(f"\nResultados y figuras guardados en: {config['output_dir']}")


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Tarea 2 - Bootstrapping")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CONFIG["csv_path"])
    parser.add_argument("--k", type=int, default=DEFAULT_CONFIG["k_neighbors"])
    parser.add_argument("--n_iterations", type=int, default=DEFAULT_CONFIG["n_iterations"])
    parser.add_argument("--random_state", type=int, default=DEFAULT_CONFIG["random_state"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["csv_path"] = args.csv_path
    config["k_neighbors"] = args.k
    config["n_iterations"] = args.n_iterations
    config["random_state"] = args.random_state
    return config


if __name__ == "__main__":
    main(parse_args())
