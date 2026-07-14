# Clasificacion de correos: embeddings + ChromaDB (Tarea 1) y Bootstrapping (Tarea 2)

## Estructura de ficheros

```
utils_embeddings.py        # Modulo compartido: carga de datos, embeddings,
                            # almacenamiento en ChromaDB, clasificador k-NN
                            # basado en ChromaDB, metricas y graficas.
task1_knn_chromadb.py      # TAREA 1 (ejecutable de forma independiente)
task2_bootstrapping.py     # TAREA 2 (ejecutable de forma independiente)
requirements.txt
```

## Dataset esperado

Un CSV con, como minimo, las columnas `email_id`, `email_corpus` y
`class_label` (el resto de columnas descritas en el enunciado son
opcionales y no se usan salvo que se decida enriquecer el modelo con
ellas). Coloca el fichero real y apunta `--csv_path` hacia el, por
ejemplo:

```bash
pip install -r requirements.txt
python task1_knn_chromadb.py --csv_path emails_dataset.csv --k 5
python task2_bootstrapping.py --csv_path emails_dataset.csv --k 5 --n_iterations 100
```

## Decisiones de diseno clave

1. **Reutilizacion de los notebooks del curso**: la generacion de
   embeddings usa exactamente el mismo patron que `L1-student.ipynb` /
   `L2-student.ipynb` (`SentenceTransformerEmbeddingFunction` de
   `chromadb.utils.embedding_functions`), y el almacenamiento sigue el
   mismo flujo `client.create_collection(...) -> collection.add(...)`.

2. **k-NN "nativo" de ChromaDB**: en lugar de reimplementar la busqueda
   de vecinos, `ChromaKNNClassifier` (en `utils_embeddings.py`) usa el
   propio indice de ChromaDB (`collection.query`) para encontrar los k
   vecinos mas cercanos y predice por voto mayoritario de sus
   `class_label`. Esto cumple literalmente el requisito "clasifique los
   correos utilizando k-NN sobre los embeddings [almacenados en
   ChromaDB]".

3. **Cache de embeddings**: `generate_embeddings` guarda los embeddings
   calculados en disco (`.embedding_cache/`) indexados por un hash del
   contenido, para no recalcular ~20.000 embeddings dos veces si se
   ejecutan ambas tareas sobre el mismo dataset.

4. **Metricas macro y weighted**: con 35 clases (probablemente
   desbalanceadas) se reportan tanto el promedio *macro* (todas las
   clases pesan igual, sensible a clases minoritarias) como *weighted*
   (pondera por soporte), ademas de accuracy y la matriz de confusion.

5. **Por que Bootstrapping en la Tarea 2**: ver el docstring inicial de
   `task2_bootstrapping.py`. En resumen: da una estimacion de
   media +/- desviacion estandar (y un intervalo de confianza) del
   rendimiento del modelo, mucho mas robusta frente a un unico split
   Holdout, especialmente relevante con 35 clases potencialmente
   desbalanceadas y datos sinteticos de dificultad variable (columnas
   `is_ambiguous`, `is_multi_intent`, `noise_level`, etc.).

6. **Modularidad**: ambos scripts son ejecutables de forma
   independiente (`python taskX_....py`), cada uno con su propio
   `main()` y CLI (`argparse`), pero comparten el modulo de utilidades
   para evitar duplicar codigo (carga de datos, embeddings, metricas).

## Salidas generadas

- `outputs_task1/`: `confusion_matrix_holdout.png`,
  `confusion_matrix_cv.png`, `metrics_comparison_holdout_vs_cv.png`.
- `outputs_task2/`: `bootstrap_raw_results.csv` (metricas por
  iteracion), `bootstrap_summary.csv` (media/std/IC por metrica),
  `bootstrap_metric_boxplot.png`, `bootstrap_metric_history.png`.
- `chroma_db_emails/`: base de datos ChromaDB persistida en disco con
  los embeddings de todos los correos (Tarea 1, requisito 2).
