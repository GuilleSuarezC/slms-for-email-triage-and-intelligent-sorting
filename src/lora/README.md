# Clasificador de Emails con Qwen + LoRA → GGUF/LM Studio

Pipeline completo y modular para entrenar un clasificador de emails de **35 clases**
mediante fine-tuning LoRA (SFT) sobre Qwen2.5, evaluarlo con cross-validation,
exportarlo a GGUF para LM Studio y probarlo en una demo Streamlit.

- **Entrada:** columna `email_corpus`
- **Salida:** columna `class_label` (una de 35 etiquetas, salida cerrada)
- Cualquier otra columna del CSV se **ignora**.

> Lee **`docs/DECISIONS.md`** antes de entrenar. En particular, el plan pide el
> modelo *VL* (multimodal); para texto puro el modelo de **texto**
> `Qwen/Qwen2.5-7B-Instruct` es más ligero y exporta a GGUF sin fricción.
> Se cambia con dos líneas en `config.yaml`.

## Estructura

```
email-classifier-lora/
├── config.yaml              # toda la configuración del pipeline
├── requirements.txt
├── email_dataset.csv        # <-- coloca aquí tu CSV (separado por ;)
├── src/                     # lógica reutilizable
│   ├── config.py            # config, seeds, logging, IO
│   ├── data.py              # limpieza, validación, splits, K-Fold
│   ├── prompts.py           # prompt del clasificador estricto
│   ├── modeling.py          # carga modelo, 4-bit, LoRA, encoder de chat
│   ├── dataset.py           # tokenización SFT con enmascarado + collator
│   ├── train.py             # entrenamiento LoRA reutilizable
│   ├── infer.py             # inferencia controlada + normalización
│   └── evaluate.py          # accuracy, F1 macro/weighted, confusión
├── scripts/                 # CLI por pasos
│   ├── prepare_data.py
│   ├── run_cv.py
│   ├── train_final.py
│   ├── evaluate_test.py
│   ├── merge_lora.py
│   └── export_gguf.sh
├── app/streamlit_app.py     # demo (HF local | LM Studio)
├── docs/
│   ├── DECISIONS.md
│   └── EXPORT_GGUF.md
└── outputs/                 # artefactos generados (labels, splits, métricas, pesos)
```

## Requisitos

- Python 3.10+
- **GPU NVIDIA** con VRAM suficiente para QLoRA 4-bit de un 7B (≈ 12–16 GB).
  Sin GPU puedes preparar datos y revisar el código, pero no entrenar.
- `pip install -r requirements.txt`

## Uso (orden de ejecución)

```bash
# 0) coloca email_dataset.csv en la raíz y ajusta config.yaml si hace falta
pip install -r requirements.txt

# 1) limpieza + labels.json + split 80/20
python scripts/prepare_data.py

# 2) (opcional) cross-validation estratificada k=5
python scripts/run_cv.py

# 3) entrenar el adaptador LoRA final con el 80%
python scripts/train_final.py

# 4) evaluación única sobre el 20% test
python scripts/evaluate_test.py

# 5) exportar a GGUF para LM Studio
python scripts/merge_lora.py
bash scripts/export_gguf.sh outputs/merged_model outputs/gguf Q4_K_M

# 6) demo
streamlit run app/streamlit_app.py
```

## Artefactos que genera

- `outputs/labels.json` — lista cerrada de etiquetas
- `outputs/train.csv`, `outputs/test.csv` — splits
- `outputs/cv_metrics.json` — métricas de CV (media ± desviación)
- `outputs/final_adapter/` — adaptador LoRA final
- `outputs/test_metrics.json`, `outputs/test_confusion_matrix.{csv,png}`,
  `outputs/test_classification_report.json` — evaluación final
- `outputs/merged_model/`, `outputs/gguf/` — para LM Studio

## Notas de coste

CV con k=5 entrena el 7B cinco veces. Si solo quieres el modelo, pon
`cv.enabled: false` en `config.yaml` y ejecuta directamente los pasos 1, 3, 4.
