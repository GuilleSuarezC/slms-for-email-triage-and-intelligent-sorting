# Exportación a GGUF y uso en LM Studio

## Flujo resumido

```
LoRA adapter ──merge──► modelo fusionado (HF) ──convert──► GGUF f16 ──quantize──► GGUF Q4/Q5 ──► LM Studio
```

## 1. Fusionar LoRA con el base

```bash
python scripts/merge_lora.py
# -> outputs/merged_model/   (modelo completo en fp16 + tokenizer)
```

## 2. Convertir a GGUF y cuantizar

```bash
bash scripts/export_gguf.sh outputs/merged_model outputs/gguf Q4_K_M
# Cuantizaciones habituales: Q4_K_M (equilibrio), Q5_K_M (más calidad), Q8_0 (casi sin pérdida)
```

El script clona llama.cpp, convierte con `convert_hf_to_gguf.py`, compila
`llama-quantize` y produce `outputs/gguf/model-Q4_K_M.gguf`.

## 3. Importar en LM Studio

1. Copia el `.gguf` a la carpeta de modelos de LM Studio (o usa "Import model").
2. Cárgalo y abre la pestaña **Developer / Local Server**.
3. Arranca el servidor (por defecto en `http://localhost:1234/v1`), compatible
   con la API de OpenAI.
4. En la app Streamlit, elige el modo **LM Studio (API)** y apunta a esa URL.

## ⚠️ Aviso sobre Qwen2.5-VL (multimodal)

`convert_hf_to_gguf.py` está pensado para modelos de texto. Para los VL hay que
generar además un proyector de visión (mmproj) y el soporte en llama.cpp y en
LM Studio es **parcial y cambiante**.

- Si entrenaste con el modelo de **texto** (`Qwen/Qwen2.5-7B-Instruct`), el flujo
  de arriba funciona tal cual. **Es la opción recomendada para desplegar en LM Studio.**
- Si te ceñiste al VL y la conversión falla, revisa los issues recientes de
  llama.cpp sobre `qwen2_5_vl` y la versión de tu LM Studio; puede que necesites
  generar el `mmproj` por separado o que aún no esté soportado.

Para clasificación de texto, no pierdes nada usando la variante de texto, y
ahorras todo este dolor.
