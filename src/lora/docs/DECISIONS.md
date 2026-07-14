# Decisiones clave del diseño

Explicación breve de por qué el pipeline está montado así.

## 1. Modelo base: VL vs. texto (léelo antes de entrenar)

El plan pide `Qwen/Qwen2.5-VL-7B-Instruct`. Lo he dejado como valor por defecto
en `config.yaml` para respetarlo, pero conviene saber que:

- **Qwen2.5-VL-7B es multimodal** (visión + texto). Para clasificar emails de
  texto puro no aprovechas la parte de visión; solo añades peso y fragilidad.
- **El export a GGUF para LM Studio es el punto crítico.** Los modelos de texto
  se convierten con `convert_hf_to_gguf.py` sin fricción. Los VL requieren además
  generar un proyector de visión (mmproj) y el soporte en llama.cpp/LM Studio es
  parcial y cambiante.

Recomendación: para este caso de uso, `Qwen/Qwen2.5-7B-Instruct` (solo texto) es
más ligero, igual de capaz para clasificar y exporta limpio. Cambiar cuesta dos
líneas en `config.yaml` (`model.id` e `model.is_vl: false`). El código soporta
ambas rutas.

## 2. Clasificación como generación controlada (SFT)

En vez de una cabeza de clasificación, reformulamos el problema como chat
supervisado: el modelo aprende a emitir **solo la etiqueta**. Ventajas: reutiliza
el conocimiento del LLM instruido y permite desplegar el mismo artefacto en
LM Studio. El riesgo (deriva generativa) se controla con:
- prompt estricto con la lista cerrada de 35 etiquetas,
- enmascarado del prompt en el loss (solo se entrena la etiqueta),
- decodificación corta + normalización contra la lista válida en inferencia.

## 3. LoRA / QLoRA

LoRA entrena solo matrices de bajo rango y congela el modelo base: mucho menos
coste de cómputo y memoria, y un adaptador de pocas decenas de MB en lugar de
re-guardar 7B de parámetros. Con cuantización 4-bit (QLoRA) cabe en GPUs
modestas. Parámetros por defecto: `r=16, alpha=32, dropout=0.05`, atacando las
proyecciones de atención y MLP de Qwen.

## 4. Enmascarado del prompt en la pérdida

Solo los tokens de la etiqueta del `assistant` cuentan para el loss (el resto se
pone a `-100`). Así el modelo no malgasta capacidad reescribiendo el prompt y se
centra en producir la etiqueta correcta. El email se trunca por tokens (nunca la
etiqueta) para respetar `max_seq_len`.

## 5. Splits y validación

- **80/20 estratificado**: el 20% de test se aparta y no se toca hasta el final
  (una única evaluación, sin tuning posterior).
- **StratifiedKFold (k=5)** sobre el 80% para estimar el rendimiento con
  intervalos (media ± desviación). Mantener la proporción de las 35 clases es
  esencial con desbalance.
- **Coste**: k=5 implica entrenar el 7B cinco veces. Por eso la CV usa
  `cv_epochs` reducido y es desactivable. Es legítimo usar la CV solo para fijar
  hiperparámetros y luego entrenar el modelo final con todo el 80%.

## 6. Métricas

Con 35 clases y probable desbalance, **F1 macro es la métrica principal** (trata
a todas las clases por igual). Reportamos también accuracy y F1 weighted, más la
matriz de confusión y el classification report por clase. Lectura rápida:
- F1 macro bajo con accuracy alto → las clases minoritarias fallan (desbalance).
- Gran brecha macro vs. weighted → el modelo acierta lo frecuente y falla la cola.

## 7. Inferencia cerrada

`normalize_prediction` garantiza que la salida final siempre sea una de las 35
etiquetas: match exacto → case-insensitive → subcadena → similitud difflib →
solapamiento de tokens → fallback determinista. Nunca inventa una clase nueva.

## 8. Reproducibilidad

`seed` fija random/numpy/torch; el tokenizer se guarda junto al modelo fusionado;
los splits y las etiquetas se serializan a disco. Recomendado versionar el
dataset y los `outputs/*.json`.
