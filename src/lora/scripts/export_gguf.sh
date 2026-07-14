#!/usr/bin/env bash
# ============================================================================
#  Paso 5b: convertir el modelo fusionado a GGUF y cuantizar (para LM Studio).
#
#  IMPORTANTE sobre Qwen2.5-VL:
#  llama.cpp convierte modelos de TEXTO sin problemas. Para los VL (multimodal)
#  el soporte es parcial y requiere generar además un proyector de visión
#  (mmproj). Si solo clasificas texto, la opción robusta es haber entrenado con
#  Qwen/Qwen2.5-7B-Instruct (texto) -> el flujo de abajo funciona tal cual.
# ============================================================================
set -euo pipefail

MERGED_DIR="${1:-outputs/merged_model}"
OUT_DIR="${2:-outputs/gguf}"
QUANT="${3:-Q4_K_M}"   # alternativas comunes: Q5_K_M, Q8_0

mkdir -p "$OUT_DIR"

# 1) Clonar llama.cpp si no existe
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git
fi
pip install -r llama.cpp/requirements.txt

# 2) Convertir HF -> GGUF (fp16)
python llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" \
  --outfile "$OUT_DIR/model-f16.gguf" \
  --outtype f16

# 3) Compilar las herramientas de cuantización
cmake -B llama.cpp/build llama.cpp
cmake --build llama.cpp/build --config Release -j --target llama-quantize

# 4) Cuantizar
llama.cpp/build/bin/llama-quantize \
  "$OUT_DIR/model-f16.gguf" \
  "$OUT_DIR/model-${QUANT}.gguf" \
  "$QUANT"

echo "GGUF listo: $OUT_DIR/model-${QUANT}.gguf"
echo "Impórtalo en LM Studio (carpeta de modelos) y arranca el servidor local."
