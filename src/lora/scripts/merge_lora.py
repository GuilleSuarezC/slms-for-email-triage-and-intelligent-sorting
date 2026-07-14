#!/usr/bin/env python
"""Paso 5a: fusionar el adaptador LoRA con el modelo base (merge) en fp16.

El modelo fusionado en outputs/merged_model es el que luego se convierte a GGUF.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel

from src.config import get_logger, load_config
from src.train import make_encoder

log = get_logger()


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    # Para fusionar cargamos el base SIN cuantizar (en fp16) y le pegamos el LoRA.
    if cfg.model.is_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    else:
        from transformers import AutoModelForCausalLM as ModelCls

    log.info("Cargando base en fp16 para merge...")
    base = ModelCls.from_pretrained(
        cfg.model.id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, cfg.paths.final_adapter_dir)
    log.info("Fusionando pesos LoRA...")
    model = model.merge_and_unload()

    out = cfg.paths.merged_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    # Guardar tokenizer/processor junto al modelo fusionado
    enc = make_encoder(cfg)
    enc.tokenizer.save_pretrained(out)
    if cfg.model.is_vl:
        enc.processor.save_pretrained(out)

    log.info("Modelo fusionado guardado en %s", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
