"""Carga del modelo base, cuantización 4-bit, LoRA y un encoder de chat unificado.

Soporta dos rutas:
  - Texto  (recomendada): AutoModelForCausalLM + AutoTokenizer.
  - VL      (la pedida):   Qwen2_5_VLForConditionalGeneration + AutoProcessor.

La ruta de TEXTO está validada y es la más robusta. La ruta VL funciona en
modo solo-texto (sin imágenes) pero es más delicada y, sobre todo, complica el
export a GGUF (ver docs/EXPORT_GGUF.md).
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
#  Encoder de chat: oculta las diferencias entre tokenizer (texto) y           #
#  processor (VL). Siempre devuelve listas de token ids.                       #
# --------------------------------------------------------------------------- #
class ChatEncoder:
    def __init__(self, model_id: str, is_vl: bool):
        self.is_vl = is_vl
        if is_vl:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self._tmpl = self.processor
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._tmpl = self.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def to_ids(self, messages, add_generation_prompt: bool) -> list[int]:
        # tokenize=False -> string con la plantilla de chat ya aplicada (special
        # tokens incluidos). Luego tokenizamos sin volver a añadir specials.
        text = self._tmpl.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode_text(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
#  Carga del modelo                                                            #
# --------------------------------------------------------------------------- #
def _bnb_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(cfg, for_training: bool = True):
    """Carga el modelo base (cuantizado si procede)."""
    quant = _bnb_config(cfg.model.load_in_4bit)
    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float16
    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if quant is not None:
        kwargs["quantization_config"] = quant

    if cfg.model.is_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration

        log.warning("Cargando modelo VL (multimodal) en modo solo-texto: %s", cfg.model.id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model.id, **kwargs)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(cfg.model.id, **kwargs)

    if for_training:
        model.config.use_cache = False
    return model


def attach_lora(model, cfg):
    """Prepara el modelo para QLoRA y añade los adaptadores LoRA."""
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model
