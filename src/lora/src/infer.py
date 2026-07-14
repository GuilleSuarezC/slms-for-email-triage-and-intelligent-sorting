"""Inferencia controlada + normalización contra la lista cerrada de etiquetas.

Aunque el modelo se entrena para emitir solo la etiqueta, en inferencia puede
desviarse. Por eso:
  1. Decodificamos pocos tokens (la etiqueta es corta), greedy.
  2. Forzamos la salida a una de las etiquetas válidas con un matcher robusto.
"""
from __future__ import annotations

import difflib
import re

import torch

from .prompts import build_messages


# --------------------------------------------------------------------------- #
#  Normalización contra lista cerrada                                          #
# --------------------------------------------------------------------------- #
def normalize_prediction(raw: str, labels: list[str]) -> str:
    """Mapea texto libre a la etiqueta válida más plausible. Nunca inventa."""
    if raw is None:
        return labels[0]
    cand = raw.strip().strip("\"'`.").strip()

    lower = {l.lower(): l for l in labels}

    # 1) match exacto (case-insensitive)
    if cand.lower() in lower:
        return lower[cand.lower()]

    # 2) primera línea / primera "palabra-etiqueta"
    first_line = cand.splitlines()[0].strip() if cand else ""
    if first_line.lower() in lower:
        return lower[first_line.lower()]

    # 3) alguna etiqueta aparece como subcadena (la más larga gana)
    hits = [l for l in labels if l.lower() in cand.lower()]
    if hits:
        return max(hits, key=len)

    # 4) similitud aproximada (difflib)
    close = difflib.get_close_matches(cand.lower(), list(lower.keys()), n=1, cutoff=0.6)
    if close:
        return lower[close[0]]

    # 5) por tokens compartidos con cada etiqueta
    cand_tokens = set(re.findall(r"\w+", cand.lower()))
    if cand_tokens:
        best, score = labels[0], -1
        for l in labels:
            ov = len(cand_tokens & set(re.findall(r"\w+", l.lower())))
            if ov > score:
                best, score = l, ov
        if score > 0:
            return best

    # 6) fallback determinista
    return labels[0]


# --------------------------------------------------------------------------- #
#  Generación                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_one(model, encoder, email: str, labels: list[str], cfg, return_raw: bool = False):
    is_vl = cfg.model.is_vl
    messages = build_messages(email, labels, is_vl, label=None)
    input_ids = encoder.to_ids(messages, add_generation_prompt=True)
    input_ids = torch.tensor([input_ids], device=model.device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=cfg.infer.max_new_tokens,
        do_sample=not cfg.infer.greedy,
        num_beams=1,
        pad_token_id=encoder.pad_token_id,
        eos_token_id=encoder.eos_token_id,
    )
    gen = out[0][input_ids.shape[1]:]
    raw = encoder.decode(gen)
    pred = normalize_prediction(raw, labels)
    return (pred, raw) if return_raw else pred


@torch.no_grad()
def predict_many(model, encoder, texts, labels, cfg):
    """Inferencia secuencial (sencilla y segura). Para grandes volúmenes se
    podría batchear con padding a la izquierda."""
    return [predict_one(model, encoder, t, labels, cfg) for t in texts]
