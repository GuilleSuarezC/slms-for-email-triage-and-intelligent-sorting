"""Construcción del dataset de entrenamiento (SFT).

Cada ejemplo se convierte en input_ids + labels donde:
  - El prompt (system + user) se enmascara con -100 -> no contribuye a la pérdida.
  - Solo la etiqueta del assistant cuenta para el loss.

Además truncamos el EMAIL (no la etiqueta) para respetar max_seq_len, de modo
que la respuesta supervisada nunca se corte.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .prompts import build_messages


def _truncate_email(email: str, labels, encoder, max_len: int, is_vl: bool) -> str:
    """Recorta el email por tokens para que el ejemplo completo quepa en max_len."""
    # Coste fijo: todo menos el texto del email (system+plantilla+etiqueta).
    skeleton = build_messages("", labels, is_vl, label="X")
    overhead = len(encoder.to_ids(skeleton, add_generation_prompt=False))
    budget = max(16, max_len - overhead)

    email_ids = encoder.encode_text(email)
    if len(email_ids) <= budget:
        return email
    return encoder.decode_text(email_ids[:budget])


def build_supervised_example(email, label, labels, encoder, max_len, is_vl):
    email = _truncate_email(email, labels, encoder, max_len, is_vl)

    prompt_msgs = build_messages(email, labels, is_vl, label=None)
    full_msgs = build_messages(email, labels, is_vl, label=label)

    prompt_ids = encoder.to_ids(prompt_msgs, add_generation_prompt=True)
    full_ids = encoder.to_ids(full_msgs, add_generation_prompt=False)

    # full_ids debería empezar por prompt_ids; enmascaramos esa parte.
    labels_ids = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Salvaguarda por si la plantilla difiere en longitud
    if len(labels_ids) != len(full_ids):
        labels_ids = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels_ids = labels_ids[: len(full_ids)]

    return {"input_ids": full_ids, "labels": labels_ids}


class SFTDataset(Dataset):
    def __init__(self, df, labels, encoder, cfg):
        self.rows = df.to_dict("records")
        self.labels = labels
        self.encoder = encoder
        self.max_len = cfg.model.max_seq_len
        self.is_vl = cfg.model.is_vl

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return build_supervised_example(
            r["text"], r["label"], self.labels, self.encoder, self.max_len, self.is_vl
        )


class Collator:
    """Padding dinámico a la longitud máxima del batch."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
