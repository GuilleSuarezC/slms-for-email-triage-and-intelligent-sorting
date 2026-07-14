"""Ingeniería de prompt.

Reformulamos la clasificación como *generación controlada* tipo chat:
  system -> instrucciones estrictas + lista de etiquetas válidas
  user   -> el email
  assistant -> exactamente UNA etiqueta

El objetivo del prompt es minimizar la deriva generativa: nada de explicaciones,
nada de etiquetas inventadas, solo una de las 35 clases.
"""
from __future__ import annotations

SYSTEM_TEMPLATE = (
    "Eres un clasificador de emails estricto. Clasifica el email del usuario en "
    "EXACTAMENTE UNA de las siguientes categorías. Responde ÚNICAMENTE con el "
    "texto de la categoría, sin explicaciones, sin comillas y sin texto adicional. "
    "No inventes categorías nuevas.\n\n"
    "Categorías válidas:\n{labels}"
)


def format_label_block(labels: list[str]) -> str:
    return "\n".join(f"- {lbl}" for lbl in labels)


def build_system(labels: list[str]) -> str:
    return SYSTEM_TEMPLATE.format(labels=format_label_block(labels))


def _content(text: str, is_vl: bool):
    """El procesador VL espera contenido como lista de bloques; el modelo de
    texto espera un string plano."""
    if is_vl:
        return [{"type": "text", "text": text}]
    return text


def build_messages(email: str, labels: list[str], is_vl: bool, label: str | None = None):
    """Construye la lista de mensajes de chat.

    Si `label` es None -> prompt de inferencia (sin respuesta del assistant).
    Si `label` viene dado -> ejemplo supervisado completo (para SFT).
    """
    messages = [
        {"role": "system", "content": _content(build_system(labels), is_vl)},
        {"role": "user", "content": _content(email, is_vl)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": _content(label, is_vl)})
    return messages
