"""
llm_email_classifier_cv.py
==========================
Pipeline de clasificación de correos electrónicos con LLM local (LM Studio).

ENTRADA PREDICTIVA: únicamente la columna `email_corpus`.
Ninguna otra columna del dataset se incluye en el prompt enviado al LLM.

Estrategias comparadas: Zero-shot y Few-shot
Evaluación: Validación cruzada estratificada (StratifiedKFold)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
import json
import re
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo (compatible con entornos headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

import threading
import concurrent.futures
import lmstudio as lms


# =============================================================================
# 2. CONFIGURACIÓN GLOBAL
# =============================================================================
MODEL_NAME = "qwen/qwen2.5-vl-7b"
N_SPLITS = 5
RANDOM_STATE = 42
SHUFFLE = True
N_FEW_SHOT_PER_CLASS = 1
MAX_FEW_SHOT_EXAMPLES = 7    # Con n_ctx=4096 y MAX_EXAMPLE_CORPUS_CHARS=400: ~600 tokens libres
MAX_ROWS = None          # None = usar todas las filas; int = límite para pruebas
REUSE_CACHE = True
TEMPERATURE = 0.0        # Temperatura baja para respuestas deterministas
MAX_RETRIES = 2
LLM_TIMEOUT_SECONDS      = 90   # Timeout por llamada. Few-shot con prompts largos puede
                                # tardar 60-80s en un modelo de 20B en CPU/GPU mixta.
                                # Subir a 120 si el hardware es lento.
LLM_TIMEOUT_FEW_SHOT     = 120  # Timeout específico para few-shot (prompt más largo).
LLM_PAUSE_AFTER_TIMEOUT  = 30   # Pausa LARGA tras timeout: LM Studio necesita ~20-30s
                                # para abortar la inferencia interna antes de aceptar
                                # una nueva llamada sin estado corrupto.
LLM_RECONNECT_RETRIES    = 3    # Reintentos de reconexión al modelo tras cuelgue
CONTEXT_RESET_EVERY  = 0     # DESACTIVADO: ya no se recrea el objeto modelo.
                             # El contexto fresco se garantiza creando un lms.Chat()
                             # nuevo en cada llamada (ver _call_model_with_timeout).
                             # Recrear lms.llm() repetidamente causaba fuga de memoria.
MAX_CORPUS_CHARS         = 3000  # Truncar email_corpus (correo A CLASIFICAR) a este
                                 # número de caracteres. ~750 tokens con ratio 4 chars/token.
MAX_EXAMPLE_CORPUS_CHARS = 400   # Truncar corpus de los EJEMPLOS few-shot. Más corto porque
                                 # hay hasta MAX_FEW_SHOT_EXAMPLES ejemplos en el mismo prompt.
                                 # 400 chars * 10 ejemplos = ~1000 tokens para todos los ejemplos.
                                 # Subir solo si amplías el contexto del modelo en LM Studio.

# Columna exclusivamente usada para inferencia
PREDICTIVE_INPUT_COLUMN = "email_corpus"

# Columnas conservadas solo para auditoría/trazabilidad (NUNCA en el prompt)
AUDIT_ONLY_COLUMNS = [
    "email_id", "subject", "secondary_label", "is_multi_intent",
    "is_ambiguous", "is_spam", "language", "tone", "sender_profile",
    "email_length", "noise_level", "channel", "temperature", "top_p",
    "top_k", "seed", "model_assigned", "generation_time_s",
    "prompt_used", "timestamp",
]

# Columna objetivo
TARGET_COLUMN = "class_label"


# =============================================================================
# 3. DEFINICIÓN DE CLASES
# =============================================================================
CLASS_DEFINITIONS = {
    "factura_error":           "El cliente reporta un error en el importe, fechas o conceptos de su factura eléctrica o de gas",
    "corte_suministro":        "Aviso o queja sobre un corte de suministro eléctrico o de gas, ya sea programado o por impago",
    "alta_consumo_anomalo":    "El cliente detecta un consumo inusualmente alto que no corresponde a su uso habitual",
    "solicitud_cambio_tarifa": "Petición para cambiar a otra tarifa eléctrica (PVPC, tarifa fija, discriminación horaria, etc.)",
    "averia_contador":         "Reporte de contador roto, con lectura incorrecta o que no funciona correctamente",
    "solicitud_nuevo_contrato":"Solicitud para dar de alta un nuevo contrato de luz o gas en un domicilio o local",
    "baja_contrato":           "Solicitud de cancelación o baja del contrato de suministro energético",
    "cambio_titular":          "Petición para cambiar el titular del contrato (herencia, compraventa, divorcio, etc.)",
    "reclamacion_factura":     "Reclamación formal sobre el cobro incorrecto de conceptos en la factura",
    "consulta_autoconsumo":    "Preguntas sobre instalación de placas solares, autoconsumo compartido o vertido a red",
    "instalacion_solar":       "Consulta o solicitud de información sobre instalación de paneles fotovoltaicos",
    "consulta_tarifas_pvpc":   "Preguntas sobre el Precio Voluntario al Pequeño Consumidor y precios de mercado",
    "fraude_energetico":       "Denuncia o sospecha de fraude en el suministro, enganche ilegal o manipulación de contador",
    "emergencia_gas":          "Situación de emergencia relacionada con gas: olor a gas, posible fuga, explosión",
    "fuga_gas":                "Detección o sospecha de fuga de gas en instalación doméstica o industrial",
    "consulta_eficiencia":     "Preguntas sobre cómo reducir el consumo energético y mejorar la eficiencia del hogar",
    "programa_fidelizacion":   "Consulta o queja sobre puntos, descuentos o beneficios del programa de fidelización",
    "solicitud_aplazamiento":  "Petición para aplazar o fraccionar el pago de una factura pendiente",
    "incidencia_apagón":       "Reporte de apagón o microinterrupción en la zona o edificio del cliente",
    "cambio_domiciliacion":    "Solicitud de cambio de cuenta bancaria para el cargo de las facturas",
    "factura_duplicada":       "El cliente ha recibido dos facturas del mismo período o un cobro duplicado",
    "consulta_lectura_contador":"Preguntas sobre cómo leer el contador, enviar lectura o interpretar los datos",
    "bono_social":             "Consulta o solicitud relacionada con el bono social eléctrico para hogares vulnerables",
    "contrato_empresarial":    "Gestión de contratos de suministro para empresas, polígonos industriales o grandes consumidores",
    "vehiculo_electrico":      "Preguntas sobre tarifas especiales, puntos de recarga o instalación de cargador para vehículo eléctrico",
    "reclamacion_calidad":     "Queja sobre la calidad del suministro: fluctuaciones de tensión, micro-cortes frecuentes",
    "denuncia_proveedor":      "Denuncia formal contra la comercializadora por prácticas abusivas o incumplimiento contractual",
    "phishing_energia":        "El cliente ha recibido un correo o llamada fraudulenta suplantando a la empresa energética",
    "newsletter":              "Boletín informativo de la empresa enviado al cliente",
    "oferta_comercial":        "Comunicación comercial con ofertas, promociones o nuevas tarifas",
    "respuesta_automatica":    "Respuesta automática del sistema o de un cliente ausente de la oficina",
    "spam_energia":            "Correo no deseado relacionado con servicios energéticos de terceros",
    "solicitud_certificado":   "Petición de certificado de consumo, contrato o instalación para trámites administrativos",
    "consulta_juridica":       "Consulta sobre aspectos legales del contrato, LOPD, o reclamaciones ante organismos reguladores",
    "otro_irrelevante":        "Correo que no tiene relación con servicios energéticos o está fuera del dominio",
}

VALID_LABELS = list(CLASS_DEFINITIONS.keys())   # Orden fijo para métricas y matrices

# Aliases conocidos: etiquetas que el LLM tiende a inventar -> etiqueta valida real
# Ampliar esta tabla si aparecen nuevas variantes en los logs.
LABEL_ALIASES: dict = {
    # Variantes de consulta_autoconsumo
    "solicitud_autoconsumo":       "consulta_autoconsumo",
    "autoconsumo":                 "consulta_autoconsumo",
    "consulta_autoconsumo_solar":  "consulta_autoconsumo",
    # Variantes de instalacion_solar
    "solicitud_instalacion_solar": "instalacion_solar",
    "instalacion_fotovoltaica":    "instalacion_solar",
    "placas_solares":              "instalacion_solar",
    # Variantes de factura_error
    "error_factura":               "factura_error",
    "factura_incorrecta":          "factura_error",
    "error_en_factura":            "factura_error",
    # Variantes de reclamacion_factura
    "reclamacion":                 "reclamacion_factura",
    "reclamacion_cobro":           "reclamacion_factura",
    # Variantes de solicitud_cambio_tarifa
    "cambio_tarifa":               "solicitud_cambio_tarifa",
    "cambio_de_tarifa":            "solicitud_cambio_tarifa",
    # Variantes de corte_suministro
    "corte":                       "corte_suministro",
    "corte_luz":                   "corte_suministro",
    "corte_gas":                   "corte_suministro",
    # Variantes de alta_consumo_anomalo
    "consumo_anomalo":             "alta_consumo_anomalo",
    "consumo_elevado":             "alta_consumo_anomalo",
    "consumo_alto":                "alta_consumo_anomalo",
    # Variantes de baja_contrato
    "baja":                        "baja_contrato",
    "cancelacion_contrato":        "baja_contrato",
    "baja_de_contrato":            "baja_contrato",
    # Variantes de solicitud_nuevo_contrato
    "alta_contrato":               "solicitud_nuevo_contrato",
    "nuevo_contrato":              "solicitud_nuevo_contrato",
    "alta_suministro":             "solicitud_nuevo_contrato",
    # Variantes de vehiculo_electrico
    "coche_electrico":             "vehiculo_electrico",
    "recarga_electrica":           "vehiculo_electrico",
    "punto_recarga":               "vehiculo_electrico",
    # Variantes de phishing_energia
    "phishing":                    "phishing_energia",
    "fraude_comunicacion":         "phishing_energia",
    # Variantes de incidencia_apagon
    "apagon":                      "incidencia_apagón",
    "incidencia_apagon":           "incidencia_apagón",
    "microinterrupcion":           "incidencia_apagón",
    # Variantes de emergencia_gas / fuga_gas
    "fuga":                        "fuga_gas",
    "olor_gas":                    "emergencia_gas",
    "emergencia":                  "emergencia_gas",
}


# Descripción legible para los prompts
CLASS_DEFINITIONS_STR = "\n".join(
    f'  - "{k}": {v}' for k, v in CLASS_DEFINITIONS.items()
)

# JSON vacío de class_scores (plantilla para el prompt)
_EMPTY_CLASS_SCORES = json.dumps(
    {k: 0.0 for k in VALID_LABELS}, indent=4, ensure_ascii=False
)


# =============================================================================
# 4. CONFIGURACIÓN DE LOGGING
# =============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    """Configura logging a consola y a fichero."""
    log_path = output_dir / "run.log"
    logger = logging.getLogger("classifier")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler de consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Handler de fichero
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# =============================================================================
# 5. CARGA Y VALIDACIÓN DEL DATASET
# =============================================================================
def load_and_validate_dataset(
    input_path: str,
    max_rows: Optional[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Carga el CSV separado por `;` y valida columnas obligatorias."""
    logger.info(f"Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path, sep=";", encoding="utf-8", low_memory=False)

    required = [PREDICTIVE_INPUT_COLUMN, TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas obligatorias ausentes: {missing}")

    logger.info(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

    if max_rows is not None:
        df = df.head(max_rows).copy()
        logger.info(f"Limitando a {max_rows} filas para pruebas")

    # Filtrar filas con class_label inválido
    before = len(df)
    df = df[df[TARGET_COLUMN].isin(VALID_LABELS)].copy()
    after = len(df)
    if before != after:
        logger.warning(f"Eliminadas {before - after} filas con class_label no reconocido")

    df = df.reset_index(drop=True)

    logger.info(f"Filas válidas para clasificación: {len(df)}")
    class_dist = df[TARGET_COLUMN].value_counts()
    logger.info(f"Distribución de clases:\n{class_dist.to_string()}")

    return df


# =============================================================================
# 6. PREPROCESAMIENTO DE email_corpus
# =============================================================================
def clean_email_corpus(text) -> str:
    """
    Limpieza básica del texto de email_corpus.
    - Normaliza espacios sin eliminar información semántica.
    - Trunca a MAX_CORPUS_CHARS para evitar context overflow en el LLM.
      El truncado se hace por palabras completas para no cortar a mitad de token.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Normalizar espacios en blanco
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    # Truncar a MAX_CORPUS_CHARS si es necesario (corte por palabra completa)
    if len(text) > MAX_CORPUS_CHARS:
        truncated = text[:MAX_CORPUS_CHARS]
        last_space = truncated.rfind(" ")
        if last_space > MAX_CORPUS_CHARS * 0.8:   # corte limpio si hay espacio cercano
            truncated = truncated[:last_space]
        text = truncated + " [TEXTO TRUNCADO]"
    return text


def build_input_text(row: pd.Series) -> str:
    """
    Construye el texto de entrada para el LLM.
    RESTRICCIÓN: usa EXCLUSIVAMENTE la columna email_corpus.
    """
    return clean_email_corpus(row[PREDICTIVE_INPUT_COLUMN])


# =============================================================================
# 7. CONSTRUCCIÓN DE PROMPTS
# =============================================================================
def build_zero_shot_prompt(email_corpus: str) -> str:
    """
    Construye el prompt Zero-shot.
    El LLM recibe únicamente el texto del correo, sin ejemplos etiquetados.
    """
    prompt = f"""Eres un clasificador experto de correos electrónicos del sector energético.

Tu tarea es clasificar el siguiente texto de correo electrónico en exactamente una de las clases válidas.

Reglas obligatorias:
- Usa exclusivamente el texto del correo proporcionado.
- No asumas información que no aparezca en el texto.
- Devuelve únicamente JSON válido.
- No añadas explicaciones, comentarios ni texto adicional.
- No inventes clases.
- La clave `predicted_label` debe ser exactamente una de las clases válidas.
- La clave `confidence` debe ser un número entre 0 y 1.
- La clave `class_scores` debe contener todas las clases válidas.
- Los valores de `class_scores` deben representar una distribución de confianza.
- Si el correo no contiene información suficiente o está fuera del dominio energético, usa `otro_irrelevante`.

Clases válidas:
{CLASS_DEFINITIONS_STR}

Texto del correo:
{email_corpus}

Devuelve exclusivamente este JSON (rellena los valores 0.0 con tus estimaciones reales):
{{
  "predicted_label": "una_clase_valida",
  "confidence": 0.0,
  "class_scores": {_EMPTY_CLASS_SCORES}
}}"""
    return prompt


def build_few_shot_examples_text(
    train_df: pd.DataFrame,
    n_per_class: int,
    max_total: int,
    rng: np.random.RandomState,
) -> str:
    """
    Construye el bloque de texto de ejemplos Few-shot desde el fold de entrenamiento.

    RESTRICCIONES:
    - Solo usa email_corpus y class_label de train_df.
    - No usa ninguna otra columna.
    - No hay data leakage: train_df proviene exclusivamente del fold de entrenamiento.
    - Preferencia por textos no vacíos y de longitud moderada (50–500 chars).
    """
    examples = []

    for label in VALID_LABELS:
        subset = train_df[train_df[TARGET_COLUMN] == label].copy()
        if subset.empty:
            continue

        # Preferir textos de longitud moderada y no vacíos
        subset["_len"] = subset[PREDICTIVE_INPUT_COLUMN].apply(
            lambda x: len(clean_email_corpus(x))
        )
        subset = subset[subset["_len"] > 10]
        if subset.empty:
            continue

        # Priorizar textos entre 50 y 500 caracteres
        preferred = subset[(subset["_len"] >= 50) & (subset["_len"] <= 500)]
        pool = preferred if not preferred.empty else subset

        n_sample = min(n_per_class, len(pool))
        sampled = pool.sample(n=n_sample, random_state=rng.randint(0, 9999))

        for _, row in sampled.iterrows():
            # Truncar el corpus del ejemplo a MAX_EXAMPLE_CORPUS_CHARS.
            # Es más corto que MAX_CORPUS_CHARS porque hay N ejemplos en el prompt.
            corpus_full    = clean_email_corpus(row[PREDICTIVE_INPUT_COLUMN])
            if len(corpus_full) > MAX_EXAMPLE_CORPUS_CHARS:
                cut = corpus_full[:MAX_EXAMPLE_CORPUS_CHARS]
                last_space = cut.rfind(" ")
                if last_space > MAX_EXAMPLE_CORPUS_CHARS * 0.8:
                    cut = cut[:last_space]
                corpus_text = cut + " [...]"
            else:
                corpus_text = corpus_full
            examples.append((label, corpus_text))

        if len(examples) >= max_total:
            break

    # Barajar para evitar sesgo por orden de clases
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    examples = [examples[i] for i in indices[:max_total]]

    # Formatear como texto
    blocks = []
    for i, (label, corpus_text) in enumerate(examples, start=1):
        blocks.append(
            f"Ejemplo {i}:\n"
            f"Texto del correo:\n{corpus_text}\n\n"
            f"Etiqueta correcta:\n{label}"
        )

    return "\n\n---\n\n".join(blocks)


def build_few_shot_prompt(email_corpus: str, few_shot_examples_text: str) -> str:
    """
    Construye el prompt Few-shot.
    Los ejemplos incluyen únicamente email_corpus y class_label.
    """
    prompt = f"""Eres un clasificador experto de correos electrónicos del sector energético.

Tu tarea es clasificar el texto final de correo electrónico en exactamente una de las clases válidas.

A continuación tienes ejemplos etiquetados. Úsalos solo como referencia de criterio de clasificación.

Importante:
- Cada ejemplo contiene únicamente el texto del correo.
- No uses información externa al texto.
- No asumas datos que no estén en el texto.
- El correo final debe clasificarse exclusivamente por su contenido textual.
- Los ejemplos proceden únicamente del fold de entrenamiento (sin data leakage).

Ejemplos:
{few_shot_examples_text}

---

Clases válidas:
{CLASS_DEFINITIONS_STR}

Reglas obligatorias:
- Usa exclusivamente el texto del correo proporcionado.
- Devuelve únicamente JSON válido.
- No añadas explicaciones, comentarios ni texto adicional.
- No inventes clases.
- La clave `predicted_label` debe ser exactamente una de las clases válidas.
- La clave `confidence` debe ser un número entre 0 y 1.
- La clave `class_scores` debe contener todas las clases válidas.
- Los valores de `class_scores` deben representar una distribución de confianza.
- Si el correo no contiene información suficiente o está fuera del dominio energético, usa `otro_irrelevante`.

Texto del correo a clasificar:
{email_corpus}

Devuelve exclusivamente este JSON (rellena los valores 0.0 con tus estimaciones reales):
{{
  "predicted_label": "una_clase_valida",
  "confidence": 0.0,
  "class_scores": {_EMPTY_CLASS_SCORES}
}}"""
    return prompt


# =============================================================================
# 8. PARSING ROBUSTO DE LA RESPUESTA DEL LLM
# =============================================================================
def _normalize_scores(scores: dict) -> dict:
    """Normaliza los scores para que sumen 1.0. Evita división por cero."""
    total = sum(scores.values())
    if total > 0:
        return {k: v / total for k, v in scores.items()}
    # Si todos son 0, distribución uniforme
    n = len(scores)
    return {k: 1.0 / n for k in scores}


def _fuzzy_match_label(candidate: str, logger: logging.Logger) -> Optional[str]:
    """
    Intenta mapear una etiqueta no reconocida a una valida usando varias heuristicas.
    Prioridad:
    0. Tabla de aliases exactos (LABEL_ALIASES) -- mas rapido y mas preciso
    1. Coincidencia exacta normalizada
    2. Substring bidireccional con coincidencia unica
    3. Similitud Jaccard sobre tokens
    """
    candidate_clean = candidate.strip().lower().replace(" ", "_")

    # 0. Tabla de aliases conocidos (maxima prioridad, sin ambiguedad)
    if candidate_clean in LABEL_ALIASES:
        resolved = LABEL_ALIASES[candidate_clean]
        logger.info(f"Alias lookup: '{candidate}' -> '{resolved}'")
        return resolved

    # 1. Exact match normalizado
    if candidate_clean in VALID_LABELS:
        return candidate_clean

    # 2. Substring bidireccional
    matches_sub = [
        lbl for lbl in VALID_LABELS
        if lbl in candidate_clean or candidate_clean in lbl
    ]
    if len(matches_sub) == 1:
        logger.debug(f"Fuzzy (substring): '{candidate}' -> '{matches_sub[0]}'")
        return matches_sub[0]

    # 3. Similitud Jaccard sobre tokens
    candidate_tokens = set(candidate_clean.replace("-", "_").split("_"))
    candidate_tokens = {t for t in candidate_tokens if len(t) > 2}

    best_label = None
    best_score = 0.0
    for lbl in VALID_LABELS:
        lbl_tokens = set(lbl.split("_"))
        if not candidate_tokens or not lbl_tokens:
            continue
        intersection = candidate_tokens & lbl_tokens
        union = candidate_tokens | lbl_tokens
        score = len(intersection) / len(union)
        if score > best_score:
            best_score = score
            best_label = lbl

    SIMILARITY_THRESHOLD = 0.4
    if best_score >= SIMILARITY_THRESHOLD and best_label is not None:
        logger.debug(f"Fuzzy (tokens, score={best_score:.2f}): '{candidate}' -> '{best_label}'")
        return best_label

    return None


def parse_llm_response(raw_response: str, logger: logging.Logger) -> dict:
    """
    Parsea la respuesta del LLM de forma robusta.

    IMPORTANTE — distinción de tipos de error:
    - parse_error=True  + needs_retry=True  → JSON completamente inválido → reintentar llamada al LLM
    - parse_error=True  + needs_retry=False → JSON válido pero etiqueta no reconocida → NO reintentar,
                                              usar fallback localmente (evita colgarse)
    - parse_error=False → todo correcto

    Devuelve un dict con:
        predicted_label, confidence, class_scores, parse_error, needs_retry
    """
    def _fallback(needs_retry: bool = False) -> dict:
        fb = {
            "predicted_label": "otro_irrelevante",
            "confidence": 0.0,
            "class_scores": {k: 0.0 for k in VALID_LABELS},
            "parse_error": True,
            "needs_retry": needs_retry,
        }
        fb["class_scores"]["otro_irrelevante"] = 1.0
        return fb

    if not raw_response or not raw_response.strip():
        logger.warning("Respuesta vacía del LLM → needs_retry=True")
        return _fallback(needs_retry=True)

    # Extraer el bloque JSON aunque haya texto antes/después
    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not json_match:
        logger.warning("No se encontró bloque JSON en la respuesta → needs_retry=True")
        return _fallback(needs_retry=True)

    json_str = json_match.group(0)

    parsed = None
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Reparación 1: quitar comas finales antes de } o ]
        json_str_fixed = re.sub(r",\s*([\}\]])", r"\1", json_str)
        try:
            parsed = json.loads(json_str_fixed)
            logger.debug("JSON reparado (comas finales)")
        except json.JSONDecodeError:
            pass

    if parsed is None:
        # Reparación 2: intentar extraer solo la parte superior del JSON (hasta predicted_label)
        mini_match = re.search(
            r'"predicted_label"\s*:\s*"([^"]+)"', json_str
        )
        if mini_match:
            candidate = mini_match.group(1).strip()
            mapped = _fuzzy_match_label(candidate, logger)
            label = mapped if mapped else "otro_irrelevante"
            logger.warning(
                f"JSON irrecuperable, predicted_label extraído por regex: "
                f"'{candidate}' → '{label}' → needs_retry=False"
            )
            fb = _fallback(needs_retry=False)
            fb["predicted_label"] = label
            fb["class_scores"] = {k: 0.0 for k in VALID_LABELS}
            fb["class_scores"][label] = 1.0
            return fb

        logger.warning("JSON inválido e irrecuperable → needs_retry=True")
        return _fallback(needs_retry=True)

    # ── JSON válido a partir de aquí ──────────────────────────────────────────

    # Validar predicted_label
    predicted_label = str(parsed.get("predicted_label", "")).strip()

    if predicted_label not in VALID_LABELS:
        mapped = _fuzzy_match_label(predicted_label, logger)
        if mapped:
            logger.info(
                f"predicted_label '{predicted_label}' mapeado a '{mapped}' (fuzzy) → needs_retry=False"
            )
            predicted_label = mapped
        else:
            # JSON válido pero etiqueta no recuperable → NO reintentar (evita el cuelgue)
            logger.warning(
                f"predicted_label '{predicted_label}' no reconocido y sin mapeo fuzzy. "
                f"Usando 'otro_irrelevante' → needs_retry=False (JSON era válido)"
            )
            fb = _fallback(needs_retry=False)
            return fb

    # Validar confidence
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Validar y completar class_scores
    raw_scores = parsed.get("class_scores", {})
    class_scores = {}
    for label in VALID_LABELS:
        if label in raw_scores:
            try:
                class_scores[label] = float(raw_scores[label])
            except (TypeError, ValueError):
                class_scores[label] = 0.0
        else:
            class_scores[label] = 0.0

    # Si la clase predicha tiene score 0, asignar el confidence
    if class_scores[predicted_label] == 0.0:
        class_scores[predicted_label] = max(confidence, 0.01)

    # Normalizar scores
    class_scores = _normalize_scores(class_scores)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_scores": class_scores,
        "parse_error": False,
        "needs_retry": False,
    }


# =============================================================================
# 9. CACHÉ DE PREDICCIONES
# =============================================================================
def _cache_key(
    strategy: str,
    fold: int,
    email_corpus: str,
    email_id: Optional[str],
    model_name: str,
) -> str:
    """Genera la clave de caché para una predicción."""
    corpus_hash = hashlib.md5(email_corpus.encode("utf-8")).hexdigest()
    id_part = str(email_id) if email_id else "noid"
    raw = f"{strategy}|fold{fold}|{id_part}|{corpus_hash}|{model_name}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_from_cache(cache_dir: Path, key: str) -> Optional[dict]:
    """Carga predicción desde caché si existe."""
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_to_cache(cache_dir: Path, key: str, data: dict) -> None:
    """Guarda predicción en caché."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # El fallo de caché no debe detener la ejecución


# =============================================================================
# 10. CLASIFICACIÓN CON LM STUDIO
# =============================================================================

def _call_model_with_timeout(model, prompt: str, timeout: int) -> str:
    """
    Llama al modelo con contexto completamente fresco en cada invocación.

    Estrategia de contexto limpio:
    - Se crea un objeto lms.Chat() nuevo en cada llamada.
    - El modelo (lms.llm) se crea UNA sola vez al inicio y se reutiliza.
    - Esto evita dos problemas simultáneos:
        a) Acumulación de historial entre clasificaciones (context overflow).
        b) Fuga de memoria por recrear lms.llm() cientos de veces.

    La llamada se ejecuta en un thread daemon con timeout duro para que
    un bloqueo en LM Studio no congele el proceso principal.
    """
    result_holder = [None]
    exc_holder    = [None]

    def _worker():
        try:
            # Chat fresco = contexto vacío, sin historial previo
            chat = lms.Chat()
            chat.add_user_message(prompt)
            result_holder[0] = model.respond(chat)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(
            f"model.respond() no respondió en {timeout}s. "
            f"LM Studio puede estar saturado."
        )

    if exc_holder[0] is not None:
        raise exc_holder[0]

    return str(result_holder[0]) if result_holder[0] is not None else ""


def _reconnect_model(model_name: str, logger: logging.Logger, retries: int = 3):
    """
    Intenta reconectar al modelo LM Studio tras un timeout persistente.
    Devuelve el nuevo objeto modelo, o el original si falla.
    """
    for attempt in range(1, retries + 1):
        wait = attempt * 3
        logger.warning(
            f"Reconectando a LM Studio (intento {attempt}/{retries}), "
            f"esperando {wait}s..."
        )
        time.sleep(wait)
        try:
            new_model = lms.llm(model_name)
            logger.info("Reconexión a LM Studio exitosa.")
            return new_model
        except Exception as e:
            logger.error(f"Reconexión fallida (intento {attempt}): {e}")
    logger.error("No se pudo reconectar. Continuando con el modelo anterior.")
    return None   # señal de fallo total


# Contenedor mutable para el modelo, accesible desde run_cross_validation
_MODEL_REF = [None]


def _is_context_exceeded(error: Exception) -> bool:
    """Detecta si el error es específicamente de contexto excedido."""
    msg = str(error).lower()
    return "context size has been exceeded" in msg or "context length" in msg


def _shrink_prompt(prompt: str, strategy: str, few_shot_text: str,
                   email_corpus_text: str, logger: logging.Logger) -> str:
    """
    Construye un prompt reducido cuando el original excede el contexto.
    Estrategia de reducción en cascada:
      1. Recortar el corpus a clasificar a la mitad
      2. Eliminar todos los ejemplos few-shot (degradar a zero-shot)
      3. Recortar el corpus a 1/4
    Devuelve el prompt reducido y loguea qué se hizo.
    """
    current_len = len(prompt)

    # Paso 1: recortar corpus a la mitad
    half = max(200, len(email_corpus_text) // 2)
    short_corpus = email_corpus_text[:half].rsplit(" ", 1)[0] + " [TRUNCADO]"
    if strategy == "zero_shot" or not few_shot_text:
        reduced = build_zero_shot_prompt(short_corpus)
    else:
        reduced = build_few_shot_prompt(short_corpus, few_shot_text)
    logger.warning(
        f"Context exceeded: corpus recortado de {len(email_corpus_text)} "
        f"a {len(short_corpus)} chars. Prompt: {current_len} → {len(reduced)} chars."
    )
    if len(reduced) < current_len * 0.85:
        return reduced

    # Paso 2: eliminar ejemplos few-shot (zero-shot puro)
    reduced_zs = build_zero_shot_prompt(short_corpus)
    logger.warning(
        f"Context exceeded: degradando a zero-shot (sin ejemplos). "
        f"Prompt: {len(reduced)} → {len(reduced_zs)} chars."
    )
    if len(reduced_zs) < len(reduced) * 0.9:
        return reduced_zs

    # Paso 3: corpus mínimo (200 chars)
    min_corpus = email_corpus_text[:200].rsplit(" ", 1)[0] + " [TRUNCADO]"
    reduced_min = build_zero_shot_prompt(min_corpus)
    logger.warning(
        f"Context exceeded: corpus reducido al mínimo (200 chars). "
        f"Prompt: {len(reduced_zs)} → {len(reduced_min)} chars."
    )
    return reduced_min


def classify_email_with_llm(
    model,
    prompt: str,
    max_retries: int,
    logger: logging.Logger,
    strategy: str = "zero_shot",
    few_shot_text: str = "",
    email_corpus_text: str = "",
) -> dict:
    """
    Envía el prompt al modelo LM Studio con timeout duro por llamada.

    Tipos de error y respuesta:
    - Context exceeded  → reducir el prompt inmediatamente y reintentar (sin reconectar)
    - TimeoutError      → espera larga + reconectar + reintentar con prompt reducido
    - JSON inválido     → reintentar si needs_retry=True, aceptar fallback si False
    - Etiqueta mala     → aceptar fallback sin reintentar

    El timeout se elige automáticamente según la estrategia:
    - zero_shot → LLM_TIMEOUT_SECONDS (más corto, prompt más ligero)
    - few_shot  → LLM_TIMEOUT_FEW_SHOT (más largo, prompt con ejemplos)
    """
    # Seleccionar timeout según estrategia
    active_timeout = LLM_TIMEOUT_FEW_SHOT if strategy == "few_shot" else LLM_TIMEOUT_SECONDS
    raw_response  = ""
    last_error    = None
    current_model = model
    current_prompt = prompt

    # Niveles progresivos de reduccion del corpus (cada context error baja un nivel)
    shrink_corpus_fns = [
        lambda c: c[:max(400, len(c)//2)].rsplit(" ", 1)[0] + " [TRUNCADO]",
        lambda c: c[:max(200, len(c)//4)].rsplit(" ", 1)[0] + " [TRUNCADO]",
        lambda c: (c[:200].rsplit(" ", 1)[0] + " [TRUNCADO]") if len(c) > 200 else c,
    ]
    shrink_level = [0]

    def make_reduced_prompt():
        lvl = min(shrink_level[0], len(shrink_corpus_fns) - 1)
        short_corpus = shrink_corpus_fns[lvl](email_corpus_text)
        shrink_level[0] += 1
        logger.warning(
            "Context exceeded: nivel %d -> corpus %d -> %d chars, zero-shot forzado.",
            lvl + 1, len(email_corpus_text), len(short_corpus)
        )
        return build_zero_shot_prompt(short_corpus)

    for attempt in range(max_retries + 1):
        try:
            raw_response = _call_model_with_timeout(
                current_model, current_prompt, timeout=active_timeout
            )
            parsed = parse_llm_response(raw_response, logger)

            if not parsed["parse_error"]:
                parsed["raw_response"] = raw_response
                parsed.pop("needs_retry", None)
                if current_model is not model:
                    _MODEL_REF[0] = current_model
                return parsed

            needs_retry = parsed.get("needs_retry", False)
            if not needs_retry:
                logger.warning(
                    "Etiqueta no recuperable (intento %d). Aceptando fallback.",
                    attempt + 1
                )
                parsed["raw_response"] = raw_response
                parsed.pop("needs_retry", None)
                return parsed

            logger.warning(
                "JSON invalido (intento %d/%d). Reintentando...",
                attempt + 1, max_retries + 1
            )
            time.sleep(0.5)

        except TimeoutError as e:
            last_error = e
            logger.error("TIMEOUT en intento %d/%d: %s", attempt + 1, max_retries + 1, e)
            if attempt < max_retries:
                # Espera larga SOLO si vamos a reintentar: LM Studio necesita tiempo
                # para abortar la inferencia interna antes de aceptar nuevas llamadas.
                # Sin esta pausa, la siguiente llamada llega con el servidor en estado
                # corrupto y falla con "Context size exceeded" aunque el prompt sea pequeño.
                logger.warning(
                    "Esperando %ds para que LM Studio libere el hilo de inferencia...",
                    LLM_PAUSE_AFTER_TIMEOUT
                )
                time.sleep(LLM_PAUSE_AFTER_TIMEOUT)
                new_model = _reconnect_model(
                    MODEL_NAME, logger, retries=LLM_RECONNECT_RETRIES
                )
                if new_model is not None:
                    current_model = new_model
                    _MODEL_REF[0] = current_model
                # Reducir prompt tras timeout: el prompt largo puede ser la causa
                if email_corpus_text:
                    current_prompt = make_reduced_prompt()

        except Exception as e:
            last_error = e
            if _is_context_exceeded(e):
                if attempt < max_retries and email_corpus_text:
                    current_prompt = make_reduced_prompt()
                    time.sleep(1.0)
                else:
                    break
            else:
                logger.error("Error en llamada al LLM (intento %d): %s", attempt + 1, e)
                time.sleep(1.0)

    logger.error(
        "Todos los reintentos fallaron. Ultimo error: %s. Devolviendo 'otro_irrelevante'.",
        last_error
    )
    return {
        "predicted_label": "otro_irrelevante",
        "confidence": 0.0,
        "class_scores": _normalize_scores({k: 0.0 for k in VALID_LABELS}),
        "raw_response": raw_response,
        "parse_error": True,
    }


# =============================================================================
# 11. VALIDACIÓN CRUZADA ESTRATIFICADA
# =============================================================================
def run_cross_validation(
    df: pd.DataFrame,
    model,
    strategy: str,
    n_splits: int,
    n_few_shot_per_class: int,
    max_few_shot_examples: int,
    random_state: int,
    reuse_cache: bool,
    cache_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Ejecuta validación cruzada estratificada para la estrategia dada.

    strategy: "zero_shot" | "few_shot"
    Devuelve un DataFrame con todas las predicciones.

    RESTRICCIÓN: el LLM recibe únicamente email_corpus.
    """
    logger.info(f"[{strategy.upper()}] Iniciando validación cruzada con {n_splits} folds")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=SHUFFLE, random_state=random_state)
    rng = np.random.RandomState(random_state)

    all_results = []

    X = df[PREDICTIVE_INPUT_COLUMN].values
    y = df[TARGET_COLUMN].values

    # Inicializar referencia mutable al modelo (permite propagación de reconexiones)
    _MODEL_REF[0] = model
    # Contador de llamadas reales al LLM (excluye cache hits) para el reset de contexto
    llm_call_counter = [0]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        logger.info(
            f"[{strategy.upper()}] Fold {fold_idx}/{n_splits} | "
            f"Train: {len(train_idx)} | Val: {len(val_idx)}"
        )

        train_df = df.iloc[train_idx].copy()
        val_df   = df.iloc[val_idx].copy()

        # Construir ejemplos Few-shot desde el fold de entrenamiento (sin data leakage)
        few_shot_text = ""
        if strategy == "few_shot":
            few_shot_text = build_few_shot_examples_text(
                train_df=train_df,
                n_per_class=n_few_shot_per_class,
                max_total=max_few_shot_examples,
                rng=rng,
            )
            logger.debug(
                f"[{strategy.upper()}] Fold {fold_idx}: "
                f"{len(few_shot_text.split('Ejemplo'))-1} ejemplos Few-shot preparados"
            )

        # ── Checkpoint: fichero CSV parcial por fold ──────────────────────────
        # Se escribe en tiempo real: si el proceso cae, el trabajo no se pierde.
        checkpoint_path = output_dir / f"_checkpoint_{strategy}_fold{fold_idx}.csv"
        checkpoint_rows = []   # buffer del fold actual

        # Clasificar cada correo del fold de validación
        for i, (idx, row) in enumerate(val_df.iterrows()):
            email_corpus_text = build_input_text(row)
            email_id_val = row.get("email_id", None) if "email_id" in row.index else None
            true_label = row[TARGET_COLUMN]

            # Caso especial: email_corpus vacío
            is_empty_corpus = len(email_corpus_text) == 0

            # Comprobar caché
            cache_key = _cache_key(
                strategy=strategy,
                fold=fold_idx,
                email_corpus=email_corpus_text,
                email_id=email_id_val,
                model_name=MODEL_NAME,
            )

            prediction = None
            if reuse_cache:
                prediction = load_from_cache(cache_dir, cache_key)
                if prediction is not None:
                    logger.debug(f"Cache hit: fold {fold_idx}, idx {idx}")

            if prediction is None:
                if is_empty_corpus:
                    logger.warning(
                        f"email_corpus vacío en idx {idx} (fold {fold_idx}). "
                        f"Clasificando como 'otro_irrelevante'"
                    )
                    prediction = {
                        "predicted_label": "otro_irrelevante",
                        "confidence": 0.0,
                        "class_scores": _normalize_scores(
                            {k: 0.0 for k in VALID_LABELS}
                        ),
                        "raw_response": "",
                        "parse_error": False,
                    }
                else:
                    # Construir prompt (SOLO email_corpus)
                    if strategy == "zero_shot":
                        prompt = build_zero_shot_prompt(email_corpus_text)
                    else:
                        prompt = build_few_shot_prompt(email_corpus_text, few_shot_text)

                    # El contexto se limpia en cada llamada mediante lms.Chat() fresco
                    # dentro de _call_model_with_timeout. No es necesario recrear lms.llm().
                    llm_call_counter[0] += 1
                    active_model = _MODEL_REF[0] if _MODEL_REF[0] is not None else model
                    prediction = classify_email_with_llm(
                        model=active_model,
                        prompt=prompt,
                        max_retries=MAX_RETRIES,
                        logger=logger,
                        strategy=strategy,
                        few_shot_text=few_shot_text,
                        email_corpus_text=email_corpus_text,
                    )

                # Guardar en caché inmediatamente
                if reuse_cache:
                    save_to_cache(cache_dir, cache_key, prediction)

            predicted_label = prediction["predicted_label"]
            confidence      = prediction["confidence"]
            class_scores    = prediction["class_scores"]
            raw_response    = prediction.get("raw_response", "")
            parse_error     = prediction.get("parse_error", False)

            # Registro de resultado
            record = {
                "fold":            fold_idx,
                "strategy":        strategy,
                "true_label":      true_label,
                "predicted_label": predicted_label,
                "confidence":      confidence,
                "is_correct":      int(predicted_label == true_label),
                "parse_error":     int(parse_error),
                "raw_response":    raw_response,
                PREDICTIVE_INPUT_COLUMN: email_corpus_text,
            }

            # Columnas de auditoría (NO usadas para inferencia)
            if "email_id" in row.index:
                record["email_id"] = row["email_id"]
            for col in ["subject", "language", "tone", "sender_profile",
                        "is_spam", "is_multi_intent", "is_ambiguous"]:
                if col in row.index:
                    record[col] = row[col]

            # class_scores individuales para ROC AUC
            for label in VALID_LABELS:
                record[f"score_{label}"] = class_scores.get(label, 0.0)

            all_results.append(record)
            checkpoint_rows.append(record)

            # ── Flush de checkpoint cada 50 predicciones ─────────────────────
            # Garantiza que si el proceso cae no se pierden más de 50 muestras.
            if len(checkpoint_rows) % 50 == 0:
                try:
                    pd.DataFrame(checkpoint_rows).to_csv(
                        checkpoint_path, index=False, encoding="utf-8"
                    )
                    logger.debug(
                        f"Checkpoint guardado: {checkpoint_path.name} "
                        f"({len(checkpoint_rows)} filas)"
                    )
                except Exception as e:
                    logger.warning(f"Error al guardar checkpoint: {e}")

            if (i + 1) % 10 == 0:
                elapsed_per_sample = ""
                logger.info(
                    f"[{strategy.upper()}] Fold {fold_idx} | "
                    f"Progreso: {i + 1}/{len(val_df)}"
                )

        # Flush final del fold completo
        try:
            pd.DataFrame(checkpoint_rows).to_csv(
                checkpoint_path, index=False, encoding="utf-8"
            )
            logger.info(
                f"[{strategy.upper()}] Fold {fold_idx} completado. "
                f"Checkpoint final: {checkpoint_path.name}"
            )
        except Exception as e:
            logger.warning(f"Error al guardar checkpoint final del fold: {e}")

        logger.info(
            f"[{strategy.upper()}] Fold {fold_idx} completado. "
            f"Accuracy parcial: "
            f"{sum(r['is_correct'] for r in all_results if r['fold'] == fold_idx) / len(val_idx):.4f}"
        )

    results_df = pd.DataFrame(all_results)
    logger.info(
        f"[{strategy.upper()}] Validación cruzada completada. "
        f"Total predicciones: {len(results_df)}"
    )
    return results_df


# =============================================================================
# 12. CÁLCULO DE MÉTRICAS
# =============================================================================
def compute_metrics_for_fold(
    true_labels: list,
    pred_labels: list,
    score_matrix: np.ndarray,
    fold_id: int,
    strategy: str,
    logger: logging.Logger,
) -> dict:
    """Calcula métricas para un fold dado."""
    metrics = {
        "strategy": strategy,
        "fold": fold_id,
        "n_samples": len(true_labels),
    }

    # Etiquetas presentes en este fold
    present_labels = sorted(set(true_labels + pred_labels))

    metrics["accuracy"] = accuracy_score(true_labels, pred_labels)

    for avg in ["macro", "weighted"]:
        metrics[f"precision_{avg}"] = precision_score(
            true_labels, pred_labels, average=avg,
            labels=VALID_LABELS, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            true_labels, pred_labels, average=avg,
            labels=VALID_LABELS, zero_division=0
        )
        metrics[f"f1_{avg}"] = f1_score(
            true_labels, pred_labels, average=avg,
            labels=VALID_LABELS, zero_division=0
        )

    # ROC AUC multiclass One-vs-Rest
    for avg in ["macro", "weighted"]:
        try:
            y_true_bin = label_binarize(true_labels, classes=VALID_LABELS)
            # Verificar que hay al menos 2 clases en este fold
            if y_true_bin.shape[1] < 2 or y_true_bin.sum(axis=0).min() == 0:
                raise ValueError("Fold sin suficiente diversidad de clases para ROC AUC")

            metrics[f"roc_auc_{avg}"] = roc_auc_score(
                y_true_bin,
                score_matrix,
                average=avg,
                multi_class="ovr",
            )
        except Exception as e:
            logger.warning(
                f"[{strategy}] Fold {fold_id}: ROC AUC ({avg}) no calculable: {e}"
            )
            metrics[f"roc_auc_{avg}"] = float("nan")

    return metrics


def compute_all_metrics(
    results_df: pd.DataFrame,
    strategy: str,
    logger: logging.Logger,
) -> tuple:
    """
    Calcula métricas por fold y métricas globales.
    Devuelve: (metrics_by_fold_list, global_metrics_dict)
    """
    score_cols = [f"score_{lbl}" for lbl in VALID_LABELS]
    metrics_by_fold = []

    for fold_id in sorted(results_df["fold"].unique()):
        fold_df = results_df[results_df["fold"] == fold_id]

        true_labels = fold_df["true_label"].tolist()
        pred_labels = fold_df["predicted_label"].tolist()
        score_matrix = fold_df[score_cols].values.astype(float)

        fold_metrics = compute_metrics_for_fold(
            true_labels=true_labels,
            pred_labels=pred_labels,
            score_matrix=score_matrix,
            fold_id=fold_id,
            strategy=strategy,
            logger=logger,
        )
        metrics_by_fold.append(fold_metrics)

    # Métricas globales (concatenación de todos los folds)
    all_true   = results_df["true_label"].tolist()
    all_pred   = results_df["predicted_label"].tolist()
    all_scores = results_df[score_cols].values.astype(float)

    global_metrics = compute_metrics_for_fold(
        true_labels=all_true,
        pred_labels=all_pred,
        score_matrix=all_scores,
        fold_id=0,  # 0 = global
        strategy=strategy,
        logger=logger,
    )
    global_metrics["fold"] = "global"

    return metrics_by_fold, global_metrics


# =============================================================================
# 13. GENERACIÓN DE GRÁFICOS
# =============================================================================
def _safe_save(fig, path: Path, logger: logging.Logger):
    """Guarda figura y cierra."""
    try:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info(f"Gráfico guardado: {path}")
    except Exception as e:
        logger.error(f"Error al guardar gráfico {path}: {e}")
    plt.close(fig)


def plot_global_comparison(
    metrics_zero: dict,
    metrics_few: dict,
    plots_dir: Path,
    logger: logging.Logger,
):
    """Gráfico 1: Comparación global Zero-shot vs Few-shot."""
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "ROC AUC\n(macro)"]

    zero_vals = [metrics_zero.get(k, 0) for k in keys]
    few_vals  = [metrics_few.get(k, 0)  for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, zero_vals, width, label="Zero-shot", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, few_vals,  width, label="Few-shot",  color="#DD8452")

    ax.set_ylabel("Valor")
    ax.set_title("Comparación Global: Zero-shot vs Few-shot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    _safe_save(fig, plots_dir / "01_global_comparison.png", logger)


def plot_boxplot_by_fold(
    all_metrics_df: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
):
    """Gráfico 2: Boxplot por fold para f1_macro, accuracy, roc_auc_macro."""
    metrics_to_plot = ["f1_macro", "accuracy", "roc_auc_macro"]
    titles = ["F1 Macro por fold", "Accuracy por fold", "ROC AUC Macro por fold"]

    fold_df = all_metrics_df[all_metrics_df["fold"] != "global"].copy()
    for col in metrics_to_plot:
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        data_to_plot = []
        group_labels = []
        for strategy in ["zero_shot", "few_shot"]:
            vals = fold_df[fold_df["strategy"] == strategy][metric].dropna().tolist()
            data_to_plot.append(vals)
            group_labels.append(strategy.replace("_", "-"))

        bp = ax.boxplot(
            data_to_plot,
            labels=group_labels,
            patch_artist=True,
            notch=False,
        )
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Métricas por Fold (Zero-shot vs Few-shot)", fontsize=12, y=1.02)
    _safe_save(fig, plots_dir / "02_boxplot_by_fold.png", logger)


def plot_confusion_matrix(
    results_df: pd.DataFrame,
    strategy: str,
    plots_dir: Path,
    logger: logging.Logger,
):
    """Gráfico 3/4: Matriz de confusión agregada."""
    true_labels = results_df["true_label"].tolist()
    pred_labels = results_df["predicted_label"].tolist()

    cm = confusion_matrix(true_labels, pred_labels, labels=VALID_LABELS)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(VALID_LABELS)))
    ax.set_yticks(np.arange(len(VALID_LABELS)))
    ax.set_xticklabels(VALID_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(VALID_LABELS, fontsize=7)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión Agregada — {strategy.replace('_', '-')}")

    # Anotaciones en celdas significativas
    thresh = cm.max() / 2.0
    for i in range(len(VALID_LABELS)):
        for j in range(len(VALID_LABELS)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > thresh else "black")

    _safe_save(fig, plots_dir / f"0{'3' if strategy == 'zero_shot' else '4'}_confusion_matrix_{strategy}.png", logger)


def plot_f1_by_class(
    results_zero: pd.DataFrame,
    results_few: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
):
    """Gráfico 5: F1-score por clase para ambas estrategias."""
    def get_f1_per_class(results_df):
        true_labels = results_df["true_label"].tolist()
        pred_labels = results_df["predicted_label"].tolist()
        report = classification_report(
            true_labels, pred_labels, labels=VALID_LABELS,
            output_dict=True, zero_division=0
        )
        return [report.get(lbl, {}).get("f1-score", 0.0) for lbl in VALID_LABELS]

    f1_zero = get_f1_per_class(results_zero)
    f1_few  = get_f1_per_class(results_few)

    x = np.arange(len(VALID_LABELS))
    width = 0.4

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - width / 2, f1_zero, width, label="Zero-shot", color="#4C72B0", alpha=0.8)
    ax.bar(x + width / 2, f1_few,  width, label="Few-shot",  color="#DD8452", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(VALID_LABELS, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score por Clase — Zero-shot vs Few-shot")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    _safe_save(fig, plots_dir / "05_f1_by_class.png", logger)


def plot_roc_curves(
    results_df: pd.DataFrame,
    strategy: str,
    plots_dir: Path,
    logger: logging.Logger,
    max_classes: int = 10,
):
    """
    Gráfico 6: Curvas ROC One-vs-Rest (primeras max_classes clases por soporte).
    """
    from sklearn.metrics import roc_curve, auc

    score_cols  = [f"score_{lbl}" for lbl in VALID_LABELS]
    true_labels = results_df["true_label"].tolist()
    score_mat   = results_df[score_cols].values.astype(float)
    y_true_bin  = label_binarize(true_labels, classes=VALID_LABELS)

    # Seleccionar las clases con más soporte
    support = y_true_bin.sum(axis=0)
    top_idx = np.argsort(support)[::-1][:max_classes]
    top_labels = [VALID_LABELS[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", len(top_labels))

    for i, (cls_idx, cls_name) in enumerate(zip(top_idx, top_labels)):
        y_true_cls = y_true_bin[:, cls_idx]
        y_score_cls = score_mat[:, cls_idx]

        if y_true_cls.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true_cls, y_score_cls)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=cmap(i), lw=1.5,
                    label=f"{cls_name} (AUC={roc_auc:.2f})")
        except Exception as e:
            logger.warning(f"ROC para {cls_name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curvas ROC (OvR) — {strategy.replace('_', '-')} (top {max_classes} clases)")
    ax.legend(loc="lower right", fontsize=7)

    _safe_save(fig, plots_dir / f"06_roc_curves_{strategy}.png", logger)


def generate_all_plots(
    results_zero: pd.DataFrame,
    results_few: pd.DataFrame,
    metrics_zero_global: dict,
    metrics_few_global: dict,
    all_metrics_df: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
):
    """Genera y guarda todos los gráficos."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_global_comparison(metrics_zero_global, metrics_few_global, plots_dir, logger)
    plot_boxplot_by_fold(all_metrics_df, plots_dir, logger)
    plot_confusion_matrix(results_zero, "zero_shot", plots_dir, logger)
    plot_confusion_matrix(results_few,  "few_shot",  plots_dir, logger)
    plot_f1_by_class(results_zero, results_few, plots_dir, logger)
    plot_roc_curves(results_zero, "zero_shot", plots_dir, logger)
    plot_roc_curves(results_few,  "few_shot",  plots_dir, logger)


# =============================================================================
# 14. EXPORTACIÓN DE RESULTADOS
# =============================================================================
def export_results(
    results_zero: pd.DataFrame,
    results_few: pd.DataFrame,
    metrics_by_fold_zero: list,
    metrics_by_fold_few: list,
    global_metrics_zero: dict,
    global_metrics_few: dict,
    output_dir: Path,
    run_config: dict,
    logger: logging.Logger,
):
    """Exporta todos los artefactos a disco."""

    # --- Predicciones ---
    # Columnas base para exportar
    base_cols = [
        "email_id", "fold", "strategy", "true_label", "predicted_label",
        "confidence", "is_correct", "parse_error", "raw_response",
        PREDICTIVE_INPUT_COLUMN,
        # Columnas de auditoría (no usadas en inferencia)
        "subject", "language", "tone", "sender_profile",
        "is_spam", "is_multi_intent", "is_ambiguous",
    ]
    # Solo incluir columnas que existan
    for df, fname in [(results_zero, "predictions_zero_shot.csv"),
                      (results_few,  "predictions_few_shot.csv")]:
        cols_to_export = [c for c in base_cols if c in df.columns]
        df[cols_to_export].to_csv(output_dir / fname, index=False, encoding="utf-8")
        logger.info(f"Exportado: {output_dir / fname}")

    # --- Métricas por fold ---
    all_metrics_records = (
        metrics_by_fold_zero + metrics_by_fold_few +
        [global_metrics_zero, global_metrics_few]
    )
    metrics_df = pd.DataFrame(all_metrics_records)
    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False, encoding="utf-8")
    logger.info(f"Exportado: {output_dir / 'metrics_by_fold.csv'}")

    # --- Resumen de métricas globales ---
    summary_records = [global_metrics_zero, global_metrics_few]
    pd.DataFrame(summary_records).to_csv(
        output_dir / "metrics_summary.csv", index=False, encoding="utf-8"
    )
    logger.info(f"Exportado: {output_dir / 'metrics_summary.csv'}")

    # --- Classification reports ---
    for results_df, strategy, fname in [
        (results_zero, "zero_shot", "classification_report_zero_shot.csv"),
        (results_few,  "few_shot",  "classification_report_few_shot.csv"),
    ]:
        report = classification_report(
            results_df["true_label"].tolist(),
            results_df["predicted_label"].tolist(),
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df.to_csv(output_dir / fname, encoding="utf-8")
        logger.info(f"Exportado: {output_dir / fname}")

    # --- Matrices de confusión ---
    for results_df, strategy, fname in [
        (results_zero, "zero_shot", "confusion_matrix_zero_shot.csv"),
        (results_few,  "few_shot",  "confusion_matrix_few_shot.csv"),
    ]:
        cm = confusion_matrix(
            results_df["true_label"].tolist(),
            results_df["predicted_label"].tolist(),
            labels=VALID_LABELS,
        )
        cm_df = pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)
        cm_df.to_csv(output_dir / fname, encoding="utf-8")
        logger.info(f"Exportado: {output_dir / fname}")

    # --- Log de errores de parsing ---
    errors_zero = results_zero[results_zero["parse_error"] == 1].copy() if "parse_error" in results_zero.columns else pd.DataFrame()
    errors_few  = results_few[results_few["parse_error"] == 1].copy()  if "parse_error" in results_few.columns  else pd.DataFrame()
    errors_all  = pd.concat([errors_zero, errors_few], ignore_index=True)

    cols_error = [c for c in ["email_id", "fold", "strategy", "true_label",
                               "predicted_label", "parse_error", "raw_response",
                               PREDICTIVE_INPUT_COLUMN] if c in errors_all.columns]
    errors_all[cols_error].to_csv(
        output_dir / "errors_log.csv", index=False, encoding="utf-8"
    )
    logger.info(f"Exportado: {output_dir / 'errors_log.csv'} ({len(errors_all)} errores)")

    # --- Configuración de la ejecución ---
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Exportado: {output_dir / 'run_config.json'}")


# =============================================================================
# 15. ARGUMENTOS POR LÍNEA DE COMANDOS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificador de correos energéticos con LLM local (LM Studio) + Validación Cruzada Estratificada"
    )
    parser.add_argument(
        "--input", type=str, default="data/emails.csv",
        help="Ruta al fichero CSV separado por ';'"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Directorio base de salida"
    )
    parser.add_argument(
        "--n-splits", type=int, default=N_SPLITS,
        help=f"Número de folds en StratifiedKFold (default: {N_SPLITS})"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limitar número de filas (útil para pruebas; default: todas)"
    )
    parser.add_argument(
        "--few-shot-per-class", type=int, default=N_FEW_SHOT_PER_CLASS,
        help=f"Máximo ejemplos Few-shot por clase (default: {N_FEW_SHOT_PER_CLASS})"
    )
    parser.add_argument(
        "--max-few-shot-examples", type=int, default=MAX_FEW_SHOT_EXAMPLES,
        help=f"Máximo total de ejemplos Few-shot (default: {MAX_FEW_SHOT_EXAMPLES})"
    )
    parser.add_argument(
        "--reuse-cache", action="store_true", default=REUSE_CACHE,
        help="Reutilizar predicciones cacheadas (default: activado)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Desactivar el uso de caché"
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help=f"Semilla para reproducibilidad (default: {RANDOM_STATE})"
    )
    return parser.parse_args()


# =============================================================================
# 16. FUNCIÓN MAIN
# =============================================================================
def main():
    args = parse_args()

    # Resolver flag de caché
    use_cache = args.reuse_cache and not args.no_cache

    # Crear estructura de directorios
    output_dir = Path(args.output)
    plots_dir  = output_dir / "plots"
    cache_dir  = output_dir / "cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("INICIO DE EJECUCIÓN — LLM Email Classifier CV")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Modelo: {MODEL_NAME}")
    logger.info(f"Estrategia de entrada predictiva: {PREDICTIVE_INPUT_COLUMN} (SOLO esta columna)")
    logger.info(f"Folds: {args.n_splits} | Random state: {args.random_state}")
    logger.info(f"Caché: {'activada' if use_cache else 'desactivada'}")
    logger.info("=" * 70)

    # Configuración de la ejecución para run_config.json
    run_config = {
        "timestamp":              datetime.now().isoformat(),
        "model":                  MODEL_NAME,
        "n_splits":               args.n_splits,
        "random_state":           args.random_state,
        "temperature":            TEMPERATURE,
        "n_few_shot_per_class":   args.few_shot_per_class,
        "max_few_shot_examples":  args.max_few_shot_examples,
        "max_rows":               args.max_rows,
        "reuse_cache":            use_cache,
        # IMPORTANTE: documentar columna predictiva
        "predictive_input_columns": [PREDICTIVE_INPUT_COLUMN],
        "audit_only_columns": AUDIT_ONLY_COLUMNS,
        "note": (
            "La clasificación del LLM se basa EXCLUSIVAMENTE en la columna "
            f"'{PREDICTIVE_INPUT_COLUMN}'. Las columnas en 'audit_only_columns' "
            "se conservan solo para trazabilidad y nunca se envían al modelo."
        ),
        "valid_labels":           VALID_LABELS,
        "n_classes":              len(VALID_LABELS),
    }

    # Cargar dataset
    df = load_and_validate_dataset(
        input_path=args.input,
        max_rows=args.max_rows,
        logger=logger,
    )

    # Inicializar modelo LM Studio
    logger.info(f"Conectando con LM Studio, modelo: {MODEL_NAME}")
    try:
        model = lms.llm(MODEL_NAME)
        logger.info("Modelo LM Studio inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar el modelo LM Studio: {e}")
        sys.exit(1)

    # =========================================================================
    # VALIDACIÓN CRUZADA — ZERO-SHOT
    # =========================================================================
    logger.info("=" * 50)
    logger.info("ESTRATEGIA: ZERO-SHOT")
    logger.info("=" * 50)
    results_zero = run_cross_validation(
        df=df,
        model=model,
        strategy="zero_shot",
        n_splits=args.n_splits,
        n_few_shot_per_class=0,
        max_few_shot_examples=0,
        random_state=args.random_state,
        reuse_cache=use_cache,
        cache_dir=cache_dir,
        output_dir=output_dir,
        logger=logger,
    )

    # =========================================================================
    # VALIDACIÓN CRUZADA — FEW-SHOT
    # =========================================================================
    logger.info("=" * 50)
    logger.info("ESTRATEGIA: FEW-SHOT")
    logger.info("=" * 50)
    results_few = run_cross_validation(
        df=df,
        model=model,
        strategy="few_shot",
        n_splits=args.n_splits,
        n_few_shot_per_class=args.few_shot_per_class,
        max_few_shot_examples=args.max_few_shot_examples,
        random_state=args.random_state,
        reuse_cache=use_cache,
        cache_dir=cache_dir,
        output_dir=output_dir,
        logger=logger,
    )

    # =========================================================================
    # MÉTRICAS
    # =========================================================================
    logger.info("Calculando métricas...")

    metrics_by_fold_zero, global_metrics_zero = compute_all_metrics(
        results_zero, "zero_shot", logger
    )
    metrics_by_fold_few, global_metrics_few = compute_all_metrics(
        results_few, "few_shot", logger
    )

    all_metrics_df = pd.DataFrame(
        metrics_by_fold_zero + metrics_by_fold_few +
        [global_metrics_zero, global_metrics_few]
    )

    # Logging de métricas globales
    logger.info("--- MÉTRICAS GLOBALES: ZERO-SHOT ---")
    for k, v in global_metrics_zero.items():
        logger.info(f"  {k}: {v}")

    logger.info("--- MÉTRICAS GLOBALES: FEW-SHOT ---")
    for k, v in global_metrics_few.items():
        logger.info(f"  {k}: {v}")

    # =========================================================================
    # GRÁFICOS
    # =========================================================================
    logger.info("Generando gráficos...")
    generate_all_plots(
        results_zero=results_zero,
        results_few=results_few,
        metrics_zero_global=global_metrics_zero,
        metrics_few_global=global_metrics_few,
        all_metrics_df=all_metrics_df,
        plots_dir=plots_dir,
        logger=logger,
    )

    # =========================================================================
    # EXPORTACIÓN
    # =========================================================================
    logger.info("Exportando resultados...")
    export_results(
        results_zero=results_zero,
        results_few=results_few,
        metrics_by_fold_zero=metrics_by_fold_zero,
        metrics_by_fold_few=metrics_by_fold_few,
        global_metrics_zero=global_metrics_zero,
        global_metrics_few=global_metrics_few,
        output_dir=output_dir,
        run_config=run_config,
        logger=logger,
    )

    logger.info("=" * 70)
    logger.info("FIN DE EJECUCIÓN")
    logger.info(f"Resultados en: {output_dir.resolve()}")
    logger.info("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()