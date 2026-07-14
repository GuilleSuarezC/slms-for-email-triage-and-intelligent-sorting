"""
================================================================================
Sistema de Generación de Dataset Sintético de Correos del Sector Energético
Versión: 11.0.0
Descripción: Fix crash Tkinter/matplotlib + corrección automática de placeholders via LLM
================================================================================
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
# CRÍTICO: forzar backend no-interactivo ANTES de cualquier import
# de matplotlib o seaborn para evitar el crash de Tkinter en hilos.
# ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # noqa: E402  — debe ir antes de pyplot

import threading
import asyncio
import csv
import json
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

try:
    import lmstudio as lms
    LMS_AVAILABLE = True
except ImportError:
    LMS_AVAILABLE = False
    logging.warning("lmstudio SDK no disponible. Instala con: pip install lmstudio")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("chromadb no disponible. Usando fallback Parquet/CSV.")

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN CENTRALIZADA
# ─────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    "num_samples": 500,
    "output_dir": "./energy_dataset",
    "metadata_file": "metadata.csv",
    "dataset_file": "email_dataset.csv",
    "dataset_json": "email_dataset.json",
    "parquet_fallback": "email_dataset.parquet",
    "model_strategy": "round-robin",
    "max_workers": 4,
    "max_retries": 3,
    "retry_delay": 2.0,
    "request_timeout": 120,
    "default_temperature": 0.85,
    "default_top_p": 0.92,
    "default_top_k": 50,
    "max_tokens": 2000,
    "multi_intent_ratio": 0.20,
    "ambiguous_ratio": 0.15,
    "spam_ratio": 0.08,
    "english_ratio": 0.20,
    "plot_style": "seaborn-v0_8-whitegrid",
    "figure_dpi": 150,
    "chroma_collection": "energy_emails",
    "chroma_persist_dir": "./chroma_db",
}

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EnergyEmailDataset")

# ─────────────────────────────────────────────────────────────────
# CLASES DE DATOS
# ─────────────────────────────────────────────────────────────────
@dataclass
class EmailRecord:
    email_id: str
    subject: str
    email_corpus: str
    class_label: str
    secondary_label: Optional[str]
    is_multi_intent: bool
    is_ambiguous: bool
    is_spam: bool
    language: str
    tone: str
    sender_profile: str
    email_length: str
    noise_level: str
    channel: str
    temperature: float
    top_p: float
    top_k: int
    seed: int
    model_assigned: str
    generation_time_s: float
    prompt_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MetadataRow:
    email_id: str
    class_label: str
    secondary_label: Optional[str]
    is_multi_intent: bool
    is_ambiguous: bool
    is_spam: bool
    language: str
    tone: str
    sender_profile: str
    email_length: str
    noise_level: str
    channel: str
    temperature: float
    top_p: float
    top_k: int
    seed: int
    model_assigned: str = ""


# ─────────────────────────────────────────────────────────────────
# DATOS DEL DOMINIO – SECTOR ENERGÉTICO
# ─────────────────────────────────────────────────────────────────
ENERGY_CLASSES_WEIGHTED: List[Tuple[str, float]] = [
    ("factura_error",             0.08),
    ("corte_suministro",          0.07),
    ("alta_consumo_anomalo",      0.06),
    ("solicitud_cambio_tarifa",   0.06),
    ("averia_contador",           0.05),
    ("solicitud_nuevo_contrato",  0.04),
    ("baja_contrato",             0.04),
    ("cambio_titular",            0.04),
    ("reclamacion_factura",       0.04),
    ("consulta_autoconsumo",      0.04),
    ("instalacion_solar",         0.03),
    ("consulta_tarifas_pvpc",     0.03),
    ("fraude_energetico",         0.03),
    ("emergencia_gas",            0.03),
    ("fuga_gas",                  0.03),
    ("consulta_eficiencia",       0.025),
    ("programa_fidelizacion",     0.025),
    ("solicitud_aplazamiento",    0.025),
    ("incidencia_apagón",         0.025),
    ("cambio_domiciliacion",      0.02),
    ("factura_duplicada",         0.02),
    ("consulta_lectura_contador", 0.02),
    ("bono_social",               0.02),
    ("contrato_empresarial",      0.02),
    ("vehiculo_electrico",        0.02),
    ("reclamacion_calidad",       0.015),
    ("denuncia_proveedor",        0.015),
    ("phishing_energia",          0.01),
    ("newsletter",                0.01),
    ("oferta_comercial",          0.01),
    ("respuesta_automatica",      0.01),
    ("spam_energia",              0.01),
    ("solicitud_certificado",     0.01),
    ("consulta_juridica",         0.01),
    ("otro_irrelevante",          0.005),
]

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "factura_error": "El cliente reporta un error en el importe, fechas o conceptos de su factura eléctrica o de gas",
    "corte_suministro": "Aviso o queja sobre un corte de suministro eléctrico o de gas, ya sea programado o por impago",
    "alta_consumo_anomalo": "El cliente detecta un consumo inusualmente alto que no corresponde a su uso habitual",
    "solicitud_cambio_tarifa": "Petición para cambiar a otra tarifa eléctrica (PVPC, tarifa fija, discriminación horaria, etc.)",
    "averia_contador": "Reporte de contador roto, con lectura incorrecta o que no funciona correctamente",
    "solicitud_nuevo_contrato": "Solicitud para dar de alta un nuevo contrato de luz o gas en un domicilio o local",
    "baja_contrato": "Solicitud de cancelación o baja del contrato de suministro energético",
    "cambio_titular": "Petición para cambiar el titular del contrato (herencia, compraventa, divorcio, etc.)",
    "reclamacion_factura": "Reclamación formal sobre el cobro incorrecto de conceptos en la factura",
    "consulta_autoconsumo": "Preguntas sobre instalación de placas solares, autoconsumo compartido o vertido a red",
    "instalacion_solar": "Consulta o solicitud de información sobre instalación de paneles fotovoltaicos",
    "consulta_tarifas_pvpc": "Preguntas sobre el Precio Voluntario al Pequeño Consumidor y precios de mercado",
    "fraude_energetico": "Denuncia o sospecha de fraude en el suministro, enganche ilegal o manipulación de contador",
    "emergencia_gas": "Situación de emergencia relacionada con gas: olor a gas, posible fuga, explosión",
    "fuga_gas": "Detección o sospecha de fuga de gas en instalación doméstica o industrial",
    "consulta_eficiencia": "Preguntas sobre cómo reducir el consumo energético y mejorar la eficiencia del hogar",
    "programa_fidelizacion": "Consulta o queja sobre puntos, descuentos o beneficios del programa de fidelización",
    "solicitud_aplazamiento": "Petición para aplazar o fraccionar el pago de una factura pendiente",
    "incidencia_apagón": "Reporte de apagón o microinterrupción en la zona o edificio del cliente",
    "cambio_domiciliacion": "Solicitud de cambio de cuenta bancaria para el cargo de las facturas",
    "factura_duplicada": "El cliente ha recibido dos facturas del mismo período o un cobro duplicado",
    "consulta_lectura_contador": "Preguntas sobre cómo leer el contador, enviar lectura o interpretar los datos",
    "bono_social": "Consulta o solicitud relacionada con el bono social eléctrico para hogares vulnerables",
    "contrato_empresarial": "Gestión de contratos de suministro para empresas, polígonos industriales o grandes consumidores",
    "vehiculo_electrico": "Preguntas sobre tarifas especiales, puntos de recarga o instalación de cargador para vehículo eléctrico",
    "reclamacion_calidad": "Queja sobre la calidad del suministro: fluctuaciones de tensión, micro-cortes frecuentes",
    "denuncia_proveedor": "Denuncia formal contra la comercializadora por prácticas abusivas o incumplimiento contractual",
    "phishing_energia": "El cliente ha recibido un correo o llamada fraudulenta suplantando a la empresa energética",
    "newsletter": "Boletín informativo de la empresa enviado al cliente",
    "oferta_comercial": "Comunicación comercial con ofertas, promociones o nuevas tarifas",
    "respuesta_automatica": "Respuesta automática del sistema o de un cliente ausente de la oficina",
    "spam_energia": "Correo no deseado relacionado con servicios energéticos de terceros",
    "solicitud_certificado": "Petición de certificado de consumo, contrato o instalación para trámites administrativos",
    "consulta_juridica": "Consulta sobre aspectos legales del contrato, LOPD, o reclamaciones ante organismos reguladores",
    "otro_irrelevante": "Correo que no tiene relación con servicios energéticos o está fuera del dominio",
}

TONES = ["formal", "informal", "urgente", "agresivo", "técnico", "confuso", "cortés", "desesperado"]
SENDER_PROFILES = [
    "cliente_residencial", "cliente_empresarial", "abogado", "jubilado", "autónomo", "particular_enfadado"
]
EMAIL_LENGTHS = ["muy_corto", "corto", "medio", "largo", "muy_largo"]
NOISE_LEVELS = ["limpio", "leve", "moderado", "alto"]
CHANNELS = ["webform", "email_directo", "app_movil", "reenvio_tercero", "respuesta_cadena"]

LENGTH_TOKENS = {
    "muy_corto": (50, 100),
    "corto": (100, 200),
    "medio": (200, 350),
    "largo": (350, 500),
    "muy_largo": (500, 700),
}

NOISE_HINTS = {
    "limpio": "ninguno, texto correcto y profesional",
    "leve": "algún acento faltante o puntuación descuidada",
    "moderado": "errores tipográficos, frases cortadas, mayúsculas al azar",
    "alto": "faltas de ortografía graves, lenguaje muy informal, texto difícil de leer",
}

LENGTH_HINTS = {
    "muy_corto": "30-60",
    "corto": "80-130",
    "medio": "150-250",
    "largo": "280-400",
    "muy_largo": "450-600",
}

# ─────────────────────────────────────────────────────────────────
# PLANTILLAS MAESTRAS REFINADAS
# ─────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: List[str] = [

    # PLANTILLA 1: Formato "Briefing Narrativo"
    """Tu objetivo es asumir la identidad de un {sender_profile} y redactar un texto en {language} para {channel}. 

Motivo: {class_description} (Categoría: {class_label}). 
Estilo: No vayas directo al grano. Narra cómo te diste cuenta del problema. Tono {tone}. Extensión: {length_hint} palabras. Ruido: {noise_hint}.

Condiciones especiales:
- Si is_multi_intent=True ({is_multi_intent}): Mezcla con {secondary_label}.
- Si is_ambiguous=True ({is_ambiguous}): Sé contradictorio.
- Si is_spam=True ({is_spam}): Estilo promocional agresivo.

REGLAS CRÍTICAS DE FORMATO:
1. Genera ÚNICAMENTE los campos SUBJECT y BODY. No digas "Aquí tienes el correo" ni añadas comentarios.
2. NO UTILICES PLACEHOLDERS EN EL EMAIL CORPUS BAJO NINGÚN CONCEPTO.
3. PROHIBIDO EL USO DE MARKDOWN: No uses asteriscos (*), almohadillas (#), ni guiones de lista que generen formato. 
4. No uses negritas ni cursivas. El texto debe ser PLANO.
5. Inventa nombres, DNI, direcciones y cifras reales. NADA de corchetes [ ] ni placeholders.

SUBJECT: [Asunto]
BODY: [Texto narrativo plano]""",

    # PLANTILLA 2: Formato "Ficha Técnica / System Prompt"
    """[INSTRUCCIÓN DE SISTEMA: GENERACIÓN DE TEXTO PLANO]
EMISOR: {sender_profile} | IDIOMA: {language} | CANAL: {channel} | TEMA: {class_label}
TONO: {tone} | RUIDO: {noise_hint} | LONGITUD: {length_hint} palabras.

MODIFICADORES: Multi-intención={is_multi_intent} ({secondary_label}), Ambigüedad={is_ambiguous}, Spam={is_spam}.

REQUISITOS TÉCNICOS DE SALIDA:
- Prohibido incluir metadatos, introducciones del asistente o despedidas.
- NO UTILICES PLACEHOLDERS EN EL EMAIL CORPUS BAJO NINGÚN CONCEPTO.
- Prohibido el formato Markdown. No uses asteriscos para negritas. Si necesitas enumerar, usa números (1., 2.) o guiones simples (-) sin estilos adicionales.
- Debes inventar: Número de contrato (ej. CTR-7721), fecha, empresa energética real y teléfono.
- Inventa nombres, DNI, direcciones y cifras reales. NADA de corchetes [ ] ni placeholders.

ENTREGA EXCLUSIVAMENTE:
SUBJECT: [Asunto conciso]
BODY: [Cuerpo con puntos en texto plano, sin negritas]""",

    # PLANTILLA 3: Formato "Roleplay / Guion de Situación"
    """ESCENARIO: Eres un {sender_profile} contactando con su comercializadora por {channel} en {language}.
SITUACIÓN: {class_label} ({class_description}). Tono: {tone}. Ruido: {noise_hint}.

MISIÓN: Escribe un relato cronológico (pasado, presente y futuro).
- Si is_multi_intent=True ({is_multi_intent}): Involucra {secondary_label}.
- Si is_ambiguous=True ({is_ambiguous}): Orden caótico.
- Si is_spam=True ({is_spam}): Falsa alerta para robo de datos.

REGLA DE ORO DE LA RESPUESTA:
- NO escribas nada fuera de SUBJECT y BODY.
- NO UTILICES PLACEHOLDERS EN EL EMAIL CORPUS BAJO NINGÚN CONCEPTO.
- ELIMINA CUALQUIER INTENTO DE FORMATO DE TEXTO (asteriscos, negritas, subrayados).
- Inventa e integra: DNI con letra, referencia de ticket (TCK-XXXX), dirección física exacta y nombre completo.
- Inventa nombres, DNI, direcciones y cifras reales. NADA de corchetes [ ] ni placeholders.

SUBJECT: [Asunto]
BODY: [El relato secuencial en texto plano puro]""",

    # PLANTILLA 4: Formato "Checklist del Director"
    """¡Acción! Escribe como un {sender_profile} viviendo esto: {class_label} ({class_description}).
Configuración: Idioma {language}, Canal {channel}, Tono {tone}, Longitud {length_hint}, Ruido {noise_hint}.

Modificadores: Multi-intent={is_multi_intent} ({secondary_label}), Ambigüedad={is_ambiguous}, Spam={is_spam}.

RESTRICCIONES DE PRODUCCIÓN (OBLIGATORIO):
- Entrega SOLO el contenido final. No hables conmigo (el usuario).
- NO UTILICES PLACEHOLDERS EN EL EMAIL CORPUS BAJO NINGÚN CONCEPTO.
- NADA DE FORMATO RICH TEXT O MARKDOWN. Solo texto plano. No uses asteriscos para resaltar.
- Inventa: Nombre propio, ciudad, importe de factura y CUPS.
- Inventa nombres, DNI, direcciones y cifras reales. NADA de corchetes [ ] ni placeholders.

SALIDA DE CÁMARA:
SUBJECT: [Asunto]
BODY: [El mensaje crudo sin ningún tipo de marcador ni formato]""",

    # PLANTILLA 5: Formato "Orden de Trabajo / Terminal"
    """=========================================
EJECUCIÓN DE GENERACIÓN DE TEXTO PLANO
=========================================
PERFIL: {sender_profile} | CANAL: {channel} | IDIOMA: {language}
CASO: {class_label} ({class_description})
PARÁMETROS: Tono={tone}, Longitud={length_hint}, Ruido={noise_hint}
FLAGS: Multi={is_multi_intent}, Ambiguo={is_ambiguous}, Spam={is_spam}

PROTOCOLO DE SEGURIDAD DE DATOS:
- Genera datos sintéticos: SN de equipo, técnico, marca de contador y dirección exacta. No uses [ ].

PROTOCOLO DE SALIDA ESTRICTO:
- No incluyas NADA que no sea el SUBJECT y el BODY.
- NO UTILICES PLACEHOLDERS EN EL EMAIL CORPUS BAJO NINGÚN CONCEPTO.
- Prohibido el uso de caracteres de formato Markdown (asteriscos, negritas).
- El texto debe ser una cadena de caracteres limpia, sin estilos visuales.
- Inventa nombres, DNI, direcciones y cifras reales (cualquier dato debe ser inventado). NADA de corchetes [ ] ni placeholders.

SUBJECT: [Asunto]
BODY: [Contenido del sistema en texto plano]"""
]


# ─────────────────────────────────────────────────────────────────
# CLASE: MetadataGenerator
# ─────────────────────────────────────────────────────────────────
class MetadataGenerator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.num_samples = config["num_samples"]
        self.rng = np.random.default_rng(seed=42)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classes = [c for c, _ in ENERGY_CLASSES_WEIGHTED]
        raw_weights = np.array([w for _, w in ENERGY_CLASSES_WEIGHTED])
        self.weights = raw_weights / raw_weights.sum()

    def _sample_classes(self) -> List[str]:
        return list(self.rng.choice(self.classes, size=self.num_samples, p=self.weights))

    def _assign_multi_intent(self, primary_labels: List[str]) -> List[Optional[str]]:
        secondary = []
        for label in primary_labels:
            if self.rng.random() < self.config["multi_intent_ratio"]:
                candidates = [c for c in self.classes if c != label]
                secondary.append(str(self.rng.choice(candidates)))
            else:
                secondary.append(None)
        return secondary

    def _generate_params(self) -> Tuple[List[float], List[float], List[int], List[int]]:
        temps = self.rng.uniform(0.65, 1.05, self.num_samples).round(2).tolist()
        top_ps = self.rng.uniform(0.30, 0.78, self.num_samples).round(2).tolist()
        top_ks = self.rng.integers(30, 80, self.num_samples).tolist()
        seeds = self.rng.integers(1, 99999, self.num_samples).tolist()
        return temps, top_ps, top_ks, seeds

    def generate(self) -> pd.DataFrame:
        self.logger.info(f"Generando metadatos para {self.num_samples} muestras...")
        primary_labels = self._sample_classes()
        secondary_labels = self._assign_multi_intent(primary_labels)
        temps, top_ps, top_ks, seeds = self._generate_params()

        is_multi_intent = [s is not None for s in secondary_labels]
        is_ambiguous = [
            (self.rng.random() < self.config["ambiguous_ratio"]) and not mi
            for mi in is_multi_intent
        ]
        is_spam = [
            label in ("spam_energia", "otro_irrelevante", "newsletter",
                      "oferta_comercial", "respuesta_automatica", "phishing_energia")
            or self.rng.random() < self.config["spam_ratio"]
            for label in primary_labels
        ]

        tones = self.rng.choice(TONES, self.num_samples).tolist()
        profiles = self.rng.choice(SENDER_PROFILES, self.num_samples).tolist()
        lengths = self.rng.choice(
            EMAIL_LENGTHS, self.num_samples, p=[0.1, 0.25, 0.35, 0.20, 0.10]
        ).tolist()
        noises = self.rng.choice(
            NOISE_LEVELS, self.num_samples, p=[0.40, 0.30, 0.20, 0.10]
        ).tolist()
        channels = self.rng.choice(CHANNELS, self.num_samples).tolist()
        lang_choices = ["es"] * 4 + ["en"]
        languages = [random.choice(lang_choices) for _ in range(self.num_samples)]

        rows = []
        for i in range(self.num_samples):
            row = MetadataRow(
                email_id=str(uuid.uuid4())[:12],
                class_label=primary_labels[i],
                secondary_label=secondary_labels[i],
                is_multi_intent=is_multi_intent[i],
                is_ambiguous=is_ambiguous[i],
                is_spam=is_spam[i],
                language=languages[i],
                tone=tones[i],
                sender_profile=profiles[i],
                email_length=lengths[i],
                noise_level=noises[i],
                channel=channels[i],
                temperature=temps[i],
                top_p=top_ps[i],
                top_k=top_ks[i],
                seed=seeds[i],
            )
            rows.append(asdict(row))

        df = pd.DataFrame(rows)
        self.logger.info(
            f"Metadatos generados: {len(df)} filas | "
            f"multi-intent: {df['is_multi_intent'].sum()} | "
            f"ambiguous: {df['is_ambiguous'].sum()} | "
            f"spam: {df['is_spam'].sum()}"
        )
        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        self.logger.info(f"Metadatos guardados: {path}")


# ─────────────────────────────────────────────────────────────────
# CLASE: StatsPlotter
# ─────────────────────────────────────────────────────────────────
class StatsPlotter:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["output_dir"]) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            plt.style.use(config.get("plot_style", "seaborn-v0_8-whitegrid"))
        except OSError:
            plt.style.use("ggplot")

    def plot_all(self, df: pd.DataFrame) -> None:
        self.logger.info("Generando visualizaciones EDA...")
        self._plot_class_distribution(df)
        self._plot_noise_vs_tone(df)
        self._plot_length_distribution(df)
        self._plot_flags_summary(df)
        self._plot_language_channel(df)
        self.logger.info(f"Plots guardados en: {self.output_dir}")

    def _plot_class_distribution(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(14, 10))
        counts = df["class_label"].value_counts()
        colors = sns.color_palette("husl", len(counts))
        counts.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Distribución de Clases – Sector Energético", fontsize=14, fontweight="bold")
        ax.set_xlabel("Número de muestras")
        ax.set_ylabel("Clase")
        for i, v in enumerate(counts.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(self.output_dir / "01_class_distribution.png", dpi=self.config["figure_dpi"])
        plt.close(fig)

    def _plot_noise_vs_tone(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = pd.crosstab(df["noise_level"], df["tone"])
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title("Ruido vs Tono del Correo", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "02_noise_vs_tone.png", dpi=self.config["figure_dpi"])
        plt.close(fig)

    def _plot_length_distribution(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        length_order = ["muy_corto", "corto", "medio", "largo", "muy_largo"]
        counts_all = df["email_length"].value_counts().reindex(length_order, fill_value=0)
        counts_all.plot(kind="bar", ax=axes[0], color=sns.color_palette("Blues_d", 5), edgecolor="white")
        axes[0].set_title("Distribución de Longitud")
        axes[0].set_xlabel("Longitud")
        axes[0].set_ylabel("Muestras")
        axes[0].tick_params(axis="x", rotation=45)

        lang_length = df.groupby(["language", "email_length"]).size().unstack(fill_value=0)
        lang_length.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="white")
        axes[1].set_title("Longitud por Idioma")
        axes[1].set_xlabel("Idioma")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(loc="upper right", fontsize=8)

        plt.suptitle("Análisis de Longitud de Correos", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / "03_length_distribution.png", dpi=self.config["figure_dpi"])
        plt.close(fig)

    def _plot_flags_summary(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        flags = [
            ("is_multi_intent", "Multi-Intención", "#2196F3"),
            ("is_ambiguous", "Ambiguos", "#FF9800"),
            ("is_spam", "Spam/Irrelevante", "#F44336"),
        ]
        for ax, (col, title, color) in zip(axes, flags):
            counts = df[col].value_counts()
            labels = ["Sí" if k else "No" for k in counts.index]
            colors_pie = [color, "#ECEFF1"]
            ax.pie(counts.values, labels=labels, autopct="%1.1f%%",
                   colors=colors_pie, startangle=90, textprops={"fontsize": 11})
            ax.set_title(f"{title}\n({counts.get(True, 0)} / {len(df)})", fontsize=11, fontweight="bold")
        plt.suptitle("Distribución de Casos Especiales", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "04_flags_summary.png", dpi=self.config["figure_dpi"])
        plt.close(fig)

    def _plot_language_channel(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        channel_counts = df["channel"].value_counts()
        channel_counts.plot(kind="bar", ax=axes[0],
                            color=sns.color_palette("Pastel1", len(channel_counts)), edgecolor="gray")
        axes[0].set_title("Canal de Envío")
        axes[0].set_xlabel("Canal")
        axes[0].tick_params(axis="x", rotation=40)

        lang_counts = df["language"].value_counts()
        lang_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%",
                         colors=["#4CAF50", "#2196F3"], textprops={"fontsize": 12})
        axes[1].set_title("Distribución por Idioma")
        axes[1].set_ylabel("")
        plt.suptitle("Canal e Idioma", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "05_language_channel.png", dpi=self.config["figure_dpi"])
        plt.close(fig)

    def print_summary(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("  RESUMEN ESTADÍSTICO DEL DATASET DE METADATOS")
        print("=" * 60)
        print(f"  Total muestras:      {len(df)}")
        print(f"  Clases únicas:       {df['class_label'].nunique()}")
        print(f"  Multi-intención:     {df['is_multi_intent'].sum()} ({df['is_multi_intent'].mean()*100:.1f}%)")
        print(f"  Ambiguos:            {df['is_ambiguous'].sum()} ({df['is_ambiguous'].mean()*100:.1f}%)")
        print(f"  Spam/irrelevante:    {df['is_spam'].sum()} ({df['is_spam'].mean()*100:.1f}%)")
        print(f"  Idioma ES:           {(df['language']=='es').sum()}")
        print(f"  Idioma EN:           {(df['language']=='en').sum()}")
        print(f"  Top clase:           {df['class_label'].value_counts().index[0]} ({df['class_label'].value_counts().iloc[0]})")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────
# CLASE: LLMClient
# ─────────────────────────────────────────────────────────────────
class LLMClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_models: List[str] = []
        self._model_index: int = 0
        self._model_weights: Optional[np.ndarray] = None
        self._thread_local = threading.local()
        self._handle_creation_lock = threading.Lock()

        if not LMS_AVAILABLE:
            raise RuntimeError(
                "La librería 'lmstudio' no está instalada. "
                "Instálala con: pip install lmstudio"
            )

    def connect(self) -> bool:
        try:
            self.available_models = self._discover_models()
            if not self.available_models:
                self.logger.warning(
                    "No se detectaron modelos cargados. "
                    "Carga modelos con 'lms load <modelo>' y vuelve a ejecutar."
                )
                return False
            self.logger.info(f"Modelos listos para usar: {self.available_models}")
            self._init_weights()
            return True
        except Exception as e:
            self.logger.error(f"Error conectando a LM Studio: {e}")
            return False

    def _discover_models(self) -> List[str]:
        model_ids: List[str] = []

        # Estrategia 1: client.llm.list_loaded()
        try:
            with lms.Client() as client:
                loaded = client.llm.list_loaded()
                for m in loaded:
                    mid = (
                        getattr(m, "identifier", None)
                        or getattr(m, "model_key", None)
                        or getattr(m, "id", None)
                    )
                    if mid:
                        model_ids.append(mid)
            if model_ids:
                self.logger.info(f"Modelos detectados via client.llm.list_loaded(): {model_ids}")
                return model_ids
        except Exception as e:
            self.logger.debug(f"client.llm.list_loaded() falló: {e}")

        # Estrategia 2: lms.list_loaded_models()
        try:
            loaded = lms.list_loaded_models()
            for m in loaded:
                mid = (
                    getattr(m, "identifier", None)
                    or getattr(m, "model_key", None)
                    or getattr(m, "id", None)
                )
                if mid:
                    model_ids.append(mid)
            if model_ids:
                self.logger.info(f"Modelos detectados via lms.list_loaded_models(): {model_ids}")
                return model_ids
        except Exception as e:
            self.logger.debug(f"lms.list_loaded_models() falló: {e}")

        # Estrategia 3: handle por defecto
        try:
            handle = lms.llm()
            mid = (
                getattr(handle, "identifier", None)
                or getattr(handle, "model_key", None)
                or getattr(getattr(handle, "model_info", None), "identifier", None)
            )
            if mid:
                self.logger.info(f"Modelo por defecto detectado via lms.llm(): {mid}")
                return [mid]
            self.logger.info("Modelo por defecto detectado (identificador no accesible).")
            return ["__default__"]
        except Exception as e:
            self.logger.debug(f"lms.llm() sin argumentos falló: {e}")

        self.logger.warning(
            "No se encontró ningún modelo cargado en LM Studio. "
            "Asegúrate de haber cargado al menos un modelo con 'lms load <modelo>'."
        )
        return []

    def _init_weights(self) -> None:
        n = len(self.available_models)
        self._model_weights = np.ones(n) / n

    def assign_models(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.available_models:
            raise RuntimeError("No hay modelos disponibles para asignar.")
        strategy = self.config["model_strategy"]
        n = len(df)
        assigned: List[str] = []
        if strategy == "round-robin":
            for i in range(n):
                assigned.append(self.available_models[i % len(self.available_models)])
        elif strategy == "weighted":
            chosen = np.random.choice(self.available_models, size=n, p=self._model_weights)
            assigned = list(chosen)
        else:
            assigned = [random.choice(self.available_models) for _ in range(n)]
        df = df.copy()
        df["model_assigned"] = assigned
        self.logger.info(
            f"Modelos asignados | estrategia='{strategy}' | "
            f"distribución={pd.Series(assigned).value_counts().to_dict()}"
        )
        return df

    def _get_handle(self, model_id: str) -> Any:
        if not hasattr(self._thread_local, "handles"):
            self._thread_local.handles = {}
        if model_id not in self._thread_local.handles:
            with self._handle_creation_lock:
                if model_id not in self._thread_local.handles:
                    self.logger.debug(
                        f"[{threading.current_thread().name}] Creando handle para '{model_id}'"
                    )
                    if model_id in ("__default__", "llm", None):
                        handle = lms.llm()
                    else:
                        handle = lms.llm(model_id)
                    self._thread_local.handles[model_id] = handle
        return self._thread_local.handles[model_id]

    def generate_text(
        self,
        prompt: str,
        model_id: str,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        max_tokens: int = 2000,
        seed: Optional[int] = None,
    ) -> str:
        models_to_try = [model_id] + [m for m in self.available_models if m != model_id]
        last_error: Optional[Exception] = None
        infer_config: Dict[str, Any] = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxTokens": max_tokens,
        }
        if seed is not None:
            infer_config["seed"] = seed

        for attempt, current_model in enumerate(models_to_try[: self.config["max_retries"] + 1]):
            if attempt > 0:
                delay = self.config["retry_delay"] * (2 ** (attempt - 1))
                self.logger.debug(f"Reintento {attempt} con '{current_model}', espera {delay:.1f}s")
                time.sleep(delay)
            try:
                model_handle = self._get_handle(current_model)
                result = model_handle.respond(prompt, config=infer_config)
                text = result if isinstance(result, str) else str(result)
                text = text.strip()
                if text:
                    if attempt > 0:
                        self.logger.info(f"Éxito en reintento {attempt} con '{current_model}'")
                    return text
                self.logger.warning(f"Respuesta vacía del modelo '{current_model}'")
            except Exception as e:
                last_error = e
                self.logger.warning(f"Error generando con '{current_model}' (intento {attempt + 1}): {e}")
                if hasattr(self._thread_local, "handles"):
                    self._thread_local.handles.pop(current_model, None)

        self.logger.error(f"Todos los intentos fallaron. Último error: {last_error}")
        return ""


# ─────────────────────────────────────────────────────────────────
# CLASE: DatasetGenerator
# ─────────────────────────────────────────────────────────────────
class DatasetGenerator:
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient) -> None:
        self.config = config
        self.llm = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = PROMPT_TEMPLATES
        self._template_idx = 0

    def _next_template(self) -> Tuple[str, str]:
        idx = self._template_idx % len(self.templates)
        tpl = self.templates[idx]
        template_id = f"plantilla_{idx + 1}"
        self._template_idx += 1
        return tpl, template_id

    def _build_prompt(self, row: pd.Series) -> Tuple[str, str]:
        template, template_id = self._next_template()
        class_label = row["class_label"]
        class_desc = CLASS_DESCRIPTIONS.get(class_label, "Correo del sector energético")

        secondary_hint = ""
        if row["is_multi_intent"] and row.get("secondary_label"):
            sec_desc = CLASS_DESCRIPTIONS.get(str(row["secondary_label"]), "")
            secondary_hint = f"Intención secundaria también presente: {row['secondary_label']} ({sec_desc})"

        ambiguity_hint = ""
        if row["is_ambiguous"]:
            ambiguity_hint = "IMPORTANTE: El correo debe ser ambiguo, difícil de clasificar con certeza."

        lang_instruction = "en inglés (English)" if row["language"] == "en" else "en español"

        prompt = template.format(
            class_label=class_label,
            class_description=class_desc,
            sender_profile=row["sender_profile"],
            tone=row["tone"],
            language=lang_instruction,
            length_hint=LENGTH_HINTS.get(row["email_length"], "150-250"),
            noise_hint=NOISE_HINTS.get(row["noise_level"], "leve"),
            channel=row["channel"],
            secondary_intent_hint=secondary_hint,
            ambiguity_hint=ambiguity_hint,
            is_multi_intent=row["is_multi_intent"],
            is_ambiguous=row["is_ambiguous"],
            is_spam=row["is_spam"],
            secondary_label=row.get("secondary_label") or "N/A",
        )

        anti_preamble = (
            "INSTRUCCIÓN CRÍTICA: Responde ÚNICAMENTE con el correo solicitado. "
            "No escribas frases introductorias como 'Claro, aquí tienes', 'Por supuesto', "
            "'Entendido', ni cadenas de razonamiento previas del tipo '<think>...</think>', "
            "'<|channel>thought ... <channel|>' o cualquier explicación de lo que vas a hacer. "
            "Tu respuesta debe comenzar DIRECTAMENTE con 'SUBJECT:' y nada más.\n"
            "PROHIBICIÓN ABSOLUTA DE PLACEHOLDERS: No uses corchetes en ningún caso. "
            "Está terminantemente prohibido escribir [DNI], [Nombre], [CUPS], [Fecha], [importe], "
            "[contrato], [empresa], [titular] ni ningún otro texto entre corchetes. "
            "Todos los datos deben estar rellenos con valores ficticios inventados por ti ahora mismo.\n\n"
        )
        prompt = anti_preamble + prompt
        return prompt, template_id

    def _parse_response(self, raw: str) -> Tuple[str, str, List[str]]:
        if not raw:
            return "[Sin asunto]", "[Generación fallida]", []

        # 1. Eliminar bloque LM Studio: <|channel>thought...<channel|>
        if "<|channel>thought" in raw:
            closing_tag = "<channel|>"
            close_idx = raw.rfind(closing_tag)
            if close_idx == -1:
                self.logger.warning(
                    "Bloque <|channel>thought no cerrado — tokens agotados durante el "
                    "razonamiento. Aumenta CONFIG['max_tokens'] para modelos razonadores."
                )
                return "[Sin asunto]", "[Generación fallida - tokens agotados en razonamiento]", []
            raw = raw[close_idx + len(closing_tag):].strip()

        # 2. Eliminar bloques <think>...</think>
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

        # 3. Descartar razonamiento en texto plano (rfind del último SUBJECT:)
        upper_raw = raw.upper()
        last_subject_pos = upper_raw.rfind("SUBJECT:")
        if last_subject_pos == -1:
            self.logger.warning("Respuesta sin SUBJECT: — formato inválido.")
            return "[Sin asunto]", "[Generación fallida - sin formato válido]", []
        if last_subject_pos > 0:
            self.logger.debug(f"Descartados {last_subject_pos} chars de razonamiento previo.")
            raw = raw[last_subject_pos:].strip()

        # 4. Eliminar frases introductorias residuales
        raw = re.sub(
            r"^(Claro[,.]?|Por supuesto[,.]?|Entendido[,.]?|Aquí tienes[,:]?|"
            r"De acuerdo[,.]?|Perfecto[,.]?)[^\n]*\n+",
            "", raw, flags=re.IGNORECASE
        ).strip()

        # 5. Extraer subject
        subject_match = re.search(r"SUBJECT:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else ""

        # 6. Extraer body
        parts = re.split(r"---+", raw, maxsplit=1)
        if len(parts) == 2:
            body = parts[1].strip()
        else:
            body = re.sub(r"SUBJECT:\s*.+?\n?", "", raw, flags=re.IGNORECASE).strip()

        # 7. Limpiar artefactos LLM comunes
        body = re.sub(r"^\[?(BODY|CUERPO|EMAIL BODY|CORREO)[\]:]?\s*", "", body, flags=re.IGNORECASE)

        # 8. Eliminar primera línea del body si duplica el subject
        if subject and body:
            body_lines = body.split("\n")
            first_line = body_lines[0].strip()
            if first_line and (
                first_line.lower() == subject.lower()
                or subject.lower().endswith(first_line[1:].lower())
                or first_line[1:].lower() in subject.lower()
            ):
                body = "\n".join(body_lines[1:]).strip()

        # 9. Fallbacks
        if not subject:
            subject = body.split("\n")[0][:100] if body else "[Sin asunto]"
        if not body:
            body = raw

        # 10. Avisar si parece truncado
        if body and not re.search(r'[.!?»"\']\s*$', body.rstrip()):
            self.logger.warning("El correo parece truncado. Considera aumentar max_tokens.")

        # 11. Detectar placeholders sin rellenar — se devuelven para corrección en _generate_one
        placeholder_pattern = re.compile(r"\[[^\]]{2,40}\]")
        placeholders_found = list(set(placeholder_pattern.findall(body + " " + subject)))

        return subject[:200], body, placeholders_found

    def _fix_placeholders(
        self,
        subject: str,
        body: str,
        placeholders: List[str],
        model_id: str,
        temperature: float,
    ) -> Tuple[str, str]:
        """
        Segunda llamada al LLM para sustituir placeholders sin rellenar.

        Construye un prompt de corrección muy directo que entrega el texto
        completo y la lista exacta de placeholders encontrados, y pide
        únicamente el texto corregido sin explicaciones.

        Returns:
            Tuple[str, str]: (subject corregido, body corregido)
        """
        placeholders_str = ", ".join(placeholders)
        fix_prompt = (
            "El siguiente correo electrónico contiene placeholders sin rellenar "
            f"({placeholders_str}). "
            "Debes sustituir CADA UNO de esos placeholders por un valor ficticio pero realista "
            "y coherente con el contexto del correo (nombres españoles, DNIs válidos, CUPSs, "
            "importes en euros, fechas concretas, números de contrato, etc.). "
            "NO cambies nada más del texto. NO añadas explicaciones ni comentarios. "
            "Devuelve ÚNICAMENTE el correo corregido en el mismo formato:\n\n"
            f"SUBJECT: {subject}\n---\n{body}"
        )

        self.logger.debug(
            f"Lanzando corrección de {len(placeholders)} placeholder(s): {placeholders}"
        )

        raw_fixed = self.llm.generate_text(
            prompt=fix_prompt,
            model_id=model_id,
            temperature=min(temperature, 0.7),   # Temperatura baja: queremos precisión
            top_p=0.9,
            top_k=40,
            max_tokens=self.config["max_tokens"],
            seed=None,                            # Sin semilla fija: variación en los datos inventados
        )

        if not raw_fixed:
            self.logger.warning("La llamada de corrección de placeholders devolvió vacío.")
            return subject, body

        # Reutilizar el parser para extraer subject y body del texto corregido
        fixed_subject, fixed_body, remaining = self._parse_response(raw_fixed)

        if remaining:
            # Si aún quedan placeholders tras la corrección, loguear y devolver lo mejor que tenemos
            self.logger.warning(
                f"Tras corrección aún quedan {len(remaining)} placeholder(s) sin rellenar: "
                f"{remaining} — se conserva el texto corregido parcialmente."
            )

        return fixed_subject, fixed_body

    def _generate_one(self, row: pd.Series) -> EmailRecord:
        prompt, template_id = self._build_prompt(row)
        t_start = time.time()

        raw_response = self.llm.generate_text(
            prompt=prompt,
            model_id=row["model_assigned"],
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            top_k=int(row["top_k"]),
            max_tokens=self.config["max_tokens"],
            seed=int(row["seed"]),
        )

        t_elapsed = time.time() - t_start
        subject, body, placeholders = self._parse_response(raw_response)

        # ── Corrección automática de placeholders ─────────────────────────────
        if placeholders:
            self.logger.warning(
                f"[{row['email_id']}] {len(placeholders)} placeholder(s) detectado(s): "
                f"{placeholders} — lanzando corrección automática."
            )
            subject, body = self._fix_placeholders(
                subject=subject,
                body=body,
                placeholders=placeholders,
                model_id=row["model_assigned"],
                temperature=float(row["temperature"]),
            )
            t_elapsed = time.time() - t_start   # Actualizar tiempo incluyendo la corrección

        return EmailRecord(
            email_id=row["email_id"],
            subject=subject,
            email_corpus=body,
            class_label=row["class_label"],
            secondary_label=row.get("secondary_label"),
            is_multi_intent=bool(row["is_multi_intent"]),
            is_ambiguous=bool(row["is_ambiguous"]),
            is_spam=bool(row["is_spam"]),
            language=row["language"],
            tone=row["tone"],
            sender_profile=row["sender_profile"],
            email_length=row["email_length"],
            noise_level=row["noise_level"],
            channel=row["channel"],
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            top_k=int(row["top_k"]),
            seed=int(row["seed"]),
            model_assigned=row["model_assigned"],
            generation_time_s=round(t_elapsed, 2),
            prompt_used=template_id,
        )

    def generate_all(self, df: pd.DataFrame) -> List[EmailRecord]:
        n_workers = self.config["max_workers"]
        worker_label = "worker" if n_workers == 1 else "workers"
        self.logger.info(f"Iniciando generación de {len(df)} correos con {n_workers} {worker_label}...")
        records: List[EmailRecord] = []
        failed = 0
        rows = [df.iloc[i] for i in range(len(df))]

        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = {executor.submit(self._generate_one, row): i for i, row in enumerate(rows)}
            with tqdm(total=len(futures), desc="Generando correos", unit="email") as pbar:
                for future in as_completed(futures):
                    try:
                        record = future.result(timeout=self.config["request_timeout"])
                        records.append(record)
                        if not record.email_corpus or record.email_corpus == "[Generación fallida]":
                            failed += 1
                    except Exception as e:
                        self.logger.error(f"Error en generación: {e}")
                        failed += 1
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({"fallos": failed})

        self.logger.info(f"Generación completada: {len(records)} registros, {failed} fallos")
        return records


# ─────────────────────────────────────────────────────────────────
# CLASE: DatabaseManager
# ─────────────────────────────────────────────────────────────────
class DatabaseManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._chroma_client: Optional[Any] = None
        self._chroma_collection: Optional[Any] = None

    def records_to_dataframe(self, records: List[EmailRecord]) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in records])

    def save_csv(self, df: pd.DataFrame) -> str:
        path = self.output_dir / self.config["dataset_file"]
        df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL, sep=";")
        self.logger.info(f"Dataset CSV guardado: {path} ({len(df)} filas)")
        return str(path)

    def save_json(self, df: pd.DataFrame) -> str:
        path = self.output_dir / self.config["dataset_json"]
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        self.logger.info(f"Dataset JSON guardado: {path}")
        return str(path)

    def save_parquet(self, df: pd.DataFrame) -> str:
        path = self.output_dir / self.config["parquet_fallback"]
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except ImportError:
            df.to_parquet(path, index=False, engine="fastparquet")
        self.logger.info(f"Dataset Parquet guardado: {path}")
        return str(path)

    def _init_chromadb(self) -> bool:
        if not CHROMA_AVAILABLE:
            return False
        try:
            persist_dir = self.config.get("chroma_persist_dir", "./chroma_db")
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.config["chroma_collection"],
                metadata={"description": "Energy sector synthetic email dataset"},
            )
            self.logger.info(f"ChromaDB inicializado: {persist_dir}")
            return True
        except Exception as e:
            self.logger.warning(f"Error iniciando ChromaDB: {e}")
            return False

    def index_chromadb(self, records: List[EmailRecord], batch_size: int = 100) -> bool:
        if not self._init_chromadb():
            self.logger.warning("ChromaDB no disponible. Usando fallback Parquet.")
            df = self.records_to_dataframe(records)
            self.save_parquet(df)
            return False

        self.logger.info(f"Indexando {len(records)} documentos en ChromaDB...")
        total_indexed = 0
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            try:
                ids = [r.email_id for r in batch]
                documents = [r.email_corpus for r in batch]
                metadatas = [
                    {
                        "subject": r.subject,
                        "class_label": r.class_label,
                        "secondary_label": r.secondary_label or "",
                        "is_multi_intent": str(r.is_multi_intent),
                        "is_ambiguous": str(r.is_ambiguous),
                        "is_spam": str(r.is_spam),
                        "language": r.language,
                        "tone": r.tone,
                        "sender_profile": r.sender_profile,
                        "model_assigned": r.model_assigned,
                        "timestamp": r.timestamp,
                    }
                    for r in batch
                ]
                self._chroma_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_indexed += len(batch)
            except Exception as e:
                self.logger.error(f"Error indexando batch {i//batch_size}: {e}")

        self.logger.info(f"ChromaDB: {total_indexed}/{len(records)} documentos indexados")
        return total_indexed > 0

    def save_all(self, records: List[EmailRecord]) -> Dict[str, str]:
        df = self.records_to_dataframe(records)
        paths = {}
        paths["csv"] = self.save_csv(df)
        paths["json"] = self.save_json(df)
        paths["parquet"] = self.save_parquet(df)
        self.logger.info("Dataset guardado en todos los formatos.")
        return paths


# ─────────────────────────────────────────────────────────────────
# MODO DEMO (sin LM Studio)
# ─────────────────────────────────────────────────────────────────
class DemoLLMClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.available_models = ["demo-model-v1", "demo-model-v2"]
        self.logger = logging.getLogger(self.__class__.__name__)
        self._idx = 0

    def connect(self) -> bool:
        self.logger.warning("Usando DemoLLMClient (sin LM Studio real).")
        return True

    def assign_models(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["model_assigned"] = [self.available_models[i % 2] for i in range(len(df))]
        return df

    def generate_text(self, prompt: str, model_id: str, **kwargs) -> str:
        class_match = re.search(r"Clase[:\s]+(\w+)", prompt)
        class_label = class_match.group(1) if class_match else "consulta"
        templates_demo = [
            f"SUBJECT: Consulta urgente sobre {class_label.replace('_', ' ')}\n---\nEstimados señores,\n\nMe dirijo a ustedes para informarles de una situación relacionada con {class_label.replace('_', ' ')} en mi suministro. El número de contrato es CT-{random.randint(10000,99999)}. Quedo a su disposición.\n\nAtentamente,\nJosé García",
            f"SUBJECT: Problema con mi contrato - {class_label}\n---\nBuenos días,\n\nLlevo varios días intentando resolver este problema (referencia CUPS: ES00{random.randint(10000000,99999999)}AA). {CLASS_DESCRIPTIONS.get(class_label, 'Problemas con el servicio')}.\n\nEspero su pronta respuesta.\n\nSaludos,\nMaría López",
            f"SUBJECT: RE: Incidencia #{random.randint(100000,999999)}\n---\nHola,\n\nOs escribo de nuevo porque no he recibido respuesta. El asunto afecta a {class_label.replace('_', ' ')}. Por favor, contacten conmigo al 612{random.randint(100000,999999)}.\n\nGracias,\nAntonio Ruiz",
        ]
        time.sleep(0.05)
        return random.choice(templates_demo)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def run_pipeline(config: Dict[str, Any], use_demo: bool = False) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SISTEMA DE GENERACIÓN DE DATASET SINTÉTICO – SECTOR ENERGÉTICO")
    print("=" * 70 + "\n")

    print("📋 FASE 1-2: Generando metadatos del dataset...")
    metadata_gen = MetadataGenerator(config)
    df_meta = metadata_gen.generate()
    metadata_path = output_dir / config["metadata_file"]
    metadata_gen.save(df_meta, str(metadata_path))
    print(f"   ✓ Metadatos guardados en: {metadata_path}\n")

    print("📊 FASE 3: Análisis estadístico exploratorio (EDA)...")
    plotter = StatsPlotter(config)
    plotter.print_summary(df_meta)
    plotter.plot_all(df_meta)
    print(f"   ✓ Gráficos guardados en: {output_dir / 'plots'}\n")

    print("🤖 FASE 4: Inicializando cliente LLM...")
    if use_demo or not LMS_AVAILABLE:
        llm = DemoLLMClient(config)
    else:
        llm = LLMClient(config)

    connected = llm.connect()
    if not connected:
        print("   ⚠️  No se pudo conectar a LM Studio. Usando modo demo.")
        llm = DemoLLMClient(config)
        llm.connect()

    df_meta = llm.assign_models(df_meta)
    metadata_gen.save(df_meta, str(metadata_path))
    print(f"   ✓ Modelos asignados. Estrategia: {config['model_strategy']}\n")

    print("✉️  Generando correos electrónicos...")
    generator = DatasetGenerator(config, llm)
    records = generator.generate_all(df_meta)

    if not records:
        print("   ❌ No se generaron registros. Abortando.")
        return
    print(f"   ✓ {len(records)} correos generados.\n")

    print("💾 FASE 5: Guardando dataset...")
    db = DatabaseManager(config)
    paths = db.save_all(records)
    for fmt, path in paths.items():
        print(f"   ✓ {fmt.upper()}: {path}")

    print("\n🔍 Indexando en ChromaDB...")
    indexed = db.index_chromadb(records)
    if indexed:
        print(f"   ✓ Indexación completada en: {config['chroma_persist_dir']}")
    else:
        print("   ⚠️  ChromaDB no disponible. Fallback Parquet guardado.")

    df_final = db.records_to_dataframe(records)
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"  Total correos generados:   {len(df_final)}")
    print(f"  Tasa de éxito:             {(df_final['email_corpus'] != '[Generación fallida]').mean()*100:.1f}%")
    print(f"  Clases únicas:             {df_final['class_label'].nunique()}")
    print(f"  Tiempo promedio/correo:    {df_final['generation_time_s'].mean():.2f}s")
    print(f"  Directorio de salida:      {output_dir.resolve()}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generador de Dataset Sintético de Correos del Sector Energético",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python energy35C_email_dataset_generator_v10.py
  python energy35C_email_dataset_generator_v10.py --samples 1000 --workers 8
  python energy35C_email_dataset_generator_v10.py --demo --samples 50
  python energy35C_email_dataset_generator_v10.py --strategy weighted --output ./mi_dataset
        """
    )
    parser.add_argument("--samples", type=int, default=CONFIG["num_samples"])
    parser.add_argument("--workers", type=int, default=CONFIG["max_workers"])
    parser.add_argument("--strategy", choices=["random", "round-robin", "weighted"],
                        default=CONFIG["model_strategy"])
    parser.add_argument("--output", type=str, default=CONFIG["output_dir"])
    parser.add_argument("--demo", action="store_true")

    args = parser.parse_args()
    CONFIG["num_samples"] = args.samples
    CONFIG["max_workers"] = args.workers
    CONFIG["model_strategy"] = args.strategy
    CONFIG["output_dir"] = args.output

    run_pipeline(CONFIG, use_demo=args.demo)