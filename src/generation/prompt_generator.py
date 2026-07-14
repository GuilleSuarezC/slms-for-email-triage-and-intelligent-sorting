"""
╔══════════════════════════════════════════════════════════════════╗
║          EMAIL PROMPT GENERATOR — LM Studio Edition             ║
║  Genera, envía y almacena prompts para correos con LLMs locales  ║
╚══════════════════════════════════════════════════════════════════╝

Módulos:
  1. PromptTemplate    — Plantillas universales para correos
  2. PromptBuilder     — Constructor y combinador de parámetros
  3. LMStudioClient    — Cliente para la API compatible con OpenAI
  4. VectorStore       — Almacenamiento en ChromaDB o pandas DataFrame
  5. ResultExporter    — Exportación a CSV / JSON
  6. EmailPromptSystem — Orquestador principal

Uso rápido:
    python prompt_generator.py --demo
    python prompt_generator.py --interactive
"""

import os
import json
import csv
import uuid
import hashlib
import logging
import argparse
import itertools
from datetime import datetime
from typing import Optional

import requests

# ── Dependencias opcionales ────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠  pandas no disponible — instálalo con: pip install pandas")

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠  ChromaDB no disponible — usando pandas como almacén alternativo")

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmailPromptGen")


# ══════════════════════════════════════════════════════════════════
# 1. PLANTILLAS DE PROMPT
# ══════════════════════════════════════════════════════════════════

class PromptTemplate:
    """
    Colección de plantillas universales para la generación de correos.
    Cada plantilla es una cadena con marcadores {variable}.
    """

    # Plantilla general (cubre todos los casos de uso)
    UNIVERSAL = """Eres un redactor profesional de correos electrónicos.

TAREA: Redacta UN correo electrónico completo con las siguientes características:

- Tipo de correo  : {email_type}
- Tono            : {tone}
- Longitud        : {length}
- Idioma          : {language}
- Contexto / Propósito: {context}

RESTRICCIONES:
- Incluye: Asunto, Saludo, Cuerpo y Despedida.
- Ajusta ESTRICTAMENTE el número de palabras según la longitud indicada:
    * Muy corto : 30–60 palabras
    * Corto     : 80–120 palabras
    * Medio     : 150–250 palabras
    * Largo     : 300–450 palabras
- No añadas explicaciones fuera del correo.
- El correo debe ser coherente, natural y listo para enviar.

CORREO:"""

    # Plantilla orientada a ventas / persuasión
    SALES = """Actúa como un copywriter experto en email marketing.

OBJETIVO: Escribe un correo de {email_type} altamente persuasivo.

Parámetros:
  Tono      → {tone}
  Longitud  → {length}
  Idioma    → {language}
  Contexto  → {context}

El correo DEBE:
1. Captar atención en el asunto (máx. 8 palabras).
2. Abrir con un gancho emocional o dato sorprendente.
3. Presentar el valor/beneficio de forma clara.
4. Incluir una llamada a la acción (CTA) concreta.
5. Mantener el tono "{tone}" de principio a fin.

Longitudes orientativas: muy corto=50 palabras, corto=100, medio=200, largo=400.

CORREO COMPLETO (Asunto + Cuerpo):"""

    # Plantilla para soporte / atención al cliente
    SUPPORT = """Eres un agente de soporte al cliente empático y resolutivo.

TAREA: Redacta un correo de {email_type} para el siguiente caso:

  Situación : {context}
  Tono      : {tone}
  Longitud  : {length}
  Idioma    : {language}

Pautas:
- Reconoce el problema o consulta del cliente.
- Ofrece una solución clara o próximos pasos.
- Cierra de forma cordial e invita a más preguntas.
- NO uses jerga técnica innecesaria.
- Longitud: muy corto≈50 palabras, corto≈100, medio≈200, largo≈380.

CORREO (incluye Asunto):"""

    # Plantilla para correos de seguimiento
    FOLLOWUP = """Redacta un correo de seguimiento profesional.

Contexto previo: {context}

Configuración:
  Tipo     → {email_type}
  Tono     → {tone}
  Longitud → {length}
  Idioma   → {language}

Instrucciones:
- Referencia brevemente la comunicación anterior.
- Indica claramente el motivo del seguimiento.
- Propón un paso concreto o fecha de reunión si aplica.
- Sé {tone}, sin resultar insistente ni agresivo.

Extensión: muy corto≈45 palabras, corto≈90, medio≈180, largo≈350.

CORREO (Asunto + Cuerpo):"""

    @classmethod
    def get(cls, template_name: str = "universal") -> str:
        """Devuelve la plantilla por nombre (universal, sales, support, followup)."""
        mapping = {
            "universal": cls.UNIVERSAL,
            "sales":     cls.SALES,
            "support":   cls.SUPPORT,
            "followup":  cls.FOLLOWUP,
        }
        return mapping.get(template_name.lower(), cls.UNIVERSAL)

    @classmethod
    def list_templates(cls) -> list[str]:
        return ["universal", "sales", "support", "followup"]


# ══════════════════════════════════════════════════════════════════
# 2. CONSTRUCTOR DE PROMPTS
# ══════════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Construye prompts individuales y genera combinaciones automáticas
    a partir de los parámetros disponibles.
    """

    # Valores predefinidos para cada dimensión
    EMAIL_TYPES = ["spam", "ventas", "soporte", "seguimiento", "informativo",
                   "bienvenida", "recordatorio", "agradecimiento", "queja", "propuesta"]

    TONES = ["amable", "educado", "formal", "informal", "persuasivo",
             "urgente", "enfadado", "entusiasta", "empático", "neutro"]

    LENGTHS = ["muy corto", "corto", "medio", "largo"]

    LANGUAGES = ["español", "inglés", "francés", "alemán", "portugués"]

    def __init__(self):
        self.templates = PromptTemplate

    # ── Constructor individual ─────────────────────────────────────
    def build(
        self,
        email_type: str,
        tone: str,
        length: str,
        context: str,
        language: str = "español",
        template_name: str = "universal",
    ) -> dict:
        """
        Construye un prompt completo y devuelve un diccionario con
        el prompt y todos sus metadatos.
        """
        template = self.templates.get(template_name)
        prompt_text = template.format(
            email_type=email_type,
            tone=tone,
            length=length,
            language=language,
            context=context,
        )

        # ID determinista basado en el contenido (evita duplicados exactos)
        fingerprint = hashlib.md5(prompt_text.encode()).hexdigest()

        return {
            "id":            str(uuid.uuid4()),
            "fingerprint":   fingerprint,
            "prompt":        prompt_text,
            "template":      template_name,
            "email_type":    email_type,
            "tone":          tone,
            "length":        length,
            "language":      language,
            "context":       context,
            "created_at":    datetime.utcnow().isoformat(),
        }

    # ── Generador de combinaciones ─────────────────────────────────
    def generate_combinations(
        self,
        email_types:   Optional[list[str]] = None,
        tones:         Optional[list[str]] = None,
        lengths:       Optional[list[str]] = None,
        languages:     Optional[list[str]] = None,
        contexts:      Optional[list[str]] = None,
        template_name: str = "universal",
        max_combos:    int = 20,
    ) -> list[dict]:
        """
        Genera hasta `max_combos` combinaciones únicas de parámetros.
        Si no se proporcionan listas, se usan los valores predefinidos.
        """
        et = email_types or self.EMAIL_TYPES[:3]
        tn = tones       or self.TONES[:3]
        ln = lengths     or self.LENGTHS[:2]
        lg = languages   or ["español"]
        cx = contexts    or ["Contexto genérico de ejemplo"]

        all_combos = list(itertools.product(et, tn, ln, lg, cx))
        selected   = all_combos[:max_combos]

        prompts = []
        seen_fingerprints = set()

        for email_type, tone, length, language, context in selected:
            p = self.build(email_type, tone, length, context, language, template_name)
            if p["fingerprint"] not in seen_fingerprints:
                seen_fingerprints.add(p["fingerprint"])
                prompts.append(p)

        logger.info(f"Generadas {len(prompts)} combinaciones únicas de prompts")
        return prompts


# ══════════════════════════════════════════════════════════════════
# 3. CLIENTE LM STUDIO
# ══════════════════════════════════════════════════════════════════

class LMStudioClient:
    """
    Cliente HTTP para la API de LM Studio (compatible con OpenAI).
    Soporta múltiples modelos en el mismo servidor.

    Configuración por defecto:
        base_url : http://localhost:1234/v1
        timeout  : 120 s

    LM Studio expone los modelos cargados en /v1/models.
    Cada petición de completado va a /v1/chat/completions.
    """

    def __init__(
        self,
        base_url:    str  = "http://localhost:1234/v1",
        timeout:     int  = 120,
        api_key:     str  = "lm-studio",   # LM Studio no requiere clave real
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    # ── Listar modelos disponibles ─────────────────────────────────
    def list_models(self) -> list[str]:
        """Devuelve los IDs de los modelos cargados en LM Studio."""
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"No se pudo obtener la lista de modelos: {e}")
            return []

    # ── Completado de chat ─────────────────────────────────────────
    def complete(
        self,
        prompt:      str,
        model:       str,
        temperature: float = 0.7,
        top_p:       float = 0.95,
        max_tokens:  int   = 512,
    ) -> dict:
        """
        Envía un prompt al modelo y devuelve un diccionario con:
            text        — respuesta generada
            model       — modelo utilizado
            tokens_used — tokens de entrada + salida
            latency_ms  — latencia en milisegundos
            error       — mensaje de error (si lo hay)
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p":       top_p,
            "max_tokens":  max_tokens,
            "stream":      False,
        }

        start = datetime.utcnow()
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data       = r.json()
            elapsed_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            text       = data["choices"][0]["message"]["content"].strip()
            usage      = data.get("usage", {})

            return {
                "text":        text,
                "model":       model,
                "tokens_used": usage.get("total_tokens", 0),
                "latency_ms":  elapsed_ms,
                "error":       None,
            }

        except requests.exceptions.ConnectionError:
            return {
                "text":        "",
                "model":       model,
                "tokens_used": 0,
                "latency_ms":  0,
                "error":       "⚠ LM Studio no accesible — ¿está en ejecución?",
            }
        except Exception as e:
            return {
                "text":        "",
                "model":       model,
                "tokens_used": 0,
                "latency_ms":  0,
                "error":       str(e),
            }

    # ── Enviar a múltiples modelos ─────────────────────────────────
    def complete_multi(
        self,
        prompt:      str,
        models:      list[str],
        temperature: float = 0.7,
        top_p:       float = 0.95,
        max_tokens:  int   = 512,
    ) -> list[dict]:
        """Envía el mismo prompt a varios modelos y agrega las respuestas."""
        results = []
        for model in models:
            logger.info(f"  → Consultando modelo: {model}")
            resp = self.complete(prompt, model, temperature, top_p, max_tokens)
            results.append(resp)
            if resp["error"]:
                logger.warning(f"    ✗ Error en {model}: {resp['error']}")
            else:
                logger.info(f"    ✓ {resp['tokens_used']} tokens · {resp['latency_ms']} ms")
        return results


# ══════════════════════════════════════════════════════════════════
# 4. ALMACÉN VECTORIAL (ChromaDB / pandas)
# ══════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Almacena prompts y respuestas.

    Backends disponibles:
        "chroma"  — ChromaDB (persistente, búsqueda semántica)
        "pandas"  — DataFrame en memoria (exportable a CSV/JSON)

    Se selecciona automáticamente según las dependencias instaladas,
    o se puede forzar con el parámetro `backend`.
    """

    def __init__(
        self,
        backend:    str = "auto",
        chroma_dir: str = "./chroma_db",
        collection: str = "email_prompts",
    ):
        self.records: list[dict] = []   # buffer universal

        # Selección de backend
        if backend == "auto":
            backend = "chroma" if HAS_CHROMA else "pandas"

        self.backend = backend
        logger.info(f"Backend de almacenamiento: {self.backend.upper()}")

        if self.backend == "chroma" and HAS_CHROMA:
            self._init_chroma(chroma_dir, collection)
        elif self.backend == "pandas" and not HAS_PANDAS:
            raise RuntimeError("pandas no está instalado. Instálalo con: pip install pandas")

    def _init_chroma(self, chroma_dir: str, collection: str):
        """Inicializa el cliente y la colección de ChromaDB."""
        self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB inicializado en '{chroma_dir}' · colección '{collection}'")

    # ── Añadir un registro ─────────────────────────────────────────
    def add(self, prompt_meta: dict, response: dict) -> str:
        """
        Almacena un par (prompt, respuesta) con sus metadatos.
        Devuelve el ID del registro.
        """
        record_id = str(uuid.uuid4())
        metadata  = {
            "prompt_id":   prompt_meta["id"],
            "fingerprint": prompt_meta["fingerprint"],
            "template":    prompt_meta["template"],
            "email_type":  prompt_meta["email_type"],
            "tone":        prompt_meta["tone"],
            "length":      prompt_meta["length"],
            "language":    prompt_meta["language"],
            "context":     prompt_meta["context"][:200],   # ChromaDB limita metadatos
            "model":       response["model"],
            "tokens_used": response["tokens_used"],
            "latency_ms":  response["latency_ms"],
            "has_error":   bool(response["error"]),
            "created_at":  datetime.utcnow().isoformat(),
        }

        full_record = {
            **metadata,
            "record_id": record_id,
            "prompt":    prompt_meta["prompt"],
            "response":  response["text"],
            "error":     response.get("error") or "",
        }

        # ── ChromaDB ───────────────────────────────────────────────
        if self.backend == "chroma" and HAS_CHROMA:
            # El documento almacenado es la respuesta (para búsqueda semántica)
            doc = response["text"] if response["text"] else prompt_meta["prompt"]
            self._collection.add(
                documents=[doc],
                metadatas=[metadata],
                ids=[record_id],
            )

        # ── pandas / memoria ───────────────────────────────────────
        self.records.append(full_record)
        return record_id

    # ── Búsqueda semántica (solo ChromaDB) ────────────────────────
    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Busca registros por similitud semántica (requiere ChromaDB)."""
        if self.backend != "chroma" or not HAS_CHROMA:
            logger.warning("La búsqueda semántica solo está disponible con ChromaDB.")
            return []
        results = self._collection.query(query_texts=[query], n_results=n_results)
        return results.get("metadatas", [[]])[0]

    # ── DataFrame ─────────────────────────────────────────────────
    def to_dataframe(self):
        """Convierte todos los registros a un DataFrame de pandas."""
        if not HAS_PANDAS:
            raise RuntimeError("pandas no instalado.")
        return pd.DataFrame(self.records)

    # ── Estadísticas ───────────────────────────────────────────────
    def stats(self) -> dict:
        """Resumen rápido del almacén."""
        total   = len(self.records)
        errors  = sum(1 for r in self.records if r.get("has_error"))
        models  = list({r["model"] for r in self.records})
        types   = list({r["email_type"] for r in self.records})
        return {
            "total_records": total,
            "errors":        errors,
            "success_rate":  f"{((total-errors)/total*100):.1f}%" if total else "N/A",
            "models_used":   models,
            "email_types":   types,
        }


# ══════════════════════════════════════════════════════════════════
# 5. EXPORTADOR DE RESULTADOS
# ══════════════════════════════════════════════════════════════════

class ResultExporter:
    """Exporta los registros del almacén a CSV o JSON."""

    @staticmethod
    def to_csv(records: list[dict], path: str = "results.csv") -> str:
        """Exporta a CSV. Devuelve la ruta del archivo."""
        if not records:
            logger.warning("No hay registros para exportar.")
            return ""
        fieldnames = list(records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Exportado a CSV: {path} ({len(records)} registros)")
        return path

    @staticmethod
    def to_json(records: list[dict], path: str = "results.json", indent: int = 2) -> str:
        """Exporta a JSON. Devuelve la ruta del archivo."""
        if not records:
            logger.warning("No hay registros para exportar.")
            return ""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=indent)
        logger.info(f"Exportado a JSON: {path} ({len(records)} registros)")
        return path


# ══════════════════════════════════════════════════════════════════
# 6. ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class EmailPromptSystem:
    """
    Orquesta todo el flujo:
        1. Construir prompts
        2. Enviarlos a los modelos
        3. Almacenar resultados
        4. (Opcional) Exportar

    Ejemplo básico:
        system = EmailPromptSystem(models=["llama3", "mistral"])
        system.run_demo()
    """

    # Modelos de ejemplo para LM Studio (ajusta según los que tengas cargados)
    DEFAULT_MODELS = [
        "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "lmstudio-community/Phi-3-mini-4k-instruct-GGUF",
    ]

    def __init__(
        self,
        models:      Optional[list[str]] = None,
        base_url:    str   = "http://localhost:1234/v1",
        backend:     str   = "auto",
        temperature: float = 0.7,
        top_p:       float = 0.95,
        max_tokens:  int   = 512,
    ):
        self.models      = models or self.DEFAULT_MODELS
        self.temperature = temperature
        self.top_p       = top_p
        self.max_tokens  = max_tokens

        self.builder  = PromptBuilder()
        self.client   = LMStudioClient(base_url=base_url)
        self.store    = VectorStore(backend=backend)
        self.exporter = ResultExporter()

    # ── Flujo principal ────────────────────────────────────────────
    def run(
        self,
        prompts:       list[dict],
        models:        Optional[list[str]] = None,
        export_csv:    bool = False,
        export_json:   bool = False,
        output_prefix: str  = "email_results",
    ) -> list[dict]:
        """
        Ejecuta el ciclo completo para una lista de prompts construidos.
        Devuelve todos los registros almacenados.
        """
        models = models or self.models
        logger.info(f"▶ Iniciando ciclo: {len(prompts)} prompts × {len(models)} modelos")

        for i, prompt_meta in enumerate(prompts, 1):
            logger.info(
                f"[{i}/{len(prompts)}] tipo={prompt_meta['email_type']} "
                f"tono={prompt_meta['tone']} longitud={prompt_meta['length']}"
            )
            responses = self.client.complete_multi(
                prompt=prompt_meta["prompt"],
                models=models,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            for resp in responses:
                self.store.add(prompt_meta, resp)

        # Exportar si se solicita
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_csv:
            self.exporter.to_csv(self.store.records, f"{output_prefix}_{ts}.csv")
        if export_json:
            self.exporter.to_json(self.store.records, f"{output_prefix}_{ts}.json")

        # Estadísticas finales
        stats = self.store.stats()
        logger.info("══════════════════════════════════")
        logger.info(f"  Total registros : {stats['total_records']}")
        logger.info(f"  Tasa de éxito   : {stats['success_rate']}")
        logger.info(f"  Modelos usados  : {', '.join(stats['models_used'])}")
        logger.info("══════════════════════════════════")

        return self.store.records

    # ── Demo rápida ────────────────────────────────────────────────
    def run_demo(self, export: bool = True):
        """
        Ejecuta una demostración con 6 combinaciones predefinidas
        y exporta los resultados a JSON.
        """
        logger.info("═══════ MODO DEMO ═══════")

        contexts = [
            "Presentar nuestro nuevo servicio de consultoría a una PYME del sector logístico.",
            "Recordar al cliente que su suscripción vence en 3 días.",
            "Dar la bienvenida a un nuevo usuario registrado en nuestra plataforma SaaS.",
        ]

        prompts = self.builder.generate_combinations(
            email_types=["ventas", "recordatorio", "bienvenida"],
            tones=["persuasivo", "amable"],
            lengths=["corto", "medio"],
            languages=["español"],
            contexts=contexts,
            template_name="universal",
            max_combos=6,
        )

        return self.run(
            prompts=prompts,
            export_csv=export,
            export_json=export,
            output_prefix="demo_results",
        )

    # ── Modo interactivo ───────────────────────────────────────────
    def run_interactive(self):
        """Guía al usuario para construir y enviar un prompt paso a paso."""
        print("\n╔══════════════════════════════════════╗")
        print("║   EMAIL PROMPT GENERATOR — Modo CLI  ║")
        print("╚══════════════════════════════════════╝\n")

        def pick(label: str, options: list[str]) -> str:
            print(f"\n{label}:")
            for i, o in enumerate(options, 1):
                print(f"  {i}. {o}")
            while True:
                try:
                    idx = int(input("  Elige (número): ")) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                except ValueError:
                    pass
                print("  ⚠ Opción inválida.")

        email_type = pick("Tipo de correo", PromptBuilder.EMAIL_TYPES)
        tone       = pick("Tono",           PromptBuilder.TONES)
        length     = pick("Longitud",       PromptBuilder.LENGTHS)
        language   = pick("Idioma",         PromptBuilder.LANGUAGES)
        template   = pick("Plantilla",      PromptTemplate.list_templates())

        context = input("\nDescribe el contexto o propósito del correo:\n> ").strip()
        if not context:
            context = "Correo de ejemplo genérico."

        prompt_meta = self.builder.build(
            email_type=email_type,
            tone=tone,
            length=length,
            context=context,
            language=language,
            template_name=template,
        )

        print("\n── PROMPT GENERADO ──────────────────────────────")
        print(prompt_meta["prompt"])
        print("─────────────────────────────────────────────────")

        # Detectar modelos disponibles
        available = self.client.list_models()
        if available:
            print(f"\nModelos detectados en LM Studio: {available}")
            models_to_use = available[:5]
        else:
            print("\n⚠ No se detectaron modelos. Usando nombres de ejemplo (modo offline).")
            models_to_use = ["modelo-demo"]

        records = self.run(
            prompts=[prompt_meta],
            models=models_to_use,
            export_json=True,
        )

        print("\n── RESPUESTAS ───────────────────────────────────")
        for r in records:
            print(f"\n▸ Modelo: {r['model']}")
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
            else:
                print(r["response"])
        print("─────────────────────────────────────────────────")
        return records


# ══════════════════════════════════════════════════════════════════
# 7. PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Email Prompt Generator — LM Studio Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python prompt_generator.py --demo
  python prompt_generator.py --interactive
  python prompt_generator.py --type ventas --tone persuasivo --length medio \\
      --context "Lanzamiento de producto" --models llama3 mistral
        """
    )
    p.add_argument("--demo",        action="store_true",  help="Ejecutar demo automática")
    p.add_argument("--interactive", action="store_true",  help="Modo interactivo CLI")
    p.add_argument("--type",        default="ventas",     help="Tipo de correo")
    p.add_argument("--tone",        default="persuasivo", help="Tono del correo")
    p.add_argument("--length",      default="medio",      help="Longitud del correo")
    p.add_argument("--language",    default="español",    help="Idioma")
    p.add_argument("--context",     default="Ejemplo de correo generado automáticamente.")
    p.add_argument("--template",    default="universal",  help="Plantilla a usar")
    p.add_argument("--models",      nargs="+",            help="IDs de modelos LM Studio")
    p.add_argument("--base-url",    default="http://localhost:1234/v1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p",       type=float, default=0.95)
    p.add_argument("--max-tokens",  type=int,   default=512)
    p.add_argument("--backend",     default="auto", choices=["auto", "chroma", "pandas"])
    p.add_argument("--export-csv",  action="store_true")
    p.add_argument("--export-json", action="store_true")
    p.add_argument("--combos",      type=int,   default=0,
                   help="Generar N combinaciones automáticas en lugar de un solo prompt")
    return p.parse_args()


def main():
    args = parse_args()

    system = EmailPromptSystem(
        models=args.models,
        base_url=args.base_url,
        backend=args.backend,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    if args.demo:
        system.run_demo()

    elif args.interactive:
        system.run_interactive()

    elif args.combos > 0:
        prompts = system.builder.generate_combinations(
            template_name=args.template,
            max_combos=args.combos,
        )
        system.run(
            prompts=prompts,
            export_csv=args.export_csv,
            export_json=args.export_json,
        )

    else:
        # Prompt único con los parámetros dados
        prompt_meta = system.builder.build(
            email_type=args.type,
            tone=args.tone,
            length=args.length,
            context=args.context,
            language=args.language,
            template_name=args.template,
        )
        print("\n── PROMPT ──────────────────────────────────────")
        print(prompt_meta["prompt"])
        print("────────────────────────────────────────────────\n")
        system.run(
            prompts=[prompt_meta],
            export_csv=args.export_csv,
            export_json=args.export_json,
        )


if __name__ == "__main__":
    main()
