"""
examples.py — Ejemplos de uso del Email Prompt Generator
=========================================================
Ejecuta este archivo directamente para ver todos los ejemplos:
    python examples.py
"""

from prompt_generator import (
    PromptTemplate,
    PromptBuilder,
    LMStudioClient,
    VectorStore,
    ResultExporter,
    EmailPromptSystem,
)

# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 1 — Prompt único manual
# ═══════════════════════════════════════════════════════════════════
def ejemplo_prompt_unico():
    """Construye y muestra un único prompt sin llamar a ningún modelo."""
    print("\n" + "═"*60)
    print("EJEMPLO 1 — Prompt único")
    print("═"*60)

    builder = PromptBuilder()
    prompt  = builder.build(
        email_type  = "ventas",
        tone        = "persuasivo",
        length      = "corto",
        context     = "Promoción de 30% de descuento en suscripción anual de software contable.",
        language    = "español",
        template_name = "sales",
    )

    print(f"ID          : {prompt['id']}")
    print(f"Huella MD5  : {prompt['fingerprint']}")
    print(f"Plantilla   : {prompt['template']}")
    print(f"\n── PROMPT ──\n{prompt['prompt']}")


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 2 — Generación automática de combinaciones
# ═══════════════════════════════════════════════════════════════════
def ejemplo_combinaciones():
    """Genera 12 combinaciones y muestra sus metadatos (sin modelos)."""
    print("\n" + "═"*60)
    print("EJEMPLO 2 — Combinaciones automáticas")
    print("═"*60)

    builder = PromptBuilder()
    prompts = builder.generate_combinations(
        email_types   = ["soporte", "seguimiento", "informativo"],
        tones         = ["amable", "formal", "urgente"],
        lengths       = ["muy corto", "corto"],
        languages     = ["español"],
        contexts      = ["Incidencia técnica con el servidor de correo corporativo."],
        template_name = "support",
        max_combos    = 12,
    )

    for i, p in enumerate(prompts, 1):
        print(f"  [{i:02d}] tipo={p['email_type']:<14} tono={p['tone']:<10} longitud={p['length']}")

    print(f"\nTotal generados: {len(prompts)}")


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 3 — Listar plantillas disponibles
# ═══════════════════════════════════════════════════════════════════
def ejemplo_plantillas():
    """Muestra todas las plantillas disponibles."""
    print("\n" + "═"*60)
    print("EJEMPLO 3 — Plantillas disponibles")
    print("═"*60)

    for name in PromptTemplate.list_templates():
        template = PromptTemplate.get(name)
        # Mostrar solo las primeras 3 líneas de cada plantilla
        preview = "\n".join(template.strip().splitlines()[:3])
        print(f"\n▸ '{name}':\n{preview}\n  ...")


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 4 — Almacén VectorStore con datos simulados
# ═══════════════════════════════════════════════════════════════════
def ejemplo_vector_store():
    """Simula el almacenamiento sin necesitar LM Studio activo."""
    print("\n" + "═"*60)
    print("EJEMPLO 4 — VectorStore con datos simulados")
    print("═"*60)

    builder = PromptBuilder()
    store   = VectorStore(backend="pandas")   # pandas siempre disponible

    # Datos simulados
    fake_responses = [
        {"model": "llama3",   "text": "Estimado cliente, ...", "tokens_used": 120, "latency_ms": 800,  "error": None},
        {"model": "mistral",  "text": "Hola, nos complace...", "tokens_used": 95,  "latency_ms": 650,  "error": None},
        {"model": "phi3mini", "text": "",                      "tokens_used": 0,   "latency_ms": 0,    "error": "Timeout"},
    ]

    prompt_meta = builder.build(
        email_type="ventas", tone="amable", length="corto",
        context="Ejemplo simulado", language="español",
    )

    for resp in fake_responses:
        store.add(prompt_meta, resp)

    print(f"Registros almacenados: {len(store.records)}")
    stats = store.stats()
    print(f"Estadísticas: {stats}")

    # Exportar
    exporter = ResultExporter()
    exporter.to_csv(store.records,  "ejemplo_simulado.csv")
    exporter.to_json(store.records, "ejemplo_simulado.json")
    print("✓ Archivos exportados: ejemplo_simulado.csv / .json")


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 5 — Flujo completo con LM Studio (requiere servidor activo)
# ═══════════════════════════════════════════════════════════════════
def ejemplo_flujo_completo():
    """
    Ejemplo real: detecta modelos en LM Studio, genera prompts y
    almacena resultados. Si LM Studio no está activo, muestra los
    prompts generados sin enviarlos.
    """
    print("\n" + "═"*60)
    print("EJEMPLO 5 — Flujo completo con LM Studio")
    print("═"*60)

    system = EmailPromptSystem(
        base_url    = "http://localhost:1234/v1",
        backend     = "auto",
        temperature = 0.75,
        top_p       = 0.92,
        max_tokens  = 400,
    )

    # Detectar modelos disponibles
    available = system.client.list_models()
    if not available:
        print("⚠ LM Studio no está disponible. Mostrando solo los prompts generados.\n")
        prompts = system.builder.generate_combinations(
            email_types=["ventas", "soporte"],
            tones=["persuasivo", "formal"],
            lengths=["corto"],
            contexts=["Ejemplo sin modelo activo."],
            max_combos=4,
        )
        for p in prompts:
            print(f"\n── Prompt [{p['email_type']} / {p['tone']}] ──")
            print(p["prompt"][:300] + "...")
        return

    print(f"Modelos detectados: {available}\n")
    models = available[:3]   # Usar hasta 3 modelos

    prompts = system.builder.generate_combinations(
        email_types=["ventas", "seguimiento"],
        tones=["persuasivo", "amable"],
        lengths=["corto", "medio"],
        contexts=["Propuesta de colaboración B2B en el sector tecnológico."],
        max_combos=4,
    )

    records = system.run(
        prompts     = prompts,
        models      = models,
        export_csv  = True,
        export_json = True,
    )

    print(f"\n✓ {len(records)} registros guardados.")


# ═══════════════════════════════════════════════════════════════════
# EJEMPLO 6 — Control de diversidad (temperatura / top-p)
# ═══════════════════════════════════════════════════════════════════
def ejemplo_diversidad():
    """
    Muestra cómo ajustar temperatura y top-p para controlar
    la creatividad de las respuestas (requiere LM Studio activo).
    """
    print("\n" + "═"*60)
    print("EJEMPLO 6 — Control de diversidad")
    print("═"*60)

    builder = PromptBuilder()
    client  = LMStudioClient()
    store   = VectorStore(backend="pandas")

    prompt_meta = builder.build(
        email_type="informativo",
        tone="neutro",
        length="corto",
        context="Anuncio de nueva funcionalidad en plataforma SaaS.",
    )

    configs = [
        {"temperature": 0.3, "top_p": 0.85, "label": "Conservador"},
        {"temperature": 0.7, "top_p": 0.95, "label": "Equilibrado"},
        {"temperature": 1.1, "top_p": 0.98, "label": "Creativo"},
    ]

    available = client.list_models()
    if not available:
        print("⚠ LM Studio no disponible. Mostrando configuraciones sin ejecutar.\n")
        for c in configs:
            print(f"  {c['label']:12} → temperature={c['temperature']}, top_p={c['top_p']}")
        return

    model = available[0]
    for cfg in configs:
        print(f"\n▸ Modo {cfg['label']} (temp={cfg['temperature']}, top_p={cfg['top_p']})")
        resp = client.complete(
            prompt=prompt_meta["prompt"],
            model=model,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        store.add(prompt_meta, resp)
        if resp["error"]:
            print(f"  ERROR: {resp['error']}")
        else:
            print(f"  {resp['text'][:200]}...")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ejemplo_prompt_unico()
    ejemplo_combinaciones()
    ejemplo_plantillas()
    ejemplo_vector_store()
    ejemplo_flujo_completo()
    ejemplo_diversidad()

    print("\n\n✅ Todos los ejemplos completados.")
    print("   Para el modo interactivo ejecuta: python prompt_generator.py --interactive")
    print("   Para la demo automática ejecuta:  python prompt_generator.py --demo")
