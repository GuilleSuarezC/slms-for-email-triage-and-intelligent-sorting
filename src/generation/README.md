# Email Prompt Generator — LM Studio

Generador automático de prompts para correos electrónicos con modelos de lenguaje locales.

---

## Instalación rápida

```bash
pip install -r requirements.txt
```

> **Mínimo obligatorio:** `requests`  
> **Recomendado:** `requests pandas chromadb`

---

## Uso en 30 segundos

```bash
# Demo automática (6 combinaciones, 3 modelos)
python prompt_generator.py --demo

# Modo interactivo guiado
python prompt_generator.py --interactive

# Prompt único directo
python prompt_generator.py \
  --type ventas --tone persuasivo --length medio \
  --context "Lanzamiento de nuevo producto SaaS" \
  --models llama3 mistral phi3 \
  --export-json

# Generar 20 combinaciones automáticas
python prompt_generator.py --combos 20 --export-csv --export-json
```

---

## Arquitectura

```
EmailPromptSystem          ← Orquestador principal
├── PromptTemplate         ← 4 plantillas (universal, sales, support, followup)
├── PromptBuilder          ← Constructor y combinador de parámetros
├── LMStudioClient         ← API compatible con OpenAI (LM Studio)
├── VectorStore            ← ChromaDB (semántico) o pandas (memoria)
└── ResultExporter         ← CSV y JSON
```

---

## Parámetros disponibles

| Dimensión     | Valores predefinidos |
|---------------|----------------------|
| `email_type`  | spam, ventas, soporte, seguimiento, informativo, bienvenida, recordatorio, agradecimiento, queja, propuesta |
| `tone`        | amable, educado, formal, informal, persuasivo, urgente, enfadado, entusiasta, empático, neutro |
| `length`      | muy corto (30-60 palabras), corto (80-120), medio (150-250), largo (300-450) |
| `language`    | español, inglés, francés, alemán, portugués |
| `template`    | universal, sales, support, followup |

---

## Configuración de LM Studio

1. Abre LM Studio y carga entre 1 y 5 modelos
2. Ve a **Local Server** → **Start Server** (puerto por defecto: `1234`)
3. Verifica que el servidor esté activo en `http://localhost:1234/v1/models`

```python
from prompt_generator import LMStudioClient
client = LMStudioClient()
print(client.list_models())   # ['llama3', 'mistral', ...]
```

---

## API Python

```python
from prompt_generator import EmailPromptSystem, PromptBuilder

# Sistema completo
system = EmailPromptSystem(
    models      = ["llama3", "mistral", "phi3"],
    temperature = 0.7,
    top_p       = 0.95,
    max_tokens  = 512,
)

# Un prompt personalizado
builder     = PromptBuilder()
prompt_meta = builder.build(
    email_type = "ventas",
    tone       = "persuasivo",
    length     = "corto",
    context    = "Descuento del 40% en nuestro plan Pro.",
    language   = "español",
    template_name = "sales",
)

records = system.run(
    prompts     = [prompt_meta],
    export_json = True,
)

# Múltiples combinaciones
prompts = builder.generate_combinations(
    email_types = ["ventas", "soporte"],
    tones       = ["amable", "formal"],
    lengths     = ["corto", "medio"],
    max_combos  = 10,
)
records = system.run(prompts=prompts, export_csv=True)

# Exportar DataFrame
df = system.store.to_dataframe()
print(df[["email_type", "tone", "model", "tokens_used"]].head())
```

---

## Salida almacenada

Cada registro incluye:

| Campo          | Descripción                          |
|----------------|--------------------------------------|
| `record_id`    | UUID único del registro              |
| `fingerprint`  | Hash MD5 del prompt (anti-duplicados)|
| `prompt`       | Texto completo del prompt            |
| `response`     | Correo generado por el modelo        |
| `email_type`   | Tipo de correo                       |
| `tone`         | Tono utilizado                       |
| `length`       | Longitud objetivo                    |
| `language`     | Idioma                               |
| `model`        | Modelo LM Studio usado               |
| `tokens_used`  | Tokens consumidos                    |
| `latency_ms`   | Latencia de la llamada               |
| `has_error`    | Boolean de error                     |
| `created_at`   | Timestamp UTC                        |

---

## Búsqueda semántica (ChromaDB)

```python
results = system.store.search(
    query="correo de soporte técnico urgente",
    n_results=5,
)
for r in results:
    print(r["email_type"], r["tone"], r["model"])
```

---

## Estructura del proyecto

```
email_prompt_generator/
├── prompt_generator.py   ← Código principal (6 módulos)
├── examples.py           ← 6 ejemplos de uso comentados
├── requirements.txt      ← Dependencias
└── README.md             ← Esta documentación
```
