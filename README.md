# Generador offline de Quiz y Presentaciones desde PDF

Aplicación **100 % local** que toma un PDF académico y genera, con
inferencia local vía **Ollama**:

1. una presentación `.pptx` sobre una plantilla universitaria, y
2. un quiz de opción múltiple con exportación a JSON y a PDF imprimible.

Ni el PDF ni el texto extraído salen nunca de la máquina del usuario.

---

## Tabla de contenidos

1. [Principios de diseño](#1-principios-de-diseño)
2. [Arquitectura del pipeline](#2-arquitectura-del-pipeline)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Módulos](#4-módulos)
5. [Instalación y ejecución](#5-instalación-y-ejecución)
6. [Ejecución con Docker](#6-ejecución-con-docker)
7. [Benchmark](#7-benchmark)
8. [Decisiones de diseño relevantes](#8-decisiones-de-diseño-relevantes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Principios de diseño

| Principio | Implicación práctica |
|---|---|
| **Privacidad total** | Cero llamadas a servicios cloud. Inferencia con Ollama local, procesado de PDF con PyMuPDF, generación de PPTX/PDF con `python-pptx` y `reportlab`. |
| **Anclaje al documento** | El LLM nunca genera contenido "de memoria". Todo pasa por una *Knowledge Base* (KB) intermedia construida desde el PDF. |
| **Calidad sobre cantidad** | Mejor un quiz de 8 preguntas bien hechas que uno de 20 mediocres. Un revisor determinista descarta o regenera lo dudoso. |
| **Salidas validadas** | Cada respuesta del LLM se valida con Pydantic + JSON Schema (modo *structured outputs* de Ollama). |
| **Fallo recuperable** | Excepciones de dominio explícitas (`src/exceptions.py`); fallbacks deterministas en KB, plan de quiz y plan de slides para modelos pequeños. |

---

## 2. Arquitectura del pipeline

```
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│   PDF        │───▶ │ PDF → Markdown │───▶ │ Map-Reduce LLM   │
│  (subido)    │     │  (pymupdf4llm) │     │ + Extractores    │
└──────────────┘     └────────────────┘     └────────┬─────────┘
                                                     ▼
                                          ┌────────────────────┐
                                          │  KnowledgeBase     │  ◀── fuente única
                                          │  (JSON / Pydantic) │      de verdad
                                          └─────────┬──────────┘
                                                    │
                       ┌────────────────────────────┴───────────────────────────┐
                       ▼                                                        ▼
            ┌──────────────────────┐                              ┌──────────────────────┐
            │  SlidePlan (LLM)     │                              │  QuizPlan (LLM)      │
            │  + sanitización      │                              │  + sanitización      │
            └──────────┬───────────┘                              └──────────┬───────────┘
                       ▼                                                     ▼
            ┌──────────────────────┐                              ┌──────────────────────┐
            │ Bullets por slide    │                              │ Pregunta por pregunta│
            │ (1 llamada por slide)│                              │ (1 llamada por preg.)│
            └──────────┬───────────┘                              └──────────┬───────────┘
                       ▼                                                     ▼
            ┌──────────────────────┐                              ┌──────────────────────┐
            │ Revisor / refinado   │                              │ Revisor / refinado   │
            │ (determinista)       │                              │ (determinista)       │
            └──────────┬───────────┘                              └──────────┬───────────┘
                       ▼                                                     ▼
            ┌──────────────────────┐                              ┌──────────────────────┐
            │ Render PPTX          │                              │ Quiz interactivo     │
            │ (python-pptx)        │                              │ + PDF (reportlab)    │
            └──────────────────────┘                              └──────────────────────┘
```

### Etapas

| # | Fase | Entrada | Salida | Módulo |
|---|---|---|---|---|
| 1 | Extracción PDF | bytes | `ProcessedPDF` (Markdown enriquecido) | `src/pdf_processor.py` |
| 2 | Chunking por headings | Markdown | `list[Chunk]` | `src/map_reduce.py` |
| 3 | Hints literales | Markdown | `LiteralHints` | `src/extractors.py` |
| 4 | MAP (resumen por chunk) | `Chunk` | resumen parcial | `src/map_reduce.py` |
| 5 | REDUCE → KB | parciales + hints | `KnowledgeBase` | `src/map_reduce.py` |
| 6a | Plan de slides | KB | `SlidePlan` | `src/pptx_generator.py` + `src/plans.py` |
| 6b | Plan de quiz | KB | `QuizPlan` | `src/quiz_generator.py` + `src/plans.py` |
| 7a | Redacción de bullets | `PlannedSlide` + KB | `BuiltSlide` | `src/pptx_generator.py` |
| 7b | Redacción de preguntas | `PlannedQuestion` + KB | `QuizQuestion` | `src/quiz_generator.py` |
| 8 | Revisor + refinamiento | artefacto + KB | artefacto pulido | `src/critics.py` |
| 9a | Render PPTX | `PresentationPlan` + plantilla | bytes `.pptx` | `src/pptx_generator.py` |
| 9b | Export Quiz PDF | `Quiz` | bytes `.pdf` | `src/quiz_pdf_exporter.py` |
| 10 | UI + descarga | artefactos | navegador | `app.py` |

### Por qué este diseño y no un único prompt

1. **Ventana de contexto**: los modelos locales de 7-9 B tienen 4–8 k tokens útiles; un PDF de 50 páginas los desborda.
2. **Coherencia global**: una sola pasada tiende a repetir conceptos, olvidar otros y mezclar tablas.
3. **Auditoría**: con contenido intermedio estructurado (KB, planes) se puede inspeccionar y regenerar partes concretas.
4. **Reutilización**: la KB se construye una vez y sirve a la vez para quiz y para slides.
5. **Depuración**: cuando algo falla, se sabe exactamente en qué etapa.

---

## 3. Estructura del repositorio

```
tfg-local/
├── app.py                          # UI Streamlit (entrada principal)
├── plantilla_universidad.pptx      # Plantilla institucional (layouts 0 y 2)
├── requirements.txt                # Dependencias con versiones mínimas
├── Dockerfile                      # Imagen de la app (sin Ollama)
├── docker-compose.yml              # Orquestación local
├── .dockerignore                   # Filtros de build
├── .streamlit/config.toml          # Tema y maxUploadSize
├── src/                            # Núcleo del pipeline (ver §4)
└── benchmark/                      # Evaluación reproducible (ver §7)
```

`src/`:

| Archivo | Propósito |
|---|---|
| `config.py` | Constantes globales (modelos, límites, layouts, `OLLAMA_BASE_URL`). |
| `exceptions.py` | Jerarquía de excepciones de dominio. |
| `ollama_client.py` | Wrapper de Ollama con preflight y reintentos. |
| `pdf_processor.py` | PDF → Markdown, detección de escaneos y tablas, contexto visual. |
| `extractors.py` | Heurísticas literales (definiciones, código, fórmulas, términos). |
| `knowledge_base.py` | Modelo Pydantic de la KB + coerción defensiva del JSON del LLM. |
| `map_reduce.py` | Chunking + fase MAP + REDUCE → KB + anclaje literal. |
| `plans.py` | `SlidePlan` / `QuizPlan` + saneadores + fallbacks deterministas. |
| `prompts.py` | Prompts del LLM (centralizados, en español). |
| `quiz_generator.py` | Plan + redacción 1-a-1 + fallback determinista del quiz. |
| `pptx_generator.py` | Plan + redacción 1-a-1 + render PPTX sobre plantilla. |
| `critics.py` | Revisor determinista + refinamiento selectivo de quiz y slides. |
| `quiz_pdf_exporter.py` | Export del quiz a PDF imprimible (ReportLab). |

---

## 4. Módulos

### 4.1. `src/config.py`

Constantes `Final` (`AVAILABLE_MODELS`, `DEFAULT_MODEL`, `LLM_TEMPERATURE`, `NUM_CTX`, `CHUNK_SIZE_CHARS`, `MAX_INPUT_CHARS`, `MAX_BULLETS_PER_SLIDE`, `MAX_CHARS_PER_BULLET`, `LAYOUT_TITLE`, `LAYOUT_CONTENT`, etc.).

`OLLAMA_BASE_URL` se lee de la variable de entorno homónima si está definida (por defecto `http://localhost:11434`). Esto permite apuntar al host desde Docker.

### 4.2. `src/exceptions.py`

```
AppError
├── PDFError
│   ├── ScannedPDFError
│   └── PDFTooLargeError
├── OllamaError
│   ├── OllamaUnavailableError
│   └── OllamaModelNotFoundError
├── GenerationError
└── TemplateError
```

### 4.3. `src/ollama_client.py`

- `preflight()`: comprueba servidor (`/api/tags`) y modelo instalado por **tag exacto** (`qwen2.5:7b`, no `qwen2.5`).
- `generate(...)` con reintentos exponenciales (`tenacity`) sobre errores HTTP/Ollama.
- `generate_json(...)` con limpieza defensiva (elimina `` ```json `` fences, recorta a `{...}` o `[...]`).
- Acepta el parámetro `schema=...` para usar el modo *structured outputs* de Ollama.

### 4.4. `src/pdf_processor.py`

- Detección de PDFs escaneados (`_detect_scanned`): si <30 % de páginas tienen ≥40 caracteres de texto, lanza `ScannedPDFError`.
- Normalización ligera de tablas GFM (`_normalize_markdown_tables`).
- Extracción de contexto visual (`_extract_image_context`): para cada imagen relevante, anota el bloque de texto más cercano.
- Límites duros: `MAX_INPUT_PAGES`, `MAX_INPUT_CHARS`.

### 4.5. `src/extractors.py`

Heurísticas puras (sin LLM) sobre el Markdown:

- `extract_literal_definitions` (patrones `"X es Y"`, `"Se llama X a Y"`, `**Término**: definición`).
- `extract_code_fences` (bloques ```` ``` ```` con detección de lenguaje).
- `extract_formulas` (líneas con `=` y símbolos matemáticos).
- `extract_key_terms` (negritas + títulos relevantes).

Todo se agrupa en `LiteralHints`, que se inyecta como material de anclaje en el prompt REDUCE.

### 4.6. `src/knowledge_base.py`

Modelos Pydantic de los átomos:

| Tipo | Prefijo id | Campos |
|---|---|---|
| `Definition` | `def:*` | `term`, `definition`, `subtopic`, `verbatim` |
| `Example` | `ex:*` | `name`, `description`, `attributes[]`, `methods[]`, `subtopic` |
| `FormulaOrCode` | `fc:*` | `kind` (`formula`/`code`), `content`, `caption`, `language` |
| `NumericDatum` | `dt:*` | `value`, `description` |
| `Relation` | `rel:*` | `kind`, `source`, `target`, `description` |

`KnowledgeBase` agrega los átomos, expone `atom_count`, `atom_ids()`, `get_atom(id)`, `atoms_by_subtopic()`, `to_markdown()` y `to_prompt_context(max_chars)`.

`coerce_kb_payload(...)` normaliza la respuesta del LLM: tolera alias ES/EN (`tema`/`main_topic`, `definiciones`/`definitions`…), un nivel de envoltura (`{"kb": {...}}`), placeholders del propio prompt y átomos con campos parciales.

### 4.7. `src/map_reduce.py`

- `split_markdown(md)`: divide por cabeceras Markdown y empaqueta en chunks ≤ `CHUNK_SIZE_CHARS` con solapamiento controlado.
- `_map_phase`: una llamada por chunk con `MAP_SUMMARY_PROMPT`.
- `_reduce_to_kb`: consolida parciales + `LiteralHints` en JSON validado contra `KnowledgeBase`. Tres capas de resiliencia: coerción defensiva → reintento a temperatura más baja → fallback determinista (`build_fallback_kb`).
- Tras construir la KB se aplican dos pasadas anti-alucinación:
  - `_prune_ungrounded_relations`: descarta relaciones cuya `source` o `target` no aparecen literalmente en el Markdown.
  - `_prune_ungrounded_examples`: ídem para `Example`, filtrando atributos/métodos inexistentes.
- `_enrich_kb_with_literal_hints`: fusiona definiciones y bloques de código literales detectados que el LLM hubiera omitido.
- API pública: `build_knowledge_base(client, markdown, progress_cb=...)`.

### 4.8. `src/plans.py`

- `SlidePlan` con `kind ∈ {intro, definition, example, comparison, code, process, relations, outlook, conclusion}`.
- `QuizPlan` con `bloom_level` y `kind` cerrados (Bloom: recordar/comprender/aplicar/analizar/evaluar/crear).
- `BLOOM_RECOMMENDED_KINDS`: mapa orientativo Bloom → kinds.
- JSON Schemas (`SLIDE_PLAN_JSON_SCHEMA`, `QUIZ_PLAN_JSON_SCHEMA`) para forzar salida estructurada en Ollama.
- `coerce_slide_plan_payload` / `coerce_quiz_plan_payload`: aceptan alias ES/EN, un nivel de envoltura y entradas parciales.
- `resolve_atom_id`: reconciliación de ids con normalización de acentos/plurales y *prefix matching* dentro del mismo tipo.
- `sanitize_slide_plan` / `sanitize_quiz_plan`: filtran ids inválidos, deduplican, fuerzan distribución Bloom razonable.
- `build_fallback_slide_plan` / `build_fallback_quiz_plan`: planes deterministas desde la KB cuando el LLM produce JSON irrecuperable.

### 4.9. `src/quiz_generator.py`

Pipeline en dos pasos:

1. `plan_quiz(client, kb, num_questions)` con *structured outputs* y reintento a temperatura baja; si nada cuaja, fallback determinista.
2. `generate_single_question(client, kb, planned, previous)`: una llamada por pregunta con:
   - el átomo central renderizado (`_atom_markdown`),
   - contexto relacionado del mismo subtopic para distractores plausibles,
   - resumen de las últimas preguntas para evitar repeticiones.

Si el LLM falla en una pregunta concreta, `_build_deterministic_question` produce una pregunta mínima basada en el átomo. Tras el flujo principal, si quedan menos preguntas que `MIN_NUM_QUESTIONS`, se rellena con fallbacks (`_fill_min_questions_with_fallback`).

`refine=True` (por defecto) invoca `critics.refine_quiz` para regenerar las preguntas con issues `high`.

### 4.10. `src/pptx_generator.py`

- `plan_slides`: pide `SlidePlan` con structured outputs; sanea, garantiza `intro` al inicio y recorta a `[DEFAULT_NUM_SLIDES_MIN, DEFAULT_NUM_SLIDES_MAX]`.
- `render_slide_bullets`: una llamada por slide con `SLIDE_BULLETS_FROM_ATOMS_PROMPT`, incluyendo átomos asignados, índice global y bullets ya aceptados (para evitar repetición). Reintenta a temperatura más baja si la primera pasada no produce ≥3 bullets. Como último recurso, `_fallback_bullets_from_atoms` construye bullets sin LLM a partir de los átomos.
- `_clean_bullet`: descarta bullets con `…`, prefijos de átomo (`def:`, `ex:`…), aperturas meta ("en esta diapositiva", "se habla de"…), longitud insuficiente o sin cierre natural.
- `_balance_bullet_density`: equilibra longitud y deduplica.
- `render_pptx`: portada (layout 0), índice generado dinámicamente, una slide por `BuiltSlide` con bullets, y slide final de conclusiones si existe.
- `generate_presentation`: KB → plan → bullets → refinamiento → conclusión → render PPTX.

### 4.11. `src/critics.py`

Revisor determinista deliberadamente pequeño:

| Quiz | Slides |
|---|---|
| `banned_phrase` (high), `duplicate` (high), `unbalanced_options` (low), `meta_language` (low), `not_grounded` (low) | `meta_language` (low), `duplicate_content` (low), `not_grounded` (low) |

- `_strip_meta_preamble`: recorte barato por regex de muletillas ("Según el texto, …").
- `refine_quiz`: un único pase. Para issues `high`, regenera con la misma `PlannedQuestion` (rota `concept_id` si el problema es `duplicate`). Si tras regenerar sigue siendo crítica, se descarta. Luego recorta a `max_questions` priorizando las preguntas con menos issues.
- `refine_slides`: regenera solo slides con issues `high`.

### 4.12. `src/quiz_pdf_exporter.py`

ReportLab + `SimpleDocTemplate` A4 con márgenes 2 cm. Dos secciones: hoja de examen y hoja de soluciones (respuesta correcta, nivel Bloom y justificación). Estilos en tonos universitarios.

### 4.13. `app.py` (UI Streamlit)

Secciones con estado en `st.session_state`:

- `render_sidebar`: selector de modelo + rango fijo de preguntas + aviso si falta la plantilla.
- `render_upload_section`: uploader con reset al cambiar de PDF, muestra páginas/caracteres/imágenes/tablas y un *expander* con los primeros 2 000 caracteres del Markdown.
- `render_generate_section`: dos columnas (Quiz / Presentación). Cada una hace `preflight` de Ollama, asegura KB y dispara el pipeline correspondiente.
- `render_quiz_results`: quiz interactivo con `st.form`, corrección y descargas (JSON y PDF).
- `render_pptx_results`: previsualización del índice + bullets y descarga `.pptx`.

Sanitización del nombre de descarga (`_sanitize_for_filename`) para evitar caracteres inválidos en Windows/macOS/Linux (por ejemplo, el `:` de los tags de Ollama).

---

## 5. Instalación y ejecución

### 5.1. Requisitos

- Python 3.11+.
- [Ollama](https://ollama.com/download) instalado y corriendo.
- Al menos un modelo descargado (por defecto, `qwen2.5:14b`). Modelos soportados en la UI: `qwen2.5:14b`, `gemma3:12b`, `qwen2.5:7b`, `gemma2:9b`, `gemma3:4b`.
- GPU NVIDIA con ≥6 GB de VRAM recomendable.

### 5.2. Instalación

```powershell
cd tfg-local
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.3. Arranque

```powershell
# En una terminal independiente, asegúrate de que Ollama corre:
ollama serve

# Lanza la app:
streamlit run app.py
```

Abre `http://localhost:8501` y sigue el flujo: subir PDF → generar quiz y/o presentación → descargar.

---

## 6. Ejecución con Docker

La imagen contiene **solo** la app Streamlit y sus dependencias de procesamiento; **Ollama queda en el host**. La razón: los modelos LLM pueden ocupar varias decenas de GB y conviene gestionarlos con `ollama pull` en la máquina, no dentro del contenedor.

```powershell
# 1. Ollama corriendo en el host
ollama serve

# 2. Build y arranque del contenedor
docker compose build
docker compose up -d

# 3. Abrir la app
# http://localhost:8501

# 4. Parar
docker compose down
```

`docker-compose.yml` inyecta `OLLAMA_BASE_URL=http://host.docker.internal:11434` y mapea `host.docker.internal:host-gateway` para que funcione también en Linux.

Para usar una URL distinta (otro host, otra red), basta con sobreescribir la variable:

```powershell
$env:OLLAMA_BASE_URL = "http://192.168.1.20:11434"
docker compose up -d
```

---

## 7. Benchmark

El módulo `benchmark/` evalúa de forma reproducible la calidad del sistema con varios modelos y PDFs. Ver `benchmark/README.md` para el detalle.

Resumen:

- `python -m benchmark.runner` ejecuta combinaciones (PDF × modelo) y vuelca artefactos + métricas en `benchmark/results/`.
- `python -m benchmark.runner --regen-reports` regenera los CSV agregados desde los `metrics.json` existentes (útil si Excel tenía bloqueado el fichero).
- Reports en `benchmark/reports/`:
  - `benchmark_summary.csv` (una fila por ejecución, política UPSERT por `(pdf_id, model)`),
  - `model_averages.csv` (medias por modelo),
  - `manual_evaluation_template.csv` (plantilla para puntuar a mano una muestra),
  - `rubric_reference.txt` (rúbrica 1–5 para los evaluadores).

Tres niveles de evaluación:

1. **Automática**: métricas de `benchmark/metrics.py` (Bloom, duplicados, bullets/slide, coherencia índice–slides…). Sin nota agregada.
2. **Semiautomática (IA externa)**: por cada ejecución se generan dos `eval_prompts/*.txt` que se pegan en ChatGPT / Gemini junto al PDF original; la IA devuelve un JSON con rúbrica 1–5.
3. **Manual**: `manual_evaluation_template.csv` para validar cualitativamente una muestra reducida.

---

## 8. Decisiones de diseño relevantes

- **KnowledgeBase como contrato único**: separar "entender el PDF" de "generar artefactos". Permite reutilizar la KB para quiz y slides sin recalcular, auditarla manualmente (`kb.json` se guarda en cada ejecución del benchmark) y testear partes aisladas con datos sintéticos.
- **`atom_id` con prefijo tipado** (`def:`, `ex:`, `fc:`, `dt:`, `rel:`): toda la planificación referencia átomos por id; el saneado puede recuperar ids levemente erróneos del LLM (acentos, plurales) sin romper la pipeline.
- **Anclaje literal del REDUCE** (`_enrich_kb_with_literal_hints`): aunque el LLM produzca una KB pobre, se mantiene un suelo mínimo de átomos verificables extraídos por regex desde el Markdown.
- **Anti-alucinación** (`_prune_ungrounded_*`): las `Relation` y los `Example` cuyas entidades no aparecen literalmente en el PDF se descartan. Imprescindible con modelos grandes que tienden a inventar jerarquías.
- **Refinamiento selectivo**: el revisor solo regenera lo crítico (severity `high`). Un pase completo costaría tantas llamadas como elementos; este enfoque cuesta solo *k+1*.
- **Fallbacks deterministas**: cuando los modelos pequeños (<7 B) devuelven JSON inválido sistemático, la pipeline degrada con grácil a un quiz/PPTX construido sin LLM a partir de los átomos. El sistema nunca devuelve "vacío".
- **Sin score automático agregado**: las métricas automáticas son recuentos y heurísticas; la nota final 1–5 se obtiene con IA externa (rúbrica) y/o manualmente. Evita el espejismo de un "número mágico" de calidad.

---

## 9. Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| `model 'qwen2.5' not found (status 404)` | Falta el tag completo. Usar siempre `qwen2.5:7b`, `gemma2:9b`, etc. |
| `Error: listen tcp 127.0.0.1:11434: bind` | Otro `ollama.exe` ya en marcha. Cerrarlo desde la bandeja y reiniciar. |
| `ollama ps` muestra `100% CPU` | Forzar `$env:OLLAMA_LLM_LIBRARY="cuda_v12"` antes de `ollama serve`. |
| `ScannedPDFError` | El PDF no tiene capa de texto. Pre-procesarlo con `ocrmypdf`. |
| `KB válida pero sin átomos` | El LLM no extrajo nada utilizable del documento. Probar otro modelo o revisar el contenido del PDF. |
| `PermissionError` al regenerar reportes del benchmark | El CSV está abierto en Excel. Cerrarlo y volver a ejecutar `python -m benchmark.runner --regen-reports`. |
