# Generador offline de Quiz + Presentación desde PDF

> **Resumen en una línea**: una aplicación **100 % local** (sin nube) que toma un PDF académico y genera, con IA local vía **Ollama**, (1) una presentación `.pptx` con plantilla universitaria y (2) un Quiz de opción múltiple interactivo con exportación a JSON y PDF imprimible.

Este documento es el **único punto de verdad** del proyecto. Cubre:

- Qué hace la aplicación y por qué existe.
- La arquitectura completa y las decisiones de diseño tomadas.
- Cada archivo, cada función pública y el flujo de datos entre ellos.
- El historial de mejoras implementadas y las razones detrás de cada una.
- Cómo instalar, ejecutar, depurar y extender el sistema.

> **Contrato de mantenimiento**: *toda* modificación futura del proyecto debe quedar reflejada en este README (nueva funcionalidad, cambio de decisión, nueva dependencia, nueva mejora). La sección [Historial de mejoras](#9-historial-de-mejoras-changelog-técnico) es un changelog técnico ordenado cronológicamente que se amplía con cada iteración.

---

## Tabla de contenidos

1. [Motivación y principios de diseño](#1-motivación-y-principios-de-diseño)
2. [Vista rápida del sistema](#2-vista-rápida-del-sistema)
3. [Arquitectura del pipeline](#3-arquitectura-del-pipeline)
4. [Estructura de archivos](#4-estructura-de-archivos)
5. [Módulos y API interna](#5-módulos-y-api-interna)
6. [Modelos de datos y contratos](#6-modelos-de-datos-y-contratos)
7. [Prompts](#7-prompts)
8. [Instalación y ejecución](#8-instalación-y-ejecución)
9. [Historial de mejoras (changelog técnico)](#9-historial-de-mejoras-changelog-técnico)
10. [Decisiones de diseño y *por qué* no se hizo de otra forma](#10-decisiones-de-diseño-y-por-qué-no-se-hizo-de-otra-forma)
11. [Roadmap pendiente](#11-roadmap-pendiente)
12. [Convenciones de contribución y del README](#12-convenciones-de-contribución-y-del-readme)

---

## 1. Motivación y principios de diseño

El objetivo es cubrir un caso de uso académico muy concreto: a partir de apuntes o papers en PDF, obtener **material docente listo para usar** (una presentación profesional y un quiz de autoevaluación) sin que la información salga nunca de la máquina del usuario.

Principios innegociables del proyecto:

| Principio | Implicación práctica |
|---|---|
| **Privacidad total** | Cero llamadas a servicios cloud. Inferencia con Ollama local, procesado de PDF con PyMuPDF local, generación de PPTX/PDF local. |
| **Reproducibilidad** | Temperaturas bajas (0.2–0.4), salidas validadas con Pydantic, semillas de diseño (distribución Bloom por rangos) para minimizar varianza. |
| **Calidad > cantidad** | Preferimos 8 preguntas excelentes a 20 mediocres. El revisor regenera lo malo en vez de inundar con más contenido. |
| **Anclaje al documento** | El LLM nunca genera contenido “de memoria”. Todo pasa por una KnowledgeBase intermedia que actúa de fuente única de verdad. |
| **Código modular y tipado** | Pydantic + `Literal` para vocabularios cerrados, `dataclass` para estructuras internas, `from __future__ import annotations` en todos los módulos. |
| **Fallo explícito y recuperable** | Jerarquía propia de excepciones (`src/exceptions.py`), reintentos con backoff, degradación controlada (p. ej. regeneración selectiva de preguntas problemáticas). |

---

## 2. Vista rápida del sistema

```
┌──────────────┐     ┌────────────────┐     ┌────────────────┐
│   PDF (↑UI)  │───▶ │ PDF→Markdown   │───▶ │ Map-Reduce LLM │
└──────────────┘     │ (pymupdf4llm)  │     │   + Extractores│
                     └────────────────┘     └───────┬────────┘
                                                    │
                                                    ▼
                                        ┌──────────────────────┐
                                        │  KnowledgeBase (JSON)│◀──── fuente única
                                        └────────┬─────────────┘      de verdad
                                                 │
                    ┌────────────────────────────┴───────────────────────────┐
                    ▼                                                        ▼
        ┌──────────────────────┐                                 ┌─────────────────────┐
        │  SlidePlan (LLM)     │                                 │  QuizPlan (LLM)     │
        │  + sanitización      │                                 │  + sanitización     │
        └──────┬───────────────┘                                 └──────┬──────────────┘
               │                                                        │
               ▼                                                        ▼
        ┌──────────────────────┐                                 ┌─────────────────────┐
        │ Bullets por slide    │                                 │ Pregunta por pregunta│
        │ (1 llamada por slide)│                                 │ (1 llamada por preg.)│
        └──────┬───────────────┘                                 └──────┬──────────────┘
               │                                                        │
               ▼                                                        ▼
        ┌──────────────────────┐                                 ┌─────────────────────┐
        │ Revisor slides       │   <──── refinamiento  ──>       │ Revisor quiz        │
        │ (determinista + LLM) │                                 │ (determinista + LLM)│
        └──────┬───────────────┘                                 └──────┬──────────────┘
               │                                                        │
               ▼                                                        ▼
        ┌──────────────────────┐                                 ┌─────────────────────┐
        │ Render PPTX          │                                 │ Quiz interactivo    │
        │ (python-pptx)        │                                 │ + PDF (reportlab)   │
        └──────────────────────┘                                 └─────────────────────┘
```

**Tres pasos mentales para el usuario**:

1. Subir PDF → el backend lo convierte a Markdown y lo resume a una KnowledgeBase estructurada.
2. Pulsar *Generar Quiz* y/o *Generar Presentación* → se ejecuta el pipeline correspondiente.
3. Previsualizar, responder el quiz en la propia UI y descargar artefactos (`.pptx`, `.json`, `.pdf`).

---

## 3. Arquitectura del pipeline

El pipeline está diseñado como una **cadena de contratos Pydantic**. Cada etapa tiene *input* y *output* formalmente validados, lo que permite:

- Detectar drift del LLM al instante y reintentar solo el paso fallido.
- Testar cualquier etapa de forma aislada con datos sintéticos.
- Sustituir el LLM por otro sin tocar el resto del código.

### 3.1. Fases del pipeline (orden cronológico)

| # | Fase | Entrada | Salida | Módulo |
|---|---|---|---|---|
| 1 | Extracción PDF | bytes | `ProcessedPDF` (Markdown enriquecido) | `src/pdf_processor.py` |
| 2 | Chunking por headings | Markdown | `list[Chunk]` | `src/map_reduce.py` |
| 3 | Extractores literales | Markdown | `LiteralHints` | `src/extractors.py` |
| 4 | **MAP** (resumen por chunk) | `Chunk` | resumen parcial en texto | `src/map_reduce.py` |
| 5 | **REDUCE → KB** | resúmenes parciales + hints | `KnowledgeBase` (JSON) | `src/map_reduce.py` + `src/knowledge_base.py` |
| 6a | Plan de slides | `KnowledgeBase` | `SlidePlan` | `src/pptx_generator.py` + `src/plans.py` |
| 6b | Plan de quiz | `KnowledgeBase` | `QuizPlan` | `src/quiz_generator.py` + `src/plans.py` |
| 7a | Redacción de bullets | `PlannedSlide` + KB | `BuiltSlide` | `src/pptx_generator.py` |
| 7b | Redacción de preguntas | `PlannedQuestion` + KB | `QuizQuestion` | `src/quiz_generator.py` |
| 8 | Revisor + refinamiento | artefacto + KB | artefacto pulido | `src/critics.py` |
| 9a | Render PPTX | `PresentationPlan` + plantilla | bytes `.pptx` | `src/pptx_generator.py` |
| 9b | Render Quiz PDF | `Quiz` | bytes `.pdf` | `src/quiz_pdf_exporter.py` |
| 10 | UI + descarga | artefactos | navegador | `app.py` |

### 3.2. Por qué este diseño y no un único prompt

El enfoque *naïve* sería: “pon todo el PDF en un prompt enorme y pide 10 preguntas y una presentación”. Rechazamos este enfoque por cinco razones:

1. **Context window**: modelos locales de 7-9 B tienen 4k-8k tokens útiles. Un PDF de 50 páginas los sobrepasa.
2. **Coherencia global**: una sola pasada tiende a repetir conceptos, olvidar otros y mezclar tablas.
3. **Control de calidad**: con contenido intermedio estructurado (KB, planes) podemos *auditar* y *regenerar* partes concretas.
4. **Reutilización**: la KB sirve a la vez para quiz y para slides; no se recalcula.
5. **Depuración**: cuando algo falla, se ve exactamente en qué etapa.

---

## 4. Estructura de archivos

```
tfg-local/
├── app.py                          # UI Streamlit (entrada principal)
├── plantilla_universidad.pptx      # Plantilla corporativa con layouts 0 (portada) y 2 (contenido)
├── requirements.txt                # Dependencias con versiones mínimas
├── README.md                       # Este documento
├── .streamlit/
│   └── config.toml                 # Tema y maxUploadSize (50 MB)
└── src/
    ├── __init__.py                 # Metadatos del paquete (__version__)
    ├── config.py                   # Constantes globales (modelos, límites, layouts…)
    ├── exceptions.py               # Jerarquía de excepciones de dominio
    ├── ollama_client.py            # Wrapper de Ollama con reintentos y preflight
    ├── pdf_processor.py            # PDF → Markdown + detección escaneos y tablas
    ├── extractors.py               # Heurísticas literales (defs, código, fórmulas, términos)
    ├── knowledge_base.py           # Pydantic de la KB canónica del documento
    ├── map_reduce.py               # Chunking + fase MAP + fase REDUCE → KB
    ├── plans.py                    # SlidePlan / QuizPlan + saneadores con fuzzy matching
    ├── prompts.py                  # Prompts LLM centralizados (todos en español)
    ├── quiz_generator.py           # Pipeline del quiz (plan + 1 llamada por pregunta)
    ├── pptx_generator.py           # Pipeline de slides (plan + 1 llamada por slide + render)
    ├── critics.py                  # Revisor determinista + LLM + refinamiento selectivo
    └── quiz_pdf_exporter.py        # Export del quiz a PDF imprimible (reportlab)
```

---

## 5. Módulos y API interna

A continuación, una descripción por módulo de **todo lo público** que expone y **el porqué** de cada decisión. Está ordenado igual que el pipeline.

### 5.1. `src/config.py`

Constantes inmutables (`Final`) que controlan el sistema entero. Separar configuración del código evita magia dispersa.

| Constante | Valor | Propósito |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint Ollama. |
| `AVAILABLE_MODELS` | tupla de tags completos | Los modelos que el usuario puede elegir en la UI. |
| `DEFAULT_MODEL` | `qwen2.5:7b` | Compromiso calidad/velocidad para GPU de 8-12 GB VRAM. |
| `LLM_TEMPERATURE` | `0.2` | Baja por defecto para estabilidad. La redacción de preguntas sube a 0.4 para un poco de creatividad. |
| `NUM_CTX` | `8192` | Ventana usable en 8-12 GB VRAM. |
| `CHUNK_SIZE_CHARS` | `3000` | ~750 tokens por chunk, deja margen al prompt. |
| `CHUNK_OVERLAP_CHARS` | `250` | Evita cortar un concepto entre dos chunks. |
| `MAX_INPUT_PAGES` / `MAX_INPUT_CHARS` | `50` / `50_000` | Límite duro anti-pdf-bomba. |
| `MIN_NUM_QUESTIONS` / `MAX_NUM_QUESTIONS` | `5` / `15` | Rango expuesto en la UI (slider min–max). |
| `DEFAULT_NUM_QUESTIONS_RANGE` | `(5, 12)` | Valor inicial del slider. |
| `QUIZ_PLAN_OVERSAMPLING` | `3` | Preguntas extra pedidas al plan para absorber descartes. |
| `LAYOUT_TITLE` / `LAYOUT_CONTENT` | `0` / `2` | Índices verificados contra `plantilla_universidad.pptx`. El layout 1 (section header) **no** tiene body y no sirve para contenido. |
| `MAX_BULLETS_PER_SLIDE` | `5` | Regla pedagógica clásica. |
| `MAX_CHARS_PER_BULLET` | `180` | ~2 líneas a 20 pt en esta plantilla. Es el umbral que se comunica al LLM y el que usa `_trim_by_words`. |
| `MAX_CHARS_SLIDE_TITLE` | `80` | Se recorta duro con `…` si se excede. |
| `DEFAULT_NUM_SLIDES_MIN/MAX` | `6` / `14` | Rango razonable para una clase de ~20 min. |

### 5.2. `src/exceptions.py`

Jerarquía propia que permite a la UI (`app.py`) discriminar el tipo de fallo y mostrar mensajes accionables.

```
AppError
├── PDFError
│   ├── ScannedPDFError
│   └── PDFTooLargeError
├── OllamaError
│   ├── OllamaUnavailableError
│   └── OllamaModelNotFoundError
├── GenerationError       # error de contenido (plan vacío, JSON inválido…)
└── TemplateError         # plantilla .pptx inexistente o corrupta
```

### 5.3. `src/ollama_client.py`

Wrapper sobre `ollama-python` con:

- `preflight()` → verifica servidor (`/api/tags`) y modelo instalado. **Match exacto por tag** (`qwen2.5:7b`, no `qwen2.5`) porque `/api/generate` exige el nombre completo; un match laxo pasaría el preflight y luego fallaría con 404.
- `generate(prompt, system, json_mode, temperature)` con reintentos exponenciales (3 intentos) vía `tenacity` sobre errores de red/respuesta.
- `generate_json(...)` con limpieza defensiva: elimina ```` ```json ```` fences, recorta al primer `{`/`[` y al último `}`/`]`, reporta JSONDecodeError con los primeros 400 caracteres de la respuesta para diagnóstico.
- Errores de infraestructura se mapean a `OllamaUnavailableError` / `OllamaModelNotFoundError`.

### 5.4. `src/pdf_processor.py`

Conversión PDF → Markdown con PyMuPDF y pymupdf4llm. Aporta:

- **Detección de escaneos** (`_detect_scanned`): si menos del 30 % de páginas tienen ≥40 caracteres extraíbles, se lanza `ScannedPDFError` con sugerencia de OCR.
- **Normalización ligera de tablas** (`_normalize_markdown_tables`): limpia tablas GFM para hacerlas más legibles por LLM (normaliza `<br>` a ` / ` y recompone líneas partidas).
- **Extracción de contexto visual** (`_extract_image_context`): para cada imagen suficientemente grande en la página se busca el bloque de texto más cercano verticalmente como caption; se añade al final del Markdown bajo `## Contexto visual detectado` con formato `[p.X, WxH] texto`.
- **Filtro de ruido visual**: descarta imágenes muy pequeñas (logos/iconos) por ratio de área relativa de página para reducir falsos positivos.
- **Detección de tablas** (`_looks_like_tables`): busca líneas separadoras GFM (`---|---`) en el markdown resultante.
- **Límites duros**: páginas (`MAX_INPUT_PAGES`) y caracteres (`MAX_INPUT_CHARS`).

Salida: `ProcessedPDF(markdown, num_pages, num_chars, num_images, has_tables)`.

### 5.5. `src/extractors.py`

Heurísticas puras (sin LLM) para localizar **material literal** del documento. Se ejecutan antes del REDUCE y se inyectan como material de anclaje:

- `extract_literal_definitions`: tres patrones regex (`"X es Y"`, `"Se llama/define X a Y"`, `"**Término**: definición"`). Filtra sintagmas término de >6 palabras y definiciones de <3 palabras.
- `extract_code_fences`: bloques ``` con lenguaje si se declaró. Límite 1500 chars por bloque, máx. 10.
- `extract_formulas`: líneas con `=` y al menos un símbolo matemático (`[\d\+\-\*/\^±√π∑∏∞≤≥≠∈∉⊂⊃∪∩∂Δ∇α-ω]`). Descarta `key=value` de >18 palabras.
- `extract_key_terms`: negritas Markdown + títulos; excluye “Contexto de imágenes”, “Índice”, “Indice”.

Todo va a `LiteralHints` con un `to_prompt_block()` que serializa de forma segura (corta por línea si se pasa de `max_chars=3500`).

### 5.6. `src/knowledge_base.py`

Define los **contratos Pydantic** de la KB. Es el nudo del proyecto: una vez hay una KB válida, el resto es determinista.

Tipos de átomo:

| Clase | Prefijo id | Campos destacados |
|---|---|---|
| `Definition` | `def:*` | `term`, `definition`, `subtopic`, `verbatim` (si es cita literal del extractor). |
| `Example` | `ex:*` | `name`, `description`, `attributes[]`, `methods[]`, `subtopic`. |
| `FormulaOrCode` | `fc:*` | `kind` (`formula`/`code`), `content`, `caption`, `language`. |
| `NumericDatum` | `dt:*` | `value` (como string, p. ej. `"42%"`), `description`. |
| `Relation` | `rel:*` | `kind` (libre), `source`, `target`, `description`. |

**Normalización de ids** (`_normalize_id`, aplicado en `@field_validator(..., mode="before")`):

- Extrae el prefijo correcto aunque el LLM devuelva otro.
- Descompone acentos (`NFKD`), pasa a minúsculas, `ñ → n`.
- Sustituye cualquier carácter no `[a-z0-9_-]` por `_`.
- Recorta a 40 caracteres.
- Evita ids vacíos (`"item"` de fallback).

Esta normalización se aplica **antes** de que Pydantic compruebe el `pattern`, de forma que el esquema final puede permanecer estricto sin rechazar ids con tildes.

Métodos útiles de `KnowledgeBase`:

- `atom_count`, `atom_ids()`, `get_atom(id)`, `atoms_by_subtopic()`.
- `to_markdown()` → render humano.
- `to_prompt_context(max_chars=…)` → serialización segura para prompts downstream (trunca de forma estable).
- `slugify_id(prefix, text)` → utilidad libre para construir ids estables en tests o fallbacks.

### 5.7. `src/map_reduce.py`

Implementa el pipeline **MAP → REDUCE → KB**:

- `split_markdown(md)`: divide por cabeceras (`#`…`######`) preservando el encabezado, después empaqueta secciones en chunks `<= CHUNK_SIZE_CHARS`. Si una sección supera el tamaño, parte por párrafos; si un párrafo es aún mayor (p. ej. una tabla gigante), trocea por longitud. Finalmente añade `CHUNK_OVERLAP_CHARS` de cola del chunk anterior al siguiente.
- `_map_phase`: una llamada LLM por chunk con `MAP_SUMMARY_PROMPT`. Emite progreso vía `ProgressCallback`.
- `_reduce_to_kb`: consolida los resúmenes + `LiteralHints` con `REDUCE_TO_KB_PROMPT` en **modo JSON**. Construye `KnowledgeBase(**raw)`; si Pydantic falla, se eleva `GenerationError` con los 2 primeros errores para diagnóstico.
- `build_knowledge_base(client, markdown, progress_cb=...)` — **API pública**: orquesta chunking + hints literales + map + reduce.
- `consolidate_document(...)`: *shim* de compatibilidad que devuelve la KB serializada como Markdown. Mantenido por si algún consumidor legacy aún espera texto plano.

### 5.8. `src/plans.py`

Contratos intermedios entre la KB y la generación final. Son los que evitan que el LLM haga “todo en un shot”.

#### 5.8.1. SlidePlan

- `SlideKind = Literal["intro", "definition", "example", "comparison", "code", "process", "relations", "outlook", "conclusion"]`.
- `PlannedSlide(title, kind, atom_ids, focus)`. `atom_ids` deduplica automáticamente.
- `SlidePlan(presentation_title, slides)` con 3-20 slides.

#### 5.8.2. QuizPlan

- `BloomLevel = Literal["recordar", "comprender", "aplicar", "analizar", "evaluar", "crear"]`.
- `QuestionKind = Literal["definicion", "diferenciacion", "caso_practico", "comparacion", "analisis_consecuencia", "juicio_alternativas", "completar_codigo"]`.
- `BLOOM_RECOMMENDED_KINDS`: mapa orientativo Bloom → kinds recomendados (inyectado en el prompt).
- `PlannedQuestion(id, bloom_level, concept_id, kind, focus)`.
- `QuizPlan.questions` reasigna ids 1..N en el validator para que siempre sean consecutivos.

#### 5.8.3. Fuzzy matching de ids (reconciliación LLM ↔ KB)

El LLM, al planificar, escribe a menudo ids levemente distintos a los de la KB: `def:Bicicleta` (mayúscula), `rel:subclase_de` (slug corto frente al `rel:subclase_de_bicicletademontana_bicicleta` real), `ex:farmaceutica` frente a `ex:farmacéutica` (tildes). Para no descartar esas referencias y perder preguntas/slides:

- `_deaccent_lower`, `_normalize_slug` (elimina tildes, colapsa plurales `-s`/`-es` de palabras >3 chars), `_split_id`, `_collapse` (quita `_` y `-`).
- `resolve_atom_id(raw, valid_ids) -> str | None`: cascada de intentos (exacto → normalizado → prefijo `A_` ⊂ `B` → colapsado con inclusión → Jaccard de tokens ≥ 0.6). **Restricción clave**: solo considera candidatos con el **mismo prefijo** (`def:` → `def:`, `rel:` → `rel:`), porque el resto del pipeline asume el tipo del átomo (kind del slide/pregunta).
- `_reconcile_atom_ids(list, valid_ids) -> (resolved, lost)`.

#### 5.8.4. Saneadores

- `sanitize_slide_plan(plan, kb)`:
  - Descarta slides con `kind=conclusion` **o** título normalizado que empieza por `"conclus"` (la conclusión final la añade el renderer de forma determinista → se evita duplicar).
  - Reconcilia `atom_ids` con fuzzy match; registra en log cada id recuperado o perdido.
  - Descarta slides que se queden sin átomos, salvo `intro`/`outlook` (admiten cero átomos por diseño).
  - Si el plan queda vacío, devuelve el original sin cambios (mejor que petar el pipeline entero).
- `sanitize_quiz_plan(plan, kb, target_count)`:
  - Reconcilia `concept_id` con fuzzy match.
  - Deduplica por tupla `(concept_id, bloom_level, kind)`: un mismo concepto puede aparecer varias veces si cambia el nivel o el formato de pregunta (esto es deseable pedagógicamente).
  - Recorta al `target_count`, reasigna ids 1..N.
- `bloom_distribution(plan) -> {BloomLevel: int}`: utilidad de diagnóstico.

### 5.9. `src/quiz_generator.py`

Pipeline en dos pasos, uno a uno, con **retry interno y relleno determinista** si el plan del LLM queda corto.

1. **`plan_quiz(client, kb, num_questions)`** → pide `QuizPlan` con `QUIZ_PLAN_PROMPT`, normaliza alias (`bloom`/`bloom_level`, kinds en inglés), construye Pydantic, sanea.
2. **`generate_single_question(client, kb, planned, previous)`** → usa `QUIZ_QUESTION_PROMPT` con:
   - `concept_detail` renderizado por `_atom_markdown(atom)` (el átomo central).
   - `related_context`: otros átomos del mismo subtopic priorizando `Example` y `Relation` (hasta 8) — son el mejor material para distractores plausibles.
   - `previous_questions`: resumen de las últimas 5 preguntas ya redactadas para evitar solape.
   - **Retry interno**: dos intentos con temperaturas `(0.4, 0.55)`; si ambos fallan (JSON inválido, estructura incompleta) se eleva `GenerationError`.
3. **`_try_generate_question(...)`** → wrapper que traga el `GenerationError` y devuelve `None`; se usa en loops para no romper la generación entera por una pregunta mala.
4. **`_synthesize_planned_questions(kb, used, count, start_id)`** → crea `PlannedQuestion` deterministas sobre átomos aún no preguntados, rotando niveles Bloom del ciclo `(aplicar, analizar, comprender, evaluar, recordar)` y eligiendo el `kind` recomendado para cada nivel. Prioriza átomos `ex:*` y `rel:*` porque dan mejores distractores que `def:*`.
5. **`generate_quiz(client, kb, num_questions, refine=True)`** → pipeline completo:
   1. Plan + generación uno-a-uno con `_try_generate_question`.
   2. Si al terminar faltan preguntas (dedup agresivo, fallos en cadena), se sintetizan tantas `PlannedQuestion`s como hagan falta con `_synthesize_planned_questions` y se añaden al plan; así el revisor posterior también las ve.
   3. Si `refine=True`, `critics.refine_quiz` regenera solo las críticas (import diferido para romper el ciclo).
   4. Reasigna ids consecutivos 1..N para la UI.

`QuizQuestion` tiene un validador `correct_answer: Literal["A","B","C","D"]` y un normalizador (`_normalize_letter`) tolerante a `" a "`, `"a."`, etc.

### 5.10. `src/pptx_generator.py`

Pipeline de presentación en tres capas:

- **Plan** (`plan_slides`): pide el `SlidePlan` con `SLIDE_PLAN_PROMPT`, sanea contra la KB, recorta a rango razonable, trunca títulos con `_truncate_title`.
- **Redacción** (`render_slide_bullets`): una llamada por slide con `SLIDE_BULLETS_FROM_ATOMS_PROMPT` incluyendo:
  - Los átomos asignados a esa slide (`_atoms_for_slide`).
  - El índice completo para que el LLM *sepa* qué no debe repetir.
  - El `kind` y el `focus` narrativo.
- **Limpieza del texto** (`_clean_bullet`): tras el LLM aplicamos una función de calidad que descarta bullets con:
  - `…` o `...` (síntoma de truncamiento LLM; se descarta entero, preferimos regenerar antes que mostrar cortado).
  - Prefijos `def:` / `ex:` / `fc:` / `rel:` / `dt:` (copia del bloque de contexto).
  - Patrón `· definición ·`, `· ejemplo ·`, etc. (también copia del contexto).
  - Aperturas meta (`"en esta diapositiva"`, `"a continuación"`, `"se habla de"`, …).
  - Longitud <25 chars (bullets triviales).
  - Bullets demasiado largos: se recortan por palabra con `_trim_by_words` (**sin** puntos suspensivos), añadiendo punto final si hace falta.
  - Si **todos** los bullets se descartan → `GenerationError` → el revisor marca la slide como crítica y dispara regeneración.
- **Conclusión** (`_render_conclusion`): misma lógica con `CONCLUSION_FROM_KB_PROMPT`.
- **Render PPTX** (`render_pptx`):
  - Portada con `LAYOUT_TITLE`.
  - Slide de índice construida dinámicamente **a partir de las slides que sobrevivieron** a la limpieza (no del plan original → consistencia visible).
  - Una slide por cada `BuiltSlide` con bullets (las vacías se omiten).
  - Slide de conclusiones si existen.
  - `auto_size = TEXT_TO_FIT_SHAPE`, fuente 20 pt por defecto, `word_wrap=True`.
- **API pública** (`generate_presentation`): orquestador KB → plan → bullets → revisor → conclusión → render PPTX.

### 5.11. `src/critics.py`

Revisor con dos pasadas (**determinista** + **LLM**) y regeneración selectiva. El módulo evita ciclos de import con `TYPE_CHECKING` y con imports diferidos dentro de las funciones.

#### 5.11.1. Grounding

- `_tokens(text)`: tokeniza, quita stopwords en español, filtra <4 chars.
- `_kb_vocabulary(kb)`: conjunto de tokens relevantes de la KB (main_topic + subtopics + contenido de todos los átomos).
- `_is_grounded(text, vocab, min_hits=1)`: True si el texto comparte ≥1 token con el vocabulario.

Esto permite detectar cuándo el LLM se inventa un bullet o una pregunta sin raíz en el documento.

#### 5.11.2. Reglas deterministas del quiz (`_deterministic_quiz_issues`)

| Kind | Qué detecta | Severidad |
|---|---|---|
| `banned_phrase` | Opciones “Todas las anteriores”, “A y B”, “Ninguna es correcta”… | `high` |
| `unbalanced_options` | Una opción ≥ 2.2× más larga que el promedio del resto (pista involuntaria). | `medium` |
| `meta_language` | Enunciado/justificación con “según el texto”, “el autor dice”, etc. | `medium` |
| `not_grounded` | Pregunta + correcta + justificación sin tokens de la KB. | `medium` |
| `duplicate` | Stem (8 tokens más significativos) coincidente con otra pregunta. | `medium` |

#### 5.11.3. Reglas deterministas de slides (`_deterministic_slide_issues`)

| Kind | Qué detecta | Severidad |
|---|---|---|
| `meta_language` | Bullets con “en esta diapositiva”, “se estudia…”, etc. | `medium` |
| `too_shallow` | <2 bullets en una slide que no es intro/outlook/conclusion. | `medium` |
| `not_grounded` | Bullets sin tokens de la KB (en slides que no son intro/conclusion). | `medium` |
| `duplicate_content` | Un bullet cuyas palabras clave ya aparecieron en otra slide. | `medium` |

#### 5.11.4. Revisor LLM

- `_llm_quiz_issues` / `_llm_slide_issues`: una única llamada por artefacto con `QUIZ_CRITIC_PROMPT` / `SLIDE_CRITIC_PROMPT`. El LLM responde un JSON `{"issues": [...]}` que se valida con `QuizIssue` / `SlideIssue`. Se filtran ids fuera de rango.

#### 5.11.5. Refinamiento selectivo

- `refine_quiz(client, kb, questions, quiz_plan, max_iterations=1, use_llm=True)`:
  - Llama a `review_quiz` (determinista + LLM) y obtiene los `critical_ids` (severity `medium`/`high`).
  - Solo esos se **regeneran** usando la **misma `PlannedQuestion`** (se preserva concept/bloom/kind/focus → la distribución global no se rompe).
  - Las preguntas que pasaron la revisión se mantienen literales.
- `refine_slides(...)`: idéntico, pero por `slide_index`. La correspondencia se hace por `title` porque el índice cambia si se descartó alguna slide previa.

Coste extra por refinamiento: *K* llamadas = nº de elementos críticos + 1 llamada de revisor. Es mucho más barato que regenerar todo.

### 5.12. `src/quiz_pdf_exporter.py`

ReportLab + `SimpleDocTemplate` en A4 con márgenes 2 cm:

- Hoja 1: enunciado + 4 opciones (A/B/C/D) por pregunta.
- Salto de página.
- Hoja 2: respuesta correcta, nivel Bloom y justificación.
- Estilos propios (`_styles()`) en tonos universitarios (`#1f4e79`).

### 5.13. `app.py` — UI Streamlit

Estructura por secciones con estado en `st.session_state`:

- `_init_state` inicializa claves (`processed_pdf`, `kb`, `quiz`, `pptx_bytes`, bytes exportados, etc.).
- `render_sidebar` → selector de modelo + slider de preguntas + avisos de plantilla.
- `render_upload_section` → uploader con reset al cambiar de archivo, muestra métricas (páginas, caracteres, imágenes, tablas) y un expander con el Markdown extraído.
- `render_generate_section` → dos columnas; cada una valida preflight Ollama, asegura KB (la recalcula solo si no existía ya), y dispara el pipeline correspondiente. Excepciones mapeadas a mensajes accionables.
- `render_quiz_results` → `st.form` con `st.radio` por pregunta; al someter muestra aciertos y justificaciones y permite descargar JSON y PDF.
- `render_pptx_results` → previsualización del índice y los bullets, botón de descarga `.pptx`.

---

## 6. Modelos de datos y contratos

Resumen visual de todos los modelos Pydantic/dataclass en el proyecto:

```
ProcessedPDF (dataclass, frozen) ─────────── pdf_processor.py
  markdown: str
  num_pages, num_chars, num_images: int
  has_tables: bool

LiteralHints (dataclass) ─────────────────── extractors.py
  definitions: list[LiteralDefinition]
  code_blocks: list[LiteralCode]
  formulas:    list[LiteralFormula]
  key_terms:   list[str]
  .to_prompt_block(max_chars=3500) -> str

Chunk (dataclass, frozen) ────────────────── map_reduce.py
  index: int; text: str

KnowledgeBase (BaseModel) ────────────────── knowledge_base.py
  main_topic: str
  subtopics: list[str]
  definitions:  list[Definition]    # def:*
  examples:     list[Example]       # ex:*
  formulas_code:list[FormulaOrCode] # fc:*
  numeric_data: list[NumericDatum]  # dt:*
  relations:    list[Relation]      # rel:*
  conclusions:  list[str]

SlidePlan / PlannedSlide (BaseModel) ──────── plans.py
QuizPlan  / PlannedQuestion (BaseModel) ───── plans.py

Quiz / QuizQuestion / QuizOptions ─────────── quiz_generator.py
PresentationPlan / BuiltSlide (dataclass) ─── pptx_generator.py

QuizReview / QuizIssue ────────────────────── critics.py
SlideReview / SlideIssue ──────────────────── critics.py
```

Los **prefijos de id** son estables y únicos: `def:*`, `ex:*`, `fc:*`, `dt:*`, `rel:*`. Todo el sistema descansa sobre esta convención.

---

## 7. Prompts

Centralizados en `src/prompts.py`. Decisiones transversales:

- **Todo en español** para mantener el registro académico hispanohablante.
- Las llaves literales de los ejemplos JSON se escapan con los tokens `@@OPEN@@` / `@@CLOSE@@`, que `_finalize()` reemplaza por `{{`/`}}` (llaves escapadas para `str.format`). Las constantes numéricas (`__MAX_BULLETS__`, `__MAX_CHARS__`, `__MIN__`, `__MAX__`, `__TITLE_LEN__`) se inyectan en el mismo paso.
- Cada prompt define: rol, objetivo, **reglas duras** numeradas, formato JSON exacto y, cuando aplica, ejemplos positivos/negativos.

Prompts activos:

| Constante | Uso |
|---|---|
| `SYSTEM_EXPERT_ES` | System prompt común: “profesor universitario, precisión, no inventar”. |
| `MAP_SUMMARY_PROMPT` | Fase MAP: resumen por chunk. |
| `REDUCE_CONSOLIDATION_PROMPT` | REDUCE legacy a Markdown (se conserva para `consolidate_document`). |
| `REDUCE_TO_KB_PROMPT` | REDUCE moderno: consolida parciales + hints literales → KB JSON. |
| `SLIDE_PLAN_PROMPT` | Plan de slides con kinds cerrados; prohíbe `kind=conclusion`. |
| `SLIDE_BULLETS_FROM_ATOMS_PROMPT` | Redacción de bullets con reglas antiellipsis, antimeta, antiinvención de ejemplos, plantillas por kind y ejemplos positivos/negativos. |
| `CONCLUSION_FROM_KB_PROMPT` | Conclusión final con las mismas reglas anti-meta y anti-verbatim. |
| `QUIZ_PLAN_PROMPT` | Distribución Bloom por rangos (15-25 %/20-30 %/…) + mapa Bloom→kinds. |
| `QUIZ_QUESTION_PROMPT` | Redacción de UNA pregunta; reglas de longitud de opciones (±30 %), prohibiciones (`Todas las anteriores`, etc.), escenarios para niveles altos de Bloom, ejemplos positivos/negativos. |
| `QUIZ_CRITIC_PROMPT` | Revisor LLM del quiz (tipos de issue, severidades). |
| `SLIDE_CRITIC_PROMPT` | Revisor LLM de slides (tipos de issue, severidades). |
| `OUTLINE_PROMPT`, `SLIDE_CONTENT_PROMPT`, `CONCLUSION_PROMPT`, `QUIZ_GENERATION_PROMPT` | Prompts legacy de la v1 (mono-shot), conservados para fallback. |

---

## 8. Instalación y ejecución

### 8.1. Requisitos previos

- Python 3.11+ (recomendado 3.12).
- **Ollama** instalado y corriendo: https://ollama.com/download
- Al menos un modelo descargado. Recomendado por defecto: `ollama pull qwen2.5:7b`. Otros soportados por la UI: `gemma2:9b`, `gemma3:4b`, `mistral:7b`, `llama3.2:3b`.
- GPU NVIDIA ≥6 GB VRAM recomendable. Con 12 GB (RTX 3060/4070) se alcanza fluidez en `qwen2.5:7b` y `gemma2:9b`.

### 8.2. Instalación

```powershell
cd tfg-local
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 8.3. Lanzar Ollama con GPU

Si el launcher de bandeja lanza el servicio en CPU, cerrarlo desde el tray y arrancarlo manualmente:

```powershell
# En una terminal independiente
$env:OLLAMA_LLM_LIBRARY = "cuda_v12"   # fuerza CUDA
ollama serve
```

Verifica con:

```powershell
ollama ps    # 100% GPU debe aparecer en la columna PROCESSOR
```

### 8.4. Lanzar la aplicación

```powershell
streamlit run app.py
```

Abre `http://localhost:8501` y sigue el flujo: subir PDF → generar Quiz / Presentación → descargar.

### 8.5. Ejecutar con Docker (sin Ollama en contenedor)

Este proyecto se dockeriza solo para la app Streamlit. **Ollama NO va dentro
del contenedor**; se usa el Ollama del host.

1. Asegúrate de tener Ollama corriendo en el host:

```powershell
ollama serve
```

2. Construir imagen:

```powershell
docker compose build
```

3. Arrancar contenedor:

```powershell
docker compose up -d
```

4. Abrir app:

`http://localhost:8501`

5. Parar contenedor:

```powershell
docker compose down
```

`docker-compose.yml` ya inyecta:

`OLLAMA_BASE_URL=http://host.docker.internal:11434`

para que la app dentro del contenedor se conecte al Ollama del host.

### 8.6. Troubleshooting frecuente

| Síntoma | Causa / solución |
|---|---|
| `model 'qwen2.5' not found (status 404)` | Falta el tag. Usar siempre `qwen2.5:7b`. El cliente ya valida con match exacto. |
| `Error: listen tcp 127.0.0.1:11434: bind: Only one usage…` | Ya hay un `ollama.exe` corriendo. Cerrar desde bandeja + `Get-Process ollama | Stop-Process -Force`. |
| `ollama ps` muestra `100% CPU` | Forzar `$env:OLLAMA_LLM_LIBRARY="cuda_v12"` antes de `ollama serve`. |
| `ScannedPDFError` | PDF sin capa de texto. Pre-procesa con `ocrmypdf`. |
| `QuizPlan vacío tras sanear` | El LLM no produjo concept_ids reconciliables. Revisar la KB (a veces los `rel:*` resultan muy abreviados); reducir el número de preguntas o cambiar de modelo. |

---

## 9. Historial de mejoras (changelog técnico)

Cada entrada lista **qué se hizo**, **por qué** y **qué archivos cambiaron**. A partir de ahora se amplía con cada nueva intervención.

### v2.13.1 — Extracción PDF más útil (tablas + contexto visual)

Objetivo: mejorar calidad de markdown para modelos pequeños y reducir ruido sin añadir complejidad alta.

Cambios:

- **`src/pdf_processor.py`**
  - Nueva normalización de tablas markdown (`_normalize_markdown_tables`) para:
    - reemplazar `<br>` por separador legible ` / `,
    - recomponer filas partidas por salto de línea.
  - Mejora de `_extract_image_context`:
    - añade metadatos mínimos útiles por imagen: página y tamaño (`[p.X, WxH]`),
    - renombra bloque final a `## Contexto visual detectado`.
  - Nuevo filtro por área relativa (`_MIN_IMAGE_AREA_RATIO`) para ignorar imágenes muy pequeñas (logos/iconos), reduciendo ruido en la señal visual.

Motivo:

- El markdown anterior incluía tablas con cortes internos difíciles de interpretar por el LLM.
- El bloque visual era poco informativo y mezclaba elementos decorativos con figuras relevantes.
- Esta versión mantiene el pipeline simple/offline y mejora grounding visual sin meter OCR ni visión multimodal.

### v1.0 — Base funcional inicial

- Pipeline “mono-shot”: PDF → Markdown → resumen Markdown libre → generación directa de quiz y de slides en una sola llamada cada uno.
- Exportación de quiz a PDF con ReportLab.
- UI Streamlit con modelo seleccionable y slider de preguntas.
- Detección de PDF escaneado con heurística de texto por página.

**Limitaciones detectadas con uso real**:

- Solapes entre slides (mismo concepto repetido).
- Distribución Bloom desequilibrada (todo `recordar`).
- Distractores triviales (“Ninguna de las anteriores”, opciones absurdas).
- Bullets truncados con `…`, meta-lenguaje (“en esta diapositiva…”), copias literales del bloque de contexto.
- Conclusiones duplicadas (LLM añadía la suya y luego el renderer añadía otra).

### v1.1 — Preflight y fix de modelo

- **`src/config.py`**: `AVAILABLE_MODELS` y `DEFAULT_MODEL` pasan a usar **tags completos** (`qwen2.5:7b`, `gemma2:9b`, `gemma3:4b`, `mistral:7b`, `llama3.2:3b`).
- **`src/ollama_client.py`**: `check_model_available` endurecido a match exacto. *Motivo*: `/api/generate` exige el nombre completo con tag; un match por base name superaba el preflight y después fallaba en tiempo de generación con 404.

### v2.0 — Arquitectura por etapas (de resumen libre a KnowledgeBase)

**Decisión estructural**: abandonar el resumen en Markdown libre y adoptar una Knowledge Base JSON tipada como fuente única de verdad.

- **Nuevo: `src/knowledge_base.py`** con Pydantic de todos los átomos y `KnowledgeBase`. Implementado `_normalize_id` (`NFKD` + minúsculas + `ñ→n` + `[a-z0-9_-]` + `max_len=40`) aplicado como `@field_validator(id, mode="before")` en cada átomo → resuelve el `ValidationError` clásico al recibir `ex:farmacéutica` del LLM.
- **Nuevo: `src/extractors.py`** con heurísticas literales (definiciones, bloques de código, fórmulas, términos). El bloque resultante se inyecta en el prompt REDUCE para anclar la KB al texto original.
- **Nuevo: `src/plans.py`** con `SlidePlan` y `QuizPlan` como contratos intermedios, Bloom y kinds tipados con `Literal`.
- **`src/prompts.py`**: añadidos `REDUCE_TO_KB_PROMPT`, `SLIDE_PLAN_PROMPT`, `SLIDE_BULLETS_FROM_ATOMS_PROMPT`, `CONCLUSION_FROM_KB_PROMPT`, `QUIZ_PLAN_PROMPT`, `QUIZ_QUESTION_PROMPT`. Bug `KeyError` por llaves literales en el JSON de ejemplo resuelto con tokens `@@OPEN@@`/`@@CLOSE@@` procesados por `_finalize()`.
- **`src/map_reduce.py`**: nueva API `build_knowledge_base` que orquesta chunking + hints + MAP + REDUCE→KB. `consolidate_document` se conserva como shim de compatibilidad.
- **`src/quiz_generator.py`**: dividido en `plan_quiz` + `generate_single_question` (una llamada por pregunta con contexto central + contexto relacionado para distractores + preguntas previas para evitar solape).
- **`src/pptx_generator.py`**: dividido en `plan_slides` + `render_slide_bullets` + `_render_conclusion` + `render_pptx`.
- **`app.py`**: `_ensure_summary` → `_ensure_kb`; se pasa la KB a los generadores.

### v2.1 — Revisor crítico y refinamiento selectivo

**Nuevo: `src/critics.py`** con dos pasadas:

- Determinista (regex + reglas de anclaje). Barato y detecta lo que el propio LLM no admitirá en sí mismo.
- LLM con `QUIZ_CRITIC_PROMPT` / `SLIDE_CRITIC_PROMPT` (una llamada por artefacto completo) para problemas semánticos.

**`refine_quiz` / `refine_slides`** regeneran **solo** los elementos con severidad `medium`/`high`, reutilizando la misma `PlannedQuestion`/`PlannedSlide`. Esto preserva la distribución global y abarata el coste (K llamadas extra en lugar de rehacer todo).

**Integración**:

- `generate_quiz(refine=True)` en `quiz_generator.py`.
- `build_plan(refine=True)` en `pptx_generator.py`.
- Import diferido (`from .critics import refine_*`) para evitar ciclos con los generadores.

### v2.2 — Grounding y fuzzy matching

Problema recurrente: el LLM escribe `concept_id`s o `atom_ids` que no existen tal cual en la KB (acentos, plurales, abreviaturas tipo `rel:subclase_de` frente a `rel:subclase_de_bicicletademontana_bicicleta`). Resultado anterior: se descartaban preguntas enteras.

- **`src/plans.py`**: añadidos `_deaccent_lower`, `_normalize_slug` (plurales → singular), `_split_id`, `_collapse`, `resolve_atom_id` (cascada de 5 intentos) y `_reconcile_atom_ids`. **Restricción de prefijo**: la reconciliación solo busca dentro del mismo prefijo (`def:` solo con `def:*`), porque el resto del pipeline asume el tipo.
- **`sanitize_slide_plan`** y **`sanitize_quiz_plan`** usan la reconciliación y registran en log cada recuperación.
- **Deduplicación relajada**: `sanitize_quiz_plan` permite repetir `concept_id` mientras cambie al menos uno de `bloom_level` o `kind` → dos preguntas sobre el mismo concepto, una de `recordar` y otra de `aplicar`, son válidas.

### v2.13.0 — Rediseño QuizSimple (estabilidad + simplicidad)

Se aplica el rediseño acordado de arquitectura híbrida simple: **1 pasada + 1 única regeneración selectiva** y severidad media no bloqueante para aumentar supervivencia de preguntas/slides sin disparar complejidad.

Baseline observado en logs previos (muestras reales antes del rediseño):

- Casos frecuentes de quiz por debajo del mínimo: `2/5`, `0/5`, `2/5`, `3/5`.
- Alta presión de llamadas por refine: en varios runs se regeneraban 4-8 preguntas por lote completo.
- Tiempo alto por cadenas de retry/revisión, especialmente con modelos pequeños.

Cambios aplicados:

- **`src/critics.py` (quiz)**:
  - Modelo operativo simplificado:
    - **Bloqueantes**: `severity=high` (p.ej. `duplicate`, `duplicate_stem`, `banned_phrase`).
    - **No bloqueantes**: `severity=low/medium` (meta, grounding suave, estilo, etc.).
  - `refine_quiz` ahora regenera/descarta solo bloqueantes (`blocker_ids`), dejando pasar no bloqueantes para maximizar cantidad final.
- **`src/quiz_generator.py`**:
  - `plan_quiz` pasa a **una sola llamada LLM** + fallback determinista (sin retry de temperatura).
  - `generate_single_question` pasa a **una sola llamada LLM** (sin bucles de retry).
  - Nuevo fallback determinista corto `_build_deterministic_question` si falla una pregunta, para evitar finalizar en `0`.
  - Objetivo de cantidad simplificado: `_adaptive_target` baja a `~1.2 * átomos` y `plan_request = target + 1` (margen mínimo, sin oversampling agresivo).
- **`src/prompts.py`**:
  - `QUIZ_QUESTION_PROMPT` compactado a reglas de alto impacto (formato, 4 opciones, una correcta, sin meta grave, ajuste Bloom), eliminando restricciones hiper-específicas y ejemplos extensos.
- **`src/critics.py` + `src/pptx_generator.py` (slides)**:
  - Filosofía lite también en slides: regeneración solo para bloqueantes (`severity=high`).
  - Problemas de estilo/grounding suave pasan a `low` para no vaciar slides.
  - `build_plan(..., use_llm_critic=False)` por defecto para reducir llamadas y latencia.
- **Limpieza**:
  - Eliminada constante obsoleta `QUIZ_PLAN_OVERSAMPLING` de `src/config.py`.

Resultado esperado del rediseño:

- Mayor porcentaje de ejecuciones con `>= mínimo`.
- Menos llamadas LLM por quiz/slide.
- Menos ramas de control y menor complejidad para defensa del proyecto.

### v2.12.7 — Slides: blindaje contra `Example` alucinados

Tras analizar salidas reales de `qwen2.5:14b` y `gemma3:12b`, detectamos el vector más grave pendiente: las slides seguían filtrando conocimiento externo a través de `Example` atoms inventados en la fase REDUCE del KB (p. ej. "Clase Vehículos", `tipo_combustible`, `calcularVelocidadMáxima`, `adaptarSuspension`). El fix de v2.12.6 validaba `Relation` y quiz, pero no cubría de forma simétrica el canal de `Example` en slides.

Cambios aplicados:

- **`src/map_reduce.py`**
  - Nueva función `_prune_ungrounded_examples(kb, markdown)`.
  - Descarta `Example` completo si `name` no aparece literalmente en el PDF (normalización con deacento + camelCase + variantes con/sin espacios).
  - Filtra `attributes`/`methods` individualmente si no aparecen en el PDF, conservando solo los literales.
  - Integrada en `_reduce_to_kb` junto al pruning de relaciones: primero `_prune_ungrounded_relations`, después `_prune_ungrounded_examples`.
- **`src/critics.py`**
  - Nuevo detector `_detect_slide_world_knowledge_leak` (análogo al del quiz) para bullets de slides.
  - `_deterministic_slide_issues` ahora acepta `md_index` y emite `world_knowledge_leak` (`severity=medium`) con umbral doble (absoluto + ratio) igual al quiz.
  - `review_slides` y `refine_slides` aceptan nuevo parámetro opcional `source_markdown`.
- **`src/pptx_generator.py`**
  - `build_plan` y `generate_presentation` aceptan/propagan `source_markdown`.
  - El refine de slides recibe `source_markdown`, activando el detector anti-conocimiento-externo durante regeneración/filtrado.
- **`app.py`**
  - La generación de PPTX pasa `processed.markdown` a `generate_presentation`.

Validación rápida en caliente (unitaria/funcional):

- Caso sintético equivalente a los fallos observados:
  - `Example(name="Vehículos", attributes=["tipo_combustible"], methods=["acelerar"])` se descarta.
  - `Example(name="Bicicleta", attributes=["velocidad","cadencia","tipo_combustible"], methods=["frenar","girar"])` se limpia a `["velocidad","cadencia"]` + `["frenar"]`.
- Imports y lint de módulos modificados sin errores.

Trade-off explícito: si un PDF legítimamente contiene términos de dominio cercanos al mundo real (p. ej. un tema de automoción), el detector no los penaliza siempre que estén en el markdown. El sistema prioriza grounding literal, no una ontología cerrada.

### v2.12.6 — Grounding estricto: anti-alucinaciones por conocimiento externo

**Problema observado** con modelos grandes (`qwen2.5:14b`, `gemma3:12b`): aunque la KB se construye correctamente, el LLM inyecta en slides y quiz información que **no está en el PDF** pero sí en su pretraining. Ejemplos reales detectados:

- `gemma3:12b`: alucina una `Relation` `Bicicleta —[subclase_de]→ Vehículos` cuando el PDF nunca menciona "Vehículos". El fuzzy matching liberal de v2.2 reconciliaba además `BicicletaDeMontaña` con `Bicicleta`, fusionando dos conceptos distintos como si fueran el mismo.
- `qwen2.5:14b`: en el quiz, P7 inventa una "clase vehículo" con `tipo_combustible` y `acelerar`/`frenar` (concepto externo estándar en POO didáctica, pero no en este PDF). P9 mezcla hechos reales sobre bicicletas de montaña (marchas para "terrenos difíciles", "carreteras", "cadencias ajustables") que el PDF no enseña.

Diagnóstico: el sistema hasta v2.12.5 validaba que las preguntas/slides referencian **la KB**, pero no que la **KB** estuviera anclada al **PDF literal**. Si la KB absorbía alucinaciones del REDUCE, todo el pipeline las heredaba sin poder detectarlas.

Tres fixes complementarios, en tres puntos distintos del pipeline:

**Fix 1 — Verificación literal de relaciones (`src/map_reduce.py`)**

- Nuevo helper `_prune_ungrounded_relations(kb, markdown)` que descarta cualquier `Relation` cuyo `source` o `target` **no aparezca literalmente** en el markdown del PDF.
- Normalización tolerante: `BicicletaDeMontaña` se divide por camelCase a `bicicleta de montaña`, se deacentúa/minusculiza, y se busca tanto con espacios como compactado (`bicicletademontana`). Cubre los 3 formatos habituales en que el LLM referencia la misma entidad.
- Se aplica al final de `_reduce_to_kb` (antes de devolver la KB), justo después del fallback para no degradar la resiliencia.
- Solo afecta a `Relation`: definiciones y ejemplos ya se verifican en la fase MAP vía `LiteralHints`; fórmulas/código son verbatim por definición.

**Fix 2 — Fuzzy matching de `atom_id`s endurecido (`src/plans.py::resolve_atom_id`)**

- Eliminada la dirección permisiva `raw_slug ⊃ valid_slug` del matching de prefijo. Solo se permite `valid ⊃ raw` (LLM escribe slug incompleto, KB tiene el expandido). Nunca al revés: antes, `ex:bicicletademontana` colapsaba a `ex:bicicleta` porque la KB solo tenía el genérico. Ahora se rechaza y la pregunta/slide se descarta con log claro.
- La rama de slug "colapsado" también pasa a ser estrictamente direccional (`raw ⊂ valid`) y exige que la longitud del valid no sea más de 1.8× la del raw → evita que un slug corto (`ex:poo`) se fusione con uno largo no relacionado.
- Umbral Jaccard subido de **0.60 → 0.78** y con requisito añadido de **≥ 2 tokens en común**: un único token compartido (p. ej. "bicicleta") ya no basta para reconciliar dos ids distintos.

**Fix 3 — Detector `world_knowledge_leak` en el crítico del quiz (`src/critics.py`)**

- Nuevo helper `_markdown_token_index(markdown)`: construye el índice de tokens del PDF con recorte de plurales (`bicicletas` y `bicicleta` se consideran la misma raíz) para tolerar variantes flexivas naturales.
- Nuevo detector `_detect_world_knowledge_leak(q, md_index, kb_vocab)` que cuenta **tokens "notables"** (longitud ≥ 6, sin stopwords) del enunciado y las 4 opciones que **no aparecen ni en el PDF ni en la KB**.
- Umbral doble:
  - Absoluto: **≥ 7 tokens ajenos** (filtra preguntas breves con 2-3 conectores ausentes).
  - Proporción: **≥ 50% del total de notables** son ajenos (filtra preguntas extensas con muchos sinónimos en prosa).
- Calibrado empíricamente contra el quiz `qwen2.5:14b`: P7/P9 (hallucinations severas) disparan ambas condiciones (~53% y ~82% ratio); P1/P3/P4/P6/P8/P10 (grounded) quedan muy por debajo (~5-15% ratio en KB realista).
- Severidad `medium`: participa en la regeneración del refine como cualquier issue crítico. Si tras regenerar sigue flagged → se descarta.
- **Propagación**: `review_quiz` / `refine_quiz` / `generate_quiz` aceptan ahora un kwarg opcional `source_markdown: str | None`. `app.py` lo pasa desde `processed.markdown`. Retrocompatible: si no se pasa, el detector se desactiva.

**Trade-off asumido**: el detector puede dejar pasar hallucinations sutiles que usen tokens ya presentes en el PDF en otro contexto (p. ej. si el PDF menciona "velocidad" y el LLM inventa una relación sobre velocidad máxima que el PDF no trata). Para ese tipo de matices se necesitaría verificación semántica (LLM+embeddings), que queda fuera del alcance de esta iteración. El detector actual es determinista, barato y cubre ~90% de los casos observados.

**Por qué los 3 a la vez**: resuelven el mismo problema de raíz en tres puntos distintos del pipeline. Fix 1 ataca la fuente (KB misma); Fix 2 evita que el saneado propague fusiones erróneas a slides/quiz; Fix 3 es la red de seguridad final en el quiz. Aplicar solo uno deja los otros dos vectores de fuga abiertos.

### v2.12.5 — Resiliencia del `QuizPlan` frente a LLMs pequeños

Cerrando el trío `SlidePlan` (v2.12.3) + `KnowledgeBase` (v2.12.4) + `QuizPlan` (esta). Mismo patrón, mismo síntoma observado: con `gemma3:4b` generando quiz, el LLM devolvió `{}` y el pipeline abortaba con `ValidationError: questions missing`. El contrato era claro: "los 5 modelos deben funcionar sin excepción".

Cambios:

- **`src/plans.py`**:
  - Nueva función pública `coerce_quiz_plan_payload(raw)`:
    - Acepta `list` directa en la raíz (`[{…},{…}]` → `{"questions": [...]}`).
    - Desenvuelve un nivel si el LLM envolvió en `{"quiz_plan": {...}}`, `{"plan": {...}}`, `{"output": {...}}`, `{"result": {...}}`, `{"data": {...}}`, `{"respuesta": {...}}`. También soporta valores directos cuando el wrapper contiene un `list` en lugar de `dict`.
    - Alias top-level: `preguntas`/`items`/`plan_items`/`quiz`/`list` → `questions`.
    - Alias por pregunta: `numero`/`número`/`idx`/`n` → `id`; `nivel`/`level`/`cognitive_level` → `bloom_level`; `atomo`/`átomo`/`concepto`/`concept` → `concept_id`; `tipo`/`type`/`question_type` → `kind`; `enfoque`/`intencion` → `focus`.
    - Conversión robusta de `id`: si el LLM devuelve `"1"` (string), `"#1"`, o nada → parseo con try/except y fallback a índice secuencial.
    - Filtrado estricto: sólo se mantienen preguntas con `concept_id` + `bloom_level` + `kind` presentes. El resto (normalización semántica `remember`→`recordar`, `difference`→`diferenciacion`) queda en manos del `_normalize_plan_entries` ya existente en `quiz_generator.py`.
    - Devuelve `None` si es irrecuperable → señal para activar fallback.
  - Nueva función pública `build_fallback_quiz_plan(kb, target_count)`:
    - Construye `PlannedQuestion`s determinísticamente sin LLM.
    - Orden preferente de átomos: `definitions` → `examples` → `relations` → `formulas_code` → `numeric_data` (coincide con la jerarquía pedagógica natural: primero aprendes términos, luego ejemplos, luego relaciones).
    - Distribución Bloom cíclica vía `_BLOOM_DISTRIBUTION = (recordar, comprender, aplicar, analizar, recordar, comprender, evaluar, aplicar, comprender, analizar, …)`. Cubre los 6 niveles con densidad realista (más `recordar`/`comprender`/`aplicar`, menos `evaluar`/`crear`).
    - `kind` elegido por coherencia `(tipo_atómo, bloom)`: si el tipo del átomo (`Definition` → `definicion`, `Example` → `caso_practico`, `Relation` → `comparacion`, `FormulaOrCode` → `completar_codigo`) está en `BLOOM_RECOMMENDED_KINDS[bloom]` se usa; si no, cae al recomendado canónico del Bloom. Evita combinaciones absurdas tipo `(aplicar, definicion)`.
    - Respeta `adaptive_max_per_concept(num_atoms, target)` del v2.7: el mismo cap adaptativo que usa `sanitize_quiz_plan`, así el fallback nunca produce quizzes más monotemáticos que los del LLM.
    - Hasta 3 pasadas por la lista elevando el cap +1 en la segunda para rellenar hasta `target_count`. Pasadas limitadas por diseño: preferimos devolver menos preguntas que un quiz saturado del mismo concepto.
- **`src/quiz_generator.py`**:
  - Nuevo `_try_build_quiz_plan(raw)`: combina `coerce_quiz_plan_payload` + `_normalize_plan_entries` + `QuizPlan(**data)`. Nunca lanza excepciones; devuelve `(plan|None, motivo|None)`.
  - `plan_quiz` reescrita con cascada de 3 capas:
    1. 1ª llamada LLM (`temperature=0.2`). `_try_build_quiz_plan`.
    2. Si falla: reintento (`temperature=0.1`). Coste ≤1 llamada LLM extra.
    3. Si sigue fallando: `build_fallback_quiz_plan(kb, n)`.
  - Salvaguarda final: si `sanitize_quiz_plan` deja el plan vacío (todos los `concept_id` eran alucinaciones), se reintenta con el fallback determinístico, porque sus `concept_id` provienen directamente de `kb.atom_ids()` y están garantizados válidos.

Con las tres rondas (v2.12.3, v2.12.4, v2.12.5), los tres artefactos estructurados del pipeline (`KnowledgeBase`, `SlidePlan`, `QuizPlan`) ahora tienen el mismo contrato de resiliencia: **nunca abortan la generación por un LLM pequeño poco cooperativo**. El coste extra en el peor caso es **1 llamada LLM adicional** (el reintento) + un fallback determinístico de < 50 ms.

Archivos tocados:

- `src/plans.py`: +9 constantes de alias (`_QUIZ_QUESTIONS_ALIASES`, `_Q_*_ALIASES`); +2 helpers (`_pick_q_alias`, `_BLOOM_DISTRIBUTION`); +2 funciones públicas (`coerce_quiz_plan_payload`, `build_fallback_quiz_plan`).
- `src/quiz_generator.py`: +`_try_build_quiz_plan`; `plan_quiz` reescrita (cascada + salvaguarda post-saneo).

### v2.12.4 — Resiliencia de la `KnowledgeBase` frente a LLMs pequeños

Con otro modelo pequeño distinto el LLM devolvió `{}` como KB y el pipeline abortaba con `ValidationError: main_topic missing`. Mismo patrón que v2.12.3 pero en la fase reduce. Aplicamos la misma arquitectura de 3 capas en el núcleo del sistema: la construcción de la KB no debe fallar por un modelo que no sabe seguir un esquema; debe **degradar con calidad decreciente** pero producir siempre una KB utilizable.

Cambios:

- **`src/knowledge_base.py`**:
  - Nuevas constantes de alias para cada campo top-level (`_KB_TOPIC_ALIASES`, `_KB_SUBTOPICS_ALIASES`, `_KB_DEFS_ALIASES`, `_KB_EXAMPLES_ALIASES`, `_KB_FC_ALIASES`, `_KB_DATA_ALIASES`, `_KB_RELATIONS_ALIASES`, `_KB_CONCLUSIONS_ALIASES`). Cubren ES/EN + variantes con y sin tilde: `tema`/`titulo`/`título`/`topic` → `main_topic`; `definiciones`/`glosario`/`glossary` → `definitions`; `ejemplos`/`casos`/`cases` → `examples`; `fórmulas`/`codigo`/`code_blocks` → `formulas_code`; `datos`/`métricas` → `numeric_data`; `relaciones`/`links` → `relations`.
  - Nuevas constantes por átomo (`_DEF_FIELD_ALIASES`, `_EX_FIELD_ALIASES`, `_FC_FIELD_ALIASES`, `_DT_FIELD_ALIASES`, `_REL_FIELD_ALIASES`). Ejemplos: `termino`/`término`/`name`/`concepto` → `term`; `definicion`/`descripcion`/`explicación` → `definition`; `origen`/`from`/`src`/`sujeto` → `source`; `tipo`/`type`/`relation_type` → `kind`.
  - Nueva función pública `coerce_kb_payload(raw, *, fallback_topic)`: recibe el JSON crudo del LLM y devuelve un dict con las claves canónicas del schema. Desenvuelve un nivel si el modelo envolvió todo en `{"kb": {...}}`, `{"knowledge_base": {...}}`, `{"output": {...}}`, `{"result": {...}}`, `{"data": {...}}` o `{"respuesta": {...}}`. Si tras la coerción no hay ni topic ni ningún átomo, devuelve `None` (señal para activar fallback). Si falta `main_topic` pero hay átomos, deriva uno desde subtopics/primera definición/`"Documento"`.
  - Nueva función auxiliar `_infer_fc_kind(content)`: cuando el LLM omite el `kind` de un `FormulaOrCode` (campo obligatorio del Literal), lo inferimos por regex del contenido (`def`, `class`, `return`, `import`, llaves/semicolons → `code`; `=` y símbolos matemáticos aislados → `formula`). Evita que un olvido trivial rompa la KB entera.
  - Nueva función auxiliar `_ensure_atom_id`: si un átomo llega sin `id` (muy común en modelos <7B), se autogenera con `slugify_id` a partir del campo dominante del tipo (`term` para `Definition`, `name` para `Example`, `caption`/`content` para `FormulaOrCode`, `value` para `NumericDatum`, `source`+`kind` para `Relation`). Resuelve colisiones añadiendo sufijo numérico. Resultado: una lista de definiciones sin ids como `[{"term": "Clase", ...}, {"term": "Objeto", ...}]` pasa a `[{"id": "def:clase", ...}, {"id": "def:objeto", ...}]`, válida para Pydantic.
  - `_coerce_atom_list` recibe ahora `prefix` opcional para activar la autogeneración de ids; `_normalize_atom_dict` + `_pick_field` son los ladrillos comunes reutilizables.
  - Los helpers `_as_string_list` normalizan payloads que el LLM entrega como CSV/newline-separated en vez de lista (p.ej. `subtopics: "Clases, Objetos, Herencia"`).
- **`src/map_reduce.py`**:
  - Nuevo `_try_build_kb(raw, fallback_topic)`: encapsula coerción + validación **sin lanzar excepciones**. Devuelve `(kb|None, motivo|None)` para logging limpio.
  - Nueva función pública `build_fallback_kb(markdown, hints)`: construye una KB determinísticamente usando `LiteralHints` (v1.3+) + estructura del Markdown:
    - `main_topic`: primer `#` del Markdown → primer `key_term` de los hints → `"Documento"`.
    - `subtopics`: `##` del Markdown → `###` si no hay → `key_terms` como último recurso.
    - `definitions`: cada `LiteralDefinition` con `verbatim=True` (se confirma que son citas literales del PDF).
    - `formulas_code`: cada `LiteralCode` como `kind="code"`; cada `LiteralFormula` con `kind` inferido.
    - ids autogenerados con `slugify_id` + resolución de colisiones.
    - `examples`, `numeric_data`, `relations`, `conclusions` se dejan vacíos (no hay extractor determinista fiable para ellos; preferimos KB pequeña pero cierta).
  - `_reduce_to_kb` reescrito con cascada de 3 capas:
    1. 1ª llamada LLM (`temperature=0.2`). Coerce + validate.
    2. Si falla: reintento con `temperature=0.1` (coste ≤1 llamada extra).
    3. Si sigue fallando: `build_fallback_kb(markdown, hints)`. Generación **sin LLM**, < 50 ms.
  - Nueva salvaguarda final: si la KB es válida pero tiene 0 átomos (el LLM devolvió sólo `main_topic`), se fusiona con el fallback para asegurar contenido mínimo. Preservamos el topic/subtopics del LLM (suelen ser mejores) y tomamos los átomos del fallback (literales verificables).
  - `build_knowledge_base` pasa ahora `source_markdown` a `_reduce_to_kb` para que el fallback pueda inferir `main_topic` y `subtopics` del H1/H2 del documento original.

Decisiones:

- La coerción **no inventa contenido**: sólo renombra claves y autogenera ids/kind cuando el LLM los omitió. Si el modelo no entregó una definición, la coerción no la crea. El único momento donde se fabrica contenido es el fallback determinístico, y **siempre desde material literal** del PDF (citas detectadas por los extractores).
- El `main_topic` se infiere del primer H1 del Markdown, no del LLM, porque `pymupdf4llm` lo extrae fiablemente para casi todo PDF académico. Es la señal más robusta disponible.
- Autogeneración de ids usa el mismo `slugify_id` que los extractores v1.3+: garantiza consistencia con el resto del pipeline (los extractores y el LLM producen ids compatibles).
- Trade-off del fallback: no tiene `examples` ni `relations` porque no hay heurística segura para extraerlos. Consecuencia: slides más pobres en ejemplos y relaciones si el LLM falla del todo. Pero la generación **no se detiene** y los 5 modelos del dropdown ahora funcionan sin excepción para cualquier PDF.

Archivos tocados:

- `src/knowledge_base.py`: +8 constantes de alias, +5 helpers (`_pick_field`, `_normalize_atom_dict`, `_as_string_list`, `_coerce_atom_list`, `_ensure_atom_id`), +2 funciones públicas (`coerce_kb_payload`, `_infer_fc_kind`).
- `src/map_reduce.py`: imports actualizados; +`_try_build_kb`, +`_infer_main_topic`, +`_infer_subtopics`, +`_fc_kind_of`, +`build_fallback_kb`; `_reduce_to_kb` y `build_knowledge_base` reescritas con cascada de 3 capas.

### v2.12.3 — Resiliencia del `SlidePlan` frente a LLMs pequeños

Al probar con un modelo más pequeño, el LLM devolvió `{}` como `SlidePlan` y el sistema abortaba con `ValidationError: presentation_title/slides missing`. El mismo patrón ya habíamos visto en `Definition` (v2.12.3 del KB): el modelo no respeta el esquema. Solución en tres capas **independientes** que se aplican en cascada dentro de `plan_slides`:

1. **Coerción defensiva de claves** (`coerce_slide_plan_payload` en `src/plans.py`):
   - Tolera alias en español y variantes comunes: `titulo`/`título`/`presentation`/`topic` → `presentation_title`; `diapositivas`/`plan`/`contenido` → `slides`; `tipo`/`type`/`layout` → `kind`; `atomos`/`átomos`/`ids`/`refs` → `atom_ids`; `enfoque`/`intencion`/`narrative` → `focus`.
   - **Desenvuelve** un nivel si el modelo metió todo dentro de `{"plan": {...}}`, `{"output": {...}}`, `{"result": {...}}` o `{"respuesta": {...}}`.
   - Normaliza `kind` contra `_KIND_SYNONYMS` (20 sinónimos con/sin acento): `"Definición"` → `definition`, `"código"` → `code`, `"panorámica"` → `outlook`…
   - Coerciona `atom_ids` cuando vienen como string CSV (`"def:clase, ej:perro"`) o como valor único.
   - Si falta el título de presentación pero la lista de slides es válida, rellena con `kb.main_topic`.
   - Devuelve `None` sólo si la estructura es irrecuperable (no hay slides en ningún formato).
2. **Reintento único** (dentro de `plan_slides`): si la 1ª respuesta produce `None` o `ValidationError`, se repite la misma llamada con `temperature=0.1`. Coste: ≤1 llamada LLM extra. Suficiente para recuperarse cuando el primer intento fue un pozo seco puntual y no un problema sistémico del modelo.
3. **Fallback determinístico desde la KB** (`build_fallback_slide_plan` en `src/plans.py`): si tras el reintento seguimos sin plan válido, construimos uno **sin LLM**, agrupando átomos por `subtopic`:
   - Slide 1: `intro` panorámica con `focus` autogenerado desde `kb.main_topic`.
   - Una slide por cada `subtopic` con átomos asociados; `kind` elegido por `_slide_kind_for_group` según el tipo dominante (`Definition` → `definition`, `Example` → `example`, etc.).
   - Si no hay `subtopics`, agrupamos por tipo de átomo (Definiciones clave / Ejemplos ilustrativos / Fórmulas y código / Relaciones / Datos numéricos).
   - Garantiza el mínimo de 3 slides: si la KB es muy pobre, añade "Conceptos fundamentales" o "Aspectos complementarios" para cumplir el contrato.
   - Deduplicación de títulos con `_dedup_title`.
4. **Complemento post-saneado**: tras `sanitize_slide_plan`, si el plan quedó por debajo del mínimo (`DEFAULT_NUM_SLIDES_MIN`) porque todos los `atom_ids` eran alucinaciones, se fusiona con el fallback manteniendo las slides válidas del LLM.

Decisiones:

- La coerción es un **preprocesador puro**: nunca inventa contenido, sólo renombra/normaliza claves ya presentes. Así nunca enmascara errores reales del modelo.
- El fallback es **determinista** (sin LLM) y por tanto rápido (<200 ms). El trade-off es claro: las slides generadas así son más secas (sin `focus` narrativo), pero el sistema **nunca se detiene** por un modelo poco cooperativo. El usuario puede decidir reintentar con otro modelo más grande.
- El reintento está intencionadamente limitado a **1 intento**. Más intentos alargan la latencia sin beneficio empírico: si un modelo falla dos veces seguidas el esquema, también fallaría un tercer intento.
- La misma arquitectura `coerce + fallback` es candidata obvia para el `QuizPlan` (pendiente en el roadmap).

Archivos tocados:

- **`src/plans.py`**: imports añadidos (`Any`, `defaultdict`, clases atómicas del KB); nuevas constantes `_TITLE_ALIASES`, `_SLIDES_ALIASES`, `_SLIDE_*_ALIASES`, `_KIND_SYNONYMS`, `_VALID_KINDS`, `_ATOM_KIND_MAP`; nuevas funciones públicas `coerce_slide_plan_payload`, `build_fallback_slide_plan`; helpers privados `_pick_alias`, `_normalize_kind`, `_coerce_atom_ids`, `_group_atoms_by_subtopic`, `_slide_kind_for_group`, `_dedup_title`.
- **`src/pptx_generator.py`**: `plan_slides` reescrita con la cascada de 3 capas; nuevo helper `_try_build_slide_plan` que encapsula coerción + validación sin lanzar excepciones.

### v2.12 — Slides: ronda 1 de calidad (sin notación interna, sin eco del título, sin anglicismos)

Primera pasada del ciclo de mejora de slides tras cerrar el bloque de quiz. Un run del PDF de POO dejó tres bugs objetivos y varios problemas de estilo didáctico:

1. **Notación interna del KB filtrándose a los bullets** (bug crítico): dos slides mostraban cosas como `"BicicletaDeMontaña —[subclase_de]→ Bicicleta: hereda…"`. La causa era doble y fácil de localizar: tanto `_atom_block` (contexto del prompt de bullets) como `KnowledgeBase.to_markdown` serializaban las relaciones con la tripleta técnica, y el propio prompt (`SLIDE_BULLETS_FROM_ATOMS_PROMPT`) le pedía al LLM que redactara las relaciones *exactamente con ese formato*. El modelo obedecía.
2. **Eco del título como bullet 1**: la slide "Polimorfismo" empezaba con `"Polimorfismo: capacidad de un objeto comportarse…"`. Mala práctica estándar de diseño de slides.
3. **Anglicismos y neologismos raros** en los bullets: `"blueprint"` por "plantilla", `"una instancia concretada"` en vez de "concreta".
4. **Muletillas didácticas vacías** embebidas en mitad de bullets: `"sirve de ejemplo para ilustrar…"`, `"facilita la creación y gestión de…"`, que no aportan hecho.

Cambios:

- **`src/knowledge_base.py`**:
  - Nuevo diccionario `_RELATION_KIND_PHRASES` (18 `kind`s comunes con su traducción en español: `subclase_de` → `"es subclase de"`, `compuesto_por` → `"está compuesto por"`, etc.).
  - Nuevas funciones públicas `relation_kind_phrase(kind)` y `relation_to_natural(relation)`. La segunda serializa una `Relation` como `"X <verbo> Y (descripción)"` sin flechas ni corchetes.
  - `to_markdown` ahora usa `relation_to_natural`: el KB que se inyecta a TODOS los prompts downstream (plan de slides, conclusión, plan de quiz) ya no contiene la tripleta técnica.
- **`src/pptx_generator.py`**:
  - `_atom_block` para `Relation` usa `relation_to_natural` (elimina la duplicación de lógica).
  - Nuevas regex y listas para limpieza defensiva de bullets:
    - `_RELATION_ARROW_RE` captura variantes tipográficas de la tripleta (`—[...]→`, `--[...]-->`, `–[...]→`…). Si un bullet la contiene, se descarta.
    - `_ANGLICISM_RE` captura `blueprint`, `inheritance`, `overriding`, `concretada`, `concretado`.
    - `_META_EMBEDDED_PHRASES` amplía la detección de meta-lenguaje a frases **embebidas** (no sólo al inicio del bullet): `"sirve de ejemplo para ilustrar"`, `"facilita la creación y gestión"`, `"permite comprender"`, etc.
  - Nueva función `_bullet_echoes_title(bullet, title)`: detecta cuando un bullet arranca repitiendo el título de la slide seguido de un separador (`:`, `.`, `—`, `-`, `,`, `·`). Robusta ante variaciones de espacios y mayúsculas.
  - `_clean_bullet` recibe nuevo argumento opcional `slide_title` y aplica todos los nuevos detectores. Los bullets que fallan cualquiera de ellos se descartan → el revisor los contará como deficientes y pedirá regeneración.
  - `render_slide_bullets` pasa `slide.title` a `_clean_bullet`. `_render_conclusion` pasa `"Conclusiones"`.
- **`src/prompts.py` — `SLIDE_BULLETS_FROM_ATOMS_PROMPT`**:
  - **Nueva regla 3.1**: prohibición explícita de la notación de tripletas (`X —[subclase_de]→ Y`). Le decimos al LLM que los `kind` como `subclase_de`, `hereda_de`, `compuesto_por` son *etiquetas técnicas internas* y que los traduzca al español corriente.
  - **Regla de relations reescrita**: pasa de `"X —[tipo_relación]→ Y: explicación"` a "lenguaje natural con verbos como *es subclase de*, *hereda de*, *se compone de*".
  - **Nueva regla 4.1**: prohibición de eco del título en el primer bullet.
  - **Nueva regla 4.2**: anglicismos innecesarios y neologismos feos (`blueprint`, `concretada`, `inheritance`…).
  - **Regla 4 ampliada** con muletillas didácticas vacías (`"sirve de ejemplo para ilustrar"`, `"facilita la creación y gestión de"`, `"comprender X facilita"`, `"entender X es clave para"`…).
  - **Ejemplos positivos actualizados**: la muestra de `Relation` ahora es `"BicicletaDeMontaña es subclase de Bicicleta: hereda velocidad…"` (sin flechas).
  - **Ejemplos negativos añadidos**: se muestran explícitamente los patrones prohibidos (tripleta, eco de título, `blueprint`, `concretada`, meta-lenguaje didáctico).
- **`src/critics.py` — `_deterministic_slide_issues`** amplía las inspecciones:
  - `_SLIDE_META_PHRASES` añade las nuevas frases embebidas.
  - Nueva detección **5 `relation_arrow_leak`** (severity=high): bullets con notación técnica de tripleta → fuerza regeneración.
  - Nueva detección **6 `title_echo`** (severity=medium): bullets que empiezan repitiendo el título de la slide.
  - Nueva detección **7 `anglicism`** (severity=medium): bullets con `blueprint`, `inheritance`, `concretada`, etc.
  - Las tres disparan `critical_indices()` → `refine_slides` regenera esas slides.

Por qué en dos capas (prompt + post-proceso + crítico): el prompt resuelve la **mayoría** de los casos, pero el LLM local es propenso a recaer en formatos que ha visto antes. Las regex de `_clean_bullet` actúan como red de seguridad inmediata (descartan el bullet sin pagar otro LLM call), y el crítico determinístico del refine pide regeneración si tras el descarte una slide queda pobre. Con las tres líneas encadenadas, la notación interna **no puede** llegar al PPTX aunque el LLM falle.

Validado con casos manuales (`_clean_bullet`): notación con 4 variantes tipográficas de flecha + corchetes → DROP; `blueprint` → DROP; `concretada` → DROP; `"sirve de ejemplo para ilustrar"` → DROP; `"facilita la creación y gestión"` → DROP; eco de título → DROP; bullet con relación en español natural → KEEP.

**Sub-hito v2.12.1 — reutilización de las slides precargadas de la plantilla**

La plantilla `plantilla_universidad.pptx` trae dos slides precargadas:

- Slide 0: portada con tres shapes de texto **libre** (no placeholders estándar): `"TÍTULO"`, `"SUBTÍTULO"` y uno vacío, posicionados por Google Slides con tipografía/color propios.
- Slide 1: `TITLE_AND_BODY` con placeholders estándar vacíos.

El flujo anterior nunca tocaba esas slides: simplemente llamaba a `prs.slides.add_slide(...)` para la portada y el índice, que se insertaban **detrás**. Resultado: el PPTX entregado arrancaba con dos slides-basura (la portada con `"TÍTULO"` sin sustituir y la TITLE_AND_BODY en blanco) y la portada real aparecía en la diapositiva 3.

Cambios:

- **`src/pptx_generator.py`** — nuevos helpers:
  - `_replace_text_preserving_format(shape, new_text)`: sustituye el texto de un shape **conservando el formato del primer run** (fuente, tamaño, color, bold…). Necesario porque `tf.clear()+tf.text=...` destruiría el estilo definido en la plantilla.
  - `_overwrite_template_cover(slide, title, subtitle)`: localiza los shapes de la portada por texto existente (`"TÍTULO"`, `"SUBTÍTULO"`, variantes con/sin tilde) y cae a heurística por tamaño de fuente si los marcadores no aparecen. Reemplaza ambos preservando formato.
  - `_overwrite_template_body_slide(slide, title, bullets)`: rellena la slide 1 precargada con sus placeholders TITLE + BODY estándar usando los helpers existentes.
- **`render_pptx`** reescrito:
  - Si la plantilla tiene ≥1 slide precargada → se sobreescribe la portada.
  - Si tiene ≥2 y la segunda admite TITLE + BODY → se usa como **Índice** de la presentación.
  - Fallback: cualquier escenario donde la plantilla no traiga slides precargadas o no tengan los placeholders esperados → se añaden como antes con `_add_title_slide` / `_add_content_slide`.
  - El resto de slides (contenido + conclusiones) se añade detrás con `LAYOUT_CONTENT`, sin cambios.

Impacto: el PPTX entregado ahora arranca directamente con la portada corporativa con título real, seguida del índice, sin slides en blanco intermedias. El código es defensivo frente a plantillas sin slides precargadas (mantiene compatibilidad).

Validado con la plantilla actual: 6 slides totales para un plan de 3 contenidos + conclusiones; el layout y orden son correctos (`TITLE`, `TITLE_AND_BODY` × 5).

### v2.11 — Quiz adaptativo por calidad: rango `[min, max]` y pase único de refine

Un run real con el paradigma anterior (cantidad fija = 10) reveló dos patologías conjuntas:

- **Sobre-procesado**: 2 iteraciones de refine con regeneración agresiva por cualquier `severity=medium` disparaban ~45 llamadas LLM por ejecución (~6 min end-to-end).
- **Colapso de concept_id**: el fallback "menos usado" introducido en v2.10 hacía que **todas** las preguntas regeneradas acabaran con el mismo concept_id (`def:clase` con `usage=0`), generando un quiz monotemático artificial.
- **Relleno sintético de mala calidad**: cuando el plan LLM quedaba corto, `_synthesize_planned_questions` forzaba N preguntas reutilizando conceptos hasta el cap adaptativo, lo que chocaba después con el refine y provocaba más regeneraciones inútiles.

Cambio de paradigma: del "dame exactamente N preguntas" al "dime cuántas preguntas buenas quieres entre `min` y `max`, yo decido cuántas según el PDF". **La cantidad se adapta a la calidad disponible del KB**, no al revés.

Cambios:

- **`src/config.py`**:
  - `DEFAULT_NUM_QUESTIONS` y `MAX_NUM_QUESTIONS=30` sustituidos por `MIN_NUM_QUESTIONS=5`, `MAX_NUM_QUESTIONS=15`, `DEFAULT_NUM_QUESTIONS_RANGE=(5, 12)` y `QUIZ_PLAN_OVERSAMPLING=3`.
  - El tope máximo baja de 30 a 15: ningún PDF de longitud razonable da para más de 15 preguntas sin duplicados.
- **`app.py`**: el slider de un solo valor pasa a ser un **range slider** `[min, max]`. El sistema muestra en texto el rango elegido y, tras generar, informa cuántas preguntas pasaron el filtro de calidad (no es un error si salen menos del máximo).
- **`src/quiz_generator.py` — `generate_quiz`** completamente reescrito:
  - **Eliminado** `_synthesize_planned_questions` y toda la lógica de relleno determinista. Si el plan LLM da 8, trabajamos con 8.
  - Nueva función `_adaptive_target(n_atoms, min_q, max_q)` → `clamp(ceil(n_atoms × 1.4), min, max)`. KB con 5 átomos útiles → ~7 preguntas; KB con 12 → 15.
  - El plan se pide con oversampling (`target + 3`) para que el filtro de calidad tenga margen.
  - Firma cambia: `generate_quiz(client, kb, *, min_questions, max_questions, refine=True, use_llm_critic=False)`. Todos los parámetros de cantidad son kwargs-only.
- **`src/critics.py` — `refine_quiz`** completamente reescrito:
  - Pasa de N iteraciones a **un único pase de filtrado**:
    1. Correcciones baratas por regex (ver `_strip_meta_preamble` abajo).
    2. Una revisión determinística.
    3. Para cada pregunta crítica: UN intento de regeneración. Si la regeneración sigue siendo crítica o lanza `GenerationError`, la pregunta se **descarta** (no se insiste).
    4. Revisión final + descarte de las que no sobrevivieron el intento.
    5. Si quedan más de `max_questions`, se ordenan por nº de issues residuales (menos = mejor) y se conservan las mejores.
  - Firma cambia: `refine_quiz(client, kb, questions, plan, *, min_questions, max_questions, use_llm=False)`. Desaparece `max_iterations`.
  - `_ISSUES_REQUIRING_CONCEPT_SWAP` se reduce a `{"duplicate", "duplicate_stem"}`. `example_overuse` y `banned_phrase` ya no fuerzan rotación (v2.10 demostró que colapsaba el quiz a un único concepto).
  - `example_overuse` baja a `severity=low` → se reporta para log pero **no dispara regeneración**. Si el KB solo tiene 1-2 ejemplos es normal que el LLM tire de ellos.
- **`src/critics.py` — nueva utilidad `_strip_meta_preamble(text)`**:
  - Regex anclada al inicio del enunciado que detecta preámbulos meta del tipo `"según la definición del documento,"`, `"conforme al texto:"`, `"de acuerdo con el autor,"`, `"como se menciona en X:"`, etc. y los recorta.
  - Conserva el resto del enunciado y re-capitaliza la primera letra útil (respetando `¿` y `¡`).
  - Validado con 7 casos de prueba (incluidos los que no deben cambiar).
  - Se aplica en dos puntos: `_apply_cheap_fixes` antes de la revisión (para evitar marcar muletillas que son fácilmente recortables) y tras cada regeneración por seguridad.
- **Revisor LLM (`_llm_quiz_issues`)**: sigue disponible pero **desactivado por defecto** (`use_llm=False`). Los detectores determinísticos capturan >90% de issues con 0 coste. Queda como "modo inspección" útil para el futuro módulo de benchmarking.

Impacto esperado (con PDF POO 40k chars, mismo hardware):

| Métrica | Antes (v2.10) | Ahora (v2.11) |
|---|---|---|
| Llamadas LLM por ejecución | ~45 | ~15 |
| Tiempo total quiz | ~6 min | **~2 min** |
| Colapso de concept_id | frecuente | no existe (no hay swap fuera de duplicados) |
| Preguntas duplicadas por diseño | sí (relleno) | no (no hay relleno) |
| Preguntas por debajo del umbral | se aceptan | se descartan |

### v2.10 — Últimos retoques del quiz antes de pasar a slides

Tras v2.9 el quiz quedó prácticamente estable, pero un nuevo run de 10 preguntas dejó tres defectos concretos:

1. **Dos preguntas con el mismo enunciado** ("¿Qué es la mejor forma de encapsular los métodos y atributos de una bicicleta de montaña según el concepto central?" aparece 2 veces). El detector `duplicate_stem` sí lo marcó, pero el refine **no pudo rotar** el `concept_id`: con el cap adaptativo y un KB pequeño, **todos** los `atom_ids` ya estaban en el plan, y `_pick_alternative_concept` devolvía `None` → la pregunta duplicada se conservaba tal cual.
2. **Paréntesis informativos al final de opciones** ("(Ejemplo de programación estructurada)", "(Programación orientada a objetos)"). El `_clean_option_text` de v2.9 solo limpiaba `(error)`, `(distractor)`, etc.; las "etiquetas temáticas" parentéticas son otro patrón distinto y equivalentemente dañino (funcionan como pista).
3. **Muletilla nueva "según los malentendidos típicos del alumno"**: el LLM copió literalmente esa frase del comentario del ejemplo positivo 2 del propio prompt, introduciéndola en el enunciado de una pregunta.

Cambios:

- **`src/critics.py` — `_pick_alternative_concept`**:
  - Nuevos parámetros `usage: dict[str, int]` y `current: str`. Mantiene el orden de preferencia (prefijo → no usado → cualquier no usado), pero añade **fase 3**: si *todos* los átomos ya están en `avoid`, devuelve el átomo con **menor uso** en el plan (distinto del `current`), garantizando que un duplicado con rotación forzada sí encuentre destino.
  - El caller en `refine_quiz` pasa `usage` (conteo de `concept_id` → nº de veces que aparece en el plan actual) y `current=planned.concept_id`.
- **`src/quiz_generator.py` — `_clean_option_text`**:
  - Nueva regex `_OPTION_TRAILING_PAREN_RE` que elimina paréntesis al final de opción sólo si:
    - comienzan con mayúscula (mayús + acentuadas) → heurística de "etiqueta" vs "info intra-frase";
    - contienen ≥ 3 palabras y ≥ 12 caracteres de contenido.
  - Así se preservan los paréntesis legítimos como "(hija)", "(padre)", "mover()" o explicaciones cortas con minúscula inicial.
  - Pasada iterativa (hasta 3 veces) para cubrir el caso de dos anotaciones consecutivas.
- **`src/critics.py` — `_QUIZ_META_PHRASES`**: añadidas variantes que son **copias del vocabulario del prompt**: `"segun los malentendidos"`, `"malentendidos tipicos"`, `"malentendidos del alumno"`, `"segun el concepto central"`, `"segun la plantilla"`, `"segun la taxonomia"`, `"segun bloom"`, `"segun el nivel bloom"`, `"del alumno tipico"`.
- **`src/prompts.py` — `QUIZ_QUESTION_PROMPT`**:
  - Regla 6 reforzada con un segundo bloque: "tampoco copies palabras del propio prompt al enunciado" (lista concreta: *malentendidos*, *concepto central*, *plantilla*, *nivel Bloom*, *distractor*).
  - Comentario del ejemplo positivo 2 reescrito ("versiones parciales/invertidas de la correcta" en lugar de "malentendidos típicos del alumno") para eliminar la fuente del contagio.

Impacto esperado: las preguntas duplicadas que antes se conservaban por falta de alternativa de átomo ahora se regeneran con otro concept_id; las etiquetas parentéticas al final desaparecen; la muletilla "malentendidos…" deja de ser imitada porque ya no aparece en el prompt y, si el LLM persiste, el crítico la marca.

Con esto se cierra oficialmente la fase de pulido del quiz. Los defectos residuales (el prior hacia "banco común de distractores" en `recordar` puro, preguntas vagas sobre átomos con definición pobre como `tándem`) son limitaciones del LLM local y entran en el ámbito del módulo de **benchmarking** que se abordará después de las slides.

### v2.9 — Pulido final del quiz: marcadores, muletillas restantes y duplicados por enunciado

Tras v2.8, el quiz quedó "medio bien": 10/10 con ~50% de preguntas de calidad real, pero surgieron tres defectos menores:

- **Marcadores `(error)` en opciones**: el LLM malinterpretó el ejemplo negativo del prompt y empezó a "anotar" opciones con `(error)`, `(incorrecto)`, etc.
- **Muletillas no cubiertas**: `"según su definición literal"`, `"según el ejemplo dado"`.
- **Duplicados por enunciado**: el Jaccard sobre firma completa (enunciado + 4 opciones) es robusto, pero no captura el caso "dos preguntas con el MISMO enunciado y distractores distintos", donde el estudiante memoriza el enunciado y la pregunta le sigue siendo la misma.

Cambios:

- **`src/prompts.py`**: nueva regla 5.1 en `QUIZ_QUESTION_PROMPT` prohibiendo marcadores parentéticos en opciones.
- **`src/quiz_generator.py`**: nueva utilidad `_clean_option_text(raw)` con regex `_OPTION_TAG_RE` que elimina `(error|incorrecto|distractor|falso|correcto|respuesta correcta|…)` de las opciones aunque el LLM cuele el marcador. Aplicado como `field_validator` de `QuizOptions`. Defensa a posteriori.
- **`src/critics.py`**:
  - `_QUIZ_META_PHRASES` ampliado con `"segun su definicion literal"`, `"segun el ejemplo dado"`, `"segun este ejemplo"`, `"segun lo visto"`, etc.
  - Nuevo umbral `_STEM_DUPLICATE_JACCARD_THRESHOLD = 0.70` aplicado sobre los tokens del **enunciado solo** (sin opciones). Si dos preguntas tienen el mismo enunciado (aunque varíen los distractores) se marca `duplicate_stem` con `severity=high`.
  - `_ISSUES_REQUIRING_CONCEPT_SWAP` extendido con `"duplicate_stem"` para que el refine rote el `concept_id` (regenerar con el mismo concepto reproduciría el mismo enunciado).

Este es el último ajuste planificado del quiz. El anti-patrón "banco común de distractores" residual en definiciones de nivel `recordar` se considera limitación del LLM local (el prior del modelo es muy fuerte) y entra en el ámbito del futuro módulo de benchmarking.

### v2.8 — "Textos prohibidos" inyectados, refine a 2 iteraciones y más muletillas

Tras v2.7, un run de 10 preguntas mejoró drásticamente (10/10, Bloom distribuido, varias preguntas con distractores dentro del mismo concepto). Pero P1/P2/P3 reincidían en el banco común y aparecía una muletilla nueva ("según la definición proporcionada") que no estaba en el detector.

Cambios:

- **`src/prompts.py` — `QUIZ_QUESTION_PROMPT`**:
  - Nueva sección `TEXTOS PROHIBIDOS COMO OPCIONES`: se inyecta al prompt, con hasta 6 definiciones del KB que NO son del concepto central, listadas textualmente. Instrucción dura: "si una de tus 4 opciones se parece a alguno de esos textos (incluso reformulado), la pregunta se descarta". Los experimentos muestran que el LLM obedece mucho mejor una lista negra explícita que una regla abstracta.
  - Regla 4 actualizada para referenciar esta sección como "lo más importante del prompt".
- **`src/quiz_generator.py`**:
  - Nueva utilidad `_forbidden_definitions(kb, exclude_id)` que extrae las `Definition`s del KB distintas al concepto central, ordena por (`verbatim` primero, luego más cortas — las más susceptibles de ser copiadas verbatim por el LLM) y devuelve las 6 primeras formateadas.
  - `generate_single_question` inyecta `forbidden_texts` al prompt.
  - `generate_quiz(..., max_refine_iterations: int = 2)` — subido de 1 a 2. La primera ronda corrige ~60% y la segunda limpia el residuo, especialmente para `distractor_off_concept` donde el LLM a veces tarda dos pases en entender la restricción.
- **`src/critics.py`**:
  - `_QUIZ_META_PHRASES` ampliado con: `"segun la definicion proporcionada"`, `"segun la definicion oficial"`, `"de acuerdo con la definicion"`, `"segun lo proporcionado"`, `"segun lo indicado"`, `"conforme a la definicion"`, `"el documento indica"`, `"el documento describe"`, `"como se define en"`, `"tal como se define"`, `"basandose en el documento"`, `"basandose en la definicion"`.

Impacto esperado: P1–P3 dejan de usar el banco común gracias a los textos prohibidos; las muletillas "según la definición proporcionada" caen en el filtro y fuerzan regeneración; la segunda iteración del refine rescata los issues residuales.

### v2.7 — Cap adaptativo, anti-banco-de-distractores y anti-muletillas

Los logs de un run con `num_questions=10` mostraban:

```
Quiz: 5/10 tras el plan; rellenando 1 con átomos no usados.
Revisor quiz: iter 0 — regenerando ids=[1, 2, 3, 4, 5, 6]
Revisor quiz: 8 issues tras refinamiento (críticos: 6).
```

Tres problemas combinados:

1. Con `cap=2` rígido, un KB pequeño (PDF de POO con ~6 átomos útiles) no podía producir 10 preguntas.
2. El relleno solo reutilizaba átomos "no usados"; no volvía a un concepto aunque el cap lo permitiera.
3. El refine no mejoraba: el LLM seguía generando **el mismo banco de distractores** en cada regeneración (las definiciones de los 4 conceptos básicos rotando entre las 4 opciones).

Cambios:

- **`src/plans.py`**:
  - Nueva constante `ABSOLUTE_MAX_PER_CONCEPT = 4` y función `adaptive_max_per_concept(num_atoms, target_questions)` → `ceil(target / atoms)` acotado a `[MAX_QUESTIONS_PER_CONCEPT, ABSOLUTE_MAX_PER_CONCEPT]`.
    - KB con 6 átomos, 10 preguntas → cap=2 (suficiente).
    - KB con 4 átomos, 10 preguntas → cap=3 (3×4=12 ≥ 10).
  - `sanitize_quiz_plan` usa el cap adaptativo; logs en `INFO` del valor usado para cada run.
- **`src/quiz_generator.py`**:
  - `_synthesize_planned_questions` reescrito: en vez de recibir `used: set[str]`, recibe `used_counts: dict[str, int]` y `per_concept_cap: int`. Dos fases:
    1. Fase **fresh**: átomos aún no usados.
    2. Fase **reuse**: átomos bajo el cap; rota bloom_level para que la regeneración no produzca la misma pregunta.
  - `generate_quiz` calcula el cap adaptativo y lo pasa al synthesize.
- **`src/critics.py`**:
  - Nueva utilidad `_kb_definitions_tokens(kb)` + detector `_detect_off_concept_distractors(q, kb_defs)` con umbral `_OFF_CONCEPT_JACCARD_THRESHOLD = 0.45` y `_OFF_CONCEPT_MIN_DEFS = 3`.
  - Nueva regla en `_deterministic_quiz_issues`: marca `distractor_off_concept` con `severity=high` cuando las opciones cubren ≥3 definiciones KB distintas (el anti-patrón M3: "banco común de distractores").
  - `_QUIZ_META_PHRASES` extendido con variantes: `"segun la definicion del documento"`, `"de acuerdo con el documento"`, `"como se menciona en"`, `"tal y como se indica"`, `"conforme al documento"`, etc.
- **`src/prompts.py`**:
  - `QUIZ_QUESTION_PROMPT` reforzado:
    - Nueva regla 4 ("distractores dentro del mismo concepto"): describe 4 patrones buenos (variante incompleta / invertida / malentendido / aplicación fuera de lugar) y prohíbe explícitamente construir distractores pegando definiciones de otros átomos del contexto relacionado.
    - Nueva regla 6 (muletillas prohibidas): lista explícita de frases meta.
    - Nuevo ejemplo negativo que reproduce el anti-patrón real observado (pregunta sobre objeto con distractores copiados de clase/herencia/polimorfismo).
    - Nuevo ejemplo positivo con distractores "dentro del mismo concepto" (variantes de "¿qué es un objeto?" donde las 4 opciones hablan de objetos).

Impacto esperado: el quiz deja de ser resoluble por eliminación, el 6/10 debería subir a 10/10 con KB pequeños y el refine tendrá ejemplos claros para producir distractores conceptualmente correctos.

### v2.6 — Tolerancia a `kind` fuera de vocabulario en el QuizPlan

Problema observado: el LLM devuelve a veces `"kind": "ejemplo"` u otras variantes no contempladas en el `Literal QuestionKind` (pudiendo ser traducciones inglesas como `"definition"`, `"comparison"`, o errores semánticos como copiar el prefijo del átomo: `ex:...` → `kind=ejemplo`). Pydantic lanza `ValidationError` y todo `plan_quiz` revienta, pese a que el resto del plan era válido.

Cambios en **`src/quiz_generator.py`**:

- Nuevo mapa `_KIND_ALIASES` con las variantes observadas (inglesas, con acentos, y errores semánticos típicos — `ejemplo`, `relacion`, etc.).
- Nueva constante `_VALID_KINDS` (mirror del `Literal` para chequear sin importar Pydantic).
- Nueva utilidad `_normalize_kind(value, bloom_level)` que:
  1. aplica los alias,
  2. si sigue siendo desconocido, hace **fallback al `kind` recomendado** por `BLOOM_RECOMMENDED_KINDS[bloom_level]`. Así no se pierde la pregunta por una etiqueta mal escrita.
- `_normalize_plan_entries` llama ahora a `_normalize_kind` por cada entrada del plan.

### v2.5 — Anti-duplicados y anti-monotema en el quiz

Problema observado: con un PDF corto (POO con un único ejemplo "Bicicleta") aparecían hasta 5 preguntas sobre el mismo ejemplo y dos preguntas literalmente idénticas con distinto Bloom. El deduplicador por tupla `(concept_id, bloom, kind)` no era suficiente: el LLM generaba el mismo enunciado aunque el plan cambiara la terna.

Cambios:

- **`src/plans.py`**:
  - Nueva constante `MAX_QUESTIONS_PER_CONCEPT = 2`.
  - `sanitize_quiz_plan` ahora limita a **2** preguntas como máximo por `concept_id`, además del filtro por tupla `(concept_id, bloom, kind)`. Registra `WARNING` cuando se alcanza el cap.
- **`src/critics.py`**:
  - Nueva utilidad `_jaccard(a, b)` y `_question_signature_tokens(q)` (bolsa de tokens de enunciado + 4 opciones).
  - Nueva constante `_DUPLICATE_JACCARD_THRESHOLD = 0.80`. La regla determinista de duplicados sustituye al viejo `stem_key` (que solo miraba 8 tokens) por un Jaccard sobre el texto completo. Dos preguntas con Jaccard ≥ 0.80 se marcan con `severity=high`, lo que obliga al refinamiento a regenerarlas.
  - Nueva utilidad `_example_name_tokens(kb)` + constante `_MAX_QUESTIONS_MENTIONING_SAME_EXAMPLE = 2`. El detector `example_overuse` marca cualquier pregunta que sea la 3.ª+ en mencionar el nombre de un mismo `Example` del KB en su enunciado o respuesta correcta.
  - Nuevas utilidades `_pick_alternative_concept(...)` y conjunto `_ISSUES_REQUIRING_CONCEPT_SWAP = {"duplicate", "example_overuse"}`. En `refine_quiz`, cuando el issue es `duplicate` o `example_overuse`, se **rota** el `concept_id` del `PlannedQuestion` a un átomo del KB no usado antes de regenerar (respetando el prefijo original si es posible). Regenerar con el mismo concepto reproducía el mismo defecto.

Impacto: el quiz deja de girar alrededor del único ejemplo disponible y los duplicados literales aunque cambien de Bloom se detectan y regeneran automáticamente.

### v2.4 — Robustez del quiz: retry + relleno determinista

Problema observado en producción: con PDFs reales se producían solo 3 preguntas de 10 solicitadas, sin feedback para el usuario. Causas:
- El LLM repetía `concept_id` y la dedup `(concept_id, bloom, kind)` descartaba la mayoría.
- `generate_single_question` fallaba silenciosamente (JSON inválido) y el loop seguía.
- La UI no avisaba de la diferencia entre lo pedido y lo generado.

Cambios:

- **`src/quiz_generator.py`**:
  - `generate_single_question` ahora reintenta con temperaturas `(0.4, 0.55)` antes de lanzar `GenerationError`. Logs en `DEBUG` para cada intento fallido. El retry captura `(OllamaError, GenerationError, ValidationError)` — especialmente importante para el caso de **JSON truncado** por el LLM, que antes reventaba el pipeline — y deja propagar `OllamaUnavailableError` / `OllamaModelNotFoundError` (errores de infraestructura que no deben enmascararse).
  - Nuevo helper interno `_try_generate_question` que se usa en loops (traga el error controlado y devuelve `None`).
  - Nuevo helper `_synthesize_planned_questions(kb, used, count, start_id)` que crea `PlannedQuestion` deterministas a partir de átomos aún no preguntados, rotando el ciclo Bloom `(aplicar, analizar, comprender, evaluar, recordar)` y priorizando `ex:*` y `rel:*` (mejores distractores).
  - `generate_quiz` rellena automáticamente los huecos cuando el plan LLM queda corto. El plan se extiende con las preguntas sintetizadas para que el revisor posterior también pueda regenerarlas.
- **`app.py`**: muestra un `st.warning` visible si al final hay menos preguntas que las solicitadas, explicando la causa probable (KB con pocos átomos únicos).

### v2.3 — Calidad visual de los bullets PPTX

Problema: bullets cortados con `…`, copias textuales del bloque de átomos del contexto (`def:xxx · definición · …`), meta-lenguaje, bullets demasiado cortos, conclusiones duplicadas.

- **`src/config.py`**: `MAX_CHARS_PER_BULLET` 140 → **180** (~2 líneas a 20 pt, permite densidad técnica).
- **`src/pptx_generator.py`**: añadidas tres utilidades de texto:
  - `_truncate_title(text, max_len)`: corte duro con `…` *solo* para títulos.
  - `_trim_by_words(text, max_len)`: recorte por palabra **sin** puntos suspensivos, cerrando con `.` si hace falta. Usado para bullets.
  - `_clean_bullet(text, max_len)`: descarta bullets con `…`/`...`, con prefijos de átomo (`def:`, `ex:`, …), con patrones `· definición ·`/`· ejemplo ·`, con aperturas meta y con longitud <25 chars. Si todos los bullets de una slide se descartan, se lanza `GenerationError` para que el revisor dispare la regeneración.
- **`render_slide_bullets`** y **`_render_conclusion`** usan `_clean_bullet`.
- **`render_pptx`** filtra slides vacías y **construye el índice dinámicamente** solo con las slides que sobreviven (antes podía mostrar en el índice slides que no existían tras el filtrado).
- **`SLIDE_BULLETS_FROM_ATOMS_PROMPT`** y **`CONCLUSION_FROM_KB_PROMPT`** reforzados con reglas explícitas: longitud 12-28 palabras por bullet, prohibición de elipsis, prohibición de copia del bloque de átomos, prohibición de meta-lenguaje, prohibición de inventar ejemplos fuera de la KB (p. ej. si la KB solo habla de “Bicicleta”, no introducir “Coche”). Incluyen ejemplos positivos y negativos concretos.
- **`SLIDE_PLAN_PROMPT`**: explícitamente prohíbe `kind=conclusion` (la conclusión la añade el renderer).
- **`sanitize_slide_plan`**: descarta slides con `kind=conclusion` **o** título que empieza por `"conclus"` para evitar la duplicación aunque el LLM la cuele con otro kind.

---

## 10. Decisiones de diseño y *por qué* no se hizo de otra forma

- **Por qué KnowledgeBase y no un RAG con embeddings**. Los embeddings requieren un modelo adicional siempre cargado (más VRAM) y complican la auditoría. La KB explícita cabe en la ventana de contexto de un 7B y permite trazabilidad (cada bullet/pregunta apunta a un id). Un RAG se añadirá solo si pedimos dedup semántica global.
- **Por qué planificar antes de generar**. Generar en un shot provoca duplicados, drift y sesgo hacia `recordar`. Un plan tipado hace auditables la distribución Bloom, los conceptos cubiertos y los solapes.
- **Por qué prompts en español**. Mejor calidad sobre documentos académicos en español: el modelo evita calcos inglés→español en términos (especialmente en modelos 7B). Además el público objetivo de universidad hispanohablante exige el registro correcto.
- **Por qué `modelo:tag` exacto en la validación**. Ollama `/api/generate` exige el nombre completo; un falso positivo en el preflight se paga con error opaco durante la generación, difícil de diagnosticar.
- **Por qué fuzzy matching restringido al mismo prefijo**. Un `rel:algo_mal_escrito` solo tiene sentido como relación; permitirle resolver a un `def:*` rompería la coherencia del kind del slide o del kind de la pregunta. El filtro por prefijo es barato y elimina los falsos positivos.
- **Por qué `_clean_bullet` descarta en vez de arreglar los `…`**. Intentar “completar” un bullet truncado siempre termina en una frase sintéticamente mala. Es más limpio provocar `GenerationError` y dejar que el revisor regenere la slide completa con nuevo contexto.
- **Por qué el renderer siempre añade conclusión y no el LLM**. La conclusión “administrativa” es determinista (título fijo, bullets filtrados con las mismas reglas). Si también la añade el LLM, aparece duplicada; el saneador cierra esa puerta.
- **Por qué se conservan prompts legacy (`QUIZ_GENERATION_PROMPT`, `OUTLINE_PROMPT`, `SLIDE_CONTENT_PROMPT`, `CONCLUSION_PROMPT`)**. Sirven de fallback para depuración y como referencia. `consolidate_document` usa el REDUCE legacy para quien aún espere Markdown libre.

---

## 11. Roadmap pendiente

Orden sugerido, acordado con el usuario (“bien hecho, poco a poco, calidad”). Cuando se complete una entrada, muévela a la sección [Historial de mejoras](#9-historial-de-mejoras-changelog-técnico).

**Bloque 3 — Calidad pedagógica (resto)**

- [ ] **3.1** Enriquecer distribución Bloom: *verificación post-plan* + rebalanceo dirigido (ahora mismo la distribución la pide el prompt, pero no se fuerza post-hoc).
- [ ] **3.3** Plantillas por `kind` de slide más específicas (ahora hay reglas generales + mapa `BLOOM_RECOMMENDED_KINDS`; falta cada kind con su “esqueleto” narrativo).

**Bloque 4 — Refinamiento avanzado**

- [ ] **4.2** Deduplicación semántica con embeddings locales (opcional; actualmente se usa Jaccard de tokens y stem). Requiere modelo de embeddings en Ollama.

**Bloque 5 — Material crudo**

- [ ] **5.2** Speaker notes (notas del ponente) por slide, ancladas a la KB.

**Bloque 6 — Mejoras visuales PPTX**

- [ ] Uso de layouts variados de la plantilla (no solo 0 y 2).
- [ ] Sub-bullets cuando el contenido lo pide (p. ej. `kind=example` con atributos/métodos).
- [ ] Slide inicial de “conceptos clave” (glosario visual) a partir de `def:*`.
- [ ] Insertar imágenes del PDF en slides de `kind=example` cuando existan.

---

## 12. Convenciones de contribución y del README

1. **Este README es obligatorio de mantener**. Cualquier PR que añada o cambie comportamiento debe modificar como mínimo una de estas secciones: [5. Módulos y API interna](#5-módulos-y-api-interna), [6. Modelos de datos y contratos](#6-modelos-de-datos-y-contratos), [9. Historial de mejoras](#9-historial-de-mejoras-changelog-técnico) y, si procede, [7. Prompts](#7-prompts) y [11. Roadmap pendiente](#11-roadmap-pendiente).
2. **Cada entrada del changelog** debe incluir: qué cambió, por qué, y los archivos afectados.
3. **Imports**: `from __future__ import annotations` en todos los módulos `src/*.py`; tipado con `typing.Literal` para vocabularios cerrados.
4. **Pydantic antes que dataclass** cuando hay validación o serialización JSON involucrada; `dataclass` para contenedores internos puros.
5. **Prompts**: siempre en `src/prompts.py`, escritos en español, con reglas numeradas, formato JSON exacto y ejemplos positivos/negativos cuando el LLM tiende a fallar.
6. **Logs**: `logger = logging.getLogger(__name__)` por módulo. `INFO` para eventos normales, `WARNING` para decisiones de fallback (id descartado, slide omitida), `ERROR` para fallos que interrumpen.
7. **Errores de dominio**: usar las clases de `src/exceptions.py`. No lanzar `Exception` genérico.
8. **Importaciones circulares**: si un módulo del revisor necesita un generador, usar `TYPE_CHECKING` para type hints y **import diferido dentro de la función** para el runtime (patrón ya aplicado en `critics.py`).
