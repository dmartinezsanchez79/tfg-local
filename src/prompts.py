"""Prompts centralizados para Map-Reduce, Quiz y Slides.

Los prompts con JSON de ejemplo se construyen SIN f-strings para evitar
conflictos con `str.format()`. Las llaves reales del JSON se escriben como
`@@OPEN@@`/`@@CLOSE@@` y se sustituyen al final por `{{`/`}}`. La
configuración dinámica se inyecta con marcadores tipo `__MAX_BULLETS__`.
"""
from __future__ import annotations

from .config import (
    DEFAULT_NUM_SLIDES_MAX,
    DEFAULT_NUM_SLIDES_MIN,
    MAX_BULLETS_PER_SLIDE,
    MAX_CHARS_PER_BULLET,
    MAX_CHARS_SLIDE_TITLE,
)

SYSTEM_EXPERT_ES = (
    "Eres un profesor universitario experto en síntesis académica y diseño "
    "instruccional. Respondes siempre en español, con precisión y sin inventar "
    "información que no esté en el texto fuente."
)

# -------------------------------------------------------- MAP (por chunk) ---

MAP_SUMMARY_PROMPT = """\
Eres un asistente de síntesis académica. A continuación tienes un fragmento
de un documento más largo (en Markdown, con posibles tablas).

Extrae de este fragmento SOLO:
1. Los conceptos clave (3-8 bullets cortos).
2. Las definiciones o fórmulas importantes (si las hay).
3. Los datos numéricos, resultados o insights de las tablas (si existen).
4. Cualquier ejemplo o caso destacable.

NO inventes información. Si una sección no aplica, omítela.
Escribe todo en español, usando bullets concisos (una línea cada uno).

--- FRAGMENTO {index}/{total} ---
{chunk}
--- FIN DEL FRAGMENTO ---
"""

# ---------------------------------- REDUCE hacia KnowledgeBase (JSON) ------

REDUCE_TO_KB_PROMPT = """\
Consolida los resúmenes parciales y el material literal de un documento
académico en una Base de Conocimiento (KB) estructurada en JSON.

REGLAS
1. Usa SIEMPRE las definiciones literales del material cuando existan:
   copia la redacción tal cual y marca `verbatim: true`.
2. Si no hay literal, formula desde los resúmenes con `verbatim: false`.
   NO inventes nada fuera del documento.
3. `id` único por átomo, minúsculas, kebab/snake:
   def:<slug> · ex:<slug> · fc:<slug> · dt:<slug> · rel:<slug>
4. `subtopic` coherente con la lista `subtopics`.
5. Los ejemplos (`examples`) son ENTIDADES concretas del documento
   (p. ej. "enzima", "algoritmo", "empresa", "célula"); extrae
   `attributes` y `methods` cuando aparezcan.
6. Relaciones: tripletas `(source, kind, target)`.
7. Conclusiones:Muy importante mínimo 3 y máximo 5 frases sustantivas.
8. Responde SOLO con JSON válido. Sin markdown. Sin texto extra. Español.

ESQUEMA
@@OPEN@@
  "main_topic": "Tema central real del documento (no copies este texto)",
  "subtopics": ["..."],
  "definitions": [@@OPEN@@ "id":"def:slug","term":"...","definition":"...","subtopic":"...","verbatim":false @@CLOSE@@],
  "examples":    [@@OPEN@@ "id":"ex:slug","name":"...","description":"...","attributes":["..."],"methods":["..."],"subtopic":"..." @@CLOSE@@],
  "formulas_code":[@@OPEN@@ "id":"fc:slug","kind":"code","content":"...","caption":"...","language":"python","subtopic":"..." @@CLOSE@@],
  "numeric_data":[@@OPEN@@ "id":"dt:slug","value":"42%","description":"...","subtopic":"..." @@CLOSE@@],
  "relations":   [@@OPEN@@ "id":"rel:slug","kind":"subclase_de","source":"A","target":"B","description":"...","subtopic":"..." @@CLOSE@@],
  "conclusions": ["..."]
@@CLOSE@@

--- RESÚMENES PARCIALES ---
{partials}
--- FIN RESÚMENES ---

--- MATERIAL LITERAL (puede estar vacío) ---
{literal_hints}
--- FIN MATERIAL LITERAL ---
"""

# ----------------------------------------- PLAN DE PRESENTACIÓN -----------

SLIDE_PLAN_PROMPT = """\
Diseñas el plan de una presentación académica.

OBJETIVO
JSON `SlidePlan` que asigna átomos a slides. NO redactes bullets aquí.

REGLAS
1. Entre __MIN__ y __MAX__ slides (sin portada, índice ni conclusión final).
2. Cada slide: `title` (máx __TITLE_LEN__ chars), `kind` cerrado, `atom_ids`.
3. Tipos válidos: intro · definition · example · comparison · code · process ·
   relations · outlook. NO uses "conclusion".
4. Cada `atom_ids` SOLO contiene IDs presentes en `IDS VÁLIDOS`. Prohibido
   inventarlos. Si dudas, deja el array vacío.
5. Cubre los subtemas relevantes; no dupliques átomos salvo que aporte valor.
6. No numeres los títulos.

EJEMPLO (formato exacto, contenido ilustrativo):
@@OPEN@@
  "presentation_title": "Programación Orientada a Objetos",
  "slides": [
    @@OPEN@@ "title":"Introducción","kind":"intro","atom_ids":[],"focus":"Visión general" @@CLOSE@@,
    @@OPEN@@ "title":"Clases y objetos","kind":"definition","atom_ids":["def:clase","def:objeto"] @@CLOSE@@
  ]
@@CLOSE@@

IDS VÁLIDOS (usa SOLO estos):
{valid_atom_ids}

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN ---
"""

# ------------------------------------- BULLETS POR SLIDE (por átomos) -----

SLIDE_BULLETS_FROM_ATOMS_PROMPT = """\
Redactas UNA diapositiva académica profesional.

Presentación: "{presentation_title}"
Slide {index}/{total}: "{slide_title}" · tipo `{kind}`
Intención: "{focus}"

ÁTOMOS ASIGNADOS (fuente única; reescribe con tus palabras, NO copies el
formato `id · tipo · …`):
{atom_details}

ÍNDICE COMPLETO (solo para no duplicar entre slides):
{outline}

CONTENIDO YA CUBIERTO EN SLIDES ANTERIORES (evítalo):
{anti_repeat_context}

REGLAS
1. Devuelve 3 o 4 bullets (5 solo si el contenido lo exige de forma clara).
   Cada bullet es UNA frase completa y cerrada, 10-22 palabras, nunca más de
   __MAX_CHARS__ caracteres. Si no cabe, reformula más breve.
   Prohibido usar puntos suspensivos.
2. Cada bullet DEBE terminar con punto final y nunca puede acabar en una
   palabra suelta como "de", "del", "la", "el", "y", "o", "en", "con", "por".
3. Prohibido copiar del bloque de átomos: ni los prefijos `def:`, `ex:`,
   `fc:`, `rel:`, `dt:`, ni el patrón "concepto · tipo · …".
4. Prohibida la notación técnica de relaciones: nunca escribas tripletas
   con flechas y corchetes ("X —[subclase_de]→ Y"). Traduce siempre al
   español natural: "X es subclase de Y", "X hereda de Y", "X se compone
   de Y", "X contiene Y", etc.
5. Prohibido lenguaje meta ("en este apartado…", "se habla de…",
   "es importante destacar…", "permite comprender…").
6. El primer bullet NO puede repetir el título de la slide: entra directo
   al contenido (atributos, ejemplos, consecuencias).
7. Sin anglicismos gratuitos ("blueprint" → "plantilla"; no "inheritance",
   "overriding" ni "concretada").
8. Cada bullet aporta información nueva (datos, atributos, comparaciones,
   consecuencias); nada de generalidades.
9. Los bullets se apoyan SOLO en los átomos asignados. No inventes
   entidades ni analogías nuevas (si los átomos hablan de la entidad X,
   no la sustituyas por otra entidad distinta Y).
10. Responde SOLO JSON válido.
11. No repitas ideas ya incluidas en "CONTENIDO YA CUBIERTO"; aporta
   información nueva para esta slide.

PAUTAS POR TIPO
- definition : definición + ejemplo concreto + consecuencia/relación.
  No antepongas "Definición:".
- example    : atributos y métodos EXACTOS + 1 bullet de insight.
- comparison : "A vs B: diferencia concreta".
- code       : 2 bullets explicativos + 1 con el fragmento en `código`.
- process    : pasos numerados ("1. …", "2. …").
- relations  : relación en español natural ("X es subclase de Y y
  sobrescribe …"). Verbos: "es subclase de", "hereda de", "implementa",
  "se compone de", "depende de", "contiene".
- intro      : anuncia el hilo con 1-2 datos concretos.
- outlook    : afirmaciones panorámicas sustantivas, no repiten detalles.

EJEMPLOS BUENOS
- "La variable dependiente cambia en función de la concentración del
  reactivo y del tiempo de exposición."
- "El algoritmo ordena la lista en tiempo O(n log n) usando una fase de
  particionado y combinaciones sucesivas."
- "La clase derivada hereda la interfaz común y redefine un método para
  adaptar el comportamiento al nuevo contexto."

FORMATO
@@OPEN@@
  "bullets": ["...", "..."]
@@CLOSE@@
"""

# ------------------------------------- CONCLUSIÓN FINAL (desde KB) --------

CONCLUSION_FROM_KB_PROMPT = """\
Diapositiva FINAL de CONCLUSIONES sobre "{presentation_title}".

Usa las conclusiones y relaciones de la KB. Evita repetir las definiciones
literalmente.

REGLAS
- Entre 3 y __MAX_BULLETS__ bullets; cada uno es UNA frase 12-28 palabras,
  máx __MAX_CHARS__ caracteres. Sin puntos suspensivos.
- Prohibido copiar el bloque de átomos (`def:`, `ex:`, "concepto · tipo · …").
- Afirmaciones concretas: síntesis, implicaciones, conexiones. No
  "es muy importante…", no "para concluir…", no "en resumen…".
- Responde SOLO JSON válido.

FORMATO
@@OPEN@@
  "bullets": ["...", "..."]
@@CLOSE@@

--- MATERIAL DE LA KB ---
{kb_context}
--- FIN ---
"""

# ------------------------------------------- PLAN DE QUIZ -----------------

QUIZ_PLAN_PROMPT = """\
Diseñas el plan de un quiz académico ANTES de redactar las preguntas.
Objetivo: {num_questions} preguntas.

REGLAS
1. Distribución Bloom orientativa sobre {num_questions}:
   recordar 15-25 % · comprender 15-25 % · aplicar 20-30 % ·
   analizar 15-25 % · evaluar 10-20 % · crear 0-10 %.
   Mínimo 1 de "aplicar" y 1 de "analizar".
2. `concept_id` DEBE estar en `IDS VÁLIDOS`. Si no, NO la incluyas.
3. `kind` coherente con `bloom_level`:
   recordar/comprender → "definicion"|"diferenciacion"
   aplicar/crear       → "caso_practico"|"completar_codigo"
   analizar            → "comparacion"|"analisis_consecuencia"
   evaluar             → "juicio_alternativas"|"analisis_consecuencia"
4. Prioriza ejemplos (ex:*) y relaciones (rel:*) en niveles altos;
   definiciones (def:*) para recordar/comprender.
5. Permitido repetir `concept_id` solo si cambia (bloom_level, kind).
6. `focus` opcional: pista narrativa corta (≤120 chars).

EJEMPLO (formato exacto, contenido ilustrativo):
@@OPEN@@
  "questions": [
    @@OPEN@@ "id":1,"bloom_level":"recordar","concept_id":"def:objeto","kind":"definicion" @@CLOSE@@,
    @@OPEN@@ "id":2,"bloom_level":"aplicar","concept_id":"ex:cuenta","kind":"caso_practico" @@CLOSE@@
  ]
@@CLOSE@@

IDS VÁLIDOS (usa SOLO estos):
{valid_atom_ids}

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN ---
"""

# ------------------------------ GENERAR UNA PREGUNTA (1 a 1) --------------

QUIZ_QUESTION_PROMPT = """\
Redactas UNA pregunta de opción múltiple universitaria.

NIVEL BLOOM: {bloom_level}
TIPO: {kind}
ENFOQUE: "{focus}"

CONCEPTO CENTRAL (obligatorio):
{concept_detail}

CONTEXTO RELACIONADO (solo para inspirar distractores, NO los copies):
{related_context}

PREGUNTAS PREVIAS (evita repetir enunciados):
{previous_questions}

REGLAS
1. Exactamente 4 opciones (A, B, C, D) y una sola correcta.
2. Distractores plausibles, longitud similar a la correcta.
3. Prohibido: "todas/ninguna de las anteriores", "A y B".
4. Prohibido lenguaje meta ("según el documento", "como se menciona en
   el texto"…).
5. Ajusta dificultad al Bloom indicado.
6. Justificación breve (1-2 frases), coherente con el concepto central.
7. Responde SOLO JSON válido.

FORMATO
@@OPEN@@
  "bloom_level": "{bloom_level}",
  "question": "...",
  "options": @@OPEN@@ "A":"...","B":"...","C":"...","D":"..." @@CLOSE@@,
  "correct_answer": "A|B|C|D",
  "justification": "..."
@@CLOSE@@
"""


def _finalize(template: str) -> str:
    """Sustituye `@@OPEN@@`/`@@CLOSE@@` por `{{`/`}}` y los marcadores
    `__FOO__` por los valores de `config`.
    """
    return (
        template.replace("@@OPEN@@", "{{")
        .replace("@@CLOSE@@", "}}")
        .replace("__MIN__", str(DEFAULT_NUM_SLIDES_MIN))
        .replace("__MAX__", str(DEFAULT_NUM_SLIDES_MAX))
        .replace("__TITLE_LEN__", str(MAX_CHARS_SLIDE_TITLE))
        .replace("__MAX_BULLETS__", str(MAX_BULLETS_PER_SLIDE))
        .replace("__MAX_CHARS__", str(MAX_CHARS_PER_BULLET))
    )


REDUCE_TO_KB_PROMPT = _finalize(REDUCE_TO_KB_PROMPT)
SLIDE_PLAN_PROMPT = _finalize(SLIDE_PLAN_PROMPT)
SLIDE_BULLETS_FROM_ATOMS_PROMPT = _finalize(SLIDE_BULLETS_FROM_ATOMS_PROMPT)
CONCLUSION_FROM_KB_PROMPT = _finalize(CONCLUSION_FROM_KB_PROMPT)
QUIZ_PLAN_PROMPT = _finalize(QUIZ_PLAN_PROMPT)
QUIZ_QUESTION_PROMPT = _finalize(QUIZ_QUESTION_PROMPT)
