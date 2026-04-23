"""Prompts centralizados para Map-Reduce, Quiz e Índice/Slides.

Todos los prompts están en español y piden explícitamente respuestas en español
para mantener coherencia con el usuario objetivo (universidad hispanohablante).

Nota de implementación: los prompts con JSON de ejemplo se construyen SIN
f-strings para evitar conflictos de escape con `str.format()` (las llaves del
JSON chocarían con placeholders). La configuración dinámica se inyecta con
`str.replace()` sobre marcadores `__LIKE_THIS__`.
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

# ---------------------------------------------- REDUCE (consolidación) -----

REDUCE_CONSOLIDATION_PROMPT = """\
Tienes una lista de resúmenes parciales de un mismo documento académico.
Consolida toda la información en un ÚNICO resumen estructurado en Markdown,
eliminando duplicados y ordenando los temas de forma lógica.

Secciones obligatorias del resumen final:
## Tema principal
## Conceptos clave
## Definiciones y fórmulas
## Datos e insights
## Ejemplos o casos
## Conclusiones

Sé exhaustivo pero sin repetir. Responde en español.

--- RESÚMENES PARCIALES ---
{partials}
--- FIN ---
"""

# ---------------------------------- REDUCE hacia KnowledgeBase (JSON) ------

REDUCE_TO_KB_PROMPT = """\
Tienes una lista de resúmenes parciales de un mismo documento académico y,
opcionalmente, un bloque de "material literal" extraído directamente del
documento (definiciones, bloques de código, fórmulas y términos clave).

OBJETIVO
Consolidar todo en una Base de Conocimiento (KB) estructurada en JSON,
apta para generar después un quiz y una presentación sin inventar nada.

REGLAS DURAS
1. Usa SIEMPRE las definiciones literales del bloque de material literal
   cuando existan. Copia la redacción tal cual y marca `verbatim: true`.
2. Si no hay literal, formula la definición a partir de los resúmenes y
   marca `verbatim: false`. NO inventes información fuera del documento.
3. Cada elemento debe tener un `id` único, en minúsculas, kebab/snake:
   - Definiciones: `def:<slug>`
   - Ejemplos:     `ex:<slug>`
   - Fórmulas/código: `fc:<slug>`
   - Datos numéricos: `dt:<slug>`
   - Relaciones:   `rel:<slug>`
4. Asigna `subtopic` coherente (p. ej. "Objeto vs Clase", "Herencia",
   "Polimorfismo"). Debe coincidir con alguno de los valores de `subtopics`.
5. Los ejemplos deben ser ENTIDADES concretas del documento (p. ej.
   "Bicicleta"), no conceptos abstractos. Extrae sus `attributes` y
   `methods` cuando aparezcan.
6. Las relaciones son tripletas (`source`, `kind`, `target`), por ejemplo:
   `@@OPEN@@"kind":"subclase_de","source":"BicicletaDeMontaña","target":"Bicicleta"@@CLOSE@@`.
7. Las conclusiones son 3-5 frases sustantivas que capturen lo esencial.
8. Responde SOLO con un objeto JSON válido. Sin markdown, sin texto extra.
9. Escribe todo en español.

ESQUEMA EXACTO (no añadas claves extra)
@@OPEN@@
  "main_topic": "Cadena breve (2-120 chars)",
  "subtopics": ["..."],
  "definitions": [
    @@OPEN@@
      "id": "def:slug",
      "term": "Término",
      "definition": "Definición concisa.",
      "subtopic": "Nombre del subtema",
      "verbatim": false
    @@CLOSE@@
  ],
  "examples": [
    @@OPEN@@
      "id": "ex:slug",
      "name": "Nombre del ejemplo",
      "description": "Qué ilustra y por qué es relevante.",
      "attributes": ["..."],
      "methods": ["..."],
      "subtopic": "Subtema"
    @@CLOSE@@
  ],
  "formulas_code": [
    @@OPEN@@
      "id": "fc:slug",
      "kind": "code",
      "content": "Contenido literal",
      "caption": "Descripción",
      "language": "python",
      "subtopic": "Subtema"
    @@CLOSE@@
  ],
  "numeric_data": [
    @@OPEN@@
      "id": "dt:slug",
      "value": "42%",
      "description": "Qué mide ese dato.",
      "subtopic": "Subtema"
    @@CLOSE@@
  ],
  "relations": [
    @@OPEN@@
      "id": "rel:slug",
      "kind": "subclase_de",
      "source": "A",
      "target": "B",
      "description": "Por qué existe esta relación.",
      "subtopic": "Subtema"
    @@CLOSE@@
  ],
  "conclusions": ["..."]
@@CLOSE@@

--- RESÚMENES PARCIALES ---
{partials}
--- FIN RESÚMENES ---

--- MATERIAL LITERAL (puede estar vacío) ---
{literal_hints}
--- FIN MATERIAL LITERAL ---
"""

# --------------------------------------------------------------- QUIZ ------

QUIZ_GENERATION_PROMPT = """\
A partir del siguiente RESUMEN CONSOLIDADO de un documento académico,
genera un quiz de EXACTAMENTE {num_questions} preguntas de opción múltiple.

REGLAS DE CALIDAD (OBLIGATORIAS):
1. Aplica la Taxonomía de Bloom. Distribuye las preguntas en los niveles:
   - "recordar" (definiciones, hechos)
   - "comprender" (explicar con otras palabras)
   - "aplicar" (usar el concepto en un caso nuevo)
   - "analizar" (comparar, descomponer)
   - "evaluar" (juzgar alternativas)
   Varía los niveles; no todas "recordar".
2. Cada pregunta tiene EXACTAMENTE 4 opciones (A, B, C, D).
3. Solo UNA opción correcta. Las otras 3 son distractores PLAUSIBLES
   (relacionados con el tema, no absurdos, no "Ninguna de las anteriores").
4. La justificación explica por qué la correcta es correcta, apoyándose
   en el resumen. Debe ser concisa (1-3 frases).
5. Nada de preguntas con "¿Cuál de las siguientes NO..." salvo que
   aporte valor pedagógico real.
6. Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional,
   sin markdown, sin code fences.

FORMATO EXACTO DE SALIDA (JSON):
@@OPEN@@
  "quiz": [
    @@OPEN@@
      "id": 1,
      "bloom_level": "aplicar",
      "question": "Enunciado claro en español.",
      "options": @@OPEN@@
        "A": "texto opción A",
        "B": "texto opción B",
        "C": "texto opción C",
        "D": "texto opción D"
      @@CLOSE@@,
      "correct_answer": "B",
      "justification": "Explicación breve apoyada en el texto."
    @@CLOSE@@
  ]
@@CLOSE@@

--- RESUMEN CONSOLIDADO ---
{summary}
--- FIN DEL RESUMEN ---
"""

# ---------------------------------------------------------- ÍNDICE SLIDES --

OUTLINE_PROMPT = """\
Diseña el ÍNDICE de una presentación académica profesional a partir del
siguiente resumen. La presentación tendrá entre __MIN__ y __MAX__
diapositivas de contenido (sin contar portada, índice ni conclusión).

REGLAS:
1. Ajusta el número de diapositivas a la densidad del contenido:
   documentos densos -> más slides; documentos ligeros -> menos.
2. Cada entrada del índice es un TÍTULO corto (máx. __TITLE_LEN__ caracteres),
   claro y descriptivo. NO uses numeración, la añadirá el sistema.
3. Los títulos deben cubrir el documento de forma completa y lógica.
4. Responde SOLO JSON válido, sin texto adicional.

FORMATO EXACTO:
@@OPEN@@
  "titulo_presentacion": "Título general de la presentación",
  "indice": [
    "Título de la diapositiva 1",
    "Título de la diapositiva 2"
  ]
@@CLOSE@@

--- RESUMEN ---
{summary}
--- FIN ---
"""

# ---------------------------------------------------------- SLIDE CONTENT --

SLIDE_CONTENT_PROMPT = """\
Estás generando el contenido de UNA diapositiva de una presentación académica.

Título de la diapositiva: "{slide_title}"
Contexto: esta es la diapositiva {index} de {total} de una presentación
cuyo tema global es "{presentation_title}".

A partir del RESUMEN del documento, genera los bullets de esta diapositiva.

REGLAS ESTRICTAS (OBLIGATORIAS):
1. Máximo __MAX_BULLETS__ bullets. Mejor 3-4 que 5.
2. Cada bullet: máximo __MAX_CHARS__ caracteres, una sola línea,
   sin saltos de línea internos, sin sub-bullets.
3. Frases directas, sin rodeos. Evita "En este apartado se habla de…".
4. Si hay tablas complejas en el resumen, resume el INSIGHT clave, NO pegues
   la tabla.
5. El contenido debe ser coherente con el título de la diapositiva.
6. No repitas información que claramente pertenece a otra diapositiva del índice.
7. Responde SOLO JSON válido, sin texto adicional.

FORMATO EXACTO:
@@OPEN@@
  "bullets": [
    "Bullet 1 corto y directo.",
    "Bullet 2 corto y directo."
  ]
@@CLOSE@@

--- RESUMEN DEL DOCUMENTO ---
{summary}
--- FIN ---

--- ÍNDICE COMPLETO (para evitar solapes) ---
{outline}
--- FIN ---
"""

# ----------------------------------------------------- CONCLUSIÓN SLIDE ----

CONCLUSION_PROMPT = """\
Genera el contenido de la diapositiva FINAL de CONCLUSIONES de una presentación
académica sobre "{presentation_title}".

A partir del RESUMEN, destila __MAX_BULLETS__ conclusiones máximas
(ideal 3-4), cada una de máximo __MAX_CHARS__ caracteres, concretas y
con valor real (no frases genéricas tipo "se ha aprendido mucho").

Responde SOLO JSON válido.
FORMATO EXACTO:
@@OPEN@@
  "bullets": [
    "Conclusión concreta 1.",
    "Conclusión concreta 2."
  ]
@@CLOSE@@

--- RESUMEN ---
{summary}
--- FIN ---
"""


def _finalize(template: str) -> str:
    """Sustituye marcadores de configuración y de llaves JSON.

    `@@OPEN@@`/`@@CLOSE@@` -> `{{`/`}}` (escapados para str.format).
    Marcadores `__FOO__` -> valores numéricos de configuración.
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


# ----------------------------------------- PLAN DE PRESENTACIÓN (v2) ------

SLIDE_PLAN_PROMPT = """\
Eres el diseñador pedagógico de una presentación académica.

Dispones de una Base de Conocimiento (KB) estructurada con subtemas,
definiciones, ejemplos, fórmulas/código, datos y relaciones. Cada átomo
tiene un `id` estable (p.ej. `def:objeto`, `ex:bicicleta`, `rel:herencia`).

OBJETIVO
Producir un SlidePlan en JSON que asigne los átomos de la KB a slides de
contenido. Tu trabajo es PLANIFICAR, no redactar los bullets.

REGLAS DURAS
1. Entre __MIN__ y __MAX__ slides de contenido (no cuentes portada,
   índice ni conclusión administrativa).
2. Cada slide tiene `title` (máx __TITLE_LEN__ chars), un `kind` de la
   lista cerrada y una lista `atom_ids` con los IDs que cubre.
3. Tipos de slide admitidos y cuándo usarlos:
   - "intro"      : abre el tema. Sin átomos obligatorios; anticipa subtemas.
   - "definition" : 1-2 definiciones centrales + 0-1 ejemplo corto.
   - "example"    : un ejemplo concreto (ex:*) con atributos/métodos.
   - "comparison" : contrasta dos conceptos del documento.
   - "code"       : bullets + un fragmento de código/fórmula (fc:*).
   - "process"    : pasos ordenados de un procedimiento.
   - "relations"  : relaciones entre entidades (rel:*).
   - "outlook"    : visión panorámica que no duplica detalles de otras.
   - "conclusion" : NO USAR. La conclusión final la añade automáticamente
                    el renderer al cerrar la presentación; si la incluyes
                    aquí, se descartará.
4. Cada átomo debe aparecer en SU slide natural. Evita asignar el mismo
   átomo a más de una slide, salvo que tenga sentido didáctico justificarlo.
5. Cubre TODOS los subtemas y la MAYORÍA de átomos relevantes. Prioriza
   ejemplos concretos y relaciones frente a repetir definiciones.
6. El `title` no debe usar numeración ("1. ..."): la añade el sistema.
7. Responde SOLO JSON válido, sin texto extra.

FORMATO EXACTO
@@OPEN@@
  "presentation_title": "Título global breve",
  "slides": [
    @@OPEN@@
      "title": "Título de la slide",
      "kind": "definition",
      "atom_ids": ["def:objeto", "ex:bicicleta"],
      "focus": "Nota corta de intención narrativa (opcional)"
    @@CLOSE@@
  ]
@@CLOSE@@

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN ---
"""

# ------------------------------------- BULLETS POR SLIDE (v2, por átomos) --

SLIDE_BULLETS_FROM_ATOMS_PROMPT = """\
Estás redactando UNA diapositiva de una presentación académica profesional.

Contexto:
- Presentación: "{presentation_title}"
- Slide {index}/{total}: "{slide_title}"
- Tipo de slide: `{kind}`
- Nota de intención: "{focus}"

Átomos ASIGNADOS a esta slide (fuente principal, reescríbelos con tus
palabras; NO los copies con el formato `id · tipo · …`):
{atom_details}

Índice completo (solo referencia para NO duplicar con otras slides):
{outline}

REGLAS DURAS DE REDACCIÓN (todas son obligatorias)
1. Longitud: máximo __MAX_BULLETS__ bullets (ideal 3-4). Cada bullet es
   UNA frase completa, entre 12 y 28 palabras, y NUNCA supera
   __MAX_CHARS__ caracteres. Si un concepto no cabe, reformúlalo más
   breve; no lo trunques.
2. Prohibidos los puntos suspensivos: nunca escribas "…" ni "..." al
   final ni en medio de un bullet. Si no cabe, escribe una frase más
   corta.
3. Prohibido copiar del bloque de átomos: no uses nunca prefijos como
   `def:`, `ex:`, `fc:`, `rel:`, `dt:`, ni patrones como
   "concepto · definición · …" o "concepto · tipo · …". Redacta cada
   bullet como afirmación natural en español, con sujeto y verbo.
3.1. PROHIBIDA la notación interna de relaciones: NUNCA escribas
   tripletas con flechas y corchetes como "X —[subclase_de]→ Y",
   "A -[compone]-> B", "X —[tipo]→ Y", ni variantes. Si un átomo de
   relación aparece en los átomos asignados, redáctalo en español
   natural (ej: "BicicletaDeMontaña es subclase de Bicicleta y
   sobrescribe cambiarMarcha"). Los identificadores como
   `subclase_de`, `hereda_de`, `compuesto_por` son etiquetas técnicas
   internas; tradúcelas al español corriente.
4. Prohibido lenguaje meta: nada de "en este apartado…",
   "en esta diapositiva…", "a continuación se verá…",
   "se habla de…", "es importante destacar…",
   "visión panorámica muestra…", "sirve de ejemplo para ilustrar…",
   "facilita la creación y gestión de…", "permite comprender…",
   "comprender X facilita…", "entender X es clave para…".
4.1. Prohibido que el PRIMER bullet repita o reformule el TÍTULO de la
   slide. El título ya se muestra arriba; el bullet debe aportar
   contenido nuevo. Ejemplos prohibidos si el título es "Polimorfismo":
   "Polimorfismo: capacidad de…", "El polimorfismo es…",
   "Polimorfismo es la capacidad de…". En su lugar, entra directo al
   contenido (atributos, ejemplos, consecuencias).
4.2. Prohibidos los anglicismos innecesarios y los neologismos feos:
   nada de "blueprint" (usa "plantilla" o "molde"), "overriding",
   "inheritance", "class" (usa "clase"), "concretada" (usa "concreta"),
   "instanciada" está OK pero evita "instanciación concretada". Si un
   término técnico no tiene traducción natural, úsalo sin traducir
   (ej: "override" en un contexto de código es aceptable, pero la
   explicación del bullet debe estar en español).
5. Cada bullet aporta INFORMACIÓN nueva: datos, atributos concretos,
   comparaciones, consecuencias. No rellenes con generalidades.
6. Los bullets deben apoyarse SOLO en los átomos asignados arriba;
   no inventes hechos que no estén en la KB.
7. NO introduzcas ejemplos, objetos o entidades nuevas que no aparezcan
   en los átomos asignados o en el índice. Si los átomos hablan de
   `Bicicleta`, no uses "Coche" / "Vehículo" / "Animal" / "Persona"
   ni inventes analogías propias: usa SOLO los nombres reales de la KB.
8. No repitas contenido claramente propio de otra slide del índice.
9. Responde SOLO JSON válido, sin texto extra.

REGLAS ESPECÍFICAS POR TIPO (aplica las que correspondan a `kind`)
- definition : 1 bullet con la DEFINICIÓN precisa + 1 con ejemplo
               concreto + 1 con consecuencia o relación con otro
               concepto. No antepongas "Definición:".
- example    : describe el ejemplo con sus atributos y métodos EXACTOS
               (no abstracciones) y 1 bullet con el insight didáctico.
- comparison : bullets con la forma "A vs B: diferencia concreta".
- code       : 2 bullets explicativos + 1 bullet con el fragmento
               literal en comillas invertidas (`así`) si cabe.
- process    : pasos numerados ("1. …", "2. …").
- relations  : bullets que describen la relación en ESPAÑOL NATURAL,
               nunca con flechas ni corchetes. Usa verbos como
               "es subclase de", "hereda de", "se compone de",
               "depende de", "implementa", "contiene". Forma típica:
               "X es subclase de Y y sobrescribe …" o
               "X hereda de Y los atributos …, y añade ….".
- intro      : bullets que anuncian el HILO conductor con 1-2 datos
               concretos, nunca "en esta presentación se verá…".
- outlook    : bullets panorámicos con afirmaciones sustantivas; no
               repitas detalles ya dados en otras slides.

EJEMPLOS POSITIVOS (estilo al que debes aspirar)
- "Un objeto agrupa estado (atributos) y comportamiento (métodos) como
   una entidad única del dominio."
- "Bicicleta: atributos velocidad, cadencia, marcha; métodos
   cambiarMarcha, frenar, cambiarCadencia."
- "BicicletaDeMontaña es subclase de Bicicleta: hereda velocidad y
   cadencia, y sobrescribe cambiarMarcha para terreno irregular."

EJEMPLOS NEGATIVOS (NO escribas así)
- "En esta diapositiva se explica el concepto de objeto."
- "def:objeto · definición · Un objeto es la entidad básica…"  ← copia del contexto
- "La programación orientada a objetos permite la modularización del código y la reutilización de código mediante el uso de objetos…"  ← con puntos suspensivos
- "Visión panorámica de la POO muestra su importancia…"  ← meta vacío
- "BicicletaDeMontaña —[subclase_de]→ Bicicleta: hereda atributos…"  ← notación interna con flecha y corchetes
- "Polimorfismo: capacidad de un objeto comportarse de distintas maneras."  ← si el título es "Polimorfismo", el bullet no debe empezar repitiéndolo
- "Una clase define el blueprint para crear objetos."  ← anglicismo gratuito (di "plantilla" o "molde")
- "Un objeto es una instancia concretada de una clase."  ← "concretada" es extraño; di "concreta" o "específica"
- "Comprender objetos vs clases facilita la creación y gestión de datos."  ← meta-lenguaje didáctico vacío
- "La bicicleta sirve de ejemplo para ilustrar cómo los objetos encapsulan datos."  ← meta-lenguaje que no aporta hecho

FORMATO EXACTO
@@OPEN@@
  "bullets": ["...", "..."]
@@CLOSE@@
"""

# ------------------------------------- CONCLUSIÓN FINAL (v2, desde KB) ----

CONCLUSION_FROM_KB_PROMPT = """\
Genera el contenido de la diapositiva FINAL de CONCLUSIONES de una
presentación sobre "{presentation_title}".

Usa como fuente principal las conclusiones y relaciones de la KB que se
adjuntan. Evita repetir textualmente las definiciones.

REGLAS DURAS
- Entre 3 y __MAX_BULLETS__ bullets. Cada uno es UNA frase completa,
  12-28 palabras, nunca más de __MAX_CHARS__ caracteres.
- Prohibidos los puntos suspensivos "…" y "..." en los bullets.
- Prohibido copiar el bloque de átomos tal cual (no uses `def:`, `ex:`,
  `fc:`, ni el patrón "concepto · tipo · …").
- Afirmaciones concretas con valor real: síntesis, implicaciones,
  conexiones. No genericidades tipo "es muy importante…".
- No empieces con "Para concluir…", "En resumen…", ni similares.
- Responde SOLO JSON válido.

FORMATO
@@OPEN@@
  "bullets": ["...", "..."]
@@CLOSE@@

--- MATERIAL DE LA KB ---
{kb_context}
--- FIN ---
"""

# ------------------------------------------- PLAN DE QUIZ (v2) ------------

QUIZ_PLAN_PROMPT = """\
Eres el diseñador del examen. Vas a crear un PLAN del quiz antes de
redactar las preguntas. Solo planificas: conceptos, niveles Bloom y tipos.

Dispones de una KB con ids estables (def:*, ex:*, fc:*, dt:*, rel:*) y de
un objetivo de {num_questions} preguntas.

REGLAS DURAS
1. Distribución Bloom por rangos (sobre {num_questions} preguntas):
   - recordar   : 15-25 %
   - comprender : 15-25 %
   - aplicar    : 20-30 %
   - analizar   : 15-25 %
   - evaluar    : 10-20 %
   - crear      : 0-10 %
   Ajusta contando entero; al menos 1 pregunta "aplicar" y 1 "analizar".
2. Cada pregunta debe tener un `concept_id` ÚNICO que exista en la KB.
   No dos preguntas con el mismo concept_id.
3. Elige `kind` coherente con el nivel Bloom:
   - recordar   -> "definicion"
   - comprender -> "definicion" o "diferenciacion"
   - aplicar    -> "caso_practico" o "completar_codigo"
   - analizar   -> "comparacion" o "analisis_consecuencia"
   - evaluar    -> "juicio_alternativas" o "analisis_consecuencia"
   - crear      -> "caso_practico"
4. Prioriza ejemplos (ex:*) y relaciones (rel:*) para los niveles altos
   (aplicar/analizar/evaluar). Las definiciones puras para recordar.
5. `focus` (opcional) es una pista narrativa de qué enfoque tomar en la
   pregunta (p.ej. "comparar subclases", "elegir la implementación adecuada").
6. Responde SOLO JSON válido, sin texto extra.

FORMATO EXACTO
@@OPEN@@
  "questions": [
    @@OPEN@@
      "id": 1,
      "bloom_level": "recordar",
      "concept_id": "def:objeto",
      "kind": "definicion",
      "focus": "Cita la definición literal del documento."
    @@CLOSE@@
  ]
@@CLOSE@@

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN ---
"""

# ------------------------------ GENERAR UNA PREGUNTA (v2, 1 a 1) ----------

QUIZ_QUESTION_PROMPT = """\
Eres un profesor universitario redactando UNA pregunta de opción múltiple.

NIVEL BLOOM: {bloom_level}
TIPO DE PREGUNTA: {kind}
ENFOQUE NARRATIVO: "{focus}"

CONCEPTO CENTRAL (obligatorio):
{concept_detail}

CONTEXTO RELACIONADO (solo para inspirar distractores):
{related_context}

PREGUNTAS PREVIAS (evita repetir enunciados):
{previous_questions}

REGLAS CLAVE
1. Devuelve EXACTAMENTE 4 opciones (A, B, C, D) y una sola correcta.
2. Las opciones deben ser plausibles y de longitud parecida.
3. Prohibido: "todas las anteriores", "ninguna de las anteriores", "A y B".
4. Prohibido lenguaje meta en enunciado/opciones ("según el documento", "como se menciona en el texto", etc.).
5. Ajusta dificultad al Bloom indicado:
   - recordar/comprender: conceptual.
   - aplicar/analizar: mini-caso o comparación.
   - evaluar/crear: criterio o diseño básico.
6. Justificación breve (1-2 frases), clara y coherente con el concepto central.
7. Responde ÚNICAMENTE un JSON válido.

FORMATO EXACTO
@@OPEN@@
  "bloom_level": "{bloom_level}",
  "question": "Enunciado claro en español.",
  "options": @@OPEN@@
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  @@CLOSE@@,
  "correct_answer": "A|B|C|D",
  "justification": "Explicación breve apoyada en el texto."
@@CLOSE@@
"""


# --------------------------------------------- REVISOR CRÍTICO: QUIZ ------

QUIZ_CRITIC_PROMPT = """\
Eres un revisor crítico de exámenes universitarios. Recibes un quiz en
formato JSON y la Base de Conocimiento del documento. Tu tarea:

- Detectar ÚNICAMENTE problemas CLAROS de las preguntas.
- Si una pregunta está bien, NO la menciones.
- No inventes problemas para rellenar; la calidad se mide por tu precisión.

TIPOS DE PROBLEMA ADMITIDOS (usa exactamente estos ids en `kind`):
- "duplicate"            : dos preguntas cubren casi el mismo concepto o
                           su enunciado es casi idéntico.
- "trivial_distractor"   : uno o más distractores son absurdos o
                           claramente imposibles.
- "wrong_bloom"          : el `bloom_level` declarado no coincide con la
                           dificultad real de la pregunta.
- "meta_language"        : la pregunta u opciones usan frases meta vacías
                           ("en este texto…", "el autor dice…").
- "banned_phrase"        : opciones tipo "Todas las anteriores",
                           "Ninguna de las anteriores", "A y B".
- "not_grounded"         : la respuesta correcta o la justificación no se
                           apoyan en la Base de Conocimiento.
- "unbalanced_options"   : una opción es claramente más larga o más rica
                           que las otras (pista involuntaria).

SEVERIDAD (usa exactamente estos valores):
- "high"   : hace la pregunta inutilizable y debe regenerarse.
- "medium" : degrada la calidad, conviene regenerarla.
- "low"    : matiz menor; puede dejarse así.

FORMATO EXACTO (JSON, sin texto adicional)
@@OPEN@@
  "issues": [
    @@OPEN@@
      "question_id": 3,
      "kind": "trivial_distractor",
      "description": "La opción C no tiene sentido técnico.",
      "severity": "medium"
    @@CLOSE@@
  ]
@@CLOSE@@

Si no hay problemas, responde exactamente:
@@OPEN@@ "issues": [] @@CLOSE@@

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN KB ---

--- QUIZ ---
{quiz_json}
--- FIN QUIZ ---
"""

# --------------------------------------------- REVISOR CRÍTICO: SLIDES ----

SLIDE_CRITIC_PROMPT = """\
Eres un revisor crítico de presentaciones académicas. Recibes un plan de
slides en formato JSON y la Base de Conocimiento del documento. Tu tarea:

- Detectar ÚNICAMENTE problemas CLAROS de las slides.
- Si una slide está bien, NO la menciones.
- Precisión > cobertura: no rellenes con problemas inexistentes.

TIPOS DE PROBLEMA (usa exactamente estos ids en `kind`):
- "duplicate_content" : un bullet aparece (casi literal) en otras slides.
- "meta_language"     : bullets con "se habla de", "en este apartado…",
                        "es importante destacar…" y similares.
- "too_shallow"       : la slide tiene <2 bullets o bullets triviales sin
                        densidad técnica.
- "not_grounded"      : bullets que no se apoyan en la KB (inventados).
- "off_topic"         : bullet no corresponde al título ni al `kind` de
                        la slide.

SEVERIDAD:
- "high"   : la slide es inutilizable.
- "medium" : degrada la calidad, conviene regenerarla.
- "low"    : matiz menor.

Los índices de slides empiezan en 1 y siguen el orden del plan recibido.

FORMATO EXACTO
@@OPEN@@
  "issues": [
    @@OPEN@@
      "slide_index": 2,
      "kind": "duplicate_content",
      "description": "El bullet sobre herencia repite la slide 4.",
      "severity": "medium"
    @@CLOSE@@
  ]
@@CLOSE@@

Si no hay problemas, responde exactamente:
@@OPEN@@ "issues": [] @@CLOSE@@

--- KNOWLEDGE BASE ---
{kb_context}
--- FIN KB ---

--- PLAN DE SLIDES ---
{plan_json}
--- FIN PLAN ---
"""


# Finalizar todos los prompts que contienen placeholders de str.format
QUIZ_GENERATION_PROMPT = _finalize(QUIZ_GENERATION_PROMPT)
OUTLINE_PROMPT = _finalize(OUTLINE_PROMPT)
SLIDE_CONTENT_PROMPT = _finalize(SLIDE_CONTENT_PROMPT)
CONCLUSION_PROMPT = _finalize(CONCLUSION_PROMPT)
REDUCE_TO_KB_PROMPT = _finalize(REDUCE_TO_KB_PROMPT)
SLIDE_PLAN_PROMPT = _finalize(SLIDE_PLAN_PROMPT)
SLIDE_BULLETS_FROM_ATOMS_PROMPT = _finalize(SLIDE_BULLETS_FROM_ATOMS_PROMPT)
CONCLUSION_FROM_KB_PROMPT = _finalize(CONCLUSION_FROM_KB_PROMPT)
QUIZ_PLAN_PROMPT = _finalize(QUIZ_PLAN_PROMPT)
QUIZ_QUESTION_PROMPT = _finalize(QUIZ_QUESTION_PROMPT)
QUIZ_CRITIC_PROMPT = _finalize(QUIZ_CRITIC_PROMPT)
SLIDE_CRITIC_PROMPT = _finalize(SLIDE_CRITIC_PROMPT)
