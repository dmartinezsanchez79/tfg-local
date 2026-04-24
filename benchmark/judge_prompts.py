"""Generadores de prompts para evaluación externa asistida por IA.

El usuario pega el fichero `.txt` en ChatGPT o Gemini **adjuntando el
PDF original**. El LLM evaluador responde con JSON siguiendo la rúbrica
1-5 definida aquí. No se hace nada en local con esa respuesta: es el
investigador quien la recopila y analiza (manualmente o con un script
posterior).

Dos prompts: uno para quiz, otro para PPTX. Comparten estructura y
criterio de cautela (evaluar mejor con el PDF a la vista).
"""
from __future__ import annotations

import json
from typing import Any

QUIZ_CRITERIA: tuple[tuple[str, str], ...] = (
    ("semantic_correctness", "¿Las respuestas marcadas como correctas son realmente correctas según el PDF?"),
    ("relevance", "¿Las preguntas son relevantes respecto a los temas principales del PDF?"),
    ("clarity", "¿El enunciado y las opciones están redactados con claridad?"),
    ("distractor_quality", "¿Los distractores son plausibles y discriminan de verdad?"),
    ("difficulty", "¿La dificultad es adecuada para material universitario? (1=trivial, 5=exigente pero justo)"),
    ("pedagogical_value", "¿El quiz ayuda a estudiar el material?"),
    ("hallucination_risk", "¿Hay afirmaciones sin respaldo en el PDF? (1=muchas, 5=ninguna)"),
    ("overall", "Valoración global del quiz."),
)

PPTX_CRITERIA: tuple[tuple[str, str], ...] = (
    ("thematic_coverage", "¿La presentación cubre los temas principales del PDF?"),
    ("narrative_coherence", "¿Hay un hilo conductor claro: portada → índice → introducción → desarrollo → conclusiones?"),
    ("clarity", "¿Los bullets son comprensibles y bien redactados?"),
    ("information_density", "¿La densidad de información por slide es adecuada (ni vacía ni saturada)?"),
    ("no_repetition", "¿Evita repetir la misma idea entre slides? (1=muy repetitiva, 5=sin repetición)"),
    ("study_utility", "¿Sirve como material de estudio por sí sola?"),
    ("overall", "Valoración global de la presentación."),
)

_RUBRIC_SCALE = (
    "Escala 1-5 (entero):\n"
    "  1 = muy deficiente · 2 = deficiente · 3 = aceptable · "
    "4 = bueno · 5 = excelente"
)


def _json_skeleton(criteria: tuple[tuple[str, str], ...]) -> str:
    skeleton: dict[str, Any] = {c: {"score": 0, "comment": "…"} for c, _ in criteria}
    skeleton["summary_comment"] = "Resumen cualitativo 2-3 frases."
    return json.dumps(skeleton, ensure_ascii=False, indent=2)


def _criteria_block(criteria: tuple[tuple[str, str], ...]) -> str:
    lines = []
    for key, question in criteria:
        lines.append(f"- `{key}`: {question}")
    return "\n".join(lines)


_PDF_NOTE = (
    "IMPORTANTE: adjunta el PDF original al chat antes de evaluar. Si no "
    "puedes adjuntarlo, evalúa de forma cautelosa y baja puntuaciones que "
    "dependan de contrastar con la fuente (corrección semántica, cobertura, "
    "alucinación). Dilo explícitamente en `summary_comment`."
)


def build_quiz_eval_prompt(
    *,
    pdf_id: str,
    pdf_title: str,
    model: str,
    quiz_json: dict,
) -> str:
    """Texto completo del prompt para evaluar el quiz con ChatGPT/Gemini."""
    quiz_pretty = json.dumps(quiz_json, ensure_ascii=False, indent=2)
    return f"""\
Eres un evaluador académico experto. Vas a valorar un quiz de opción
múltiple generado automáticamente a partir del PDF adjunto.

CONTEXTO
- PDF: {pdf_title} (id: {pdf_id})
- Modelo generador: {model}

{_PDF_NOTE}

INSTRUCCIONES
1. Lee el quiz completo (al final de este mensaje).
2. Evalúa cada criterio con la {_RUBRIC_SCALE}
3. Responde **solo** con un JSON válido siguiendo el esquema indicado.
4. Comentarios cortos (1-2 frases por criterio), en español.
5. No inventes: si no puedes juzgar algo sin el PDF, usa score=3 y dilo
   en el comentario.

CRITERIOS
{_criteria_block(QUIZ_CRITERIA)}

FORMATO DE RESPUESTA (solo JSON)
{_json_skeleton(QUIZ_CRITERIA)}

--- QUIZ GENERADO ---
{quiz_pretty}
--- FIN DEL QUIZ ---
"""


def build_pptx_eval_prompt(
    *,
    pdf_id: str,
    pdf_title: str,
    model: str,
    plan_json: dict,
) -> str:
    """Texto completo del prompt para evaluar la presentación."""
    plan_pretty = json.dumps(plan_json, ensure_ascii=False, indent=2)
    return f"""\
Eres un evaluador académico experto. Vas a valorar una presentación
(estructura de slides + bullets) generada automáticamente a partir del
PDF adjunto.

CONTEXTO
- PDF: {pdf_title} (id: {pdf_id})
- Modelo generador: {model}

{_PDF_NOTE}

INSTRUCCIONES
1. Lee el plan de la presentación (JSON al final).
2. Valora cada criterio con la {_RUBRIC_SCALE}
3. Responde **solo** con un JSON válido siguiendo el esquema.
4. Comentarios cortos en español.
5. Si no tienes el PDF, evalúa cautelosamente y dilo en `summary_comment`.

CRITERIOS
{_criteria_block(PPTX_CRITERIA)}

FORMATO DE RESPUESTA (solo JSON)
{_json_skeleton(PPTX_CRITERIA)}

--- PRESENTACIÓN GENERADA ---
{plan_pretty}
--- FIN DE LA PRESENTACIÓN ---
"""


def rubric_reference_text() -> str:
    """Texto plano con la rúbrica completa, para `reports/rubric_reference.txt`."""
    lines = [
        "Rúbrica de evaluación del benchmark",
        "=" * 44,
        "",
        _RUBRIC_SCALE,
        "",
        "QUIZ — criterios",
        "-" * 16,
    ]
    for key, q in QUIZ_CRITERIA:
        lines.append(f"  {key}: {q}")
    lines.extend(["", "PPTX — criterios", "-" * 16])
    for key, q in PPTX_CRITERIA:
        lines.append(f"  {key}: {q}")
    lines.extend([
        "",
        "Uso sugerido",
        "-" * 12,
        "1. Automática: métricas de `metrics.py` (score_quiz y score_pptx).",
        "2. Semiautomática: pegar `eval_prompts/*.txt` en ChatGPT/Gemini",
        "   adjuntando el PDF original; recopilar los JSON devueltos.",
        "3. Manual (opcional): usar `reports/manual_evaluation_template.csv`",
        "   para puntuar una muestra pequeña con la misma escala 1-5.",
        "",
    ])
    return "\n".join(lines)
