"""Generación del Quiz en dos pasos (plan + redacción 1 a 1).

Diseño:
1. `plan_quiz`: el LLM devuelve un `QuizPlan` con distribución Bloom y
   concept_ids únicos que apuntan a átomos de la KnowledgeBase.
2. `generate_single_question`: una llamada por pregunta, con el átomo
   central como contexto y el resto de la KB para alimentar distractores.

El contrato público (`Quiz`, `generate_quiz`) es estable.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import MAX_NUM_QUESTIONS, MIN_NUM_QUESTIONS
from .exceptions import (
    GenerationError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from .knowledge_base import (
    Definition,
    Example,
    FormulaOrCode,
    KnowledgeBase,
    NumericDatum,
    Relation,
)
from .ollama_client import OllamaClient
from .plans import (
    BLOOM_RECOMMENDED_KINDS,
    PlannedQuestion,
    QuizPlan,
    build_fallback_quiz_plan,
    coerce_quiz_plan_payload,
    sanitize_quiz_plan,
)
from .prompts import QUIZ_PLAN_PROMPT, QUIZ_QUESTION_PROMPT, SYSTEM_EXPERT_ES

logger = logging.getLogger(__name__)


BloomLevel = Literal["recordar", "comprender", "aplicar", "analizar", "evaluar", "crear"]
Letter = Literal["A", "B", "C", "D"]


# =========================================================================
# Limpieza de opciones
# =========================================================================

# Marcadores entre paréntesis/corchetes con tag de corrección.
_OPTION_TAG_RE = re.compile(
    r"[\s]*[\(\[][\s]*(error|incorrecto|distractor|falso|correcto|"
    r"respuesta correcta|respuesta incorrecta|answer|correct|wrong)"
    r"[\s]*[\)\]][\s]*\.?",
    re.IGNORECASE,
)
# Paréntesis al FINAL con anotación descriptiva (≥3 palabras y ≥12 chars).
# No toca paréntesis legítimos cortos como "(hija)" o "(padre)".
_OPTION_TRAILING_PAREN_RE = re.compile(
    r"\s*[\(\[]\s*(?=[A-ZÁÉÍÓÚÑ])(?=(?:[^()\[\]]*\s){2,})[^()\[\]]{12,120}[\)\]]\s*\.?\s*$"
)


def _clean_option_text(raw: str) -> str:
    """Normaliza una opción: sin anotaciones, sin espacios extra, cerrada con '.'."""
    if not isinstance(raw, str):
        return raw
    cleaned = _OPTION_TAG_RE.sub("", raw).strip()
    for _ in range(3):  # varias anotaciones acumuladas
        new = _OPTION_TRAILING_PAREN_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned.rstrip(" .") + "." if cleaned and not cleaned.endswith(".") else cleaned


# =========================================================================
# Modelos de salida
# =========================================================================

class QuizOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str

    @field_validator("A", "B", "C", "D")
    @classmethod
    def _clean(cls, v: str) -> str:
        return _clean_option_text(v)


class QuizQuestion(BaseModel):
    id: int
    bloom_level: BloomLevel
    question: str = Field(min_length=10)
    options: QuizOptions
    correct_answer: Letter
    justification: str = Field(min_length=10)

    @field_validator("question", "justification")
    @classmethod
    def _stripped(cls, v: str) -> str:
        return v.strip()


class Quiz(BaseModel):
    quiz: list[QuizQuestion] = Field(min_length=1)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# =========================================================================
# Normalización de campos del LLM
# =========================================================================

_BLOOM_ALIASES: dict[str, BloomLevel] = {
    "remember": "recordar", "understand": "comprender",
    "apply": "aplicar", "analyze": "analizar", "analyse": "analizar",
    "evaluate": "evaluar", "create": "crear",
    "comprensión": "comprender", "comprension": "comprender",
    "aplicación": "aplicar", "aplicacion": "aplicar",
    "análisis": "analizar", "analisis": "analizar",
    "evaluación": "evaluar", "evaluacion": "evaluar",
    "creación": "crear", "creacion": "crear",
}

# Alias de `kind` observados en la práctica cuando el LLM ignora el
# vocabulario cerrado. Las traducciones inglesas más variantes sin acento.
_KIND_ALIASES: dict[str, str] = {
    "definition": "definicion", "definición": "definicion",
    "differentiation": "diferenciacion", "difference": "diferenciacion",
    "diferenciación": "diferenciacion",
    "practical_case": "caso_practico", "case_study": "caso_practico",
    "application": "caso_practico",
    "ejemplo": "caso_practico", "example": "caso_practico",
    "comparison": "comparacion", "comparación": "comparacion",
    "relacion": "comparacion", "relación": "comparacion", "relation": "comparacion",
    "analysis": "analisis_consecuencia", "consequence": "analisis_consecuencia",
    "análisis": "analisis_consecuencia", "analisis": "analisis_consecuencia",
    "análisis_consecuencia": "analisis_consecuencia",
    "judgement": "juicio_alternativas", "judgment": "juicio_alternativas",
    "evaluation": "juicio_alternativas", "juicio": "juicio_alternativas",
    "code_completion": "completar_codigo", "fill_in_the_blank": "completar_codigo",
}

_VALID_KINDS: frozenset[str] = frozenset({
    "definicion", "diferenciacion", "caso_practico", "comparacion",
    "analisis_consecuencia", "juicio_alternativas", "completar_codigo",
})


def _normalize_bloom(value: Any) -> str:
    key = str(value).strip().lower()
    return _BLOOM_ALIASES.get(key, key)


def _normalize_kind(value: Any, bloom_level: str | None) -> str:
    """Normaliza `kind` usando alias; si falla, usa el recomendado por Bloom."""
    k = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    k = _KIND_ALIASES.get(k, k)
    if k in _VALID_KINDS:
        return k
    recommended = BLOOM_RECOMMENDED_KINDS.get(bloom_level, ("definicion",))  # type: ignore[arg-type]
    logger.info("kind '%s' desconocido; fallback por bloom=%s → '%s'.",
                value, bloom_level, recommended[0])
    return recommended[0]


def _normalize_plan_entries(raw: dict[str, Any]) -> None:
    for q in raw.get("questions", []):
        if "bloom" in q and "bloom_level" not in q:
            q["bloom_level"] = q.pop("bloom")
        if "bloom_level" in q:
            q["bloom_level"] = _normalize_bloom(q["bloom_level"])
        if "kind" in q:
            q["kind"] = _normalize_kind(q["kind"], q.get("bloom_level"))


def _deaccent_lower(text: str) -> str:
    n = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in n if not unicodedata.combining(ch)).lower()


# =========================================================================
# Serialización de átomos como contexto para el prompt
# =========================================================================

def _atom_markdown(
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> str:
    if isinstance(atom, Definition):
        tag = " (literal)" if atom.verbatim else ""
        return (f"- id: `{atom.id}` | tipo: definición{tag}\n"
                f"  - término: {atom.term}\n  - definición: {atom.definition}")
    if isinstance(atom, Example):
        attrs = ", ".join(atom.attributes) or "—"
        methods = ", ".join(atom.methods) or "—"
        return (f"- id: `{atom.id}` | tipo: ejemplo\n"
                f"  - nombre: {atom.name}\n  - descripción: {atom.description}\n"
                f"  - atributos: {attrs}\n  - métodos: {methods}")
    if isinstance(atom, FormulaOrCode):
        caption = atom.caption or "—"
        lang = atom.language or ""
        return (f"- id: `{atom.id}` | tipo: {atom.kind}\n  - caption: {caption}\n"
                f"  - contenido ({lang}):\n    ```{lang}\n    {atom.content}\n    ```")
    if isinstance(atom, NumericDatum):
        return (f"- id: `{atom.id}` | tipo: dato\n"
                f"  - valor: {atom.value}\n  - descripción: {atom.description}")
    if isinstance(atom, Relation):
        return (f"- id: `{atom.id}` | tipo: relación\n"
                f"  - {atom.source} —[{atom.kind}]→ {atom.target}\n"
                f"  - descripción: {atom.description or '—'}")
    return f"- id: `{getattr(atom, 'id', '?')}` (tipo desconocido)"


def _related_context(kb: KnowledgeBase, exclude_id: str, *, max_atoms: int = 8) -> str:
    """Bloque compacto con otros átomos, priorizando mismo subtopic."""
    target = kb.get_atom(exclude_id)
    target_subtopic = getattr(target, "subtopic", None) if target else None

    scored: list[tuple[int, Any]] = []
    for atom in kb._iter_atoms():  # noqa: SLF001
        if atom.id == exclude_id:
            continue
        score = 0
        if target_subtopic and getattr(atom, "subtopic", None) == target_subtopic:
            score += 2
        if isinstance(atom, (Example, Relation)):
            score += 1
        scored.append((score, atom))

    scored.sort(key=lambda x: (-x[0], x[1].id))
    chosen = [a for _, a in scored[:max_atoms]]
    if not chosen:
        return "(sin contexto adicional)"
    return "\n".join(_atom_markdown(a) for a in chosen)


def _previous_summary(questions: list[QuizQuestion], *, limit: int = 5) -> str:
    if not questions:
        return "(ninguna aún)"
    return "\n".join(f"- [{q.bloom_level}] {q.question}" for q in questions[-limit:])


# =========================================================================
# Planificación del quiz
# =========================================================================

def plan_quiz(client: OllamaClient, kb: KnowledgeBase, num_questions: int) -> QuizPlan:
    """Pide al LLM el `QuizPlan` con coerción defensiva + fallback determinístico."""
    if kb.atom_count == 0:
        raise GenerationError("La KB no contiene átomos; no se puede planificar quiz.")

    n = max(1, min(num_questions, MAX_NUM_QUESTIONS))
    prompt = QUIZ_PLAN_PROMPT.format(
        num_questions=n,
        kb_context=kb.to_prompt_context(max_chars=6000),
    )

    def _build(raw: Any) -> QuizPlan | None:
        data = coerce_quiz_plan_payload(raw)
        if data is None:
            return None
        _normalize_plan_entries(data)
        try:
            return QuizPlan(**data)
        except ValidationError as exc:
            logger.warning("QuizPlan ValidationError: %s", exc.errors()[:2])
            return None

    plan = _build(client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2))
    if plan is None:
        logger.warning("QuizPlan del LLM irrecuperable; usando fallback determinístico.")
        plan = build_fallback_quiz_plan(kb, target_count=n)

    plan = sanitize_quiz_plan(plan, kb, target_count=n)
    if not plan.questions:
        logger.warning("QuizPlan vacío tras sanear; complementando con fallback.")
        plan = sanitize_quiz_plan(build_fallback_quiz_plan(kb, target_count=n),
                                  kb, target_count=n)
        if not plan.questions:
            raise GenerationError(
                "No se pudo construir QuizPlan ni con fallback: KB sin átomos asignables."
            )

    logger.info("QuizPlan: %d preguntas planeadas (objetivo %d).", len(plan.questions), n)
    return plan


# =========================================================================
# Redacción de una pregunta (LLM + fallback determinista)
# =========================================================================

def generate_single_question(
    client: OllamaClient,
    kb: KnowledgeBase,
    planned: PlannedQuestion,
    previous: list[QuizQuestion],
) -> QuizQuestion:
    """Redacta UNA pregunta a partir de una `PlannedQuestion`."""
    atom = kb.get_atom(planned.concept_id)
    if atom is None:
        raise GenerationError(f"concept_id '{planned.concept_id}' no existe en la KB.")

    prompt = QUIZ_QUESTION_PROMPT.format(
        bloom_level=planned.bloom_level,
        kind=planned.kind,
        focus=planned.focus or "—",
        concept_detail=_atom_markdown(atom),
        related_context=_related_context(kb, planned.concept_id),
        previous_questions=_previous_summary(previous),
    )

    try:
        raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.35)
        if not isinstance(raw, dict):
            raise GenerationError(
                f"El LLM no devolvió un objeto JSON (tipo {type(raw).__name__})."
            )
        if "bloom_level" in raw:
            raw["bloom_level"] = _normalize_bloom(raw["bloom_level"])
        else:
            raw["bloom_level"] = planned.bloom_level
        if "correct_answer" in raw:
            raw["correct_answer"] = str(raw["correct_answer"]).strip().upper()[:1]
        raw["id"] = planned.id
        return QuizQuestion(**raw)
    except (OllamaUnavailableError, OllamaModelNotFoundError):
        raise
    except (OllamaError, GenerationError, ValidationError) as exc:
        raise GenerationError(
            f"Pregunta id={planned.id} falló en generación: {exc}"
        ) from exc


def _build_deterministic_question(
    planned: PlannedQuestion,
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> QuizQuestion:
    """Construye una pregunta MCQ mínima sin LLM a partir del átomo central.

    Red de seguridad para modelos <7B que a veces devuelven JSON basura
    sistemáticamente: prefiero 1 pregunta trivial pero válida a un quiz vacío.
    Los distractores genéricos quedarán marcados por el crítico si el LLM
    principal llega a funcionar después.
    """
    if isinstance(atom, Definition):
        stem = f"¿Cuál define mejor «{atom.term}»?"
        opts = {
            "A": atom.definition[:220],
            "B": f"Es una variante opuesta a {atom.term.lower()} en el documento.",
            "C": f"Es un ejemplo concreto, no la definición de {atom.term.lower()}.",
            "D": f"Es un procedimiento general sin relación directa con {atom.term.lower()}.",
        }
    elif isinstance(atom, Example):
        attrs = ", ".join(atom.attributes[:3]) or "atributos del dominio"
        methods = ", ".join(atom.methods[:2]) or "métodos asociados"
        stem = f"¿Qué opción describe mejor el ejemplo «{atom.name}»?"
        opts = {
            "A": f"Incluye {attrs} y {methods}.",
            "B": "No posee estado interno y se modela solo con funciones globales.",
            "C": "Se usa únicamente como dato numérico sin comportamiento.",
            "D": "Es una relación abstracta entre dos entidades, no un ejemplo.",
        }
    elif isinstance(atom, Relation):
        desc = atom.description or "según la Base de Conocimiento"
        stem = f"¿Qué relación es correcta entre «{atom.source}» y «{atom.target}»?"
        opts = {
            "A": f"{atom.source} {atom.kind} {atom.target} ({desc}).",
            "B": f"{atom.target} {atom.kind} {atom.source}, en sentido inverso.",
            "C": f"{atom.source} y {atom.target} no tienen relación en el documento.",
            "D": f"{atom.source} equivale exactamente a {atom.target} en todos los casos.",
        }
    elif isinstance(atom, NumericDatum):
        stem = "¿Qué dato numérico coincide con lo indicado en el documento?"
        opts = {
            "A": f"{atom.value}: {atom.description}",
            "B": "0%: valor nulo para el mismo fenómeno descrito.",
            "C": "100%: valor absoluto no matizado por el contexto.",
            "D": "No se proporciona ningún dato cuantitativo relevante.",
        }
    else:  # FormulaOrCode
        kind_es = "fórmula" if atom.kind == "formula" else "código"
        stem = "¿Qué afirmación sobre el fragmento técnico es correcta?"
        opts = {
            "A": f"Corresponde a un {kind_es} mencionado en el material.",
            "B": "Es un ejemplo inventado sin relación con el documento.",
            "C": "Es únicamente una definición textual, no un fragmento técnico.",
            "D": "Es una conclusión final y no un elemento técnico.",
        }

    justification = (
        "Se selecciona la opción A porque es la única coherente con el átomo "
        "central de la Base de Conocimiento."
    )
    try:
        return QuizQuestion(
            id=planned.id,
            bloom_level=planned.bloom_level,
            question=stem,
            options=QuizOptions(**opts),
            correct_answer="A",
            justification=justification,
        )
    except ValidationError:
        # Plantilla ultra-segura de último recurso.
        return QuizQuestion(
            id=planned.id,
            bloom_level=planned.bloom_level,
            question="¿Cuál es la opción correcta según el concepto central?",
            options=QuizOptions(
                A="La opción A resume el concepto central del documento.",
                B="La opción B contradice el concepto central.",
                C="La opción C confunde el concepto con un caso no equivalente.",
                D="La opción D no corresponde al contexto del documento.",
            ),
            correct_answer="A",
            justification=justification,
        )


def _try_generate_question(
    client: OllamaClient,
    kb: KnowledgeBase,
    planned: PlannedQuestion,
    previous: list[QuizQuestion],
) -> QuizQuestion | None:
    """Intenta LLM; si falla cae al fallback determinista (solo si el átomo existe)."""
    try:
        q = generate_single_question(client, kb, planned, previous=previous)
    except GenerationError as exc:
        logger.warning("Fallo al generar pregunta id=%s (%s): %s",
                       planned.id, planned.bloom_level, exc)
        atom = kb.get_atom(planned.concept_id)
        if atom is None:
            return None
        return _build_deterministic_question(planned, atom)
    q.id = planned.id
    return q


def _fill_min_questions_with_fallback(
    questions: list[QuizQuestion],
    kb: KnowledgeBase,
    *,
    min_q: int,
    max_q: int,
) -> list[QuizQuestion]:
    """Completa hasta `min_q` con preguntas deterministas si la salida quedó corta."""
    if len(questions) >= min_q:
        return questions[:max_q]

    logger.warning(
        "Quiz por debajo del mínimo (%d<%d). Rellenando con fallback determinístico.",
        len(questions), min_q,
    )
    used_ids = {q.id for q in questions}
    used_stems = {_deaccent_lower(q.question) for q in questions}

    emergency = sanitize_quiz_plan(
        build_fallback_quiz_plan(kb, target_count=min_q),
        kb,
        target_count=min_q,
    )
    for planned in emergency.questions:
        if len(questions) >= min_q or len(questions) >= max_q:
            break
        atom = kb.get_atom(planned.concept_id)
        if atom is None:
            continue
        q = _build_deterministic_question(planned, atom)
        stem = _deaccent_lower(q.question)
        if stem in used_stems:
            continue
        q.id = max(used_ids, default=0) + 1
        used_ids.add(q.id)
        used_stems.add(stem)
        questions.append(q)

    return questions[:max_q]


# =========================================================================
# API pública
# =========================================================================

def _adaptive_target(n_atoms: int, min_q: int, max_q: int) -> int:
    """≈ 1.2 preguntas por átomo útil, acotado al rango [min, max]."""
    n_atoms = max(1, n_atoms)
    return max(min_q, min(max_q, math.ceil(n_atoms * 1.2)))


def generate_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    *,
    min_questions: int = MIN_NUM_QUESTIONS,
    max_questions: int = MAX_NUM_QUESTIONS,
    refine: bool = True,
) -> Quiz:
    """Pipeline adaptativo: plan LLM → redacción 1-a-1 → refinado determinista."""
    min_q = max(1, min_questions)
    max_q = max(min_q, min(max_questions, MAX_NUM_QUESTIONS))

    n_atoms = len(kb.atom_ids())
    target = _adaptive_target(n_atoms, min_q, max_q)
    plan_request = min(target + 1, max_q + 1)  # +1 para absorber descartes puntuales
    logger.info("Quiz adaptativo: KB=%d átomos, target=%d, plan=%d (rango [%d, %d]).",
                n_atoms, target, plan_request, min_q, max_q)

    plan = plan_quiz(client, kb, num_questions=plan_request)

    questions: list[QuizQuestion] = []
    for planned in plan.questions:
        q = _try_generate_question(client, kb, planned, previous=questions)
        if q is not None:
            questions.append(q)

    # Fallback de emergencia: si NINGUNA pregunta sobrevivió, construimos
    # deterministamente a partir de la KB. Garantiza que el quiz nunca
    # esté vacío con modelos muy pequeños que devuelven JSON inválido.
    if not questions:
        logger.warning("Ninguna pregunta en la 1ª pasada; fallback determinista de emergencia.")
        emergency = sanitize_quiz_plan(
            build_fallback_quiz_plan(kb, target_count=min_q), kb, target_count=min_q,
        )
        for planned in emergency.questions:
            atom = kb.get_atom(planned.concept_id)
            if atom is None:
                continue
            q = _build_deterministic_question(planned, atom)
            q.id = planned.id
            questions.append(q)
        if not questions:
            raise GenerationError(
                "No se pudo generar ninguna pregunta válida a partir del QuizPlan."
            )

    if refine:
        from .critics import refine_quiz as _refine_quiz  # import diferido
        questions, review = _refine_quiz(
            client, kb, questions, plan,
            min_questions=min_q, max_questions=max_q,
        )
        logger.info("Quiz final: %d preguntas (rango [%d, %d]); %d issues residuales.",
                    len(questions), min_q, max_q, len(review.issues))
    elif len(questions) > max_q:
        questions = questions[:max_q]

    questions = _fill_min_questions_with_fallback(
        questions,
        kb,
        min_q=min_q,
        max_q=max_q,
    )

    if not questions:
        raise GenerationError("Tras filtrar por calidad no quedó ninguna pregunta aprobada.")

    for idx, q in enumerate(questions, start=1):
        q.id = idx
    return Quiz(quiz=questions)
