"""Generación del Quiz en dos pasos (plan + redacción 1 a 1).

Diseño:
1. `plan_quiz`: el LLM devuelve un `QuizPlan` con distribución Bloom y
   concept_ids únicos que apuntan a átomos de la `KnowledgeBase`.
2. `_generate_question`: una llamada por pregunta con el átomo central
   como contexto y el resto de la KB como material para distractores.

Salida: `Quiz` (contrato estable con la UI y el exportador PDF).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import (
    MAX_NUM_QUESTIONS,
    MIN_NUM_QUESTIONS,
)
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
    PlannedQuestion,
    QuizPlan,
    build_fallback_quiz_plan,
    coerce_quiz_plan_payload,
    sanitize_quiz_plan,
)
from .prompts import (
    QUIZ_PLAN_PROMPT,
    QUIZ_QUESTION_PROMPT,
    SYSTEM_EXPERT_ES,
)

logger = logging.getLogger(__name__)


BloomLevel = Literal["recordar", "comprender", "aplicar", "analizar", "evaluar", "crear"]
Letter = Literal["A", "B", "C", "D"]


# ---------------------------------------------------- modelos de salida ---


# Marcadores que el LLM a veces cuela entre paréntesis para "anotar"
# qué opción es correcta/incorrecta. Deben eliminarse: el usuario no
# debe ver pistas. Conservamos el texto útil y quitamos solo el anexo.
_OPTION_TAG_RE = re.compile(
    r"[\s]*[\(\[][\s]*(error|incorrecto|distractor|falso|correcto|"
    r"respuesta correcta|respuesta incorrecta|answer|correct|wrong)"
    r"[\s]*[\)\]][\s]*\.?",
    re.IGNORECASE,
)

# Un paréntesis al FINAL de la opción con una nota/etiqueta "informativa"
# del tipo "(Ejemplo de programación estructurada)" o "(Programación
# orientada a objetos)". Se exige:
#  - al menos 3 palabras dentro (separadas por espacios) — así no se
#    eliminan paréntesis legítimos y cortos como "(hija)" o "(padre)";
#  - longitud total ≥ 12 chars entre paréntesis.
# Ejemplos que matchea:    "(Ejemplo de programación estructurada)",
#                          "(Programación orientada a objetos)",
#                          "(Error común en programación orientada)"
# Ejemplos que NO matchea: "(hija)", "(padre)", "(mover())".
_OPTION_TRAILING_PAREN_RE = re.compile(
    r"\s*[\(\[]\s*(?=[A-ZÁÉÍÓÚÑ])(?=(?:[^()\[\]]*\s){2,})[^()\[\]]{12,120}[\)\]]\s*\.?\s*$"
)


def _clean_option_text(raw: str) -> str:
    """Normaliza una opción: sin marcadores de corrección/anotación, sin
    espacios extra. Se aplican dos pasadas porque el LLM a veces concatena
    varias anotaciones (p. ej. dos paréntesis al final)."""
    if not isinstance(raw, str):
        return raw
    cleaned = _OPTION_TAG_RE.sub("", raw).strip()
    # Quita paréntesis/corchetes al final hasta que no queden (caso de
    # anotaciones acumuladas).
    for _ in range(3):
        new = _OPTION_TRAILING_PAREN_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned.rstrip(" .") + "." if cleaned and not cleaned.endswith(".") else cleaned


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


# ---------------------------------------------------- normalización LLM ---


_BLOOM_ALIASES: dict[str, BloomLevel] = {
    "remember": "recordar",
    "understand": "comprender",
    "apply": "aplicar",
    "analyze": "analizar",
    "analyse": "analizar",
    "evaluate": "evaluar",
    "create": "crear",
    "conocimiento": "recordar",
    "comprensión": "comprender",
    "comprension": "comprender",
    "aplicación": "aplicar",
    "aplicacion": "aplicar",
    "análisis": "analizar",
    "analisis": "analizar",
    "evaluación": "evaluar",
    "evaluacion": "evaluar",
    "creación": "crear",
    "creacion": "crear",
}


def _normalize_bloom(value: str) -> str:
    key = str(value).strip().lower()
    return _BLOOM_ALIASES.get(key, key)


def _normalize_letter(value: str) -> str:
    return str(value).strip().upper()[:1]


# ----------------------------------------- serialización de átomos (ctx) -


def _atom_markdown(
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> str:
    """Renderiza un átomo como bloque Markdown compacto para prompts."""
    if isinstance(atom, Definition):
        tag = " (literal)" if atom.verbatim else ""
        return (
            f"- id: `{atom.id}` | tipo: definición{tag}\n"
            f"  - término: {atom.term}\n"
            f"  - definición: {atom.definition}"
        )
    if isinstance(atom, Example):
        attrs = ", ".join(atom.attributes) or "—"
        methods = ", ".join(atom.methods) or "—"
        return (
            f"- id: `{atom.id}` | tipo: ejemplo\n"
            f"  - nombre: {atom.name}\n"
            f"  - descripción: {atom.description}\n"
            f"  - atributos: {attrs}\n"
            f"  - métodos: {methods}"
        )
    if isinstance(atom, FormulaOrCode):
        caption = atom.caption or "—"
        lang = atom.language or ""
        return (
            f"- id: `{atom.id}` | tipo: {atom.kind}\n"
            f"  - caption: {caption}\n"
            f"  - contenido ({lang}):\n    ```{lang}\n    {atom.content}\n    ```"
        )
    if isinstance(atom, NumericDatum):
        return (
            f"- id: `{atom.id}` | tipo: dato\n"
            f"  - valor: {atom.value}\n"
            f"  - descripción: {atom.description}"
        )
    if isinstance(atom, Relation):
        desc = atom.description or "—"
        return (
            f"- id: `{atom.id}` | tipo: relación\n"
            f"  - {atom.source} —[{atom.kind}]→ {atom.target}\n"
            f"  - descripción: {desc}"
        )
    return f"- id: `{getattr(atom, 'id', '?')}` (tipo desconocido)"


def _forbidden_definitions(kb: KnowledgeBase, exclude_id: str) -> str:
    """Lista textual de definiciones del KB que NO son del concepto central.

    Se inyecta en `QUIZ_QUESTION_PROMPT` como "textos prohibidos": el LLM
    tiende a copiar/parafrasear estas definiciones cuando construye
    distractores, produciendo el anti-patrón "banco común" detectado en
    v2.7. Al verlas explícitamente prohibidas, obedece la regla con mucha
    más fiabilidad que con una regla abstracta.

    Se limita a los primeros 6 para no inflar el prompt; en KBs grandes,
    priorizamos definiciones cortas (más susceptibles de ser copiadas
    verbatim por el LLM).
    """
    defs: list[Definition] = []
    for atom in kb._iter_atoms():  # noqa: SLF001
        if isinstance(atom, Definition) and atom.id != exclude_id:
            defs.append(atom)
    if not defs:
        return "(ninguno)"
    # Preferencia: definiciones cortas (<200 chars) y con `verbatim=True`
    # porque son las que más literalmente se copian.
    defs.sort(key=lambda d: (not d.verbatim, len(d.definition)))
    out_lines: list[str] = []
    for d in defs[:6]:
        out_lines.append(f'- "{d.term}": "{d.definition}"')
    return "\n".join(out_lines)


def _related_context(kb: KnowledgeBase, exclude_id: str, *, max_atoms: int = 8) -> str:
    """Bloque compacto con átomos distintos a `exclude_id`, del mismo subtopic
    si es posible, para alimentar la generación de distractores plausibles.
    """
    target = kb.get_atom(exclude_id)
    target_subtopic = getattr(target, "subtopic", None) if target else None

    scored: list[tuple[int, object]] = []
    for atom in kb._iter_atoms():  # noqa: SLF001 — uso interno controlado
        if atom.id == exclude_id:
            continue
        score = 0
        if target_subtopic and getattr(atom, "subtopic", None) == target_subtopic:
            score += 2
        # Priorizar relaciones y ejemplos para distractores ricos.
        if isinstance(atom, (Example, Relation)):
            score += 1
        scored.append((score, atom))

    scored.sort(key=lambda x: (-x[0], x[1].id))  # type: ignore[attr-defined]
    chosen = [a for _, a in scored[:max_atoms]]
    if not chosen:
        return "(sin contexto adicional)"
    return "\n".join(_atom_markdown(a) for a in chosen)  # type: ignore[arg-type]


def _previous_summary(questions: list[QuizQuestion], *, limit: int = 5) -> str:
    if not questions:
        return "(ninguna aún)"
    tail = questions[-limit:]
    return "\n".join(
        f"- [{q.bloom_level}] {q.question}" for q in tail
    )


# ----------------------------------------- plan + generación 1 a 1 --------


def _try_build_quiz_plan(raw: Any) -> tuple[QuizPlan | None, str | None]:
    """Coerciona + normaliza + valida. Nunca lanza excepciones.

    Combina la coerción de alias (`coerce_quiz_plan_payload`) con la
    normalización semántica específica del quiz (`_normalize_plan_entries`,
    que reconcilia `bloom`/`kind` contra los vocabularios cerrados).
    """
    data = coerce_quiz_plan_payload(raw)
    if data is None:
        return None, "estructura irrecuperable tras coerción"
    _normalize_plan_entries(data)
    try:
        return QuizPlan(**data), None
    except ValidationError as exc:
        return None, f"ValidationError: {exc.errors()[:2]}"


def plan_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    num_questions: int,
) -> QuizPlan:
    """Pide al LLM el `QuizPlan` con flujo simple de una pasada.

    Mantiene coerción defensiva + fallback determinístico para no detener
    el pipeline, pero elimina el reintento por temperatura para reducir
    latencia y complejidad de control.
    """
    if kb.atom_count == 0:
        raise GenerationError("La KB no contiene átomos; no se puede planificar quiz.")

    n = max(1, min(num_questions, MAX_NUM_QUESTIONS))
    prompt = QUIZ_PLAN_PROMPT.format(
        num_questions=n,
        kb_context=kb.to_prompt_context(max_chars=6000),
    )

    raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2)
    plan, reason = _try_build_quiz_plan(raw)

    if plan is None:
        logger.warning(
            "QuizPlan inválido (%s). "
            "Se usa fallback determinístico desde la KB.",
            reason,
        )
        plan = build_fallback_quiz_plan(kb, target_count=n)

    plan = sanitize_quiz_plan(plan, kb, target_count=n)
    if not plan.questions:
        logger.warning(
            "QuizPlan vacío tras sanear. Complementando con fallback."
        )
        plan = build_fallback_quiz_plan(kb, target_count=n)
        plan = sanitize_quiz_plan(plan, kb, target_count=n)
        if not plan.questions:
            raise GenerationError(
                "No se pudo construir un QuizPlan ni con el fallback: "
                "la KB carece de átomos asignables a preguntas."
            )

    logger.info(
        "QuizPlan: %d preguntas planeadas (objetivo %d).", len(plan.questions), n
    )
    return plan


# Alias frecuentes del LLM cuando se sale del vocabulario cerrado `QuestionKind`.
# Mapear a la forma canónica evita que Pydantic tire un ValidationError que
# reventaría todo el plan. Solo se incluyen variantes observadas en la práctica.
_KIND_ALIASES: dict[str, str] = {
    # Traducciones inglesas
    "definition": "definicion",
    "differentiation": "diferenciacion",
    "difference": "diferenciacion",
    "practical_case": "caso_practico",
    "case_study": "caso_practico",
    "application": "caso_practico",
    "comparison": "comparacion",
    "analysis": "analisis_consecuencia",
    "consequence": "analisis_consecuencia",
    "judgement": "juicio_alternativas",
    "judgment": "juicio_alternativas",
    "evaluation": "juicio_alternativas",
    "code_completion": "completar_codigo",
    "fill_in_the_blank": "completar_codigo",
    # Variantes con acentos / ortografía libre
    "definición": "definicion",
    "diferenciación": "diferenciacion",
    "comparación": "comparacion",
    "análisis": "analisis_consecuencia",
    "análisis_consecuencia": "analisis_consecuencia",
    "analisis": "analisis_consecuencia",
    "juicio": "juicio_alternativas",
    # Errores semánticos típicos: el LLM confunde el "tipo de pregunta" con el
    # "tipo del átomo" y escribe el prefijo del id (`ex:...` → kind=ejemplo).
    "ejemplo": "caso_practico",
    "example": "caso_practico",
    "relacion": "comparacion",
    "relación": "comparacion",
    "relation": "comparacion",
}

_VALID_KINDS: frozenset[str] = frozenset({
    "definicion", "diferenciacion", "caso_practico", "comparacion",
    "analisis_consecuencia", "juicio_alternativas", "completar_codigo",
})


def _normalize_kind(value: Any, bloom_level: str | None) -> str:
    """Normaliza un `kind` aplicando alias y, si falla, recomendado por Bloom.

    Nunca devuelve un valor fuera de `_VALID_KINDS`: preferimos una
    pregunta con un `kind` plausible (vía Bloom) antes que perderla por
    ValidationError.
    """
    k = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    k = _KIND_ALIASES.get(k, k)
    if k in _VALID_KINDS:
        return k
    # Fallback por Bloom: `BLOOM_RECOMMENDED_KINDS` garantiza valor.
    recommended = BLOOM_RECOMMENDED_KINDS.get(bloom_level, ("definicion",))  # type: ignore[arg-type]
    logger.info(
        "kind '%s' desconocido; fallback al recomendado por bloom=%s → '%s'.",
        value, bloom_level, recommended[0],
    )
    return recommended[0]


def _normalize_plan_entries(raw: dict[str, Any]) -> None:
    """Tolera variantes del LLM (`bloom` vs `bloom_level`, alias ES/EN)."""
    for q in raw.get("questions", []):
        if "bloom" in q and "bloom_level" not in q:
            q["bloom_level"] = q.pop("bloom")
        if "bloom_level" in q:
            q["bloom_level"] = _normalize_bloom(q["bloom_level"])
        if "kind" in q:
            q["kind"] = _normalize_kind(q["kind"], q.get("bloom_level"))


def generate_single_question(
    client: OllamaClient,
    kb: KnowledgeBase,
    planned: PlannedQuestion,
    previous: list[QuizQuestion],
) -> QuizQuestion:
    """Redacta UNA pregunta a partir de una `PlannedQuestion` de la KB.

    Modo simple: una sola llamada al LLM. Si falla, el llamador decide
    aplicar fallback determinista.
    """
    atom = kb.get_atom(planned.concept_id)
    if atom is None:
        raise GenerationError(
            f"concept_id '{planned.concept_id}' no existe en la KB."
        )

    prompt = QUIZ_QUESTION_PROMPT.format(
        bloom_level=planned.bloom_level,
        kind=planned.kind,
        focus=planned.focus or "—",
        concept_detail=_atom_markdown(atom),
        related_context=_related_context(kb, planned.concept_id),
        forbidden_texts=_forbidden_definitions(kb, planned.concept_id),
        previous_questions=_previous_summary(previous),
    )

    try:
        raw = client.generate_json(
            prompt, system=SYSTEM_EXPERT_ES, temperature=0.35
        )
        if not isinstance(raw, dict):
            raise GenerationError(
                f"El LLM no devolvió un objeto JSON (tipo {type(raw).__name__})."
            )
        if "bloom_level" in raw:
            raw["bloom_level"] = _normalize_bloom(raw["bloom_level"])
        else:
            raw["bloom_level"] = planned.bloom_level
        if "correct_answer" in raw:
            raw["correct_answer"] = _normalize_letter(raw["correct_answer"])
        raw["id"] = planned.id
        return QuizQuestion(**raw)
    except (OllamaUnavailableError, OllamaModelNotFoundError):
        raise
    except (OllamaError, GenerationError, ValidationError) as exc:
        raise GenerationError(
            f"Pregunta id={planned.id} falló en generación única: {exc}"
        ) from exc


# ----------------------------------------- API pública de alto nivel ------


def _try_generate_question(
    client: OllamaClient,
    kb: KnowledgeBase,
    planned: PlannedQuestion,
    previous: list[QuizQuestion],
) -> QuizQuestion | None:
    """Envuelve `generate_single_question` tragándose fallos controlados.

    Útil en loops donde no queremos interrumpir la generación entera si
    una pregunta no sale: se registra warning y se devuelve `None`.
    """
    try:
        q = generate_single_question(client, kb, planned, previous=previous)
    except GenerationError as exc:
        logger.warning(
            "Fallo al generar pregunta id=%s (%s): %s",
            planned.id, planned.bloom_level, exc,
        )
        atom = kb.get_atom(planned.concept_id)
        if atom is None:
            return None
        # Fallback determinista de baja complejidad: evita quizzes vacíos.
        return _build_deterministic_question(planned, atom)
    q.id = planned.id
    return q


def _build_deterministic_question(
    planned: PlannedQuestion,
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> QuizQuestion:
    """Construye una pregunta MCQ mínima sin LLM a partir del átomo central.

    Se usa solo como red de seguridad cuando falla la generación LLM.
    Priorizamos estabilidad y cumplimiento de formato (4 opciones, 1 correcta).
    """
    if isinstance(atom, Definition):
        stem = f"¿Cuál define mejor «{atom.term}»?"
        correct = atom.definition[:220]
        opts = {
            "A": correct,
            "B": f"Es una variante opuesta a {atom.term.lower()} en el documento.",
            "C": f"Es un ejemplo concreto, no la definición de {atom.term.lower()}.",
            "D": f"Es un procedimiento general sin relación directa con {atom.term.lower()}.",
        }
    elif isinstance(atom, Example):
        stem = f"¿Qué opción describe mejor el ejemplo «{atom.name}»?"
        attrs = ", ".join(atom.attributes[:3]) or "atributos del dominio"
        methods = ", ".join(atom.methods[:2]) or "métodos asociados"
        opts = {
            "A": f"Incluye {attrs} y {methods}.",
            "B": "No posee estado interno y se modela solo con funciones globales.",
            "C": "Se usa únicamente como dato numérico sin comportamiento.",
            "D": "Es una relación abstracta entre dos entidades, no un ejemplo.",
        }
    elif isinstance(atom, Relation):
        stem = f"¿Qué relación es correcta entre «{atom.source}» y «{atom.target}»?"
        desc = atom.description or "según la Base de Conocimiento"
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
        stem = "¿Qué afirmación sobre el fragmento técnico es correcta?"
        kind = "fórmula" if atom.kind == "formula" else "código"
        opts = {
            "A": f"Corresponde a un {kind} mencionado en el material.",
            "B": "Es un ejemplo inventado sin relación con el documento.",
            "C": "Es únicamente una definición textual, no un fragmento técnico.",
            "D": "Es una conclusión final y no un elemento técnico.",
        }

    try:
        return QuizQuestion(
            id=planned.id,
            bloom_level=planned.bloom_level,
            question=stem,
            options=QuizOptions(**opts),
            correct_answer="A",
            justification=(
                "Se selecciona la opción A porque es la única coherente con "
                "el átomo central de la Base de Conocimiento."
            ),
        )
    except ValidationError:
        # Último recurso: plantilla mínima ultra segura.
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
            justification=(
                "La opción A es la única alineada con el concepto central "
                "seleccionado para esta pregunta."
            ),
        )


def _adaptive_target(n_atoms: int, min_q: int, max_q: int) -> int:
    """Calcula cuántas preguntas generar según la riqueza del KB.

    Heurística simple: ≈ 1.2 preguntas por átomo útil, acotado al
    rango [min, max] configurado por el usuario. Un KB con 5 átomos
    produce ~6 preguntas útiles (sin forzar reutilización); con 12
    átomos, el máximo configurado. Con <min átomos, generamos el
    mínimo y confiamos en que el refine descarte los sobrantes si
    salen demasiado parecidos.
    """
    import math
    n_atoms = max(1, n_atoms)
    target = math.ceil(n_atoms * 1.2)
    return max(min_q, min(max_q, target))


def generate_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    *,
    min_questions: int = MIN_NUM_QUESTIONS,
    max_questions: int = MAX_NUM_QUESTIONS,
    refine: bool = True,
    use_llm_critic: bool = False,
    source_markdown: str | None = None,
) -> Quiz:
    """Pipeline de quiz adaptativo por calidad (v2.11+).

    Paradigma: el usuario define un rango `[min, max]`; el sistema
    determina internamente el `target` en función del KB y genera con
    un margen (oversampling) para absorber descartes. Una sola pasada
    de regeneración + filtrado deja solo preguntas que superen los
    detectores determinísticos de calidad.

    Garantías:
    - Nunca se rellena con preguntas "sintéticas" reusando conceptos:
      mejor menos preguntas de calidad que cubrir cuota con duplicados.
    - Si tras filtrar quedan > `max_questions`, se conservan las que
      tienen menos issues residuales.
    - Si tras filtrar quedan < `min_questions`, se devuelve lo que haya
      con un aviso en log; la UI muestra el número real generado.

    El `refine` es siempre de una sola pasada (ver `critics.refine_quiz`).
    `use_llm_critic=False` por defecto porque los detectores
    determinísticos capturan >90% de los issues con cero coste de LLM.
    """
    min_q = max(1, min_questions)
    max_q = max(min_q, min(max_questions, MAX_NUM_QUESTIONS))

    n_atoms = len(kb.atom_ids())
    target = _adaptive_target(n_atoms, min_q, max_q)
    # Margen mínimo (+1) para absorber un descarte puntual sin inflar llamadas.
    plan_request = min(target + 1, max_q + 1)
    logger.info(
        "Quiz adaptativo: KB=%d átomos, target=%d, plan=%d (rango usuario [%d, %d]).",
        n_atoms, target, plan_request, min_q, max_q,
    )

    plan = plan_quiz(client, kb, num_questions=plan_request)

    questions: list[QuizQuestion] = []
    for planned in plan.questions:
        q = _try_generate_question(client, kb, planned, previous=questions)
        if q is not None:
            questions.append(q)

    if not questions:
        logger.warning(
            "No se generó ninguna pregunta en la primera pasada. "
            "Activando fallback determinista de emergencia."
        )
        emergency_plan = build_fallback_quiz_plan(kb, target_count=min_q)
        emergency_plan = sanitize_quiz_plan(
            emergency_plan, kb, target_count=min_q
        )
        for planned in emergency_plan.questions:
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
        from .critics import refine_quiz as _refine_quiz

        questions, review = _refine_quiz(
            client, kb, questions, plan,
            min_questions=min_q,
            max_questions=max_q,
            use_llm=use_llm_critic,
            source_markdown=source_markdown,
        )
        logger.info(
            "Quiz final: %d preguntas (rango [%d, %d]); %d issues residuales.",
            len(questions), min_q, max_q, len(review.issues),
        )
    else:
        # Sin refine: recorta al tope duro por seguridad.
        if len(questions) > max_q:
            questions = questions[:max_q]

    if not questions:
        raise GenerationError(
            "Tras filtrar por calidad no quedó ninguna pregunta aprobada."
        )

    for idx, q in enumerate(questions, start=1):
        q.id = idx
    return Quiz(quiz=questions)
