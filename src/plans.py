"""Modelos de planificación para slides y quiz.

Un `SlidePlan` y un `QuizPlan` son **contratos intermedios**: el LLM los
produce a partir de la `KnowledgeBase` y especifican *qué* se va a generar
antes de generarlo. Ese paso extra evita el patrón "todo en un shot",
que es la fuente principal de duplicados, drift y bullets genéricos.

Responsabilidades:
- Definir los tipos de slide y de pregunta admitidos (vocabulario controlado).
- Validar estructura con Pydantic.
- Ofrecer utilidades para comprobar y reparar los planes frente a la KB.

Este módulo es agnóstico de UI/LLM y puede evolucionar sin romper
downstream gracias a los `Literal` tipados.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .knowledge_base import (
    Definition,
    Example,
    FormulaOrCode,
    KnowledgeBase,
    NumericDatum,
    Relation,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- slides ---

SlideKind = Literal[
    "intro",
    "definition",
    "example",
    "comparison",
    "code",
    "process",
    "relations",
    "outlook",
    "conclusion",
]


class PlannedSlide(BaseModel):
    """Especificación de UNA diapositiva de contenido.

    - `atom_ids` apunta a átomos existentes en la `KnowledgeBase`.
    - `focus` es una nota breve sobre la intención narrativa de la slide.
    """

    title: str = Field(min_length=2, max_length=80)
    kind: SlideKind
    atom_ids: list[str] = Field(default_factory=list, max_length=12)
    focus: str | None = Field(default=None, max_length=200)

    @field_validator("title", "focus")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @field_validator("atom_ids")
    @classmethod
    def _dedup_atoms(cls, vs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in vs:
            x = x.strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out


class SlidePlan(BaseModel):
    """Plan completo de la presentación.

    No incluye portada/índice/conclusión-administrativas: esas se
    construyen determinísticamente en el renderer PPTX.
    """

    presentation_title: str = Field(min_length=3, max_length=160)
    slides: list[PlannedSlide] = Field(min_length=3, max_length=20)

    @field_validator("presentation_title")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# -------- Coerción defensiva: tolerar claves en español y anidamientos ---

# Aliases que un LLM pequeño suele usar en lugar de los nombres canónicos.
_TITLE_ALIASES: tuple[str, ...] = (
    "presentation_title",
    "title",
    "titulo",
    "título",
    "presentation",
    "presentacion",
    "presentación",
    "topic",
    "tema",
    "nombre",
)
_SLIDES_ALIASES: tuple[str, ...] = (
    "slides",
    "diapositivas",
    "plan",
    "slide_list",
    "presentation_slides",
    "content",
    "contenido",
)
_SLIDE_TITLE_ALIASES: tuple[str, ...] = ("title", "titulo", "título", "heading", "nombre")
_SLIDE_KIND_ALIASES: tuple[str, ...] = ("kind", "type", "tipo", "layout", "category", "categoria")
_SLIDE_ATOMS_ALIASES: tuple[str, ...] = (
    "atom_ids",
    "atomos",
    "átomos",
    "ids",
    "atom_id",
    "atoms",
    "references",
    "refs",
)
_SLIDE_FOCUS_ALIASES: tuple[str, ...] = (
    "focus",
    "enfoque",
    "intention",
    "intencion",
    "intención",
    "narrative",
    "idea",
)

# Mapeo de variantes hacia los `kind` admitidos por `SlideKind`.
_KIND_SYNONYMS: dict[str, str] = {
    "introduccion": "intro",
    "introducción": "intro",
    "introduction": "intro",
    "definicion": "definition",
    "definición": "definition",
    "definitions": "definition",
    "ejemplo": "example",
    "ejemplos": "example",
    "examples": "example",
    "comparacion": "comparison",
    "comparación": "comparison",
    "comparativa": "comparison",
    "codigo": "code",
    "código": "code",
    "proceso": "process",
    "flujo": "process",
    "relacion": "relations",
    "relación": "relations",
    "relaciones": "relations",
    "panoramica": "outlook",
    "panorámica": "outlook",
    "outlook": "outlook",
    "vision": "outlook",
    "visión": "outlook",
    "conclusion": "conclusion",
    "conclusión": "conclusion",
    "conclusiones": "conclusion",
}

# Conjunto de valores válidos extraídos del `Literal` `SlideKind`.
_VALID_KINDS: frozenset[str] = frozenset(
    {
        "intro",
        "definition",
        "example",
        "comparison",
        "code",
        "process",
        "relations",
        "outlook",
        "conclusion",
    }
)


def _pick_alias(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """Devuelve el primer valor no-None encontrado entre los aliases."""
    for key in aliases:
        if key in data and data[key] not in (None, "", []):
            return data[key]
    return None


def _normalize_kind(raw_kind: Any) -> str | None:
    """Mapea un `kind` en distintos idiomas/mayúsculas al vocabulario válido."""
    if not isinstance(raw_kind, str):
        return None
    k = raw_kind.strip().lower()
    if k in _VALID_KINDS:
        return k
    return _KIND_SYNONYMS.get(k)


def _coerce_atom_ids(raw: Any) -> list[str]:
    """Acepta lista, string único o CSV y devuelve lista de strings no vacíos."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
        return parts
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def coerce_slide_plan_payload(
    raw: Any, kb: KnowledgeBase
) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema canónico de `SlidePlan`.

    - Tolera claves en español (`titulo`, `diapositivas`, `tipo`, `atomos`…).
    - Soporta anidamientos comunes (`{"plan": {...}}`, `{"output": {...}}`).
    - Normaliza `kind` a `SlideKind`.
    - Rellena `presentation_title` con `kb.main_topic` si falta o queda vacío.
    - Devuelve `None` si la estructura es irrecuperable (e.g. no hay slides).
    """
    if not isinstance(raw, dict):
        return None

    # Desenvoltura de un nivel si el LLM metió todo dentro de una clave.
    for wrapper in ("plan", "output", "result", "data", "respuesta"):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and any(
            k in inner for k in _TITLE_ALIASES + _SLIDES_ALIASES
        ):
            raw = inner
            break

    title = _pick_alias(raw, _TITLE_ALIASES)
    slides_raw = _pick_alias(raw, _SLIDES_ALIASES)

    if not isinstance(slides_raw, list) or not slides_raw:
        return None

    if not isinstance(title, str) or not title.strip():
        title = kb.main_topic or "Presentación"

    slides_out: list[dict[str, Any]] = []
    for item in slides_raw:
        if not isinstance(item, dict):
            continue
        s_title = _pick_alias(item, _SLIDE_TITLE_ALIASES)
        s_kind = _normalize_kind(_pick_alias(item, _SLIDE_KIND_ALIASES))
        s_atoms = _coerce_atom_ids(_pick_alias(item, _SLIDE_ATOMS_ALIASES))
        s_focus = _pick_alias(item, _SLIDE_FOCUS_ALIASES)

        if not isinstance(s_title, str) or not s_title.strip():
            continue
        if s_kind is None:
            # Heurística: si hay átomos mayoritariamente de un tipo los
            # usamos; si no, `outlook` es el más laxo.
            s_kind = "outlook"

        slide_dict: dict[str, Any] = {
            "title": s_title.strip()[:80],
            "kind": s_kind,
            "atom_ids": s_atoms[:12],
        }
        if isinstance(s_focus, str) and s_focus.strip():
            slide_dict["focus"] = s_focus.strip()[:200]
        slides_out.append(slide_dict)

    if not slides_out:
        return None

    return {"presentation_title": title.strip()[:160], "slides": slides_out}


# -------- Fallback determinístico: construir SlidePlan sin LLM -----------

# Mapa tipo-de-átomo → `kind` recomendado cuando agrupamos por contenido.
_ATOM_KIND_MAP: dict[type, str] = {
    Definition: "definition",
    Example: "example",
    FormulaOrCode: "code",
    NumericDatum: "outlook",
    Relation: "relations",
}


def _group_atoms_by_subtopic(
    kb: KnowledgeBase,
) -> dict[str, list[tuple[str, type]]]:
    """Agrupa ids de átomos por `subtopic`; deja en "" los que no lo tienen."""
    groups: dict[str, list[tuple[str, type]]] = defaultdict(list)
    for atom_list in (
        kb.definitions,
        kb.examples,
        kb.formulas_code,
        kb.numeric_data,
        kb.relations,
    ):
        for atom in atom_list:
            subtopic = (getattr(atom, "subtopic", "") or "").strip()
            groups[subtopic].append((atom.id, type(atom)))
    return groups


def _slide_kind_for_group(atom_types: list[type]) -> str:
    """Elige un `kind` basándose en la composición del grupo de átomos."""
    if not atom_types:
        return "outlook"
    counts = Counter(atom_types)
    dominant, _ = counts.most_common(1)[0]
    return _ATOM_KIND_MAP.get(dominant, "outlook")


def _dedup_title(title: str, used: set[str]) -> str:
    """Garantiza que el título no choca con otros ya usados en el fallback."""
    base = title.strip()[:78] or "Contenido"
    candidate = base
    i = 2
    while candidate.lower() in used:
        suffix = f" ({i})"
        candidate = (base[: 78 - len(suffix)] + suffix)
        i += 1
    used.add(candidate.lower())
    return candidate


def build_fallback_slide_plan(kb: KnowledgeBase) -> SlidePlan:
    """Construye un `SlidePlan` mínimo pero válido directamente desde la KB.

    Se usa cuando el LLM devuelve JSON vacío o irrecuperable. No depende de
    llamadas adicionales al modelo, así que es determinista y rápido.

    Estrategia:
    1. Slide intro panorámica (sin átomos, `focus` = resumen automático).
    2. Una slide por cada `subtopic` con átomos asociados (máx. 10 átomos).
    3. Si no hay subtopics, una slide por tipo de átomo (defs/ejemplos/…).
    4. Se garantiza el rango `[3, DEFAULT_NUM_SLIDES_MAX]`.
    """
    title = (kb.main_topic or "Presentación").strip()[:160]

    slides: list[PlannedSlide] = []
    used_titles: set[str] = set()

    intro_focus = (
        f"Panorámica general sobre {kb.main_topic}."
        if kb.main_topic
        else "Panorámica general del documento."
    )
    slides.append(
        PlannedSlide(
            title=_dedup_title("Introducción", used_titles),
            kind="intro",
            atom_ids=[],
            focus=intro_focus[:200],
        )
    )

    groups = _group_atoms_by_subtopic(kb)
    labeled_groups = [(k, v) for k, v in groups.items() if k]

    if labeled_groups:
        # Preferimos el orden de `kb.subtopics` si existe, luego el resto.
        order_index = {s: i for i, s in enumerate(kb.subtopics or [])}
        labeled_groups.sort(key=lambda kv: order_index.get(kv[0], 10_000))
        for subtopic, items in labeled_groups:
            atom_ids = [aid for aid, _ in items[:10]]
            atom_types = [t for _, t in items]
            slides.append(
                PlannedSlide(
                    title=_dedup_title(subtopic[:78], used_titles),
                    kind=_slide_kind_for_group(atom_types),  # type: ignore[arg-type]
                    atom_ids=atom_ids,
                    focus=None,
                )
            )
    else:
        # Sin subtopics: agrupamos por tipo.
        buckets: list[tuple[str, str, list[str]]] = [
            (
                "Definiciones clave",
                "definition",
                [d.id for d in kb.definitions[:8]],
            ),
            (
                "Ejemplos ilustrativos",
                "example",
                [e.id for e in kb.examples[:6]],
            ),
            (
                "Fórmulas y código",
                "code",
                [f.id for f in kb.formulas_code[:6]],
            ),
            (
                "Relaciones entre conceptos",
                "relations",
                [r.id for r in kb.relations[:8]],
            ),
            (
                "Datos numéricos",
                "outlook",
                [d.id for d in kb.numeric_data[:6]],
            ),
        ]
        for s_title, s_kind, ids in buckets:
            if not ids:
                continue
            slides.append(
                PlannedSlide(
                    title=_dedup_title(s_title, used_titles),
                    kind=s_kind,  # type: ignore[arg-type]
                    atom_ids=ids,
                    focus=None,
                )
            )

    # Garantizar mínimo 3 slides: si hay muy pocas definiciones, dividimos.
    if len(slides) < 3 and kb.definitions:
        extra_ids = [d.id for d in kb.definitions[:6]]
        if extra_ids:
            slides.append(
                PlannedSlide(
                    title=_dedup_title("Conceptos fundamentales", used_titles),
                    kind="definition",
                    atom_ids=extra_ids,
                    focus=None,
                )
            )

    # Relleno final si todavía faltan slides (KB extremadamente pobre).
    while len(slides) < 3:
        slides.append(
            PlannedSlide(
                title=_dedup_title("Aspectos complementarios", used_titles),
                kind="outlook",
                atom_ids=[],
                focus="Ideas adicionales derivadas del material.",
            )
        )

    # Recorte superior para respetar max_length=20.
    if len(slides) > 20:
        slides = slides[:20]

    logger.info(
        "Fallback SlidePlan construido desde KB: %d slides (sin LLM).",
        len(slides),
    )
    return SlidePlan(presentation_title=title, slides=slides)


# ----------------------------------------------------------------- quiz ---

BloomLevel = Literal[
    "recordar",
    "comprender",
    "aplicar",
    "analizar",
    "evaluar",
    "crear",
]

QuestionKind = Literal[
    "definicion",
    "diferenciacion",
    "caso_practico",
    "comparacion",
    "analisis_consecuencia",
    "juicio_alternativas",
    "completar_codigo",
]


# Mapa orientativo Bloom -> kinds recomendados (para el prompt y validación).
BLOOM_RECOMMENDED_KINDS: dict[BloomLevel, tuple[QuestionKind, ...]] = {
    "recordar": ("definicion",),
    "comprender": ("definicion", "diferenciacion"),
    "aplicar": ("caso_practico", "completar_codigo"),
    "analizar": ("comparacion", "analisis_consecuencia"),
    "evaluar": ("juicio_alternativas", "analisis_consecuencia"),
    "crear": ("caso_practico",),
}


# Cap por defecto de preguntas sobre el mismo concepto. Aplica cuando el KB
# tiene suficientes átomos como para no "vaciarse"; si el KB es pobre, se
# eleva automáticamente (ver `adaptive_max_per_concept`) para no devolver
# menos preguntas que las pedidas.
MAX_QUESTIONS_PER_CONCEPT = 2

# Límite absoluto: más de 4 preguntas sobre un mismo concept_id es
# indistinguible de un quiz monotemático, aunque el KB sea minúsculo.
ABSOLUTE_MAX_PER_CONCEPT = 4


def adaptive_max_per_concept(num_atoms: int, target_questions: int) -> int:
    """Cap dinámico por concept_id según riqueza del KB.

    Regla: `ceil(target / num_atoms)` acotado por
    `[MAX_QUESTIONS_PER_CONCEPT, ABSOLUTE_MAX_PER_CONCEPT]`.

    Ejemplos:
    - KB con 6 átomos, 10 preguntas → cap=2 (2*6=12 ≥ 10).
    - KB con 4 átomos, 10 preguntas → cap=3 (3*4=12 ≥ 10).
    - KB con 2 átomos, 10 preguntas → cap=4 (absoluto); se quedará corto
      pero no repetirá el mismo concepto más de 4 veces.
    """
    if num_atoms <= 0:
        return MAX_QUESTIONS_PER_CONCEPT
    import math
    needed = math.ceil(target_questions / num_atoms)
    return max(
        MAX_QUESTIONS_PER_CONCEPT,
        min(needed, ABSOLUTE_MAX_PER_CONCEPT),
    )


class PlannedQuestion(BaseModel):
    """Especificación de UNA pregunta antes de redactarla."""

    id: int = Field(ge=1)
    bloom_level: BloomLevel
    concept_id: str = Field(min_length=3, max_length=80)
    kind: QuestionKind
    focus: str | None = Field(default=None, max_length=200)

    @field_validator("concept_id", "focus")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v


class QuizPlan(BaseModel):
    """Plan global del quiz: matriz Bloom × concepto antes de redactar."""

    questions: list[PlannedQuestion] = Field(min_length=1, max_length=40)

    @field_validator("questions")
    @classmethod
    def _reassign_ids(cls, vs: list[PlannedQuestion]) -> list[PlannedQuestion]:
        for i, q in enumerate(vs, start=1):
            q.id = i
        return vs


# ----------- Coerción defensiva del payload de QuizPlan del LLM -----------

_QUIZ_QUESTIONS_ALIASES: tuple[str, ...] = (
    "questions",
    "preguntas",
    "items",
    "plan",
    "plan_items",
    "quiz",
    "list",
)
_Q_ID_ALIASES: tuple[str, ...] = ("id", "numero", "número", "idx", "index", "n")
_Q_BLOOM_ALIASES: tuple[str, ...] = (
    "bloom_level", "bloom", "nivel", "level", "cognitive_level", "nivel_bloom",
)
_Q_CONCEPT_ALIASES: tuple[str, ...] = (
    "concept_id", "atom_id", "concepto", "atomo", "átomo", "atom",
    "concept", "id_concepto", "ref",
)
_Q_KIND_ALIASES: tuple[str, ...] = (
    "kind", "type", "tipo", "question_type", "tipo_pregunta",
)
_Q_FOCUS_ALIASES: tuple[str, ...] = (
    "focus", "enfoque", "intention", "intencion", "intención", "tema",
)


def _pick_q_alias(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for k in aliases:
        if k in data and data[k] not in (None, ""):
            return data[k]
    return None


def coerce_quiz_plan_payload(raw: Any) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema canónico de `QuizPlan`.

    - Acepta `list` directa como `questions`.
    - Tolera alias ES/EN en top-level (`preguntas`, `items`, `plan`…).
    - Desenvuelve un nivel si el LLM metió todo en `{"plan": {...}}`,
      `{"output": {...}}`, `{"quiz_plan": {...}}`, etc.
    - Por cada pregunta, mapea alias de sus campos a los canónicos.
    - Devuelve `None` si es irrecuperable.
    """
    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict):
        return None

    for wrapper in (
        "quiz_plan", "quiz", "plan", "output", "result", "data", "respuesta",
    ):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and any(
            k in inner for k in _QUIZ_QUESTIONS_ALIASES
        ):
            raw = inner
            break
        if isinstance(inner, list):
            raw = {"questions": inner}
            break

    questions_raw = _pick_q_alias(raw, _QUIZ_QUESTIONS_ALIASES)
    if not isinstance(questions_raw, list) or not questions_raw:
        return None

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(questions_raw, start=1):
        if not isinstance(item, dict):
            continue
        q: dict[str, Any] = {}

        q_id = _pick_q_alias(item, _Q_ID_ALIASES)
        try:
            q["id"] = int(q_id) if q_id is not None else idx
        except (ValueError, TypeError):
            q["id"] = idx

        bloom = _pick_q_alias(item, _Q_BLOOM_ALIASES)
        if bloom is not None:
            q["bloom_level"] = bloom

        concept = _pick_q_alias(item, _Q_CONCEPT_ALIASES)
        if concept is not None:
            q["concept_id"] = str(concept).strip()

        kind = _pick_q_alias(item, _Q_KIND_ALIASES)
        if kind is not None:
            q["kind"] = kind

        focus = _pick_q_alias(item, _Q_FOCUS_ALIASES)
        if isinstance(focus, str) and focus.strip():
            q["focus"] = focus.strip()[:200]

        # Sólo añadimos preguntas con lo mínimo viable; el resto (normalización
        # de bloom/kind y verificación contra la KB) lo hace el caller.
        if "concept_id" in q and "bloom_level" in q and "kind" in q:
            out.append(q)

    if not out:
        return None
    return {"questions": out}


# --------- Fallback determinístico de QuizPlan sin LLM --------------------

_BLOOM_DISTRIBUTION: tuple[BloomLevel, ...] = (
    "recordar", "comprender", "aplicar", "analizar",
    "recordar", "comprender", "evaluar", "aplicar",
    "comprender", "analizar", "recordar", "comprender",
    "aplicar", "analizar", "evaluar",
)


def build_fallback_quiz_plan(
    kb: KnowledgeBase, target_count: int
) -> "QuizPlan":
    """Construye un `QuizPlan` determinísticamente desde la KB, sin LLM.

    Se usa cuando el LLM devuelve JSON vacío o irrecuperable. Prioriza
    definiciones (base del quiz), luego ejemplos, luego relaciones. La
    distribución Bloom se toma cíclica de `_BLOOM_DISTRIBUTION` hasta
    alcanzar `target_count` o agotar los átomos (≤2 preguntas por átomo).
    """
    atom_ids = kb.atom_ids()
    if not atom_ids:
        # Última línea: creamos 1 pregunta mínima con main_topic como concept_id
        # placeholder. Esto no debería ocurrir si el caller verificó kb.atom_count.
        return QuizPlan(
            questions=[
                PlannedQuestion(
                    id=1,
                    bloom_level="recordar",
                    concept_id="main_topic",
                    kind="definicion",
                    focus=f"Idea central: {kb.main_topic}",
                )
            ]
        )

    # Orden preferente: definitions > examples > relations > formulas > data.
    ordered: list[str] = []
    ordered.extend(d.id for d in kb.definitions)
    ordered.extend(e.id for e in kb.examples)
    ordered.extend(r.id for r in kb.relations)
    ordered.extend(f.id for f in kb.formulas_code)
    ordered.extend(n.id for n in kb.numeric_data)

    cap = adaptive_max_per_concept(len(ordered), target_count)
    usage: Counter[str] = Counter()
    questions: list[PlannedQuestion] = []

    atom_kind_map: dict[str, QuestionKind] = {}
    for d in kb.definitions:
        atom_kind_map[d.id] = "definicion"
    for e in kb.examples:
        atom_kind_map[e.id] = "caso_practico"
    for r in kb.relations:
        atom_kind_map[r.id] = "comparacion"
    for f in kb.formulas_code:
        atom_kind_map[f.id] = "completar_codigo"
    for n in kb.numeric_data:
        atom_kind_map[n.id] = "caso_practico"

    bloom_cycle = list(_BLOOM_DISTRIBUTION)
    bi = 0
    # Hasta 3 pasadas por la lista para respetar el cap sin duplicar demasiado.
    for pass_idx in range(3):
        if len(questions) >= target_count:
            break
        for aid in ordered:
            if len(questions) >= target_count:
                break
            if usage[aid] >= cap:
                continue
            bloom = bloom_cycle[bi % len(bloom_cycle)]
            bi += 1
            atom_kind = atom_kind_map.get(aid, "definicion")
            # Elegir kind coherente con el bloom (si el recomendado casa con
            # el tipo del átomo, lo usamos; si no, caemos al del átomo).
            recommended = BLOOM_RECOMMENDED_KINDS.get(bloom, ("definicion",))
            kind: QuestionKind = (
                atom_kind if atom_kind in recommended else recommended[0]
            )
            questions.append(
                PlannedQuestion(
                    id=len(questions) + 1,
                    bloom_level=bloom,
                    concept_id=aid,
                    kind=kind,
                    focus=None,
                )
            )
            usage[aid] += 1
        if pass_idx == 0 and len(questions) < target_count:
            # Segunda pasada: subimos el cap efectivo para rellenar.
            cap = min(cap + 1, ABSOLUTE_MAX_PER_CONCEPT)

    logger.info(
        "Fallback QuizPlan: %d preguntas generadas sin LLM (objetivo %d, cap=%d).",
        len(questions), target_count, cap,
    )
    return QuizPlan(questions=questions or [
        PlannedQuestion(
            id=1,
            bloom_level="recordar",
            concept_id=ordered[0],
            kind=atom_kind_map.get(ordered[0], "definicion"),
        )
    ])


# ---------------------------------------------- validación contra la KB ---


def _valid_atom_ids(kb: KnowledgeBase) -> set[str]:
    return set(kb.atom_ids())


# --------------------------- resolución tolerante de IDs (fuzzy matching) -
#
# El LLM, al planificar, escribe ids abreviados o con variantes ortográficas
# (acentos, plurales, underscores) que no existen como tales en la KB pero
# apuntan inequívocamente a un átomo real. Para no perder esas preguntas /
# slides, intentamos reconciliarlas antes de descartar.
#
# Estrategia en cascada (primera coincidencia gana):
# 1. Exacto.
# 2. Exacto tras normalización (deacentuar, lower, singularizar plurales).
# 3. El id pedido es **prefijo** del id válido (slug) — caso típico
#    `rel:subclase_de` → `rel:subclase_de_bicicletademontana_bicicleta`.
#    Solo en esa dirección: nunca colapsamos una entidad más específica
#    (`ex:bicicletademontana`) a una genérica (`ex:bicicleta`), porque
#    son conceptos distintos (bug detectado en v2.12.6 con `gemma3:12b`).
# 4. El slug "colapsado" (sin `_`) del id pedido está contenido en el
#    id válido (dirección raw ⊆ valid); nunca al revés.
# 5. Jaccard de tokens ≥ 0.78 entre los slugs, **y** con al menos 2
#    tokens en común (umbral endurecido en v2.12.6).


_SLUG_NORM_RE = re.compile(r"[^a-z0-9_\-]+")


def _deaccent_lower(text: str) -> str:
    n = unicodedata.normalize("NFKD", text)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _normalize_slug(slug: str) -> str:
    """Forma canónica: minúsculas, sin acentos, ñ→n, plurales → singular.

    Heurística simple para plurales: si un token de más de 3 caracteres
    termina en `s` o `es`, se recorta. No es gramática de laboratorio pero
    cubre los casos típicos del LLM (`bicicletas` → `bicicleta`).
    """
    s = _deaccent_lower(slug).replace("ñ", "n")
    s = _SLUG_NORM_RE.sub("_", s).strip("_-")
    parts = [p for p in s.split("_") if p]
    out: list[str] = []
    for p in parts:
        if len(p) > 4 and p.endswith("es"):
            p = p[:-2]
        elif len(p) > 3 and p.endswith("s"):
            p = p[:-1]
        out.append(p)
    return "_".join(out)


def _split_id(atom_id: str) -> tuple[str, str]:
    if ":" in atom_id:
        p, _, s = atom_id.partition(":")
        return p.strip().lower(), s.strip()
    return "", atom_id.strip()


def _collapse(slug: str) -> str:
    return slug.replace("_", "").replace("-", "")


def resolve_atom_id(raw_id: str, valid_ids: set[str]) -> str | None:
    """Intenta mapear un id posiblemente inexacto a uno presente en la KB.

    Devuelve `None` si no hay una candidatura suficientemente buena. No es
    una búsqueda global: privilegiamos ids con el mismo prefijo; sólo si
    ninguno encaja ampliamos al universo completo.
    """
    if not raw_id or not valid_ids:
        return None
    raw_id = raw_id.strip()
    if raw_id in valid_ids:
        return raw_id

    prefix, slug = _split_id(raw_id)
    if not slug:
        return None

    norm_slug = _normalize_slug(slug)
    norm_tokens = {t for t in norm_slug.split("_") if t}
    collapsed = _collapse(norm_slug)

    def _candidates(scope: list[str]) -> str | None:
        best: tuple[float, str | None] = (0.0, None)
        for v in scope:
            _, v_slug = _split_id(v)
            v_norm = _normalize_slug(v_slug)
            if v_norm == norm_slug:
                return v
            v_tokens = {t for t in v_norm.split("_") if t}
            v_collapsed = _collapse(v_norm)
            score = 0.0
            # Prefijo SOLO en dirección raw ⊆ valid: el LLM escribió un
            # slug incompleto (p. ej. `rel:subclase_de`) y la KB tiene el
            # expandido completo. Nunca al revés: si el LLM especifica
            # `ex:bicicletademontana` y la KB solo tiene `ex:bicicleta`,
            # NO colapsamos — son conceptos semánticamente distintos.
            if v_norm.startswith(norm_slug + "_"):
                score = max(score, 0.92)
            elif v_collapsed and collapsed and collapsed in v_collapsed:
                # Solo raw ⊆ valid (nunca al revés), y pedimos que el
                # "extra" del valid sea razonable (< 1.8× del raw): evita
                # colapsar slugs muy cortos contra nombres largos.
                if len(v_collapsed) <= max(12, int(len(collapsed) * 1.8)):
                    score = max(score, 0.8)
            if v_tokens and norm_tokens:
                inter = len(v_tokens & norm_tokens)
                union = len(v_tokens | norm_tokens)
                if union and inter >= 2:
                    # Jaccard solo cuenta si hay ≥ 2 tokens en común;
                    # con un único token compartido (p. ej. "bicicleta")
                    # el riesgo de fusión semántica errónea es alto.
                    score = max(score, inter / union)
            if score > best[0]:
                best = (score, v)
        return best[1] if best[0] >= 0.78 else None

    # Solo buscamos en átomos del MISMO prefix: si el LLM pidió `rel:X` no
    # tiene sentido devolverle un `def:Y` aunque sus tokens encajen, porque
    # el resto del pipeline (quiz `kind`, slide `kind`) asume el tipo
    # declarado. Si no hay candidatos, devolvemos None y que el saneador
    # descarte la entrada.
    same_prefix = [v for v in valid_ids if v.startswith(f"{prefix}:")]
    if not same_prefix:
        return None
    return _candidates(same_prefix)


def _reconcile_atom_ids(raw_ids: list[str], valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """Resuelve cada id contra la KB; separa resueltos y perdidos.

    El orden se conserva y se deduplican resueltos manteniendo la primera
    aparición (importante para preservar intención del LLM).
    """
    resolved: list[str] = []
    lost: list[str] = []
    seen: set[str] = set()
    for rid in raw_ids:
        hit = resolve_atom_id(rid, valid_ids)
        if hit is None:
            lost.append(rid)
            continue
        if hit not in seen:
            seen.add(hit)
            resolved.append(hit)
    return resolved, lost


def sanitize_slide_plan(plan: SlidePlan, kb: KnowledgeBase) -> SlidePlan:
    """Filtra `atom_ids` inexistentes y elimina slides que queden vacías.

    Antes de descartar un id se intenta reconciliar con la KB (acentos,
    plurales, prefijos). No lanza excepciones: el objetivo es tolerancia
    ante alucinaciones del LLM. Registra en log cada decisión.
    """
    valid = _valid_atom_ids(kb)
    cleaned: list[PlannedSlide] = []

    for slide in plan.slides:
        # La conclusión final la añade el renderer PPTX de forma determinista
        # a partir de `CONCLUSION_FROM_KB_PROMPT`. Descartamos cualquier
        # slide con kind=conclusion o cuyo título empiece por "conclus…"
        # (cubre variantes tipo "Conclusión general") para evitar duplicar.
        norm_title = _deaccent_lower(slide.title).strip()
        if slide.kind == "conclusion" or norm_title.startswith("conclus"):
            logger.info(
                "SlidePlan: slide '%s' (kind=%s) descartada (la conclusión la añade el renderer).",
                slide.title,
                slide.kind,
            )
            continue

        resolved, lost = _reconcile_atom_ids(slide.atom_ids, valid)
        if lost:
            logger.warning(
                "SlidePlan: slide '%s' descartó ids no reconciliables: %s",
                slide.title,
                lost,
            )
        recovered = [
            rid for rid, r in zip(slide.atom_ids, [resolve_atom_id(x, valid) for x in slide.atom_ids])
            if r is not None and r != rid
        ]
        if recovered:
            logger.info(
                "SlidePlan: slide '%s' recuperó ids vía fuzzy match: %s",
                slide.title,
                recovered,
            )
        if not resolved and slide.kind not in {"intro", "outlook"}:
            logger.warning(
                "SlidePlan: slide '%s' (%s) se descarta por quedarse sin átomos",
                slide.title,
                slide.kind,
            )
            continue
        cleaned.append(
            slide.model_copy(update={"atom_ids": resolved})
        )

    if not cleaned:
        # Plan vacío: devolvemos el original para que el generador decida.
        logger.error("SlidePlan quedó vacío tras saneado; se devuelve sin cambios.")
        return plan

    atom_usage = Counter(a for s in cleaned for a in s.atom_ids)
    duplicated = [a for a, n in atom_usage.items() if n > 1]
    if duplicated:
        logger.info("SlidePlan: átomos usados por varias slides: %s", duplicated)

    return SlidePlan(
        presentation_title=plan.presentation_title,
        slides=cleaned,
    )


def sanitize_quiz_plan(
    plan: QuizPlan, kb: KnowledgeBase, *, target_count: int
) -> QuizPlan:
    """Filtra concept_ids inválidos (con fuzzy match) y deduplica.

    Reglas clave:
    - Un `concept_id` mal escrito por el LLM se intenta reconciliar contra
      la KB antes de descartar (acentos, plurales, prefijos).
    - Se permite que un MISMO concepto aparezca en varias preguntas si
      cambia la pareja `(bloom_level, kind)`: una pregunta `recordar` y
      otra `aplicar` sobre el mismo átomo son preguntas distintas.
    - Si el plan supera el objetivo, se recorta al final (no al principio)
      priorizando la distribución Bloom recibida.
    """
    valid = _valid_atom_ids(kb)
    kept: list[PlannedQuestion] = []
    seen: set[tuple[str, str, str]] = set()
    per_concept: Counter[str] = Counter()
    cap = adaptive_max_per_concept(len(valid), target_count)
    logger.info(
        "QuizPlan: cap por concepto = %d (KB con %d átomos, objetivo %d preguntas).",
        cap, len(valid), target_count,
    )

    for q in plan.questions:
        resolved = resolve_atom_id(q.concept_id, valid)
        if resolved is None:
            logger.warning(
                "QuizPlan: pregunta id=%s descartada, concept_id desconocido '%s'",
                q.id,
                q.concept_id,
            )
            continue
        if resolved != q.concept_id:
            logger.info(
                "QuizPlan: pregunta id=%s reconcilió concept_id '%s' -> '%s'",
                q.id,
                q.concept_id,
                resolved,
            )
            q = q.model_copy(update={"concept_id": resolved})

        key = (q.concept_id, q.bloom_level, q.kind)
        if key in seen:
            logger.warning(
                "QuizPlan: pregunta id=%s descartada por repetir (concept_id, bloom, kind)=%s",
                q.id,
                key,
            )
            continue
        if per_concept[q.concept_id] >= cap:
            logger.warning(
                "QuizPlan: pregunta id=%s descartada, ya hay %d preguntas sobre '%s' (cap=%d)",
                q.id,
                per_concept[q.concept_id],
                q.concept_id,
                cap,
            )
            continue
        seen.add(key)
        per_concept[q.concept_id] += 1
        kept.append(q)

    if len(kept) > target_count:
        kept = kept[:target_count]

    for i, q in enumerate(kept, start=1):
        q.id = i

    if not kept:
        logger.error(
            "QuizPlan quedó vacío tras saneado; se usa fallback determinístico."
        )
        return build_fallback_quiz_plan(kb, target_count=target_count)

    return QuizPlan(questions=kept)


def bloom_distribution(plan: QuizPlan) -> dict[BloomLevel, int]:
    """Cuenta la distribución real de Bloom en un `QuizPlan`."""
    counter: Counter[BloomLevel] = Counter(q.bloom_level for q in plan.questions)
    return dict(counter)  # type: ignore[return-value]
