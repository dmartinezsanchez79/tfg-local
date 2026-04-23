"""Modelos de planificación para slides y quiz.

`SlidePlan` y `QuizPlan` son contratos intermedios entre la KB y la
generación final. El LLM los produce; este módulo los valida, repara
(coerce + fuzzy match) y, como red de seguridad para modelos pequeños
(<7B), sabe construir un plan mínimo desde la KB sin LLM.
"""
from __future__ import annotations

import logging
import math
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


# =========================================================================
# SLIDES
# =========================================================================

SlideKind = Literal[
    "intro", "definition", "example", "comparison",
    "code", "process", "relations", "outlook", "conclusion",
]
_VALID_SLIDE_KINDS: frozenset[str] = frozenset(SlideKind.__args__)  # type: ignore[attr-defined]


class PlannedSlide(BaseModel):
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
    presentation_title: str = Field(min_length=3, max_length=160)
    slides: list[PlannedSlide] = Field(min_length=3, max_length=20)

    @field_validator("presentation_title")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# ---------------- coerción tolerante del payload del LLM -------------------

_SLIDE_ALIASES: dict[str, tuple[str, ...]] = {
    "title":  ("presentation_title", "title", "titulo", "título", "tema", "topic"),
    "slides": ("slides", "diapositivas", "plan", "content", "contenido"),
}
_SLIDE_ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "title":  ("title", "titulo", "título", "heading", "nombre"),
    "kind":   ("kind", "type", "tipo", "layout"),
    "atoms":  ("atom_ids", "atomos", "átomos", "ids", "atoms", "refs", "references"),
    "focus":  ("focus", "enfoque", "intention", "intencion", "idea"),
}
_KIND_SYNONYMS: dict[str, str] = {
    "introduccion": "intro", "introducción": "intro", "introduction": "intro",
    "definicion": "definition", "definición": "definition", "definitions": "definition",
    "ejemplo": "example", "ejemplos": "example", "examples": "example",
    "comparacion": "comparison", "comparación": "comparison", "comparativa": "comparison",
    "codigo": "code", "código": "code",
    "proceso": "process", "flujo": "process",
    "relacion": "relations", "relación": "relations", "relaciones": "relations",
    "panoramica": "outlook", "panorámica": "outlook", "vision": "outlook", "visión": "outlook",
    "conclusion": "conclusion", "conclusión": "conclusion", "conclusiones": "conclusion",
}


def _pick(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in data and data[k] not in (None, "", []):
            return data[k]
    return None


def _normalize_kind(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    k = raw.strip().lower()
    if k in _VALID_SLIDE_KINDS:
        return k
    return _KIND_SYNONYMS.get(k)


def _coerce_id_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _unwrap(raw: dict[str, Any], wrappers: tuple[str, ...], probe_keys: tuple[str, ...]) -> dict[str, Any]:
    """Desenvuelve un nivel si el LLM metió el payload dentro de una clave."""
    for w in wrappers:
        inner = raw.get(w)
        if isinstance(inner, dict) and any(k in inner for k in probe_keys):
            return inner
        if isinstance(inner, list) and "questions" in probe_keys:
            return {"questions": inner}
    return raw


def coerce_slide_plan_payload(raw: Any, kb: KnowledgeBase) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema canónico de `SlidePlan`.

    Tolera claves ES/EN, un nivel de anidamiento y listas de slides parciales.
    Devuelve `None` si la estructura es irrecuperable.
    """
    if not isinstance(raw, dict):
        return None
    probe = _SLIDE_ALIASES["title"] + _SLIDE_ALIASES["slides"]
    raw = _unwrap(raw, ("plan", "output", "result", "data", "respuesta"), probe)

    slides_raw = _pick(raw, _SLIDE_ALIASES["slides"])
    if not isinstance(slides_raw, list) or not slides_raw:
        return None

    title = _pick(raw, _SLIDE_ALIASES["title"])
    if not isinstance(title, str) or not title.strip():
        title = kb.main_topic or "Presentación"

    slides_out: list[dict[str, Any]] = []
    for item in slides_raw:
        if not isinstance(item, dict):
            continue
        s_title = _pick(item, _SLIDE_ITEM_ALIASES["title"])
        if not isinstance(s_title, str) or not s_title.strip():
            continue
        kind = _normalize_kind(_pick(item, _SLIDE_ITEM_ALIASES["kind"])) or "outlook"
        atoms = _coerce_id_list(_pick(item, _SLIDE_ITEM_ALIASES["atoms"]))[:12]
        entry: dict[str, Any] = {
            "title": s_title.strip()[:80],
            "kind": kind,
            "atom_ids": atoms,
        }
        focus = _pick(item, _SLIDE_ITEM_ALIASES["focus"])
        if isinstance(focus, str) and focus.strip():
            entry["focus"] = focus.strip()[:200]
        slides_out.append(entry)

    if not slides_out:
        return None
    return {"presentation_title": title.strip()[:160], "slides": slides_out}


# ---------------- fallback determinístico sin LLM (para modelos pequeños) --

_ATOM_KIND_MAP: dict[type, str] = {
    Definition: "definition",
    Example: "example",
    FormulaOrCode: "code",
    NumericDatum: "outlook",
    Relation: "relations",
}


def build_fallback_slide_plan(kb: KnowledgeBase) -> SlidePlan:
    """Construye un `SlidePlan` válido desde la KB sin llamar al LLM.

    Se dispara cuando el modelo devuelve JSON irrecuperable (típico con
    modelos <7B). Agrupa átomos por `subtopic` si existen; si no, por tipo.
    """
    title = (kb.main_topic or "Presentación").strip()[:160]
    slides: list[PlannedSlide] = [
        PlannedSlide(
            title="Introducción",
            kind="intro",
            focus=f"Panorámica general sobre {kb.main_topic}."[:200],
        )
    ]
    used: set[str] = {_deaccent_lower("Introducción")}

    def add(title: str, kind: str, ids: list[str], focus: str | None = None) -> None:
        base = (_humanize_title(title).strip()[:78] or "Contenido")
        cand, i = base, 2
        while _deaccent_lower(cand) in used:
            cand = f"{base[:78-4]} ({i})"
            i += 1
        used.add(_deaccent_lower(cand))
        slides.append(PlannedSlide(title=cand, kind=kind, atom_ids=ids, focus=focus))  # type: ignore[arg-type]

    # Agrupamos por subtopic si la KB los tiene.
    groups: dict[str, list[tuple[str, type]]] = defaultdict(list)
    for atom in kb._iter_atoms():  # noqa: SLF001
        key = (getattr(atom, "subtopic", "") or "").strip()
        if key:
            groups[key].append((atom.id, type(atom)))

    if groups:
        order = {s: i for i, s in enumerate(kb.subtopics or [])}
        for subtopic, items in sorted(groups.items(), key=lambda kv: order.get(kv[0], 10_000)):
            types = [t for _, t in items]
            dominant = Counter(types).most_common(1)[0][0]
            kind = _ATOM_KIND_MAP.get(dominant, "outlook")
            add(subtopic, kind, [aid for aid, _ in items[:10]])
    else:
        buckets: list[tuple[str, str, list[str]]] = [
            ("Definiciones clave", "definition", [d.id for d in kb.definitions[:8]]),
            ("Ejemplos ilustrativos", "example", [e.id for e in kb.examples[:6]]),
            ("Fórmulas y código", "code", [f.id for f in kb.formulas_code[:6]]),
            ("Relaciones entre conceptos", "relations", [r.id for r in kb.relations[:8]]),
            ("Datos numéricos", "outlook", [d.id for d in kb.numeric_data[:6]]),
        ]
        for t, k, ids in buckets:
            if ids:
                add(t, k, ids)

    while len(slides) < 3:
        add("Aspectos complementarios", "outlook", [],
            focus="Ideas adicionales derivadas del material.")
    if len(slides) > 20:
        slides = slides[:20]

    logger.info("Fallback SlidePlan desde KB: %d slides (sin LLM).", len(slides))
    return SlidePlan(presentation_title=title, slides=slides)


# =========================================================================
# QUIZ
# =========================================================================

BloomLevel = Literal["recordar", "comprender", "aplicar", "analizar", "evaluar", "crear"]
QuestionKind = Literal[
    "definicion", "diferenciacion", "caso_practico", "comparacion",
    "analisis_consecuencia", "juicio_alternativas", "completar_codigo",
]

BLOOM_RECOMMENDED_KINDS: dict[BloomLevel, tuple[QuestionKind, ...]] = {
    "recordar": ("definicion",),
    "comprender": ("definicion", "diferenciacion"),
    "aplicar": ("caso_practico", "completar_codigo"),
    "analizar": ("comparacion", "analisis_consecuencia"),
    "evaluar": ("juicio_alternativas", "analisis_consecuencia"),
    "crear": ("caso_practico",),
}

MAX_QUESTIONS_PER_CONCEPT = 2
ABSOLUTE_MAX_PER_CONCEPT = 4


def adaptive_max_per_concept(num_atoms: int, target_questions: int) -> int:
    """Cap dinámico `ceil(target/atoms)` acotado a [2, 4]."""
    if num_atoms <= 0:
        return MAX_QUESTIONS_PER_CONCEPT
    needed = math.ceil(target_questions / num_atoms)
    return max(MAX_QUESTIONS_PER_CONCEPT, min(needed, ABSOLUTE_MAX_PER_CONCEPT))


class PlannedQuestion(BaseModel):
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
    questions: list[PlannedQuestion] = Field(min_length=1, max_length=40)

    @field_validator("questions")
    @classmethod
    def _reassign_ids(cls, vs: list[PlannedQuestion]) -> list[PlannedQuestion]:
        for i, q in enumerate(vs, start=1):
            q.id = i
        return vs


# ---------------- coerción tolerante del payload del LLM -------------------

_QUIZ_ALIASES: dict[str, tuple[str, ...]] = {
    "questions": ("questions", "preguntas", "items", "plan", "quiz"),
    "id":        ("id", "numero", "número", "idx", "index", "n"),
    "bloom":     ("bloom_level", "bloom", "nivel", "level", "nivel_bloom"),
    "concept":   ("concept_id", "atom_id", "concepto", "atom", "concept", "ref"),
    "kind":      ("kind", "type", "tipo", "question_type"),
    "focus":     ("focus", "enfoque", "intention", "tema"),
}


def coerce_quiz_plan_payload(raw: Any) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema canónico de `QuizPlan`."""
    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict):
        return None
    raw = _unwrap(raw, ("quiz_plan", "quiz", "plan", "output", "result", "data"),
                  _QUIZ_ALIASES["questions"])

    qs_raw = _pick(raw, _QUIZ_ALIASES["questions"])
    if not isinstance(qs_raw, list) or not qs_raw:
        return None

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(qs_raw, start=1):
        if not isinstance(item, dict):
            continue
        q: dict[str, Any] = {}
        try:
            raw_id = _pick(item, _QUIZ_ALIASES["id"])
            q["id"] = int(raw_id) if raw_id is not None else idx
        except (ValueError, TypeError):
            q["id"] = idx

        bloom = _pick(item, _QUIZ_ALIASES["bloom"])
        concept = _pick(item, _QUIZ_ALIASES["concept"])
        kind = _pick(item, _QUIZ_ALIASES["kind"])
        if bloom is None or concept is None or kind is None:
            continue
        q["bloom_level"] = bloom
        q["concept_id"] = str(concept).strip()
        q["kind"] = kind
        focus = _pick(item, _QUIZ_ALIASES["focus"])
        if isinstance(focus, str) and focus.strip():
            q["focus"] = focus.strip()[:200]
        out.append(q)

    if not out:
        return None
    return {"questions": out}


# ---------------- fallback determinístico sin LLM (para modelos pequeños) --

_BLOOM_CYCLE: tuple[BloomLevel, ...] = (
    "recordar", "comprender", "aplicar", "analizar",
    "recordar", "comprender", "evaluar", "aplicar",
    "comprender", "analizar", "recordar", "comprender",
    "aplicar", "analizar", "evaluar",
)


def build_fallback_quiz_plan(kb: KnowledgeBase, target_count: int) -> QuizPlan:
    """Construye un `QuizPlan` desde la KB sin llamar al LLM."""
    # Orden de preferencia: defs > ejemplos > relaciones > código > datos.
    ordered: list[str] = [
        *[d.id for d in kb.definitions],
        *[e.id for e in kb.examples],
        *[r.id for r in kb.relations],
        *[f.id for f in kb.formulas_code],
        *[n.id for n in kb.numeric_data],
    ]
    if not ordered:
        return QuizPlan(questions=[PlannedQuestion(
            id=1, bloom_level="recordar", concept_id="main_topic",
            kind="definicion", focus=f"Idea central: {kb.main_topic}",
        )])

    atom_kind: dict[str, QuestionKind] = {}
    for d in kb.definitions:     atom_kind[d.id] = "definicion"
    for e in kb.examples:        atom_kind[e.id] = "caso_practico"
    for r in kb.relations:       atom_kind[r.id] = "comparacion"
    for f in kb.formulas_code:   atom_kind[f.id] = "completar_codigo"
    for n in kb.numeric_data:    atom_kind[n.id] = "caso_practico"

    cap = adaptive_max_per_concept(len(ordered), target_count)
    usage: Counter[str] = Counter()
    questions: list[PlannedQuestion] = []
    bi = 0

    for pass_idx in range(3):
        if len(questions) >= target_count:
            break
        for aid in ordered:
            if len(questions) >= target_count:
                break
            if usage[aid] >= cap:
                continue
            bloom = _BLOOM_CYCLE[bi % len(_BLOOM_CYCLE)]
            bi += 1
            recommended = BLOOM_RECOMMENDED_KINDS.get(bloom, ("definicion",))
            base_kind = atom_kind.get(aid, "definicion")
            kind: QuestionKind = base_kind if base_kind in recommended else recommended[0]
            questions.append(PlannedQuestion(
                id=len(questions) + 1, bloom_level=bloom, concept_id=aid, kind=kind,
            ))
            usage[aid] += 1
        if pass_idx == 0 and len(questions) < target_count:
            cap = min(cap + 1, ABSOLUTE_MAX_PER_CONCEPT)

    logger.info("Fallback QuizPlan: %d preguntas sin LLM (objetivo %d, cap=%d).",
                len(questions), target_count, cap)
    if not questions:
        questions = [PlannedQuestion(
            id=1, bloom_level="recordar", concept_id=ordered[0],
            kind=atom_kind.get(ordered[0], "definicion"),
        )]
    return QuizPlan(questions=questions)


# =========================================================================
# VALIDACIÓN CONTRA LA KB: resolver de IDs + saneadores
# =========================================================================

# El LLM puede escribir ids con acentos, plurales o mayúsculas. Cascada:
#   1. Exacto.
#   2. Exacto tras deaccent + lower.
#   3. El slug pedido es prefijo del slug válido (dentro del mismo tipo).
# Si ninguna encaja, la entrada se descarta en el saneador.

_SLUG_NORM_RE = re.compile(r"[^a-z0-9_\-]+")
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9\+]{3,}")
_TITLE_STOPWORDS = {
    "de", "del", "la", "las", "el", "los", "en", "para", "por", "con", "sin", "una",
    "uno", "unos", "unas", "the", "and", "for", "with", "from", "into",
}


def _deaccent_lower(text: str) -> str:
    n = unicodedata.normalize("NFKD", text)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _normalize_slug(slug: str) -> str:
    s = _deaccent_lower(slug).replace("ñ", "n")
    return _SLUG_NORM_RE.sub("_", s).strip("_-")


def _humanize_title(text: str) -> str:
    clean = re.sub(r"[_\-]+", " ", text or "").strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return text.strip()
    out: list[str] = []
    for w in clean.split(" "):
        low = w.lower()
        if low in {"poo", "oop", "c++", "java"}:
            out.append(w.upper() if low != "java" else "Java")
        elif w.isupper() and len(w) <= 5:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def _title_signature(text: str) -> set[str]:
    norm = _deaccent_lower(_normalize_slug(text)).replace("_", " ")
    toks = {t for t in _TITLE_TOKEN_RE.findall(norm) if t not in _TITLE_STOPWORDS}
    return toks


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_duplicate_slide(candidate: PlannedSlide, existing: list[PlannedSlide]) -> bool:
    cand_title = _deaccent_lower(candidate.title).strip()
    cand_sig = _title_signature(candidate.title)
    cand_atoms = set(candidate.atom_ids)
    for prev in existing:
        prev_title = _deaccent_lower(prev.title).strip()
        if cand_title == prev_title:
            return True
        if cand_sig and _jaccard(cand_sig, _title_signature(prev.title)) >= 0.72:
            return True
        prev_atoms = set(prev.atom_ids)
        if (
            candidate.kind == prev.kind
            and cand_atoms
            and prev_atoms
            and len(cand_atoms & prev_atoms) / max(1, min(len(cand_atoms), len(prev_atoms))) >= 0.8
        ):
            return True
    return False


def _split_id(atom_id: str) -> tuple[str, str]:
    if ":" in atom_id:
        p, _, s = atom_id.partition(":")
        return p.strip().lower(), s.strip()
    return "", atom_id.strip()


def resolve_atom_id(raw_id: str, valid_ids: set[str]) -> str | None:
    """Mapea un id posiblemente inexacto a uno presente en la KB."""
    if not raw_id or not valid_ids:
        return None
    raw_id = raw_id.strip()
    if raw_id in valid_ids:
        return raw_id
    prefix, slug = _split_id(raw_id)
    if not slug:
        return None
    norm = _normalize_slug(slug)
    same = [v for v in valid_ids if v.startswith(f"{prefix}:")]
    for v in same:
        if _normalize_slug(_split_id(v)[1]) == norm:
            return v
    for v in same:
        if _normalize_slug(_split_id(v)[1]).startswith(norm + "_"):
            return v
    return None


def _reconcile(raw_ids: list[str], valid: set[str]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    lost: list[str] = []
    seen: set[str] = set()
    for rid in raw_ids:
        hit = resolve_atom_id(rid, valid)
        if hit is None:
            lost.append(rid)
        elif hit not in seen:
            seen.add(hit)
            resolved.append(hit)
    return resolved, lost


def sanitize_slide_plan(plan: SlidePlan, kb: KnowledgeBase) -> SlidePlan:
    """Filtra `atom_ids` inexistentes y descarta slides que queden vacías.

    La slide de conclusión la inserta el renderer PPTX: si el LLM la incluyó,
    se descarta aquí para evitar duplicarla.
    """
    valid = set(kb.atom_ids())
    cleaned: list[PlannedSlide] = []

    has_intro = False
    for slide in plan.slides:
        norm_title = _deaccent_lower(slide.title).strip()
        if slide.kind == "conclusion" or norm_title.startswith("conclus"):
            continue
        intro_like = slide.kind == "intro" or norm_title.startswith("introduc")
        if intro_like and has_intro:
            continue
        resolved, lost = _reconcile(slide.atom_ids, valid)
        if lost:
            logger.info("SlidePlan: '%s' descartó ids: %s", slide.title, lost)
        if not resolved and slide.kind not in {"intro", "outlook"}:
            continue
        repaired = slide.model_copy(
            update={"title": _humanize_title(slide.title), "atom_ids": resolved}
        )
        if _is_duplicate_slide(repaired, cleaned):
            logger.info("SlidePlan: slide redundante descartada: '%s'", repaired.title)
            continue
        cleaned.append(repaired)
        if intro_like:
            has_intro = True

    if len(cleaned) < 3:
        logger.warning(
            "SlidePlan demasiado corto tras saneado (%d<3); se devuelve el plan original.",
            len(cleaned),
        )
        return plan
    return SlidePlan(presentation_title=plan.presentation_title, slides=cleaned)


def sanitize_quiz_plan(plan: QuizPlan, kb: KnowledgeBase, *, target_count: int) -> QuizPlan:
    """Filtra concept_ids inválidos, deduplica y respeta el cap por concepto.

    Se permiten varias preguntas sobre el mismo `concept_id` siempre que
    cambie la pareja `(bloom_level, kind)`.
    """
    valid = set(kb.atom_ids())
    kept: list[PlannedQuestion] = []
    seen: set[tuple[str, str, str]] = set()
    per_concept: Counter[str] = Counter()
    cap = adaptive_max_per_concept(len(valid), target_count)

    for q in plan.questions:
        resolved = resolve_atom_id(q.concept_id, valid)
        if resolved is None:
            continue
        if resolved != q.concept_id:
            q = q.model_copy(update={"concept_id": resolved})
        key = (q.concept_id, q.bloom_level, q.kind)
        if key in seen or per_concept[q.concept_id] >= cap:
            continue
        seen.add(key)
        per_concept[q.concept_id] += 1
        kept.append(q)

    if len(kept) > target_count:
        kept = kept[:target_count]
    for i, q in enumerate(kept, start=1):
        q.id = i

    if not kept:
        logger.warning("QuizPlan vacío tras saneado; usando fallback determinístico.")
        return build_fallback_quiz_plan(kb, target_count=target_count)
    return QuizPlan(questions=kept)
