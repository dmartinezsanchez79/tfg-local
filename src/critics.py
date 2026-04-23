"""Revisor crítico determinista para Quiz y Presentación.

Diseño deliberadamente pequeño:
- 5 detectores para quiz: `banned_phrase` (high), `duplicate` (high),
  `unbalanced_options` (low), `meta_language` (low), `not_grounded` (low).
- 3 detectores para slides: `duplicate_content` (low), `meta_language`
  (low), `not_grounded` (low). Los demás anti-patrones (flechas, eco del
  título, anglicismos) se filtran ya en `_clean_bullet`.
- Un recorte barato por regex para preámbulos meta en enunciados de
  quiz: corrige sin regenerar cuando basta con eliminar una muletilla.
- Una sola pasada de regeneración para issues `high`. Si tras
  regenerar la pregunta sigue siendo crítica, se descarta.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Iterable, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .pptx_generator import BuiltSlide
    from .quiz_generator import QuizQuestion

from .exceptions import GenerationError
from .knowledge_base import (
    Definition,
    Example,
    FormulaOrCode,
    KnowledgeBase,
    NumericDatum,
    Relation,
)
from .ollama_client import OllamaClient
from .plans import PlannedQuestion, PlannedSlide, QuizPlan, SlidePlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- tipos ----

Severity = Literal["low", "medium", "high"]


class QuizIssue(BaseModel):
    question_id: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=3, max_length=400)
    severity: Severity = "medium"


class QuizReview(BaseModel):
    issues: list[QuizIssue] = Field(default_factory=list)

    def blocker_ids(self) -> set[int]:
        return {i.question_id for i in self.issues if i.severity == "high"}


class SlideIssue(BaseModel):
    slide_index: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=3, max_length=400)
    severity: Severity = "medium"


class SlideReview(BaseModel):
    issues: list[SlideIssue] = Field(default_factory=list)

    def blocker_indices(self) -> set[int]:
        return {i.slide_index for i in self.issues if i.severity == "high"}


# ------------------------------------------------------- tokenización ------

_WORD_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "de", "la", "el", "en", "los", "las", "un", "una", "unos", "unas",
    "por", "para", "con", "sin", "que", "cual", "como", "se", "su", "sus",
    "ser", "son", "sus", "del", "al", "lo", "le", "les", "y", "o", "u", "e",
    "no", "si", "pero", "esto", "este", "esta", "estos", "estas", "ese",
    "eso", "esa", "esos", "esas",
}


def _deaccent_lower(text: str) -> str:
    n = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in n if not unicodedata.combining(ch)).lower()


def _tokens(text: str, *, min_len: int = 4) -> set[str]:
    out: set[str] = set()
    for m in _WORD_RE.finditer(_deaccent_lower(text)):
        tok = m.group(0)
        if len(tok) >= min_len and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _atom_text(
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> str:
    if isinstance(atom, Definition):
        return f"{atom.term} {atom.definition}"
    if isinstance(atom, Example):
        return " ".join([atom.name, atom.description, *atom.attributes, *atom.methods])
    if isinstance(atom, FormulaOrCode):
        return " ".join(filter(None, [atom.caption or "", atom.content]))
    if isinstance(atom, NumericDatum):
        return f"{atom.value} {atom.description}"
    if isinstance(atom, Relation):
        return " ".join(filter(None, [atom.source, atom.target, atom.kind, atom.description or ""]))
    return ""


def _kb_vocabulary(kb: KnowledgeBase) -> set[str]:
    vocab: set[str] = set()
    vocab.update(_tokens(kb.main_topic))
    for st in kb.subtopics:
        vocab.update(_tokens(st))
    for atom in kb._iter_atoms():  # noqa: SLF001
        vocab.update(_tokens(_atom_text(atom)))
    return vocab


def _is_grounded(text: str, vocab: set[str]) -> bool:
    if not vocab:
        return True
    return bool(_tokens(text) & vocab)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------- detectores: QUIZ ----

_QUIZ_BANNED_OPTION_PHRASES = (
    "todas las anteriores", "todas las opciones",
    "ninguna de las anteriores", "ninguna de las opciones",
    "a y b", "b y c", "a y c",
    "ninguna es correcta", "solo a", "solo b",
)

_QUIZ_META_PHRASES = (
    "segun el texto", "segun el documento", "segun el autor",
    "segun la definicion", "de acuerdo con el documento",
    "de acuerdo con el texto", "conforme al documento",
    "el autor dice", "el texto menciona", "el documento indica",
    "como se menciona en", "basandose en el documento",
    "en este apartado", "en esta pregunta",
)

# Preámbulos que pueden recortarse sin regenerar:
#   "Según la definición del documento, ¿qué es X?" → "¿qué es X?"
_QUIZ_META_PREAMBLE_RE = re.compile(
    r"^\s*(?:segun|conforme\s+a[l]?|de\s+acuerdo\s+con|basandose\s+en|como\s+se\s+(?:indica|menciona|define)\s+en)\b"
    r"[^,.:;¿?]{1,80}[,:;]\s*",
    re.IGNORECASE,
)


def _strip_meta_preamble(text: str) -> str:
    if not isinstance(text, str):
        return text
    m = _QUIZ_META_PREAMBLE_RE.match(_deaccent_lower(text))
    if not m:
        return text
    remainder = text[m.end():].lstrip()
    if len(remainder) < 10:
        return text
    if remainder and remainder[0] in "¿¡" and len(remainder) > 1 and remainder[1].islower():
        remainder = remainder[0] + remainder[1].upper() + remainder[2:]
    elif remainder and remainder[0].islower():
        remainder = remainder[0].upper() + remainder[1:]
    return remainder


# Umbral empírico: dos preguntas con >=0.75 de tokens en común (enunciado
# + 4 opciones, sin stopwords) son variantes de la misma pregunta.
_DUPLICATE_JACCARD_THRESHOLD = 0.75


def _question_signature_tokens(q: "QuizQuestion") -> set[str]:
    opts = [getattr(q.options, L) for L in ("A", "B", "C", "D")]
    return _tokens(" ".join([q.question, *opts]))


def _deterministic_quiz_issues(
    kb: KnowledgeBase,
    quiz_questions: list["QuizQuestion"],
    vocab: set[str],
) -> list[QuizIssue]:
    issues: list[QuizIssue] = []
    signatures: list[tuple[int, set[str]]] = []

    for q in quiz_questions:
        qtext_norm = _deaccent_lower(q.question)
        opt_texts = [getattr(q.options, L) for L in ("A", "B", "C", "D")]
        opt_norms = [_deaccent_lower(t) for t in opt_texts]

        for L, n in zip("ABCD", opt_norms):
            if any(b in n for b in _QUIZ_BANNED_OPTION_PHRASES):
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="banned_phrase",
                    description=f"Opción {L} usa frase prohibida.",
                    severity="high",
                ))
                break

        lens = [len(t) for t in opt_texts]
        if lens:
            maxlen = max(lens)
            others = sorted(lens)[:-1]
            avg = (sum(others) / len(others)) if others else 0
            if maxlen >= 40 and avg > 0 and maxlen >= 2.2 * avg:
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="unbalanced_options",
                    description="Una opción es sensiblemente más larga que el resto.",
                    severity="low",
                ))

        full_norm = qtext_norm + " " + _deaccent_lower(q.justification)
        if any(m in full_norm for m in _QUIZ_META_PHRASES):
            issues.append(QuizIssue(
                question_id=q.id,
                kind="meta_language",
                description="Lenguaje meta en el enunciado o la justificación.",
                severity="low",
            ))

        correct = getattr(q.options, q.correct_answer)
        if not _is_grounded(f"{q.question} {correct} {q.justification}", vocab):
            issues.append(QuizIssue(
                question_id=q.id,
                kind="not_grounded",
                description="La pregunta no contiene términos de la Base de Conocimiento.",
                severity="low",
            ))

        sig = _question_signature_tokens(q)
        for prev_id, prev_sig in signatures:
            sim = _jaccard(sig, prev_sig)
            if sim >= _DUPLICATE_JACCARD_THRESHOLD:
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="duplicate",
                    description=f"Muy similar a la pregunta {prev_id} (Jaccard={sim:.2f}).",
                    severity="high",
                ))
                break
        signatures.append((q.id, sig))

    return issues


# -------------------------------------------------- detectores: SLIDES ----

_SLIDE_META_PHRASES = (
    "en este apartado", "en esta diapositiva",
    "se habla de", "se trata de", "se estudia", "se explica",
    "es importante destacar", "a continuacion",
    "el autor explica", "el documento",
    "facilita la comprension de", "ayuda a comprender",
    "permite entender", "permite comprender",
)


def _deterministic_slide_issues(
    kb: KnowledgeBase,
    built_slides: list["BuiltSlide"],
    vocab: set[str],
) -> list[SlideIssue]:
    issues: list[SlideIssue] = []
    seen: dict[str, int] = {}

    for idx, s in enumerate(built_slides, start=1):
        bullets_norm = [_deaccent_lower(b) for b in s.bullets]

        hits = [
            s.bullets[i][:60]
            for i, b in enumerate(bullets_norm)
            if any(m in b for m in _SLIDE_META_PHRASES)
        ]
        if hits:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="meta_language",
                description=f"Bullets con lenguaje meta: {hits[:2]}",
                severity="low",
            ))

        if vocab and s.kind not in {"intro", "conclusion"}:
            joined = " ".join(s.bullets)
            if joined.strip() and not _is_grounded(joined, vocab):
                issues.append(SlideIssue(
                    slide_index=idx,
                    kind="not_grounded",
                    description="Los bullets no referencian términos de la KB.",
                    severity="low",
                ))

        for b in s.bullets:
            key = " ".join(sorted(_tokens(b)))[:120]
            if not key:
                continue
            if key in seen and seen[key] != idx:
                issues.append(SlideIssue(
                    slide_index=idx,
                    kind="duplicate_content",
                    description=f"Bullet coincide con slide {seen[key]}.",
                    severity="low",
                ))
                break
            seen.setdefault(key, idx)

    return issues


# ------------------------------------------------------ API: revisar ------


def review_quiz(
    kb: KnowledgeBase, quiz_questions: list["QuizQuestion"]
) -> QuizReview:
    vocab = _kb_vocabulary(kb)
    return QuizReview(issues=_deterministic_quiz_issues(kb, quiz_questions, vocab))


def review_slides(
    kb: KnowledgeBase, built_slides: list["BuiltSlide"]
) -> SlideReview:
    vocab = _kb_vocabulary(kb)
    return SlideReview(issues=_deterministic_slide_issues(kb, built_slides, vocab))


# ------------------------------------------------------ API: refinar ------


def _pick_alternative_concept(kb: KnowledgeBase, avoid: set[str]) -> str | None:
    """Primer atom_id del KB que no esté ya usado."""
    for aid in kb.atom_ids():
        if aid not in avoid:
            return aid
    return None


def refine_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    quiz_questions: list["QuizQuestion"],
    quiz_plan: QuizPlan,
    *,
    min_questions: int = 1,
    max_questions: int | None = None,
) -> tuple[list["QuizQuestion"], QuizReview]:
    """Un único pase de refinado.

    1. Recorta preámbulos meta por regex (corrección barata).
    2. Revisa deterministamente. Para cada issue `high`, intenta regenerar
       una vez (rotando `concept_id` si el issue es `duplicate`).
    3. Si la regeneración vuelve a ser crítica, la pregunta se descarta.
    4. Si sobran, recorta a `max_questions` priorizando las que tienen
       menos issues residuales.
    """
    from .quiz_generator import generate_single_question

    for q in quiz_questions:
        q.question = _strip_meta_preamble(q.question)

    review = review_quiz(kb, quiz_questions)
    critical = review.blocker_ids()
    if not critical:
        return quiz_questions, review

    logger.info("Refinando quiz: %d preguntas bloqueantes (ids=%s).",
                len(critical), sorted(critical))

    plan_by_id = {p.id: p for p in quiz_plan.questions}
    issues_by_qid: dict[int, set[str]] = {}
    for iss in review.issues:
        if iss.severity == "high":
            issues_by_qid.setdefault(iss.question_id, set()).add(iss.kind)

    kept: list["QuizQuestion"] = []
    for q in quiz_questions:
        if q.id not in critical:
            kept.append(q)
            continue
        planned = plan_by_id.get(q.id)
        if planned is None:
            continue

        if "duplicate" in issues_by_qid.get(q.id, set()):
            avoid = {p.concept_id for p in quiz_plan.questions}
            alt = _pick_alternative_concept(kb, avoid)
            if alt is not None:
                planned = planned.model_copy(update={"concept_id": alt, "focus": None})

        try:
            newq = generate_single_question(client, kb, planned, previous=list(kept))
            newq.id = q.id
            newq.question = _strip_meta_preamble(newq.question)
            kept.append(newq)
        except GenerationError as exc:
            logger.info("Regeneración falló para pregunta %d: %s", q.id, exc)

    final_review = review_quiz(kb, kept)
    still_critical = final_review.blocker_ids()
    if still_critical:
        kept = [q for q in kept if q.id not in still_critical]

    if max_questions is not None and len(kept) > max_questions:
        issue_count: dict[int, int] = {}
        for iss in final_review.issues:
            if iss.severity == "high":
                issue_count[iss.question_id] = issue_count.get(iss.question_id, 0) + 1
        kept.sort(key=lambda q: (issue_count.get(q.id, 0), q.id))
        kept = kept[:max_questions]

    if len(kept) < min_questions:
        logger.warning("Quiz tras refinado: %d preguntas (< mínimo %d).",
                       len(kept), min_questions)

    return kept, final_review


def refine_slides(
    client: OllamaClient,
    kb: KnowledgeBase,
    built_slides: list["BuiltSlide"],
    slide_plan: SlidePlan,
) -> tuple[list["BuiltSlide"], SlideReview]:
    """Regenera únicamente las slides con issues `high`.

    Con los detectores actuales (`meta_language`, `not_grounded`,
    `duplicate_content`) las severidades son `low`, por lo que en la
    mayoría de casos no se regenera nada — solo se reporta para log.
    """
    from .pptx_generator import BuiltSlide, render_slide_bullets

    review = review_slides(kb, built_slides)
    critical = review.blocker_indices()
    if not critical:
        return list(built_slides), review

    plan_by_title = {s.title: s for s in slide_plan.slides}
    total = len(built_slides)
    new_slides: list["BuiltSlide"] = []
    for i, bs in enumerate(built_slides, start=1):
        if i not in critical:
            new_slides.append(bs)
            continue
        planned = plan_by_title.get(bs.title)
        if planned is None:
            new_slides.append(bs)
            continue
        try:
            bullets = render_slide_bullets(client, kb, planned, slide_plan, index=i, total=total)
            new_slides.append(BuiltSlide(title=bs.title, bullets=bullets, kind=bs.kind))
        except GenerationError as exc:
            logger.warning("Regeneración slide '%s' falló: %s", bs.title, exc)
            new_slides.append(bs)

    return new_slides, review
