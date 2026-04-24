"""Métricas automáticas del benchmark.

Dos familias de métricas:

* **Quiz**: se calculan sobre el dict de salida de `src.quiz_generator.Quiz`.
* **PPTX**: se calculan sobre el `PresentationPlan` (serializado a dict) ya
  que agrupa título, outline, slides y conclusión.

Las métricas son **deliberadamente simples**: recuentos, proporciones y
solapamientos por Jaccard. No pretenden ser perfectas, sino objetivas,
explicables y defendibles en memoria/tribunal. El score final es una
media ponderada documentada en `rubric_reference.txt`.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from statistics import mean
from typing import Any, Iterable

from .config import (
    PPTX_BULLET_LONG_CHARS,
    PPTX_BULLET_SHORT_CHARS,
    PPTX_MAX_BULLETS_PER_SLIDE,
    PPTX_MIN_BULLETS_PER_SLIDE,
    PPTX_REPEAT_JACCARD,
    QUIZ_BANNED_PHRASES,
    QUIZ_DUPLICATE_JACCARD,
    QUIZ_MAX_QUESTIONS,
    QUIZ_MIN_QUESTIONS,
    QUIZ_OPTION_LEN_IMBALANCE,
)

# =========================================================================
# Utilidades de texto
# =========================================================================

_WORD_RE = re.compile(r"[a-z0-9áéíóúüñ]{3,}", flags=re.IGNORECASE)
_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "u", "en", "con", "por", "para", "que", "como", "al",
    "se", "es", "son", "su", "sus", "lo", "le", "les", "este", "esta",
    "estos", "estas", "entre", "sobre", "sin", "pero", "mas", "más",
    "cual", "cuál", "cuáles", "cuando", "cuándo", "donde", "dónde",
    "mismo", "misma", "tambien", "también", "tras", "hasta",
}


def _deaccent(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokens(text: str) -> set[str]:
    """Tokens en minúscula sin stopwords ni acentos, longitud >= 3."""
    if not text:
        return set()
    norm = _deaccent(text.lower())
    return {t for t in _WORD_RE.findall(norm) if t not in _STOPWORDS and len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _contains_any(text: str, needles: Iterable[str]) -> list[str]:
    """Lista de frases prohibidas encontradas (sin duplicar)."""
    low = _deaccent(text.lower())
    found: list[str] = []
    for n in needles:
        n_norm = _deaccent(n.lower())
        if n_norm in low and n not in found:
            found.append(n)
    return found


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# =========================================================================
# Métricas del Quiz
# =========================================================================

def _bloom_distribution(questions: list[dict]) -> dict[str, int]:
    return dict(Counter(q.get("bloom_level", "?") for q in questions))


def _duplicate_pairs(questions: list[dict], threshold: float) -> int:
    """Pares (i,j) con Jaccard de stems >= threshold. Cuenta cada par una vez."""
    stems = [_tokens(q.get("question", "")) for q in questions]
    n = len(stems)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(stems[i], stems[j]) >= threshold:
                count += 1
    return count


def _option_imbalance(q: dict) -> float:
    """Ratio max/min de longitudes de opciones. 1.0 = perfectamente balanceado."""
    opts = q.get("options", {})
    lengths = [len(str(opts.get(k, ""))) for k in ("A", "B", "C", "D")]
    lengths = [x for x in lengths if x > 0]
    if not lengths:
        return 1.0
    mn = min(lengths)
    if mn == 0:
        return float("inf")
    return max(lengths) / mn


def _kb_term_coverage(questions: list[dict], kb: Any | None) -> float:
    """Fracción de términos clave de la KB que aparecen en el quiz.

    Aproximación: para cada término de definición/nombre de ejemplo, se
    considera cubierto si su primer token no-stopword aparece en el stem
    o justificación de alguna pregunta.
    """
    if kb is None:
        return 0.0
    try:
        terms: list[str] = []
        for d in getattr(kb, "definitions", []) or []:
            t = getattr(d, "term", None)
            if isinstance(t, str) and t.strip():
                terms.append(t)
        for e in getattr(kb, "examples", []) or []:
            n = getattr(e, "name", None)
            if isinstance(n, str) and n.strip():
                terms.append(n)
    except Exception:
        return 0.0
    if not terms:
        return 0.0

    quiz_tokens: set[str] = set()
    for q in questions:
        quiz_tokens |= _tokens(q.get("question", ""))
        quiz_tokens |= _tokens(q.get("justification", ""))
        opts = q.get("options", {})
        for k in ("A", "B", "C", "D"):
            quiz_tokens |= _tokens(str(opts.get(k, "")))

    hits = 0
    for term in terms:
        term_tokens = _tokens(term)
        if term_tokens and term_tokens & quiz_tokens:
            hits += 1
    return hits / len(terms)


def compute_quiz_metrics(quiz: dict, kb: Any | None = None) -> dict:
    """Métricas automáticas sobre un quiz (dict de `Quiz.to_dict()`).

    Parámetros
    ----------
    quiz:
        Dict con clave ``quiz`` conteniendo la lista de preguntas.
    kb:
        `KnowledgeBase` opcional para calcular cobertura temática
        aproximada. Si es ``None``, ``kb_term_coverage`` vale 0.
    """
    questions: list[dict] = list(quiz.get("quiz", []) or [])
    n = len(questions)

    if n == 0:
        return {
            "num_questions": 0,
            "bloom_distribution": {},
            "bloom_diversity": 0,
            "duplicate_pairs": 0,
            "unbalanced_options_count": 0,
            "option_imbalance_ratio_avg": 0.0,
            "banned_phrases_count": 0,
            "banned_phrases_examples": [],
            "pct_with_explanation": 0.0,
            "kb_term_coverage": 0.0,
            "score_quiz": 0.0,
        }

    bloom_dist = _bloom_distribution(questions)
    duplicates = _duplicate_pairs(questions, QUIZ_DUPLICATE_JACCARD)

    imbal_ratios = [_option_imbalance(q) for q in questions]
    unbalanced = sum(1 for r in imbal_ratios if r >= QUIZ_OPTION_LEN_IMBALANCE)
    avg_imbal = round(mean(r for r in imbal_ratios if r != float("inf")), 3) \
        if imbal_ratios else 0.0

    banned_hits: list[str] = []
    for q in questions:
        opts = q.get("options", {})
        text = " ".join([
            str(q.get("question", "")),
            *(str(opts.get(k, "")) for k in ("A", "B", "C", "D")),
        ])
        banned_hits.extend(_contains_any(text, QUIZ_BANNED_PHRASES))

    with_expl = sum(
        1 for q in questions
        if isinstance(q.get("justification"), str) and len(q["justification"].strip()) >= 10
    )
    pct_expl = with_expl / n
    coverage = _kb_term_coverage(questions, kb)

    # --- Score compuesto 0..1 -------------------------------------------
    s_range = 1.0 if QUIZ_MIN_QUESTIONS <= n <= QUIZ_MAX_QUESTIONS else 0.5
    s_bloom = _clip01(len(bloom_dist) / 4.0)  # 4+ niveles distintos -> 1.0
    s_expl = pct_expl
    s_dups = _clip01(1.0 - duplicates / max(1, n))
    s_banned = _clip01(1.0 - len(banned_hits) / max(1, n))
    s_balance = _clip01(1.0 - unbalanced / max(1, n))
    s_cover = coverage  # ya está 0..1

    score = (
        0.10 * s_range + 0.10 * s_bloom + 0.15 * s_expl
        + 0.15 * s_dups + 0.15 * s_banned
        + 0.15 * s_balance + 0.20 * s_cover
    )

    return {
        "num_questions": n,
        "bloom_distribution": bloom_dist,
        "bloom_diversity": len(bloom_dist),
        "duplicate_pairs": duplicates,
        "unbalanced_options_count": unbalanced,
        "option_imbalance_ratio_avg": avg_imbal,
        "banned_phrases_count": len(banned_hits),
        "banned_phrases_examples": sorted(set(banned_hits))[:5],
        "pct_with_explanation": round(pct_expl, 3),
        "kb_term_coverage": round(coverage, 3),
        "score_quiz": round(score, 3),
    }


# =========================================================================
# Métricas del PPTX (sobre PresentationPlan serializado)
# =========================================================================

def presentation_plan_to_dict(plan: Any) -> dict:
    """Serializa un `PresentationPlan` (dataclass) a dict JSON-safe."""
    return {
        "title": getattr(plan, "title", ""),
        "outline": list(getattr(plan, "outline", []) or []),
        "slides": [
            {
                "title": getattr(s, "title", ""),
                "bullets": list(getattr(s, "bullets", []) or []),
                "kind": getattr(s, "kind", "outlook"),
            }
            for s in (getattr(plan, "slides", []) or [])
        ],
        "conclusion": list(getattr(plan, "conclusion", []) or []),
    }


_SENTENCE_END_RE = re.compile(r"[.;:?!]\)?\s*$")


def _looks_truncated(bullet: str) -> bool:
    """Un bullet parece cortado si no acaba en puntuación de cierre."""
    t = (bullet or "").strip()
    if not t:
        return True
    return not bool(_SENTENCE_END_RE.search(t))


def _cross_slide_repetitions(slides: list[dict], threshold: float) -> int:
    """Pares de bullets en slides distintas con Jaccard >= threshold."""
    bags: list[tuple[int, set[str]]] = []
    for idx, s in enumerate(slides):
        for b in s.get("bullets", []) or []:
            bt = _tokens(b)
            if len(bt) >= 3:
                bags.append((idx, bt))
    count = 0
    for i in range(len(bags)):
        idx_i, ti = bags[i]
        for j in range(i + 1, len(bags)):
            idx_j, tj = bags[j]
            if idx_i == idx_j:
                continue
            if _jaccard(ti, tj) >= threshold:
                count += 1
    return count


def _index_coherence(outline: list[str], content_titles: list[str]) -> float:
    """Proporción de títulos de outline que encuentran un título de slide similar."""
    if not outline:
        return 0.0
    cont_sets = [_tokens(t) for t in content_titles]
    matched = 0
    for title in outline:
        ts = _tokens(title)
        if not ts:
            continue
        if any(_jaccard(ts, cs) >= 0.5 for cs in cont_sets):
            matched += 1
    return matched / len(outline)


def compute_pptx_metrics(plan_dict: dict) -> dict:
    """Métricas automáticas sobre un `PresentationPlan` serializado."""
    slides: list[dict] = list(plan_dict.get("slides", []) or [])
    outline: list[str] = list(plan_dict.get("outline", []) or [])
    conclusion: list[str] = list(plan_dict.get("conclusion", []) or [])

    content_slides = [s for s in slides if s.get("bullets")]
    num_content = len(content_slides)
    num_total = 2 + num_content + (1 if conclusion else 0)  # portada + índice + N + conclusión

    bullets_per_slide = [len(s.get("bullets", []) or []) for s in content_slides]
    avg_bullets = round(mean(bullets_per_slide), 2) if bullets_per_slide else 0.0

    all_bullets: list[str] = [b for s in content_slides for b in (s.get("bullets") or [])]
    total_bullets = len(all_bullets)

    too_short = sum(1 for b in all_bullets if len(b.strip()) < PPTX_BULLET_SHORT_CHARS)
    too_long = sum(1 for b in all_bullets if len(b.strip()) > PPTX_BULLET_LONG_CHARS)
    truncated = sum(1 for b in all_bullets if _looks_truncated(b))

    slides_few = sum(1 for n in bullets_per_slide if n < PPTX_MIN_BULLETS_PER_SLIDE)
    slides_many = sum(1 for n in bullets_per_slide if n > PPTX_MAX_BULLETS_PER_SLIDE)

    repetitions = _cross_slide_repetitions(content_slides, PPTX_REPEAT_JACCARD)

    content_titles = [s.get("title", "") for s in content_slides]
    coherence = _index_coherence(outline, content_titles)

    # --- Score compuesto 0..1 ------------------------------------------
    s_num = 1.0 if 6 <= num_content <= 14 else (0.6 if num_content >= 3 else 0.2)
    s_avg = 1.0 if PPTX_MIN_BULLETS_PER_SLIDE <= avg_bullets <= PPTX_MAX_BULLETS_PER_SLIDE \
        else 0.5
    denom = max(1, total_bullets)
    s_trunc = _clip01(1.0 - truncated / denom)
    s_long = _clip01(1.0 - too_long / denom)
    s_short = _clip01(1.0 - too_short / denom)
    # Normalizamos repeticiones por un factor realista: 5 pares ya es
    # rojo claro. Más pares -> 0.
    s_repeat = _clip01(1.0 - repetitions / 5.0)
    s_coh = coherence

    score = (
        0.10 * s_num + 0.15 * s_avg + 0.20 * s_trunc
        + 0.10 * s_long + 0.10 * s_short
        + 0.20 * s_repeat + 0.15 * s_coh
    )

    return {
        "num_slides_total": num_total,
        "num_content_slides": num_content,
        "avg_bullets_per_slide": avg_bullets,
        "slides_with_few_bullets": slides_few,
        "slides_with_many_bullets": slides_many,
        "bullets_total": total_bullets,
        "bullets_too_long": too_long,
        "bullets_too_short": too_short,
        "bullets_possibly_truncated": truncated,
        "cross_slide_repetition_pairs": repetitions,
        "index_coherence": round(coherence, 3),
        "has_conclusion": bool(conclusion),
        "score_pptx": round(score, 3),
    }
