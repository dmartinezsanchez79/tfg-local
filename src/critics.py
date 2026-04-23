"""Revisor crítico (self-critic) para Quiz y Presentación.

Flujo general:
1.  **Detección determinista** (sin LLM): comprueba reglas duras baratas
    como frases prohibidas, anclaje al documento, lenguaje meta en
    bullets y desequilibrio de longitud entre opciones.
2.  **Detección LLM**: una única llamada por artefacto (quiz completo
    y plan de slides completo) para capturar problemas que requieren
    juicio (duplicados semánticos, Bloom erróneo, distractores triviales,
    off-topic, etc.).
3.  **Regeneración selectiva**: solo se regenera lo marcado como
    severidad `medium`/`high`; el resto se mantiene intacto.

Por qué dos pasadas (determinista + LLM):
- La determinista es gratuita, rápida y ataca errores que el LLM
  a menudo no detecta cuando él mismo los produjo.
- La LLM aporta la capa semántica que los regex no pueden dar.

El módulo no genera contenido directamente: delega en
`quiz_generator.generate_single_question` y
`pptx_generator.render_slide_bullets` mediante import diferido para
evitar ciclos.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any, Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:  # imports solo para type-checkers (rompen ciclo en runtime)
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
from .prompts import QUIZ_CRITIC_PROMPT, SLIDE_CRITIC_PROMPT, SYSTEM_EXPERT_ES

logger = logging.getLogger(__name__)


# ------------------------------------------------------------ tipos ---------

Severity = Literal["low", "medium", "high"]


class QuizIssue(BaseModel):
    question_id: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=3, max_length=400)
    severity: Severity = "medium"


class QuizReview(BaseModel):
    issues: list[QuizIssue] = Field(default_factory=list)

    def critical_ids(self) -> set[int]:
        return {i.question_id for i in self.issues if i.severity in ("medium", "high")}

    def blocker_ids(self) -> set[int]:
        """IDs bloqueantes: solo issues de severidad alta."""
        return {i.question_id for i in self.issues if i.severity == "high"}


class SlideIssue(BaseModel):
    slide_index: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=3, max_length=400)
    severity: Severity = "medium"


class SlideReview(BaseModel):
    issues: list[SlideIssue] = Field(default_factory=list)

    def critical_indices(self) -> set[int]:
        return {i.slide_index for i in self.issues if i.severity in ("medium", "high")}

    def blocker_indices(self) -> set[int]:
        """Índices bloqueantes: solo issues de severidad alta."""
        return {i.slide_index for i in self.issues if i.severity == "high"}


# -------------------------------------------- anclaje al documento (KB) ----

_WORD_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "de", "la", "el", "en", "los", "las", "un", "una", "unos", "unas",
    "por", "para", "con", "sin", "que", "cual", "como", "se", "su", "sus",
    "al", "del", "lo", "y", "o", "u", "e", "a", "es", "son", "ser", "esta",
    "este", "estos", "estas", "ese", "esa", "eso", "si", "no", "ni", "pero",
    "también", "tambien", "entre", "sobre", "hacia", "desde", "hasta",
}


def _deaccent_lower(text: str) -> str:
    n = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in n if not unicodedata.combining(ch)).lower()


def _tokens(text: str, *, min_len: int = 4) -> set[str]:
    """Extrae tokens relevantes (>=min_len, sin stopwords)."""
    out: set[str] = set()
    for m in _WORD_RE.finditer(_deaccent_lower(text)):
        tok = m.group(0)
        if len(tok) >= min_len and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _kb_vocabulary(kb: KnowledgeBase, *, min_len: int = 4) -> set[str]:
    """Conjunto de tokens relevantes presentes en la KB.

    Se usa para grounding: si un bullet o pregunta no contiene NINGÚN
    token del vocabulario de la KB, probablemente no está anclado al
    documento.
    """
    vocab: set[str] = set()
    vocab.update(_tokens(kb.main_topic, min_len=min_len))
    for st in kb.subtopics:
        vocab.update(_tokens(st, min_len=min_len))
    for atom in kb._iter_atoms():  # noqa: SLF001
        vocab.update(_tokens(_atom_text(atom), min_len=min_len))
    return vocab


def _atom_text(
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> str:
    if isinstance(atom, Definition):
        return f"{atom.term} {atom.definition}"
    if isinstance(atom, Example):
        return " ".join([atom.name, atom.description, *atom.attributes, *atom.methods])
    if isinstance(atom, FormulaOrCode):
        return " ".join(filter(None, [atom.caption or "", atom.content, atom.language or ""]))
    if isinstance(atom, NumericDatum):
        return f"{atom.value} {atom.description}"
    if isinstance(atom, Relation):
        return " ".join(filter(None, [atom.source, atom.target, atom.kind, atom.description or ""]))
    return ""


def _is_grounded(text: str, vocab: set[str], *, min_hits: int = 1) -> bool:
    toks = _tokens(text)
    return len(toks & vocab) >= min_hits


# ----------------------------------- detectores deterministas: QUIZ -------

_QUIZ_BANNED_OPTION_PHRASES = (
    "todas las anteriores",
    "todas las opciones",
    "ninguna de las anteriores",
    "ninguna de las opciones",
    "a y b",
    "b y c",
    "a y c",
    "ninguna es correcta",
    "solo a",
    "solo b",
)

_QUIZ_META_PHRASES = (
    "segun el texto",
    "segun el documento",
    "segun el autor",
    "segun la definicion",
    "segun la definicion del documento",
    "segun la definicion proporcionada",
    "segun la definicion dada",
    "segun la definicion oficial",
    "de acuerdo con el documento",
    "de acuerdo con el texto",
    "de acuerdo con la definicion",
    "segun lo proporcionado",
    "segun lo indicado",
    "segun su definicion literal",
    "segun la definicion literal",
    "segun el ejemplo dado",
    "segun el ejemplo",
    "segun su ejemplo",
    "segun este ejemplo",
    "segun lo visto",
    "conforme al documento",
    "conforme al texto",
    "conforme a la definicion",
    "el autor dice",
    "el texto menciona",
    "el documento indica",
    "el documento describe",
    "como se menciona en",
    "como se define en",
    "tal y como se indica",
    "tal como se define",
    "basandose en el documento",
    "basandose en la definicion",
    "en este apartado",
    "en esta pregunta",
    # Copiadas literalmente del prompt (indican fuga del "meta-lenguaje"
    # de instrucciones hacia el enunciado visible al estudiante).
    "segun los malentendidos",
    "segun las malentendidos",
    "malentendidos tipicos",
    "malentendidos del alumno",
    "malentendidos del estudiante",
    "segun el concepto central",
    "segun la plantilla",
    "segun la taxonomia",
    "segun bloom",
    "segun el nivel bloom",
    "del alumno tipico",
    "del estudiante tipico",
)


# Muletillas de PREÁMBULO que aparecen al principio del enunciado seguidas
# de coma o dos puntos. Estas se pueden *recortar* sin alterar el sentido:
#     "Según la definición del documento, ¿qué es X?"  →  "¿Qué es X?"
#     "Conforme al texto: ¿cuándo se aplica Y?"       →  "¿Cuándo se aplica Y?"
# Se usan como corrección post-generación barata, evitando la regeneración
# completa por LLM cuando basta un recorte.
_QUIZ_META_PREAMBLE_RE = re.compile(
    r"^\s*(?:segun|conforme\s+a[l]?|de\s+acuerdo\s+con|basandose\s+en|como\s+se\s+(?:indica|menciona|define)\s+en)\b"
    r"[^,.:;¿?]{1,80}[,:;]\s*",
    re.IGNORECASE,
)


def _strip_meta_preamble(text: str) -> str:
    """Recorta un preámbulo meta-lingüístico del inicio de un enunciado.

    Solo actúa si tras recortar queda al menos 10 caracteres útiles; en
    caso contrario devuelve el texto intacto (más vale conservar una
    muletilla que dejar un enunciado truncado).
    """
    if not isinstance(text, str):
        return text
    # Necesitamos deacentuar para aplicar la regex sin acentos, pero
    # conservar el texto original para el recorte real.
    deacc = _deaccent_lower(text)
    m = _QUIZ_META_PREAMBLE_RE.match(deacc)
    if not m:
        return text
    cut = m.end()
    # Traduce el offset deacentuado al texto original: como `_deaccent_lower`
    # preserva la longitud de cada carácter (salvo casos patológicos), el
    # offset es aplicable tal cual.
    remainder = text[cut:].lstrip()
    if len(remainder) < 10:
        return text
    # Capitaliza la primera letra si procede (preserva "¿", "¡").
    if remainder and remainder[0] in "¿¡":
        # Signo de puntuación + minúscula después.
        if len(remainder) > 1 and remainder[1].islower():
            remainder = remainder[0] + remainder[1].upper() + remainder[2:]
    elif remainder and remainder[0].islower():
        remainder = remainder[0].upper() + remainder[1:]
    return remainder


def _jaccard(a: set[str], b: set[str]) -> float:
    """Similitud de Jaccard entre dos conjuntos de tokens."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Umbral determinado empíricamente: dos preguntas con >= 0.80 de tokens en
# común (enunciado + 4 opciones, sin stopwords) son variantes de la misma
# pregunta — aunque tengan Bloom level o concept_id distintos.
_DUPLICATE_JACCARD_THRESHOLD = 0.80

# Umbral para similitud SOLO de enunciado (sin opciones). Más estricto
# (menor) que el de la firma completa: dos preguntas con el MISMO
# enunciado y distractores distintos siguen siendo la misma pregunta
# desde la perspectiva del estudiante (la mente recuerda el enunciado,
# no las opciones).
_STEM_DUPLICATE_JACCARD_THRESHOLD = 0.70


def _kb_definitions_tokens(kb: KnowledgeBase) -> dict[str, set[str]]:
    """Indexa cada `Definition` del KB por sus tokens relevantes.

    Usado para detectar cuándo las opciones de una pregunta son "pegadas"
    de definiciones de otros conceptos (el anti-patrón M3: tres distractores
    que son exactamente las definiciones de tres conceptos distintos).
    """
    out: dict[str, set[str]] = {}
    for atom in kb._iter_atoms():  # noqa: SLF001
        if isinstance(atom, Definition):
            toks = _tokens(atom.definition, min_len=4)
            if toks:
                out[atom.id] = toks
    return out


# Si una opción comparte al menos este Jaccard con una definición del KB,
# se considera "alineada" con esa definición (no importa el orden de
# palabras ni los conectores: basta con el solapamiento conceptual).
_OFF_CONCEPT_JACCARD_THRESHOLD = 0.45

# Número mínimo de definiciones KB distintas cubiertas por las opciones
# para considerar que el quiz usa "banco común de distractores".
# 3+ definiciones significa que las 4 opciones son 4 conceptos diferentes,
# o 3 + una reformulación: en ambos casos el estudiante resuelve por
# eliminación en lugar de por comprensión del concepto objetivo.
_OFF_CONCEPT_MIN_DEFS = 3


def _detect_off_concept_distractors(
    q: "QuizQuestion", kb_defs: dict[str, set[str]]
) -> tuple[bool, dict[str, list[str]]]:
    """Devuelve (es_banco_mezcla, mapeo letra→definiciones alineadas)."""
    if not kb_defs:
        return (False, {})
    matches: dict[str, list[str]] = {}
    for L in ("A", "B", "C", "D"):
        opt_tokens = _tokens(getattr(q.options, L), min_len=4)
        if not opt_tokens:
            continue
        best_def: str | None = None
        best_sim = 0.0
        for def_id, def_tokens in kb_defs.items():
            sim = _jaccard(opt_tokens, def_tokens)
            if sim > best_sim:
                best_sim = sim
                best_def = def_id
        if best_def is not None and best_sim >= _OFF_CONCEPT_JACCARD_THRESHOLD:
            matches.setdefault(best_def, []).append(L)
    # Distintas definiciones KB cubiertas por las opciones.
    return (len(matches) >= _OFF_CONCEPT_MIN_DEFS, matches)


def _question_signature_tokens(q: "QuizQuestion") -> set[str]:
    """Bolsa de tokens que representa la pregunta para comparar duplicados.

    Incluye enunciado y las 4 opciones (no solo la correcta) porque dos
    preguntas pueden tener enunciados parecidos pero distractores muy
    distintos: en ese caso no son duplicadas semánticas.
    """
    opts = [getattr(q.options, L) for L in ("A", "B", "C", "D")]
    return _tokens(" ".join([q.question, *opts]), min_len=4)


# Máximo de preguntas que pueden girar en torno al MISMO ejemplo (por nombre).
# Si en un quiz de 10 preguntas 6 mencionan "Bicicleta", el quiz parece
# monotemático aunque los concept_ids sean distintos. 2 es el umbral útil.
_MAX_QUESTIONS_MENTIONING_SAME_EXAMPLE = 2


def _example_name_tokens(kb: KnowledgeBase) -> dict[str, set[str]]:
    """Por cada Example de la KB devuelve el conjunto de tokens de su nombre.

    Ej: `Example(name="Bicicleta de montaña")` →
        {"bicicleta de montaña": {"bicicleta", "montana"}}.
    Permite detectar cuándo múltiples preguntas giran alrededor del mismo
    ejemplo aunque el enunciado lo reformule.
    """
    out: dict[str, set[str]] = {}
    for atom in kb._iter_atoms():  # noqa: SLF001
        if isinstance(atom, Example):
            toks = _tokens(atom.name, min_len=4)
            if toks:
                out[atom.name] = toks
    return out


def _mentions_example(qtokens: set[str], ex_tokens: set[str]) -> bool:
    """True si la pregunta contiene todos los tokens del nombre del ejemplo.

    Usamos "todos" (no Jaccard) porque los nombres de ejemplos suelen ser
    cortos (1-3 tokens relevantes) y queremos evitar falsos positivos por
    palabras comunes.
    """
    return bool(ex_tokens) and ex_tokens.issubset(qtokens)


# ---------- Detector anti-world-knowledge (grounding contra el PDF) ------
#
# Incluso cuando una pregunta supera el anclaje contra la KB, el LLM puede
# colar conocimiento del mundo (sacado de su pretraining) que nunca aparece
# en el PDF. Ej: con un PDF de POO que usa la bicicleta como ejemplo, un
# modelo grande puede inventar distractores sobre "carreteras", "terrenos
# difíciles", "tipo_combustible" — conceptos plausibles pero ajenos al
# documento que el usuario quiere estudiar.
#
# Detectamos esto contando tokens "notables" (longitud ≥ 6, sin stopwords)
# que NO aparecen ni en el markdown original ni en la KB. Si una pregunta
# supera el umbral de tokens ajenos, se marca como `world_knowledge_leak`.

# Umbral doble: una pregunta se marca como `world_knowledge_leak` cuando
# AMBOS son ciertos:
#   1. Al menos `_WORLD_KNOWLEDGE_LEAK_ABS_MIN` tokens notables son ajenos
#      (no aparecen ni en el PDF ni en la KB).
#   2. La proporción de ajenos / notables totales es ≥ `_WKL_RATIO_MIN`.
#
# La combinación evita falsos positivos: una pregunta bien anclada puede
# usar sinónimos o conectores ausentes del PDF (ratio bajo pero ajenos
# altos en absoluto) o ser corta y usar 1-2 conectores fuera de la KB
# (ratio alto pero ajenos pocos). Solo cuando ambas condiciones se
# cruzan la evidencia de conocimiento externo es fuerte.
#
# Calibrado contra P7/P9 del quiz `qwen2.5:14b` (ratio ≥ 0.5, ajenos ≥ 7)
# y contra P1/P2/P3/P4/P6/P8/P10 bien ancladas (ratio ≤ 0.15 con KB real
# completa). El umbral ABS evita falsos positivos en preguntas breves con
# 2-3 conectores fuera de la KB; el RATIO los evita en preguntas extensas
# con muchos sinónimos en prosa.
_WORLD_KNOWLEDGE_LEAK_ABS_MIN = 7
_WKL_RATIO_MIN = 0.50

# Tokens mínimos considerados "notables": longitud 6+ filtra conjunciones,
# artículos largos (`aquellos`) y verbos frecuentes cortos (`tiene`).
_NOTABLE_TOKEN_MIN_LEN = 6


def _markdown_token_index(markdown: str) -> set[str]:
    """Tokens del markdown con recorte de plural (normalizados).

    Reusa la heurística de `_tokens` pero SIN filtro de longitud (los
    tokens cortos también forman vocabulario grounded). El recorte de
    plural (terminación `s`/`es`) evita falsos positivos por inflexión:
    si el PDF dice `bicicleta` y la pregunta escribe `bicicletas`, se
    considera presente.
    """
    toks: set[str] = set()
    if not markdown:
        return toks
    for m in _WORD_RE.finditer(_deaccent_lower(markdown)):
        tok = m.group(0)
        if len(tok) < 3:
            continue
        toks.add(tok)
        # Singulariza plurales habituales (misma heurística que `plans._normalize_slug`).
        if len(tok) > 4 and tok.endswith("es"):
            toks.add(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s"):
            toks.add(tok[:-1])
    return toks


def _token_in_index(tok: str, index: set[str]) -> bool:
    """True si `tok` (o su raíz singular) aparece en el índice."""
    if tok in index:
        return True
    if len(tok) > 4 and tok.endswith("es") and tok[:-2] in index:
        return True
    if len(tok) > 3 and tok.endswith("s") and tok[:-1] in index:
        return True
    return False


def _detect_world_knowledge_leak(
    q: "QuizQuestion", md_index: set[str], kb_vocab: set[str]
) -> tuple[list[str], int]:
    """Devuelve (tokens_ajenos, total_notables).

    Un token es "ajeno" cuando:
    1. Tiene longitud ≥ `_NOTABLE_TOKEN_MIN_LEN` y no es stopword.
    2. No aparece (ni como raíz singular) en el markdown del PDF.
    3. No aparece en el vocabulario de la KB.

    El llamador aplica el umbral doble (absoluto + ratio) — ver
    `_WORLD_KNOWLEDGE_LEAK_ABS_MIN` y `_WKL_RATIO_MIN`.
    """
    if not md_index:
        return [], 0
    texts = [q.question] + [getattr(q.options, L) for L in ("A", "B", "C", "D")]
    qtokens = _tokens(" ".join(texts), min_len=_NOTABLE_TOKEN_MIN_LEN)
    alien: list[str] = []
    for tok in qtokens:
        if _token_in_index(tok, md_index):
            continue
        if _token_in_index(tok, kb_vocab):
            continue
        alien.append(tok)
    return alien, len(qtokens)


def _deterministic_quiz_issues(
    kb: KnowledgeBase,
    quiz_questions: list["QuizQuestion"],
    vocab: set[str],
    *,
    md_index: set[str] | None = None,
) -> list[QuizIssue]:
    issues: list[QuizIssue] = []
    signatures: list[tuple[int, set[str]]] = []
    stems: list[tuple[int, set[str]]] = []
    example_names = _example_name_tokens(kb)
    # Cuenta cuántas preguntas han mencionado ya cada ejemplo; usamos el
    # name (str) como clave. Cada pregunta incrementa como máximo una vez
    # por ejemplo (aunque lo mencione varias).
    example_mentions: dict[str, list[int]] = {name: [] for name in example_names}
    kb_defs = _kb_definitions_tokens(kb)
    # Índice mínimo para `world_knowledge_leak`. Si no se pasa markdown,
    # el detector se desactiva (retrocompatibilidad).
    md_idx: set[str] = md_index or set()

    for q in quiz_questions:
        qtext_norm = _deaccent_lower(q.question)
        opt_texts = [getattr(q.options, L) for L in ("A", "B", "C", "D")]
        opt_norms = [_deaccent_lower(t) for t in opt_texts]

        # 1) Frases prohibidas en las opciones
        for L, n in zip("ABCD", opt_norms):
            if any(b in n for b in _QUIZ_BANNED_OPTION_PHRASES):
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="banned_phrase",
                    description=f"Opción {L} usa frase prohibida ('{opt_texts['ABCD'.index(L)][:60]}').",
                    severity="high",
                ))
                break

        # 2) Opciones desequilibradas: una mucho más larga que el resto
        lens = [len(t) for t in opt_texts]
        if lens:
            maxlen = max(lens)
            others = sorted(lens)[:-1]
            avg_others = (sum(others) / max(1, len(others))) if others else 0
            if maxlen >= 40 and avg_others > 0 and maxlen >= 2.2 * avg_others:
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="unbalanced_options",
                    description=(
                        f"Una opción es ~{maxlen/max(1,avg_others):.1f}× más larga "
                        f"que el promedio del resto (pista involuntaria)."
                    ),
                    severity="low",
                ))

        # 3) Lenguaje meta en el enunciado o la justificación
        full_norm = qtext_norm + " " + _deaccent_lower(q.justification)
        if any(m in full_norm for m in _QUIZ_META_PHRASES):
            issues.append(QuizIssue(
                question_id=q.id,
                kind="meta_language",
                description="Pregunta o justificación usa lenguaje meta ('según el texto…').",
                severity="low",
            ))

        # 4) Anclaje a la KB (enunciado + opción correcta + justificación)
        correct_opt = getattr(q.options, q.correct_answer)
        grounding_text = f"{q.question} {correct_opt} {q.justification}"
        if vocab and not _is_grounded(grounding_text, vocab):
            issues.append(QuizIssue(
                question_id=q.id,
                kind="not_grounded",
                description="La pregunta no menciona términos clave de la Base de Conocimiento.",
                severity="low",
            ))

        # 5) Duplicados semánticos: dos chequeos complementarios.
        #
        #    a) Firma completa (enunciado + 4 opciones). Umbral alto (0.80)
        #       porque dos preguntas con enunciado parecido pero
        #       distractores muy distintos pueden ser legítimamente
        #       distintas.
        #    b) Solo enunciado (stem). Umbral medio (0.70) porque una
        #       pregunta con el MISMO enunciado y otros distractores es,
        #       desde el punto de vista didáctico, la misma pregunta:
        #       el estudiante memoriza el enunciado, no las opciones.
        sig = _question_signature_tokens(q)
        stem = _tokens(q.question, min_len=4)
        duplicate_found = False
        for prev_id, prev_sig in signatures:
            sim = _jaccard(sig, prev_sig)
            if sim >= _DUPLICATE_JACCARD_THRESHOLD:
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="duplicate",
                    description=(
                        f"Muy similar a la pregunta {prev_id} "
                        f"(Jaccard={sim:.2f}); es la misma pregunta con "
                        f"reformulación superficial."
                    ),
                    severity="high",
                ))
                duplicate_found = True
                break
        if not duplicate_found:
            for prev_id, prev_stem in stems:
                sim = _jaccard(stem, prev_stem)
                if sim >= _STEM_DUPLICATE_JACCARD_THRESHOLD:
                    issues.append(QuizIssue(
                        question_id=q.id,
                        kind="duplicate_stem",
                        description=(
                            f"Enunciado prácticamente idéntico al de la "
                            f"pregunta {prev_id} (Jaccard={sim:.2f} sobre "
                            f"el enunciado); aunque los distractores "
                            f"difieran, el estudiante reconoce la "
                            f"misma pregunta."
                        ),
                        severity="high",
                    ))
                    break
        signatures.append((q.id, sig))
        stems.append((q.id, stem))

        # 6) Distractores "off-concept": las opciones son definiciones de
        #    distintos conceptos del KB en lugar de variantes del concepto
        #    objetivo. Indica banco común de distractores.
        is_bank, matches = _detect_off_concept_distractors(q, kb_defs)
        if is_bank:
            pretty = ", ".join(
                f"{def_id}→{'+'.join(letters)}"
                for def_id, letters in sorted(matches.items())
            )
            issues.append(QuizIssue(
                question_id=q.id,
                kind="distractor_off_concept",
                description=(
                    f"Las opciones cubren {len(matches)} definiciones del KB "
                    f"distintas ({pretty}): son un 'banco común' de "
                    f"distractores; el estudiante resuelve por eliminación."
                ),
                severity="medium",
            ))

        # 7) Conocimiento del mundo colado (grounding contra el PDF).
        #    Solo si disponemos de `md_index`: evita flaggear cuando se
        #    usa la API legacy sin markdown.
        if md_idx:
            alien, total_notable = _detect_world_knowledge_leak(q, md_idx, vocab)
            ratio = len(alien) / total_notable if total_notable else 0.0
            if (
                len(alien) >= _WORLD_KNOWLEDGE_LEAK_ABS_MIN
                and ratio >= _WKL_RATIO_MIN
            ):
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="world_knowledge_leak",
                    description=(
                        f"La pregunta introduce {len(alien)} términos ajenos "
                        f"al PDF y a la KB ({ratio:.0%} del total notable), "
                        f"p. ej. {sorted(alien)[:5]}: probable alucinación "
                        "con conocimiento externo del modelo."
                    ),
                    severity="low",
                ))

        # 8) Sobreuso del mismo ejemplo del KB (sesgo monotemático).
        #    Miramos enunciado + opción correcta (no distractores: si el
        #    ejemplo aparece solo como distractor no es un sesgo).
        qtokens_full = _tokens(
            f"{q.question} {getattr(q.options, q.correct_answer)}", min_len=4
        )
        for name, ex_tokens in example_names.items():
            if not _mentions_example(qtokens_full, ex_tokens):
                continue
            prev = example_mentions[name]
            if len(prev) >= _MAX_QUESTIONS_MENTIONING_SAME_EXAMPLE:
                issues.append(QuizIssue(
                    question_id=q.id,
                    kind="example_overuse",
                    description=(
                        f"El ejemplo «{name}» ya aparece en las preguntas "
                        f"{prev}; la pregunta {q.id} lo reutiliza. Usa otro "
                        f"átomo del KB para diversificar."
                    ),
                    # severity=low: se reporta para diagnóstico pero NO
                    # dispara regeneración. Si el KB tiene 1-2 ejemplos,
                    # el LLM va a tirar de ellos casi obligatoriamente;
                    # forzar rotación colapsa el quiz a un único concepto.
                    severity="low",
                ))
            prev.append(q.id)

    return issues


# ----------------------------------- detectores deterministas: SLIDES -----

_SLIDE_META_PHRASES = (
    "en este apartado",
    "en esta diapositiva",
    "se habla de",
    "se trata de",
    "se estudia",
    "se explica",
    "importancia del",
    "es importante destacar",
    "a continuacion",
    "a continuacion se",
    "en este texto",
    "el autor explica",
    "el documento",
    "sirve de ejemplo para ilustrar",
    "sirve para ilustrar",
    "facilita la creacion y gestion",
    "facilita la creacion y la gestion",
    "facilita la comprension de",
    "ayuda a comprender",
    "permite entender",
    "permite comprender",
    "es clave para entender",
    "es clave para comprender",
    "ilustra como",
)

# Notación interna de tripletas del KB que NUNCA debe llegar a un bullet.
# Ej: "BicicletaDeMontaña —[subclase_de]→ Bicicleta". Si aparece, el bullet
# es basura (copia del formato técnico).
_RELATION_ARROW_RE = re.compile(
    r"(?:—|--|-|–)\s*\[[^\]]+\]\s*(?:→|->|—>|–>)",
)

# Anglicismos y neologismos que degradan la calidad didáctica del bullet.
_SLIDE_ANGLICISM_RE = re.compile(
    r"\b(blueprint|inheritance|overriding|concretada|concretado)\b",
    re.IGNORECASE,
)


def _bullet_echoes_slide_title(bullet: str, slide_title: str) -> bool:
    """True si el bullet repite el título de la slide como preámbulo.

    Ej: title="Polimorfismo", bullet="Polimorfismo: capacidad de…" → True.
    """
    if not bullet or not slide_title:
        return False
    title_norm = " ".join(slide_title.split()).strip().lower()
    bullet_norm = " ".join(bullet.split()).strip().lower()
    if not title_norm or not bullet_norm:
        return False
    if len(title_norm) < 6 and len(title_norm.split()) < 2:
        return False
    if not bullet_norm.startswith(title_norm):
        return False
    tail = bullet_norm[len(title_norm) :].lstrip()
    if not tail:
        return True
    return tail[:1] in {":", ".", "—", "–", "-", ",", "·"}


def _detect_slide_world_knowledge_leak(
    bullets: list[str], md_index: set[str], kb_vocab: set[str]
) -> tuple[list[str], int]:
    """Variante de `_detect_world_knowledge_leak` para slides.

    Agrega todos los bullets en una única bolsa y cuenta tokens notables
    ajenos al PDF + KB. Umbral doble idéntico al del quiz: evita
    falsos positivos en bullets con prosa natural rica (sinónimos,
    conectores largos) y captura slides inventadas como "Clase Vehículos
    … tipo_motor, calcularVelocidadMáxima".
    """
    if not md_index or not bullets:
        return [], 0
    joined = " ".join(bullets)
    qtokens = _tokens(joined, min_len=_NOTABLE_TOKEN_MIN_LEN)
    alien: list[str] = []
    for tok in qtokens:
        if _token_in_index(tok, md_index):
            continue
        if _token_in_index(tok, kb_vocab):
            continue
        alien.append(tok)
    return alien, len(qtokens)


def _deterministic_slide_issues(
    kb: KnowledgeBase,
    built_slides: list["BuiltSlide"],
    vocab: set[str],
    *,
    md_index: set[str] | None = None,
) -> list[SlideIssue]:
    issues: list[SlideIssue] = []
    seen_bullets: dict[str, int] = {}
    md_idx: set[str] = md_index or set()

    for idx, s in enumerate(built_slides, start=1):
        bullets_norm = [_deaccent_lower(b) for b in s.bullets]

        # 1) Lenguaje meta
        hits = [
            s.bullets[i][:60]
            for i, b in enumerate(bullets_norm)
            if any(m in b for m in _SLIDE_META_PHRASES)
        ]
        if hits:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="meta_language",
                description=f"Bullets con lenguaje meta: {hits[:3]}",
                severity="low",
            ))

        # 2) Slide demasiado superficial
        non_empty = [b for b in s.bullets if b and b.strip()]
        if len(non_empty) < 2 and s.kind not in {"intro", "outlook", "conclusion"}:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="too_shallow",
                description=f"La slide tiene {len(non_empty)} bullet(s); aporta poca densidad.",
                severity="low",
            ))

        # 3) Anclaje a la KB (ignoramos slides tipo intro/conclusion porque
        #    suelen ser panorámicas y no siempre contienen términos KB).
        if vocab and s.kind not in {"intro", "conclusion"}:
            joined = " ".join(s.bullets)
            if joined.strip() and not _is_grounded(joined, vocab):
                issues.append(SlideIssue(
                    slide_index=idx,
                    kind="not_grounded",
                    description="Los bullets no referencian términos clave de la KB.",
                    severity="low",
                ))

        # 4) Duplicados literales entre slides
        for b in s.bullets:
            key = " ".join(sorted(_tokens(b)))[:120]
            if not key:
                continue
            if key in seen_bullets and seen_bullets[key] != idx:
                issues.append(SlideIssue(
                    slide_index=idx,
                    kind="duplicate_content",
                    description=(
                        f"Un bullet coincide semánticamente con slide "
                        f"{seen_bullets[key]}: “{b[:60]}”."
                    ),
                    severity="low",
                ))
                break
            seen_bullets.setdefault(key, idx)

        # 5) Notación interna del KB (flecha con corchetes) filtrada al bullet.
        #    Es un bug del LLM: copia la tripleta "X —[kind]→ Y" tal cual.
        arrow_hits = [b[:80] for b in s.bullets if _RELATION_ARROW_RE.search(b)]
        if arrow_hits:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="relation_arrow_leak",
                description=(
                    "Bullets con notación técnica de relación "
                    f"(tripleta con flecha y corchetes): {arrow_hits[:2]}"
                ),
                severity="high",
            ))

        # 6) Eco del título en el primer bullet (repetir el título en texto
        #    es pobre diseño didáctico).
        echo_hits = [
            s.bullets[i][:80]
            for i, b in enumerate(s.bullets)
            if _bullet_echoes_slide_title(b, s.title)
        ]
        if echo_hits:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="title_echo",
                description=(
                    "Bullet(s) repiten el título de la slide como preámbulo: "
                    f"{echo_hits[:2]}"
                ),
                severity="low",
            ))

        # 7) Anglicismos y neologismos raros.
        anglicism_hits = [
            b[:80] for b in s.bullets if _SLIDE_ANGLICISM_RE.search(b)
        ]
        if anglicism_hits:
            issues.append(SlideIssue(
                slide_index=idx,
                kind="anglicism",
                description=(
                    "Anglicismos o neologismos poco didácticos: "
                    f"{anglicism_hits[:2]}"
                ),
                severity="low",
            ))

        # 8) Conocimiento del mundo colado (grounding contra el PDF).
        #    Solo si disponemos de `md_index`: evita flaggear cuando se
        #    usa la API legacy sin markdown. Red de seguridad para cuando
        #    el LLM inventa atributos/métodos/ejemplos que sobreviven al
        #    pruning de Examples del KB.
        if md_idx and s.kind not in {"intro", "conclusion"}:
            alien, total_notable = _detect_slide_world_knowledge_leak(
                list(s.bullets), md_idx, vocab
            )
            ratio = len(alien) / total_notable if total_notable else 0.0
            if (
                len(alien) >= _WORLD_KNOWLEDGE_LEAK_ABS_MIN
                and ratio >= _WKL_RATIO_MIN
            ):
                issues.append(SlideIssue(
                    slide_index=idx,
                    kind="world_knowledge_leak",
                    description=(
                        f"La slide introduce {len(alien)} términos ajenos al "
                        f"PDF y a la KB ({ratio:.0%} del total notable), "
                        f"p. ej. {sorted(alien)[:5]}: probable alucinación "
                        "con conocimiento externo del modelo."
                    ),
                    severity="low",
                ))

    return issues


# ------------------------------------------------- revisor vía LLM --------


def _parse_issues(raw: Any, model: type[BaseModel], log_label: str) -> list[Any]:
    """Convierte la respuesta del LLM en una lista de issues del tipo dado."""
    if not isinstance(raw, dict):
        logger.warning("%s: respuesta no es dict (%s)", log_label, type(raw).__name__)
        return []
    items = raw.get("issues")
    if not isinstance(items, list):
        return []
    out: list[Any] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            out.append(model(**it))
        except ValidationError as exc:
            logger.debug("%s: issue descartado por validación: %s", log_label, exc.errors()[:1])
    return out


def _llm_quiz_issues(
    client: OllamaClient, kb: KnowledgeBase, quiz_questions: list["QuizQuestion"]
) -> list[QuizIssue]:
    quiz_json = json.dumps(
        {"quiz": [q.model_dump() for q in quiz_questions]},
        ensure_ascii=False,
    )
    prompt = QUIZ_CRITIC_PROMPT.format(
        kb_context=kb.to_prompt_context(max_chars=4000),
        quiz_json=quiz_json,
    )
    try:
        raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Revisor LLM del quiz falló: %s", exc)
        return []
    issues = _parse_issues(raw, QuizIssue, "QuizCritic")
    # Filtramos ids inválidos
    valid_ids = {q.id for q in quiz_questions}
    return [i for i in issues if i.question_id in valid_ids]


def _llm_slide_issues(
    client: OllamaClient, kb: KnowledgeBase, built_slides: list["BuiltSlide"]
) -> list[SlideIssue]:
    plan_payload = {
        "slides": [
            {
                "index": i,
                "title": s.title,
                "kind": s.kind,
                "bullets": list(s.bullets),
            }
            for i, s in enumerate(built_slides, start=1)
        ]
    }
    prompt = SLIDE_CRITIC_PROMPT.format(
        kb_context=kb.to_prompt_context(max_chars=4000),
        plan_json=json.dumps(plan_payload, ensure_ascii=False),
    )
    try:
        raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Revisor LLM de slides falló: %s", exc)
        return []
    issues = _parse_issues(raw, SlideIssue, "SlideCritic")
    n = len(built_slides)
    return [i for i in issues if 1 <= i.slide_index <= n]


# ------------------------------------------------ API pública: revisar ----


def review_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    quiz_questions: list["QuizQuestion"],
    *,
    use_llm: bool = True,
    source_markdown: str | None = None,
) -> QuizReview:
    """Revisa el quiz con detectores determinísticos (+ LLM opcional).

    Si se proporciona `source_markdown`, se activa el detector
    `world_knowledge_leak` (grounding literal contra el PDF); si no,
    se omite para mantener compatibilidad con llamadores antiguos.
    """
    vocab = _kb_vocabulary(kb)
    md_index = _markdown_token_index(source_markdown) if source_markdown else set()
    issues = list(
        _deterministic_quiz_issues(kb, quiz_questions, vocab, md_index=md_index)
    )
    if use_llm:
        issues.extend(_llm_quiz_issues(client, kb, quiz_questions))
    issues = _dedup_quiz_issues(issues)
    return QuizReview(issues=issues)


def review_slides(
    client: OllamaClient,
    kb: KnowledgeBase,
    built_slides: list["BuiltSlide"],
    *,
    use_llm: bool = True,
    source_markdown: str | None = None,
) -> SlideReview:
    """Revisa las slides con detectores determinísticos (+ LLM opcional).

    Si se proporciona `source_markdown`, se activa el detector
    `world_knowledge_leak` (grounding literal contra el PDF); si no,
    se omite para mantener compatibilidad con llamadores antiguos.
    """
    vocab = _kb_vocabulary(kb)
    md_index = _markdown_token_index(source_markdown) if source_markdown else set()
    issues = list(
        _deterministic_slide_issues(kb, built_slides, vocab, md_index=md_index)
    )
    if use_llm:
        issues.extend(_llm_slide_issues(client, kb, built_slides))
    issues = _dedup_slide_issues(issues)
    return SlideReview(issues=issues)


def _dedup_quiz_issues(issues: Iterable[QuizIssue]) -> list[QuizIssue]:
    seen: set[tuple[int, str]] = set()
    out: list[QuizIssue] = []
    for i in issues:
        key = (i.question_id, i.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _dedup_slide_issues(issues: Iterable[SlideIssue]) -> list[SlideIssue]:
    seen: set[tuple[int, str]] = set()
    out: list[SlideIssue] = []
    for i in issues:
        key = (i.slide_index, i.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


# ------------------------------------------------ API pública: refinar ----


# Issues que implican que el CONCEPT_ID actual agota su utilidad y hay que
# rotar a otro átomo al regenerar; si solo corregimos la redacción con el
# mismo concept_id, la regeneración vuelve a chocar contra el mismo
# diagnóstico (duplicado otra vez, mismo ejemplo otra vez).
#
# Desde v2.11 se restringe a duplicados verdaderos. `banned_phrase` y
# `meta_language` se corrigen por reescritura con regex (no requieren
# otro concepto). `example_overuse` pasa a severity=low (no dispara
# regeneración): si el KB tiene pocos ejemplos, forzar rotación termina
# colapsando todo a un único concept_id (bug detectado en v2.10).
_ISSUES_REQUIRING_CONCEPT_SWAP = {"duplicate", "duplicate_stem"}


def _pick_alternative_concept(
    kb: KnowledgeBase,
    avoid: set[str],
    preferred_prefix: str | None = None,
    usage: dict[str, int] | None = None,
    current: str | None = None,
) -> str | None:
    """Busca un concept_id del KB para sustituir al actual en el refine.

    Orden de preferencia:
    1. Un átomo del `preferred_prefix` aún no usado (`not in avoid`).
    2. Cualquier átomo aún no usado.
    3. Si todos están usados (KB pequeño con cap adaptativo): el átomo
       con **menor `usage`** distinto de `current`. Devuelve algo siempre
       que haya más de un átomo en el KB, evitando el caso "refine
       silencioso" donde un duplicado se conserva por falta de alternativa.
    """
    ids = list(kb.atom_ids())
    if not ids:
        return None
    if preferred_prefix:
        for aid in ids:
            if aid.startswith(preferred_prefix) and aid not in avoid:
                return aid
    for aid in ids:
        if aid not in avoid:
            return aid
    # Fase 3: fallback por mínimo uso. Excluimos el átomo actual para
    # forzar rotación real; si solo hay 1 átomo en el KB, devolvemos
    # None (no se puede rotar).
    pool = [a for a in ids if a != current]
    if not pool:
        return None
    use = usage or {}
    pool.sort(key=lambda a: (use.get(a, 0), a))  # mínimo uso primero, luego orden lex
    return pool[0]


def _apply_cheap_fixes(questions: list["QuizQuestion"]) -> int:
    """Aplica correcciones post-generación baratas (sin LLM).

    Por ahora:
    - Recorta muletillas-preámbulo del inicio del enunciado
      ("Según la definición del documento, …" → "…").

    Devuelve el nº de preguntas modificadas.
    """
    fixed = 0
    for q in questions:
        new_q = _strip_meta_preamble(q.question)
        if new_q != q.question:
            logger.info(
                "Pregunta %d: recortado preámbulo meta ('%s…' → '%s…').",
                q.id, q.question[:40], new_q[:40],
            )
            q.question = new_q
            fixed += 1
    return fixed


def refine_quiz(
    client: OllamaClient,
    kb: KnowledgeBase,
    quiz_questions: list["QuizQuestion"],
    quiz_plan: QuizPlan,
    *,
    min_questions: int = 1,
    max_questions: int | None = None,
    use_llm: bool = False,
    source_markdown: str | None = None,
) -> tuple[list["QuizQuestion"], QuizReview]:
    """Pase único de filtrado por calidad.

    Estrategia simplificada (v2.11):
    1. Correcciones baratas vía regex (recorte de muletillas).
    2. Una única revisión con detectores determinísticos.
    3. Para cada pregunta crítica: un solo intento de regeneración
       (rotando concept_id solo para duplicados reales). Si la
       regeneración sigue siendo crítica o falla, la pregunta se
       **descarta** en lugar de insistir en iteraciones ulteriores.
    4. Si tras descartar quedan más de `max_questions`, se ordenan por
       número de issues residuales y se conservan las mejores.
    5. No hay segundas iteraciones: el coste/beneficio de seguir
       iterando es malo con LLMs locales pequeños (bug documentado en
       v2.10: el refine agresivo colapsaba todas las preguntas a un
       único concepto).

    Devuelve `(questions_finales, review_residual)` con la última
    revisión aplicada — útil para logs y UI.
    """
    # Imports diferidos para evitar ciclo.
    from .quiz_generator import QuizQuestion, generate_single_question

    # 1. Correcciones baratas (recorte de preámbulos meta).
    _apply_cheap_fixes(quiz_questions)

    # 2. Revisión única determinística.
    review = review_quiz(
        client, kb, quiz_questions,
        use_llm=use_llm,
        source_markdown=source_markdown,
    )
    critical = review.blocker_ids()
    if not critical:
        logger.info("Revisor quiz: sin issues bloqueantes, filtrado completo.")
        return quiz_questions, review

    logger.info(
        "Revisor quiz: %d preguntas bloqueantes, intentando regenerar ids=%s",
        len(critical), sorted(critical),
    )

    plan_by_id: dict[int, PlannedQuestion] = {p.id: p for p in quiz_plan.questions}
    issues_by_qid: dict[int, list[QuizIssue]] = {}
    for iss in review.issues:
        if iss.severity == "high":
            issues_by_qid.setdefault(iss.question_id, []).append(iss)

    # 3. Un solo pase de regeneración por pregunta crítica; si falla, descarta.
    kept: list[QuizQuestion] = []
    for q in quiz_questions:
        if q.id not in critical:
            kept.append(q)
            continue

        planned = plan_by_id.get(q.id)
        if planned is None:
            logger.info("Pregunta %d: sin plan, descartada.", q.id)
            continue

        issue_kinds = {i.kind for i in issues_by_qid.get(q.id, [])}
        # Rotación de concept_id SOLO para duplicados reales.
        if issue_kinds & _ISSUES_REQUIRING_CONCEPT_SWAP:
            avoid = {p.concept_id for p in quiz_plan.questions}
            usage: dict[str, int] = {}
            for p in quiz_plan.questions:
                usage[p.concept_id] = usage.get(p.concept_id, 0) + 1
            preferred = (
                planned.concept_id.split(":", 1)[0] + ":"
                if ":" in planned.concept_id else None
            )
            alt = _pick_alternative_concept(
                kb, avoid,
                preferred_prefix=preferred,
                usage=usage,
                current=planned.concept_id,
            )
            if alt is not None:
                logger.info(
                    "Pregunta %d: duplicado, rotando '%s' -> '%s' (issues=%s).",
                    q.id, planned.concept_id, alt, sorted(issue_kinds),
                )
                planned = planned.model_copy(update={"concept_id": alt, "focus": None})

        try:
            newq = generate_single_question(
                client, kb, planned,
                previous=[x for x in kept],
            )
            newq.id = q.id
            # Aplicamos las mismas correcciones baratas a la regeneración.
            newq.question = _strip_meta_preamble(newq.question)
            kept.append(newq)
            logger.info("Pregunta %d: regenerada con éxito.", q.id)
        except GenerationError as exc:
            logger.info(
                "Pregunta %d: regeneración falló (%s), descartada.",
                q.id, exc,
            )

    # 4. Re-revisión determinística tras la regeneración: algunos issues
    #    críticos pueden haberse resuelto, pero también pueden haber
    #    aparecido nuevos (un duplicate_stem encadenado, por ejemplo).
    final_review = review_quiz(
        client, kb, kept,
        use_llm=False,
        source_markdown=source_markdown,
    )
    still_critical = final_review.blocker_ids()

    # 5. Descarte de las preguntas que siguen siendo críticas tras un
    #    único intento de regeneración. Es la clave del nuevo paradigma:
    #    "mejor menos preguntas buenas que más preguntas mediocres".
    if still_critical:
        before = len(kept)
        kept = [q for q in kept if q.id not in still_critical]
        dropped = before - len(kept)
        if dropped:
            logger.info(
                "Revisor quiz: %d preguntas siguen siendo bloqueantes tras "
                "regenerar; se descartan ids=%s",
                dropped, sorted(still_critical),
            )

    # 6. Recorte por `max_questions` (si aplica): ordena por nº de
    #    issues (menos = mejor) y se queda con las primeras.
    if max_questions is not None and len(kept) > max_questions:
        issue_count: dict[int, int] = {}
        for iss in final_review.issues:
            if iss.severity == "high":
                issue_count[iss.question_id] = issue_count.get(iss.question_id, 0) + 1
        kept.sort(key=lambda q: (issue_count.get(q.id, 0), q.id))
        kept = kept[:max_questions]
        logger.info(
            "Revisor quiz: recortado a %d preguntas por tope máximo.",
            max_questions,
        )

    # Aviso (no error) si no llegamos al mínimo.
    if len(kept) < min_questions:
        logger.warning(
            "Revisor quiz: %d preguntas aprobadas (< mínimo %d). El KB del "
            "documento no tiene suficiente variedad de átomos útiles.",
            len(kept), min_questions,
        )

    return kept, final_review


def refine_slides(
    client: OllamaClient,
    kb: KnowledgeBase,
    built_slides: list["BuiltSlide"],
    slide_plan: SlidePlan,
    *,
    max_iterations: int = 1,
    use_llm: bool = True,
    source_markdown: str | None = None,
) -> tuple[list["BuiltSlide"], SlideReview]:
    """Regenera únicamente las slides con issues críticos.

    Mantiene intactas las slides que pasaron la revisión.
    """
    from .pptx_generator import BuiltSlide, render_slide_bullets

    plan_by_title: dict[str, PlannedSlide] = {s.title: s for s in slide_plan.slides}
    slides: list[BuiltSlide] = list(built_slides)
    last_review: SlideReview = SlideReview(issues=[])

    for it in range(max_iterations):
        review = review_slides(
            client, kb, slides,
            use_llm=use_llm,
            source_markdown=source_markdown,
        )
        last_review = review
        critical = review.blocker_indices()
        if not critical:
            logger.info("Revisor slides: iter %d sin issues bloqueantes.", it)
            break
        logger.info(
            "Revisor slides: iter %d — regenerando índices bloqueantes=%s",
            it,
            sorted(critical),
        )

        total = len(slides)
        new_slides: list[BuiltSlide] = []
        for i, bs in enumerate(slides, start=1):
            if i not in critical:
                new_slides.append(bs)
                continue
            planned = plan_by_title.get(bs.title)
            if planned is None:
                logger.warning(
                    "No existe PlannedSlide para '%s'; se conserva la original.",
                    bs.title,
                )
                new_slides.append(bs)
                continue
            try:
                new_bullets = render_slide_bullets(
                    client, kb, planned, slide_plan, index=i, total=total
                )
                new_slides.append(
                    BuiltSlide(title=bs.title, bullets=new_bullets, kind=bs.kind)
                )
            except GenerationError as exc:
                logger.warning("Regeneración slide '%s' falló: %s", bs.title, exc)
                new_slides.append(bs)
        slides = new_slides

    return slides, last_review
