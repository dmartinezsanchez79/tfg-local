"""Extractores deterministas de material literal desde Markdown.

Complementan al LLM: mientras el MAP-REDUCE resume y abstrae, estos
extractores localizan **fragmentos tal cual aparecen en el documento**.
Objetivos:

- `extract_literal_definitions`: frases del tipo "X es …", "Se llama X a …"
- `extract_code_fences`: bloques ``` de código.
- `extract_formulas`: ecuaciones matemáticas (con '=' y símbolos típicos).
- `extract_key_terms`: negritas/cursivas y títulos como candidatos de término.

Todos los resultados van a un `LiteralHints`, que luego se inyecta en el
prompt REDUCE como material de apoyo para construir la KnowledgeBase.

Principios de diseño:
- **Cero dependencias de LLM**: son heurísticas puras.
- **Tolerantes a ruido**: si no encuentran nada, devuelven listas vacías.
- **Deduplicación estable**: se conserva el orden de primera aparición.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# --------------------------------------------------------- contenedores ---


@dataclass(frozen=True)
class LiteralDefinition:
    """Definición candidata localizada por heurística en el texto."""

    term: str
    definition: str


@dataclass(frozen=True)
class LiteralCode:
    """Fragmento de código delimitado por fences triples."""

    content: str
    language: str | None = None


@dataclass(frozen=True)
class LiteralFormula:
    """Ecuación/fórmula detectada por patrones numéricos o símbolos."""

    content: str


@dataclass
class LiteralHints:
    """Paquete con el material literal extraído del documento."""

    definitions: list[LiteralDefinition] = field(default_factory=list)
    code_blocks: list[LiteralCode] = field(default_factory=list)
    formulas: list[LiteralFormula] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.definitions, self.code_blocks, self.formulas, self.key_terms)
        )

    def to_prompt_block(self, *, max_chars: int = 3500) -> str:
        """Render compacto para inyectar como contexto en el prompt REDUCE.

        Trunca de forma segura si el bloque excede `max_chars`, manteniendo
        una sección cerrada antes del corte.
        """
        if self.is_empty:
            return ""

        parts: list[str] = []
        if self.definitions:
            parts.append("### Definiciones literales detectadas")
            for d in self.definitions:
                parts.append(f"- **{d.term}**: {d.definition}")
        if self.code_blocks:
            parts.append("\n### Bloques de código literales")
            for c in self.code_blocks:
                fence = c.language or "text"
                parts.append(f"```{fence}\n{c.content}\n```")
        if self.formulas:
            parts.append("\n### Fórmulas/ecuaciones detectadas")
            for f in self.formulas:
                parts.append(f"- `{f.content}`")
        if self.key_terms:
            parts.append("\n### Términos candidatos (negritas y títulos)")
            parts.append(", ".join(self.key_terms[:60]))

        block = "\n".join(parts).strip()
        if len(block) <= max_chars:
            return block

        # Truncado seguro por línea.
        truncated: list[str] = []
        running = 0
        for line in block.splitlines():
            if running + len(line) + 1 > max_chars:
                break
            truncated.append(line)
            running += len(line) + 1
        truncated.append("…")
        return "\n".join(truncated)


# ------------------------------------------------------------- regexes ---


# Patrón 1: "X es Y …", "X son Y …"
# - Toma como "X" un sintagma capitalizado de hasta 6 palabras.
# - Toma como "Y" el resto de la frase hasta . ; :
_DEF_ES_ES_RE = re.compile(
    r"(?<![\w¿])"
    r"(?P<term>(?:[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ\-]*"
    r"(?:\s+[a-záéíóúüñ]{1,4}\s+|\s+)){0,3}"
    r"[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ\-]*)"
    r"\s+(?:es|son)\s+"
    r"(?P<def>[^.;:\n]{8,240}[.;:])",
)

# Patrón 2: "Se (llama|denomina|define) X a/como Y"
_DEF_SE_LLAMA_RE = re.compile(
    r"Se\s+(?:llama|denomina|define(?:\s+como)?)\s+"
    r"(?P<term>[\wÁÉÍÓÚÜÑáéíóúüñ\-]{2,60})"
    r"\s+(?:a|como)\s+"
    r"(?P<def>[^.;:\n]{8,240}[.;:])",
    re.IGNORECASE,
)

# Patrón 3: definiciones en lista tipo "**Término**: definición"
_DEF_BOLD_COLON_RE = re.compile(
    r"\*\*(?P<term>[^*\n]{2,80})\*\*\s*[:\-—]\s*(?P<def>[^.\n]{10,300}\.)",
)

# Fences de código Markdown (```lang\n...\n```)
_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+\-]*)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)

# Ecuaciones: líneas con '=' y contenido matemático típico.
# Heurística: al menos un '=' y al menos un dígito o símbolo matemático.
_FORMULA_LINE_RE = re.compile(
    r"(?m)^\s*(?P<expr>[^\n]{3,160}=[^\n]{1,160})\s*$"
)
_FORMULA_OK_CHARS_RE = re.compile(
    r"[\d\+\-\*/\^±√π∑∏∞≤≥≠∈∉⊂⊃∪∩∂Δ∇αβγδεθλμνξρστφχψω]"
)

# Negritas / términos clave en Markdown: **término**
_BOLD_TERM_RE = re.compile(r"\*\*([^\*\n]{2,60})\*\*")

# Títulos Markdown (para sacar términos candidatos).
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(?P<title>.+?)\s*$")


# ------------------------------------------------------------ funciones ---


def _dedup_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        key = x.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x.strip())
    return out


def _clean_term(term: str) -> str:
    t = re.sub(r"\s+", " ", term).strip(" .,:;-—")
    # Evita artículos pegados al inicio ("El objeto", "La clase").
    t = re.sub(
        r"^(?:el|la|los|las|un|una|unos|unas)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t


def _clean_definition(defn: str) -> str:
    d = re.sub(r"\s+", " ", defn).strip()
    # Quitar separador final ':' o ';' pero conservar '.' si cierra frase.
    if d.endswith((":", ";")):
        d = d[:-1].rstrip() + "."
    if not d.endswith("."):
        d += "."
    return d


def extract_literal_definitions(
    markdown: str, *, max_items: int = 20
) -> list[LiteralDefinition]:
    """Devuelve definiciones candidatas con heurística multi-patrón."""
    results: list[LiteralDefinition] = []

    for match in _DEF_BOLD_COLON_RE.finditer(markdown):
        term = _clean_term(match.group("term"))
        defn = _clean_definition(match.group("def"))
        if term and defn:
            results.append(LiteralDefinition(term=term, definition=defn))

    for pattern in (_DEF_ES_ES_RE, _DEF_SE_LLAMA_RE):
        for match in pattern.finditer(markdown):
            term = _clean_term(match.group("term"))
            defn = _clean_definition(match.group("def"))
            # Filtros: el "término" ha de ser corto y no una frase larga.
            if not term or len(term) > 60 or len(term.split()) > 6:
                continue
            # Evita frases tipo "La presentación es interesante porque…".
            if len(defn.split()) < 3:
                continue
            results.append(LiteralDefinition(term=term, definition=defn))

    # Deduplicar por término (primera aparición gana).
    seen_terms: set[str] = set()
    unique: list[LiteralDefinition] = []
    for d in results:
        key = d.term.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)
        unique.append(d)
        if len(unique) >= max_items:
            break
    return unique


def extract_code_fences(
    markdown: str, *, max_items: int = 10, max_chars: int = 1500
) -> list[LiteralCode]:
    """Extrae bloques ``` de código manteniendo su lenguaje si se declaró."""
    results: list[LiteralCode] = []
    for match in _CODE_FENCE_RE.finditer(markdown):
        body = match.group("body").rstrip()
        if not body.strip():
            continue
        if len(body) > max_chars:
            body = body[: max_chars - 1].rstrip() + "…"
        lang = match.group("lang") or None
        results.append(LiteralCode(content=body, language=lang))
        if len(results) >= max_items:
            break
    return results


def extract_formulas(
    markdown: str, *, max_items: int = 15
) -> list[LiteralFormula]:
    """Detecta ecuaciones a línea completa con '=' y símbolos matemáticos."""
    lines: list[str] = []
    for match in _FORMULA_LINE_RE.finditer(markdown):
        expr = match.group("expr").strip()
        # Descartamos "frases" con '=' que no son fórmulas (p.ej. URLs).
        if expr.count("=") > 3:
            continue
        if not _FORMULA_OK_CHARS_RE.search(expr):
            continue
        # Evita listas tipo "key=value" sin carga matemática.
        if len(expr.split()) > 18:
            continue
        lines.append(expr)
    lines = _dedup_preserving_order(lines)[:max_items]
    return [LiteralFormula(content=x) for x in lines]


def extract_key_terms(markdown: str, *, max_items: int = 60) -> list[str]:
    """Términos candidatos: negritas del texto + títulos de secciones."""
    terms: list[str] = []
    for m in _BOLD_TERM_RE.finditer(markdown):
        terms.append(_clean_term(m.group(1)))
    for m in _HEADING_RE.finditer(markdown):
        title = _clean_term(m.group("title"))
        # Descarta títulos administrativos (p. ej. "Contexto de imágenes").
        if title.lower() in {"contexto de imágenes", "indice", "índice"}:
            continue
        terms.append(title)
    terms = [t for t in terms if 2 <= len(t) <= 80]
    return _dedup_preserving_order(terms)[:max_items]


def extract_literal_hints(markdown: str) -> LiteralHints:
    """Orquestación: aplica todos los extractores sobre el Markdown dado."""
    hints = LiteralHints(
        definitions=extract_literal_definitions(markdown),
        code_blocks=extract_code_fences(markdown),
        formulas=extract_formulas(markdown),
        key_terms=extract_key_terms(markdown),
    )
    logger.info(
        "Hints literales extraídos: %d defs, %d code, %d formulas, %d terms",
        len(hints.definitions),
        len(hints.code_blocks),
        len(hints.formulas),
        len(hints.key_terms),
    )
    return hints
