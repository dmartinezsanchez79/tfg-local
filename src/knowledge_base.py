"""Knowledge Base estructurado del documento fuente.

Representación canónica del contenido del PDF tras el Map-Reduce. Es un
JSON tipado con Pydantic que sustituye al "resumen markdown libre".

Cada átomo (`Definition`, `Example`, `FormulaOrCode`, `NumericDatum`,
`Relation`) lleva un `id` estable tipo `prefix:slug`, lo que permite
planificar quiz y slides por id sin drift.

Este módulo es agnóstico del LLM y del PDF: solo define tipos y la
coerción defensiva que tolera respuestas imperfectas de modelos pequeños.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AtomKind = Literal["definition", "example", "formula_code", "datum", "relation"]


# =========================================================================
# Normalización de ids
# =========================================================================

_ID_ALLOWED_RE = re.compile(r"[^a-z0-9_\-]+")

# Separadores usados por LLMs pequeños cuando colapsan `term`+`definition`
# en un único campo ("Objeto: es una entidad…"). Orden: más específico primero.
_TERM_SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<head>.{2,80}?)\s*:\s*(?P<tail>.+)$"),
    re.compile(r"^(?P<head>.{2,80}?)\s+[-—–]\s+(?P<tail>.+)$"),
    re.compile(r"^(?P<head>.{2,80}?)\s+(?:es|son|se\s+refiere\s+a)\s+(?P<tail>.+)$", re.I),
)


def _needs_term_split(term: Any, definition: Any) -> bool:
    """El LLM colapsó definición en `term` (típico con modelos <7B)."""
    if not isinstance(term, str):
        return False
    if len(term) > 120:
        return True
    cur_def = (definition or "").strip() if isinstance(definition, str) else ""
    return len(term) > 60 and len(cur_def) < 10


def _split_long_term(term: str, definition: str) -> tuple[str, str]:
    """Parte un `term` colapsado en (nombre_corto, definición_extendida)."""
    if not isinstance(term, str):
        return term, definition
    cleaned = term.strip()
    cur_def = (definition or "").strip() if isinstance(definition, str) else ""

    for pattern in _TERM_SPLIT_PATTERNS:
        m = pattern.match(cleaned)
        if m:
            head = m.group("head").strip(" .,:;—–-")
            tail = m.group("tail").strip()
            if head and tail:
                merged = f"{tail} {cur_def}".strip() if cur_def else tail
                return head[:120], merged

    words = cleaned.split()
    if len(words) > 6:
        head = " ".join(words[:6]).rstrip(" .,:;—–-")
        tail = " ".join(words[6:])
        merged = f"{tail} {cur_def}".strip() if cur_def else tail
        return head[:120], merged

    overflow = cleaned[120:].strip()
    head = cleaned[:120].rstrip(" .,:;—–-")
    merged = f"{overflow} {cur_def}".strip() if overflow else cur_def
    return head, merged


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize_id(prefix: str, raw: Any, *, max_len: int = 40) -> Any:
    """Normaliza un id `prefix:slug`: minúsculas, sin acentos, `[a-z0-9_-]`."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    rest = text.partition(":")[2] if ":" in text else text
    rest = _strip_accents(rest).lower().replace("ñ", "n")
    rest = _ID_ALLOWED_RE.sub("_", rest).strip("_-") or "item"
    return f"{prefix}:{rest[:max_len]}"


# =========================================================================
# Átomos
# =========================================================================

class Definition(BaseModel):
    id: str = Field(pattern=r"^def:[a-z0-9_\-]+$")
    term: str = Field(min_length=1, max_length=120)
    definition: str = Field(min_length=5, max_length=600)
    subtopic: str | None = Field(default=None, max_length=120)
    verbatim: bool = Field(
        default=False,
        description="True si `definition` es cita literal del documento.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_term_definition(cls, data: Any) -> Any:
        """Parte `term` si contiene la definición colapsada (modelos <7B)."""
        if not isinstance(data, dict):
            return data
        if _needs_term_split(data.get("term"), data.get("definition", "")):
            new_term, new_def = _split_long_term(data["term"], data.get("definition", ""))
            data = {**data, "term": new_term, "definition": new_def}
        return data

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("def", v)

    @field_validator("term", "definition")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Example(BaseModel):
    id: str = Field(pattern=r"^ex:[a-z0-9_\-]+$")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=600)
    attributes: list[str] = Field(default_factory=list, max_length=20)
    methods: list[str] = Field(default_factory=list, max_length=20)
    subtopic: str | None = Field(default=None, max_length=120)

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("ex", v)

    @field_validator("name", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("attributes", "methods")
    @classmethod
    def _strip_list(cls, vs: list[str]) -> list[str]:
        return [x.strip() for x in vs if x and x.strip()]


class FormulaOrCode(BaseModel):
    id: str = Field(pattern=r"^fc:[a-z0-9_\-]+$")
    kind: Literal["formula", "code"]
    content: str = Field(min_length=1, max_length=2000)
    caption: str | None = Field(default=None, max_length=200)
    language: str | None = Field(default=None, max_length=30)
    subtopic: str | None = Field(default=None, max_length=120)

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("fc", v)

    @field_validator("content")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.rstrip()


class NumericDatum(BaseModel):
    id: str = Field(pattern=r"^dt:[a-z0-9_\-]+$")
    value: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=3, max_length=300)
    subtopic: str | None = Field(default=None, max_length=120)

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("dt", v)


class Relation(BaseModel):
    id: str = Field(pattern=r"^rel:[a-z0-9_\-]+$")
    kind: str = Field(min_length=1, max_length=60)
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=300)
    subtopic: str | None = Field(default=None, max_length=120)

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("rel", v)


# =========================================================================
# Serialización de relaciones en lenguaje natural
# =========================================================================

# Traducción de `kind` técnico a verbo en español: evita que el LLM copie
# la notación interna `X —[kind]→ Y` en bullets o preguntas.
_RELATION_KIND_PHRASES: dict[str, str] = {
    "subclase_de": "es subclase de", "superclase_de": "es superclase de",
    "hereda_de": "hereda de", "implementa": "implementa",
    "compuesto_por": "está compuesto por", "compone": "se compone de",
    "contiene": "contiene", "parte_de": "es parte de",
    "depende_de": "depende de", "usa": "usa",
    "instancia_de": "es instancia de",
    "asociado_con": "está asociado con", "relacionado_con": "está relacionado con",
    "causa": "causa", "precede_a": "precede a", "sigue_a": "sigue a",
    "equivalente_a": "es equivalente a", "opuesto_a": "es opuesto a",
}


def relation_kind_phrase(kind: str) -> str:
    """Expresión en español del `kind`. Si no está mapeado, reemplaza `_` por ` `."""
    key = (kind or "").strip().lower()
    if key in _RELATION_KIND_PHRASES:
        return _RELATION_KIND_PHRASES[key]
    return key.replace("_", " ").strip() or "se relaciona con"


def relation_to_natural(rel: Relation) -> str:
    """Serializa `Relation` en español natural: `source <verbo> target (desc)`."""
    base = f"{rel.source} {relation_kind_phrase(rel.kind)} {rel.target}".strip()
    return f"{base} ({rel.description})" if rel.description else base


# =========================================================================
# KnowledgeBase
# =========================================================================

class KnowledgeBase(BaseModel):
    """Representación canónica del contenido del documento."""

    main_topic: str = Field(min_length=2, max_length=200)
    subtopics: list[str] = Field(default_factory=list, max_length=30)
    definitions: list[Definition] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    formulas_code: list[FormulaOrCode] = Field(default_factory=list)
    numeric_data: list[NumericDatum] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list, max_length=10)

    @property
    def atom_count(self) -> int:
        return (len(self.definitions) + len(self.examples)
                + len(self.formulas_code) + len(self.numeric_data)
                + len(self.relations))

    def atom_ids(self) -> list[str]:
        return [a.id for a in self._iter_atoms()]

    def atoms_by_subtopic(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        for atom in self._iter_atoms():
            buckets.setdefault((atom.subtopic or "").strip(), []).append(atom.id)
        return buckets

    def get_atom(self, atom_id: str) -> Definition | Example | FormulaOrCode | NumericDatum | Relation | None:
        for atom in self._iter_atoms():
            if atom.id == atom_id:
                return atom
        return None

    def _iter_atoms(self) -> Iterable[Definition | Example | FormulaOrCode | NumericDatum | Relation]:
        yield from self.definitions
        yield from self.examples
        yield from self.formulas_code
        yield from self.numeric_data
        yield from self.relations

    def to_markdown(self) -> str:
        """Render legible en Markdown para previsualización y como contexto de prompt."""
        lines: list[str] = [f"# {self.main_topic}", ""]

        def _section(title: str, items: list[str]) -> None:
            if items:
                lines.append(f"## {title}")
                lines.extend(items)
                lines.append("")

        _section("Subtemas", [f"- {s}" for s in self.subtopics])

        def_lines = [
            f"- **{d.term}**{' *(literal)*' if d.verbatim else ''}: {d.definition}"
            for d in self.definitions
        ]
        _section("Definiciones", def_lines)

        ex_lines: list[str] = []
        for e in self.examples:
            ex_lines.append(f"- **{e.name}** — {e.description}")
            if e.attributes:
                ex_lines.append(f"  - Atributos: {', '.join(e.attributes)}")
            if e.methods:
                ex_lines.append(f"  - Métodos: {', '.join(e.methods)}")
        _section("Ejemplos", ex_lines)

        fc_lines: list[str] = []
        for fc in self.formulas_code:
            caption = f" — *{fc.caption}*" if fc.caption else ""
            fence = fc.language or ("" if fc.kind == "formula" else "text")
            fc_lines.append(f"- **{fc.kind}**{caption}:")
            fc_lines.append(f"  ```{fence}")
            for line in fc.content.splitlines() or [fc.content]:
                fc_lines.append(f"  {line}")
            fc_lines.append("  ```")
        _section("Fórmulas y código", fc_lines)

        _section("Datos e insights", [f"- **{n.value}** — {n.description}" for n in self.numeric_data])
        _section("Relaciones", [f"- {relation_to_natural(r)}" for r in self.relations])
        _section("Conclusiones", [f"- {c}" for c in self.conclusions])

        return "\n".join(lines).strip() + "\n"

    def to_prompt_context(self, *, max_chars: int | None = None) -> str:
        md = self.to_markdown()
        if max_chars is None or len(md) <= max_chars:
            return md
        return md[: max_chars - 1].rstrip() + "…"


def slugify_id(prefix: str, text: str, *, max_len: int = 40) -> str:
    """Genera un id estable tipo `def:objeto_poo` a partir de un texto."""
    return _normalize_id(prefix, text, max_len=max_len)


# =========================================================================
# Coerción defensiva del payload del LLM
# =========================================================================

# Aliases mínimos (ES canónico + EN) que cubren ~99% de lo que devuelven
# los LLMs. Los modelos que se salen del esquema caen al fallback
# determinístico vía `map_reduce.build_fallback_kb`.

_KB_TOPLEVEL_ALIASES: dict[str, tuple[str, ...]] = {
    "topic":        ("main_topic", "topic", "tema", "title"),
    "subtopics":    ("subtopics", "subtemas", "sections", "secciones"),
    "definitions":  ("definitions", "definiciones", "glossary", "glosario"),
    "examples":     ("examples", "ejemplos", "instances", "casos"),
    "formulas":     ("formulas_code", "formulas", "code", "code_blocks"),
    "data":         ("numeric_data", "data", "datos", "metrics"),
    "relations":    ("relations", "relaciones", "relationships"),
    "conclusions":  ("conclusions", "conclusiones", "key_points", "takeaways"),
}

_ATOM_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "def": {
        "id": ("id", "identifier"),
        "term": ("term", "termino", "término", "name", "nombre", "concept"),
        "definition": ("definition", "definicion", "definición", "description", "descripcion"),
        "subtopic": ("subtopic", "subtema", "section"),
        "verbatim": ("verbatim", "literal"),
    },
    "ex": {
        "id": ("id", "identifier"),
        "name": ("name", "nombre", "title"),
        "description": ("description", "descripcion", "descripción"),
        "attributes": ("attributes", "atributos", "properties"),
        "methods": ("methods", "metodos", "métodos", "operations"),
        "subtopic": ("subtopic", "subtema", "section"),
    },
    "fc": {
        "id": ("id", "identifier"),
        "kind": ("kind", "type", "tipo"),
        "content": ("content", "contenido", "code", "codigo", "formula"),
        "caption": ("caption", "title", "description"),
        "language": ("language", "lang", "idioma"),
        "subtopic": ("subtopic", "subtema", "section"),
    },
    "dt": {
        "id": ("id", "identifier"),
        "value": ("value", "valor", "number"),
        "description": ("description", "descripcion", "descripción"),
        "subtopic": ("subtopic", "subtema", "section"),
    },
    "rel": {
        "id": ("id", "identifier"),
        "kind": ("kind", "type", "tipo", "relation"),
        "source": ("source", "origen", "from", "subject"),
        "target": ("target", "destino", "to", "object"),
        "description": ("description", "descripcion", "descripción"),
        "subtopic": ("subtopic", "subtema", "section"),
    },
}

# Campo semánticamente dominante de cada tipo, usado si el LLM omite el `id`.
_ID_SEED_BY_PREFIX: dict[str, tuple[str, ...]] = {
    "def": ("term", "name"),
    "ex":  ("name", "term"),
    "fc":  ("caption", "content"),
    "dt":  ("value", "description"),
    "rel": ("source", "kind"),
}


def _pick(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in data and data[key] not in (None, "", []):
            return data[key]
    return None


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in re.split(r"[\n;,]+", raw) if p.strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _ensure_id(atom: dict[str, Any], *, prefix: str, index: int, used: set[str]) -> None:
    """Rellena `id` si falta, usando el campo semánticamente clave como seed."""
    current = atom.get("id")
    if isinstance(current, str) and current.strip():
        atom["id"] = current if ":" in current else f"{prefix}:{current}"
        return

    seed = ""
    for f in _ID_SEED_BY_PREFIX.get(prefix, ()):
        val = atom.get(f)
        if isinstance(val, str) and val.strip():
            seed = val.strip()
            break
    if not seed:
        seed = f"item_{index + 1}"

    base = _normalize_id(prefix, seed)
    candidate, suffix = base, 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    atom["id"] = candidate


def _coerce_atom_list(raw: Any, prefix: str) -> list[dict[str, Any]]:
    """Renombra claves de cada átomo según su prefix y autogenera `id` si falta."""
    if not isinstance(raw, list):
        return []
    field_map = _ATOM_ALIASES[prefix]
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        norm: dict[str, Any] = {}
        for canonical, aliases in field_map.items():
            val = _pick(item, aliases)
            if val is not None:
                norm[canonical] = val
        if not norm:
            continue
        _ensure_id(norm, prefix=prefix, index=idx, used=used)
        out.append(norm)
    return out


def _infer_fc_kind(content: str) -> str:
    """Decide si un `FormulaOrCode` es `code` o `formula` por su contenido."""
    if re.search(
        r"\b(def|class|function|public|private|return|var|let|const|import|from|if|else|while|for)\b",
        content,
    ) or re.search(r"[{};]", content):
        return "code"
    if re.search(r"[=<>]|\b\d+\s*[+\-*/]\s*\d+\b", content):
        return "formula"
    return "code"


def coerce_kb_payload(raw: Any, *, fallback_topic: str | None = None) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema de `KnowledgeBase`.

    Tolera alias ES/EN, un nivel de envoltura (`{"kb": {...}}`, etc.) y
    atómos con campos parciales. Devuelve `None` si es irrecuperable.
    """
    if not isinstance(raw, dict):
        return None

    # Desenvolver un nivel si el LLM metió todo dentro de una clave genérica.
    probe = (_KB_TOPLEVEL_ALIASES["topic"]
             + _KB_TOPLEVEL_ALIASES["definitions"]
             + _KB_TOPLEVEL_ALIASES["examples"])
    for wrapper in ("kb", "knowledge_base", "base", "output", "result", "data", "respuesta"):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and any(k in inner for k in probe):
            raw = inner
            break

    out: dict[str, Any] = {}

    topic = _pick(raw, _KB_TOPLEVEL_ALIASES["topic"])
    if isinstance(topic, str) and topic.strip():
        out["main_topic"] = topic.strip()[:200]
    elif fallback_topic:
        out["main_topic"] = fallback_topic.strip()[:200]

    subtopics = _as_string_list(_pick(raw, _KB_TOPLEVEL_ALIASES["subtopics"]))
    if subtopics:
        out["subtopics"] = subtopics[:30]

    # Átomos tipados.
    for key, prefix in (("definitions", "def"), ("examples", "ex"),
                        ("formulas", "fc"), ("data", "dt"), ("relations", "rel")):
        items = _coerce_atom_list(_pick(raw, _KB_TOPLEVEL_ALIASES[key]), prefix)
        if items:
            canonical = {"def": "definitions", "ex": "examples", "fc": "formulas_code",
                         "dt": "numeric_data", "rel": "relations"}[prefix]
            out[canonical] = items

    # FormulaOrCode: el `kind` es obligatorio; si no viene, lo inferimos.
    for fc in out.get("formulas_code", []):
        if fc.get("kind") not in {"formula", "code"}:
            fc["kind"] = _infer_fc_kind(str(fc.get("content", "")))

    conclusions = _as_string_list(_pick(raw, _KB_TOPLEVEL_ALIASES["conclusions"]))
    if conclusions:
        out["conclusions"] = conclusions[:10]

    has_content = any(out.get(k) for k in
                      ("definitions", "examples", "formulas_code", "numeric_data", "relations"))
    if "main_topic" not in out and not has_content:
        return None

    # Placeholder razonable si main_topic aún falta.
    if "main_topic" not in out:
        if subtopics:
            out["main_topic"] = subtopics[0][:200]
        elif out.get("definitions"):
            out["main_topic"] = str(out["definitions"][0].get("term", "Documento"))[:200]
        else:
            out["main_topic"] = "Documento"

    return out
