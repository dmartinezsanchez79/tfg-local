"""Modelo de Knowledge Base estructurado del documento fuente.

La KB es la representación canónica del contenido del PDF tras la fase
Map-Reduce. Sustituye al "resumen markdown libre" anterior como fuente de
verdad para las fases downstream (quiz, índice, slides).

Ventajas frente al texto libre:
- Hechos discretos y tipados: definiciones, ejemplos, fórmulas, datos…
- Cada átomo lleva su `id` estable, lo que permite planificar quiz y
  slides asignando átomos concretos (evitando drift y solapes).
- Validación estricta con Pydantic: cualquier desviación del LLM se
  detecta al instante y se puede reintentar.
- Sigue siendo serializable a JSON y renderizable a Markdown para
  previsualización o fallback a prompts legacy.

Este módulo es agnóstico del LLM y del PDF: solo define tipos.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AtomKind = Literal["definition", "example", "formula_code", "datum", "relation"]


# --------------------------------------------------- normalización de ids ---

# Ids del esquema: `prefix:slug` con slug en [a-z0-9_-]. Los LLMs suelen
# devolver tildes, mayúsculas o espacios. Normalizamos en modo "before" para
# no relajar el esquema final y seguir forzando ids limpios y estables.

_ID_ALLOWED_RE = re.compile(r"[^a-z0-9_\-]+")

# Separadores típicos que los LLMs usan al colapsar "término" y "definición"
# en un único campo: "Objeto: es una entidad...", "Objeto — es...",
# "Objeto es...". Orden importa: más específico → menos específico.
_TERM_SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<head>.{2,80}?)\s*:\s*(?P<tail>.+)$"),
    re.compile(r"^(?P<head>.{2,80}?)\s+[-—–]\s+(?P<tail>.+)$"),
    re.compile(
        r"^(?P<head>.{2,80}?)\s+(?:es|son|se\s+refiere\s+a)\s+(?P<tail>.+)$",
        re.IGNORECASE,
    ),
)


def _needs_term_split(term: str, definition: str) -> bool:
    """Heurística: ¿el LLM metió la definición en el campo `term`?

    Dos patologías típicas (observadas con modelos <7B):
    1. `term` excede el límite duro de 120 caracteres → schema lo rechaza.
    2. `term` es largo (>60) y `definition` está vacía o es muy corta (<10):
       el LLM colapsó todo en un solo campo.
    """
    if not isinstance(term, str):
        return False
    if len(term) > 120:
        return True
    current_def = (definition or "").strip() if isinstance(definition, str) else ""
    return len(term) > 60 and len(current_def) < 10


def _split_long_term(term: str, definition: str) -> tuple[str, str]:
    """Parte un `term` colapsado en nombre corto + definición extendida.

    El resto se concatena delante de la `definition` existente, preservando
    toda la información que el LLM devolvió. Tolerante a modelos pequeños
    que colapsan `term` y `definition` en un único campo.

    Devuelve `(term_corto, definicion_extendida)`. El caller es responsable
    de decidir si aplicar el split con `_needs_term_split`.
    """
    if not isinstance(term, str):
        return term, definition
    cleaned = term.strip()
    current_def = (definition or "").strip() if isinstance(definition, str) else ""

    for pattern in _TERM_SPLIT_PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        head = m.group("head").strip(" .,:;—–-")
        tail = m.group("tail").strip()
        if head and tail:
            merged_def = f"{tail} {current_def}".strip() if current_def else tail
            return head[:120], merged_def

    # Sin separador claro: tomar las primeras 6 palabras como nombre y el
    # resto como continuación de la definición.
    words = cleaned.split()
    if len(words) > 6:
        head = " ".join(words[:6]).rstrip(" .,:;—–-")
        tail = " ".join(words[6:])
        merged_def = f"{tail} {current_def}".strip() if current_def else tail
        return head[:120], merged_def

    # Último recurso: truncar sin perder demasiada información; el resto se
    # conserva en la definición para no perder contenido.
    overflow = cleaned[120:].strip()
    head = cleaned[:120].rstrip(" .,:;—–-")
    merged_def = f"{overflow} {current_def}".strip() if overflow else current_def
    return head, merged_def


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize_id(prefix: str, raw: Any, *, max_len: int = 40) -> Any:
    """Normaliza un id tipo `prefix:slug`, tolerante a tildes/espacios/case.

    - Si `raw` no es str, se deja tal cual (que Pydantic lance el error).
    - Mantiene el prefix pedido; si viene otro o ninguno, lo corrige.
    - Slug: minúsculas, sin acentos, solo `[a-z0-9_-]`, sin guiones colgantes.
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if ":" in text:
        _, _, rest = text.partition(":")
    else:
        rest = text
    rest = _strip_accents(rest).lower().replace("ñ", "n")
    rest = _ID_ALLOWED_RE.sub("_", rest).strip("_-")
    if not rest:
        rest = "item"
    return f"{prefix}:{rest[:max_len]}"


# ------------------------------------------------------------------ átomos ---


class Definition(BaseModel):
    """Definición de un término clave del documento."""

    id: str = Field(pattern=r"^def:[a-z0-9_\-]+$")
    term: str = Field(min_length=1, max_length=120)
    definition: str = Field(min_length=5, max_length=600)
    subtopic: str | None = Field(default=None, max_length=120)
    verbatim: bool = Field(
        default=False,
        description=(
            "True si `definition` es cita literal del documento "
            "(extraída por heurística), False si fue reformulada por el LLM."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_term_definition(cls, data: Any) -> Any:
        """Tolerancia frente a LLMs pequeños que colapsan `term`+`definition`.

        Caso típico observado con modelos <7B: el LLM devuelve
        `term="Un objeto es una instancia concreta de una clase, que…"`
        (la definición completa) y deja `definition` vacío o corto. El
        schema rechaza `term > 120` → toda la KB se invalida. Aquí
        detectamos esa patología y partimos `term` en nombre corto +
        resto concatenado a `definition`.
        """
        if not isinstance(data, dict):
            return data
        term = data.get("term")
        definition = data.get("definition", "")
        if _needs_term_split(term, definition):
            new_term, new_def = _split_long_term(term, definition)
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
    """Ejemplo concreto mencionado en el documento (p.ej. 'Bicicleta')."""

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
    """Fórmula matemática o fragmento de código literal del documento."""

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
    """Dato numérico relevante (porcentaje, magnitud, métrica…)."""

    id: str = Field(pattern=r"^dt:[a-z0-9_\-]+$")
    value: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=3, max_length=300)
    subtopic: str | None = Field(default=None, max_length=120)

    @field_validator("id", mode="before")
    @classmethod
    def _norm_id(cls, v: Any) -> Any:
        return _normalize_id("dt", v)


class Relation(BaseModel):
    """Relación semántica entre entidades (herencia, composición, causa, etc.)."""

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


# Traducción de los `kind` internos de relación al verbo/frase natural en
# español. Se usa al serializar relaciones para prompts del LLM: evita que
# el modelo copie la notación técnica `X —[kind]→ Y` en los bullets o en
# los enunciados de preguntas. Reutilizable desde cualquier generador.
_RELATION_KIND_PHRASES: dict[str, str] = {
    "subclase_de": "es subclase de",
    "superclase_de": "es superclase de",
    "hereda_de": "hereda de",
    "implementa": "implementa",
    "compuesto_por": "está compuesto por",
    "compone": "se compone de",
    "contiene": "contiene",
    "parte_de": "es parte de",
    "depende_de": "depende de",
    "usa": "usa",
    "instancia_de": "es instancia de",
    "asociado_con": "está asociado con",
    "relacionado_con": "está relacionado con",
    "causa": "causa",
    "precede_a": "precede a",
    "sigue_a": "sigue a",
    "equivalente_a": "es equivalente a",
    "opuesto_a": "es opuesto a",
}


def relation_kind_phrase(kind: str) -> str:
    """Devuelve la expresión natural en español del `kind` de una relación.

    Si el `kind` no está en el diccionario, se normalizan los guiones bajos
    (`se_relaciona_con` → `"se relaciona con"`) y se devuelve; si queda
    vacío, se cae a `"se relaciona con"`.
    """
    key = (kind or "").strip().lower()
    if key in _RELATION_KIND_PHRASES:
        return _RELATION_KIND_PHRASES[key]
    return key.replace("_", " ").strip() or "se relaciona con"


def relation_to_natural(rel: Relation) -> str:
    """Serializa una `Relation` en español natural, sin notación técnica.

    Ejemplo:
        Relation(source="BicicletaDeMontaña", kind="subclase_de",
                 target="Bicicleta", description="hereda atributos")
        → "BicicletaDeMontaña es subclase de Bicicleta (hereda atributos)".
    """
    phrase = relation_kind_phrase(rel.kind)
    base = f"{rel.source} {phrase} {rel.target}".strip()
    if rel.description:
        return f"{base} ({rel.description})"
    return base


# --------------------------------------------------------- knowledge base ---


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

    # ---------------------------------------------------- consultas utilitarias

    @property
    def atom_count(self) -> int:
        return (
            len(self.definitions)
            + len(self.examples)
            + len(self.formulas_code)
            + len(self.numeric_data)
            + len(self.relations)
        )

    def atom_ids(self) -> list[str]:
        """Lista completa de ids de átomos, en orden estable."""
        return [
            *[d.id for d in self.definitions],
            *[e.id for e in self.examples],
            *[f.id for f in self.formulas_code],
            *[n.id for n in self.numeric_data],
            *[r.id for r in self.relations],
        ]

    def atoms_by_subtopic(self) -> dict[str, list[str]]:
        """Agrupa ids de átomos por subtopic (los sin subtopic van a '').

        Útil para alimentar el planificador de slides en el paso siguiente.
        """
        buckets: dict[str, list[str]] = {}
        for atom in self._iter_atoms():
            key = (atom.subtopic or "").strip()
            buckets.setdefault(key, []).append(atom.id)
        return buckets

    def get_atom(self, atom_id: str) -> Definition | Example | FormulaOrCode | NumericDatum | Relation | None:
        """Devuelve el átomo con el id dado, o None si no existe."""
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

    # ---------------------------------------------------- renderizado para UI

    def to_markdown(self) -> str:
        """Render legible en Markdown para previsualización y fallback.

        Mantiene compatibilidad con prompts legacy que esperan un resumen
        textual (por ejemplo la generación de slides actual).
        """
        lines: list[str] = [f"# {self.main_topic}", ""]

        if self.subtopics:
            lines.append("## Subtemas")
            for s in self.subtopics:
                lines.append(f"- {s}")
            lines.append("")

        if self.definitions:
            lines.append("## Definiciones")
            for d in self.definitions:
                tag = " *(literal)*" if d.verbatim else ""
                lines.append(f"- **{d.term}**{tag}: {d.definition}")
            lines.append("")

        if self.examples:
            lines.append("## Ejemplos")
            for e in self.examples:
                lines.append(f"- **{e.name}** — {e.description}")
                if e.attributes:
                    lines.append(f"  - Atributos: {', '.join(e.attributes)}")
                if e.methods:
                    lines.append(f"  - Métodos: {', '.join(e.methods)}")
            lines.append("")

        if self.formulas_code:
            lines.append("## Fórmulas y código")
            for fc in self.formulas_code:
                caption = f" — *{fc.caption}*" if fc.caption else ""
                fence = fc.language or ("" if fc.kind == "formula" else "text")
                lines.append(f"- **{fc.kind}**{caption}:")
                lines.append(f"  ```{fence}")
                for line in fc.content.splitlines() or [fc.content]:
                    lines.append(f"  {line}")
                lines.append("  ```")
            lines.append("")

        if self.numeric_data:
            lines.append("## Datos e insights")
            for n in self.numeric_data:
                lines.append(f"- **{n.value}** — {n.description}")
            lines.append("")

        if self.relations:
            lines.append("## Relaciones")
            for r in self.relations:
                lines.append(f"- {relation_to_natural(r)}")
            lines.append("")

        if self.conclusions:
            lines.append("## Conclusiones")
            for c in self.conclusions:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def to_prompt_context(self, *, max_chars: int | None = None) -> str:
        """Serialización compacta para inyectar en prompts downstream.

        Es Markdown semiestructurado con ids visibles, de forma que el LLM
        puede referirse a átomos por id cuando sea necesario.
        """
        md = self.to_markdown()
        if max_chars is None or len(md) <= max_chars:
            return md
        return md[: max_chars - 1].rstrip() + "…"


# --------------------------------------------------------- utilidades libres


def slugify_id(prefix: str, text: str, *, max_len: int = 40) -> str:
    """Genera un id estable tipo `def:objeto_poo` a partir de un texto.

    Útil para construir KBs manualmente (tests, extractores, fallbacks).
    Reusa la misma normalización que los validadores Pydantic para
    garantizar consistencia.
    """
    return _normalize_id(prefix, text, max_len=max_len)


# --------- Coerción defensiva del payload de KB devuelto por el LLM -------

# Aliases habituales que usan los LLMs cuando no respetan el esquema:
_KB_TOPIC_ALIASES: tuple[str, ...] = (
    "main_topic",
    "topic",
    "tema",
    "titulo",
    "título",
    "title",
    "subject",
    "nombre",
    "document_title",
)
_KB_SUBTOPICS_ALIASES: tuple[str, ...] = (
    "subtopics",
    "subtemas",
    "sub_topics",
    "secciones",
    "sections",
    "chapters",
    "capitulos",
    "capítulos",
)
_KB_DEFS_ALIASES: tuple[str, ...] = (
    "definitions",
    "definiciones",
    "defs",
    "glossary",
    "glosario",
    "terms",
    "terminos",
    "términos",
)
_KB_EXAMPLES_ALIASES: tuple[str, ...] = (
    "examples",
    "ejemplos",
    "instances",
    "instancias",
    "casos",
    "cases",
)
_KB_FC_ALIASES: tuple[str, ...] = (
    "formulas_code",
    "formulas",
    "fórmulas",
    "codigo",
    "código",
    "code",
    "code_blocks",
    "ecuaciones",
    "formulas_or_code",
    "formulas_and_code",
)
_KB_DATA_ALIASES: tuple[str, ...] = (
    "numeric_data",
    "data",
    "datos",
    "datos_numericos",
    "numeric",
    "numericos",
    "numéricos",
    "metrics",
    "metricas",
    "métricas",
)
_KB_RELATIONS_ALIASES: tuple[str, ...] = (
    "relations",
    "relaciones",
    "relationships",
    "links",
    "enlaces",
)
_KB_CONCLUSIONS_ALIASES: tuple[str, ...] = (
    "conclusions",
    "conclusiones",
    "summary_points",
    "resumen",
    "resumenes",
    "takeaways",
    "ideas_clave",
    "key_points",
)

# Alias internos de cada átomo para normalizar nombres de campo ES↔EN.
_DEF_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier", "identificador"),
    "term": ("term", "termino", "término", "name", "nombre", "concept", "concepto"),
    "definition": ("definition", "definicion", "definición", "desc", "description", "descripcion", "descripción", "explanation", "explicacion", "explicación"),
    "subtopic": ("subtopic", "subtema", "section", "seccion", "sección"),
    "verbatim": ("verbatim", "literal", "is_literal"),
}
_EX_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier"),
    "name": ("name", "nombre", "title", "titulo", "título"),
    "description": ("description", "descripcion", "descripción", "desc", "explanation", "explicacion", "explicación"),
    "attributes": ("attributes", "atributos", "attrs", "properties", "propiedades"),
    "methods": ("methods", "metodos", "métodos", "operations", "operaciones"),
    "subtopic": ("subtopic", "subtema", "section", "seccion", "sección"),
}
_FC_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier"),
    "kind": ("kind", "type", "tipo"),
    "content": ("content", "contenido", "code", "codigo", "código", "formula", "fórmula", "expression", "expresion", "expresión"),
    "caption": ("caption", "titulo", "título", "title", "description", "descripcion", "descripción", "name", "nombre"),
    "language": ("language", "lang", "idioma", "lenguaje"),
    "subtopic": ("subtopic", "subtema", "section", "seccion", "sección"),
}
_DT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier"),
    "value": ("value", "valor", "number", "numero", "número", "magnitude", "magnitud"),
    "description": ("description", "descripcion", "descripción", "desc", "meaning", "significado"),
    "subtopic": ("subtopic", "subtema", "section", "seccion", "sección"),
}
_REL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier"),
    "kind": ("kind", "type", "tipo", "relation", "relation_type"),
    "source": ("source", "origen", "from", "desde", "src", "subject", "sujeto"),
    "target": ("target", "destino", "to", "hacia", "tgt", "object", "objeto"),
    "description": ("description", "descripcion", "descripción", "desc", "explanation", "explicacion", "explicación"),
    "subtopic": ("subtopic", "subtema", "section", "seccion", "sección"),
}


def _pick_field(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """Devuelve el primer alias presente con valor truthy."""
    for key in aliases:
        if key in data and data[key] not in (None, "", []):
            return data[key]
    return None


def _normalize_atom_dict(
    raw: Any, field_map: dict[str, tuple[str, ...]]
) -> dict[str, Any] | None:
    """Renombra claves de un átomo según `field_map`. Mantiene ignorados fuera."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for canonical, aliases in field_map.items():
        val = _pick_field(raw, aliases)
        if val is not None:
            out[canonical] = val
    return out or None


def _as_string_list(raw: Any) -> list[str]:
    """Normaliza cualquier payload a lista de strings no vacíos."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[\n;,]+", raw) if p.strip()]
        return parts
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


_ID_SEED_BY_PREFIX: dict[str, tuple[str, ...]] = {
    "def": ("term", "name"),
    "ex": ("name", "term"),
    "fc": ("caption", "content"),
    "dt": ("value", "description"),
    "rel": ("source", "kind"),
}


def _ensure_atom_id(
    atom: dict[str, Any],
    *,
    prefix: str,
    index: int,
    used: set[str],
) -> dict[str, Any]:
    """Rellena `id` si falta, usando un slug del campo semánticamente clave.

    Muchos LLMs pequeños omiten por completo el `id`. En vez de descartar
    el átomo, reconstruimos un id estable a partir del campo dominante
    (`term`, `name`, etc.) o caemos a un índice secuencial.
    """
    current = atom.get("id")
    if isinstance(current, str) and current.strip():
        if ":" not in current:
            current = f"{prefix}:{current}"
        atom["id"] = current
        return atom

    seed_fields = _ID_SEED_BY_PREFIX.get(prefix, ())
    seed: str = ""
    for f in seed_fields:
        val = atom.get(f)
        if isinstance(val, str) and val.strip():
            seed = val.strip()
            break
    if not seed:
        seed = f"item_{index + 1}"

    base = _normalize_id(prefix, seed)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    atom["id"] = candidate
    return atom


def _coerce_atom_list(
    raw: Any,
    field_map: dict[str, tuple[str, ...]],
    *,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Mapea una lista de átomos a dicts con claves canónicas.

    Si se pasa `prefix`, los átomos sin `id` reciben uno autogenerado
    (`{prefix}:{slug}`) a partir del campo semánticamente clave del tipo.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx, item in enumerate(raw):
        norm = _normalize_atom_dict(item, field_map)
        if norm is None:
            continue
        if prefix is not None:
            norm = _ensure_atom_id(norm, prefix=prefix, index=idx, used=used)
        out.append(norm)
    return out


def _infer_fc_kind(content: str) -> str:
    """Decide si un `FormulaOrCode` es `code` o `formula` desde su contenido.

    Heurística local y barata (regex): si contiene palabras clave de
    lenguajes de programación (`def`, `class`, `return`, `import`,
    `public`, `function`, etc.) o llaves/semicolons → `code`. Si sólo
    tiene un `=` o símbolos matemáticos aislados → `formula`. Por
    defecto `code` (más permisivo para que el schema no rechace).
    """
    if re.search(
        r"\b(def|class|function|public|private|return|var|let|const|import|from|if|else|while|for)\b",
        content,
    ):
        return "code"
    if re.search(r"[{};]", content):
        return "code"
    if re.search(r"[=<>]|\b\d+\s*[+\-*/]\s*\d+\b", content):
        return "formula"
    return "code"


def coerce_kb_payload(
    raw: Any, *, fallback_topic: str | None = None
) -> dict[str, Any] | None:
    """Normaliza la respuesta del LLM al esquema de `KnowledgeBase`.

    - Tolera alias en español y variantes habituales (`tema`, `definiciones`,
      `fórmulas`, `relaciones`…).
    - Desenvuelve un nivel si el LLM envolvió todo en `{"kb": {...}}`,
      `{"output": {...}}`, `{"result": {...}}`, `{"data": {...}}`.
    - Acepta `subtopics` como string con separadores (`"a, b, c"`).
    - Mapea cada átomo con su tabla de alias específica.
    - Devuelve `None` si el payload es irrecuperable.
    """
    if not isinstance(raw, dict):
        return None

    for wrapper in ("kb", "knowledge_base", "base", "output", "result", "data", "respuesta"):
        inner = raw.get(wrapper)
        if isinstance(inner, dict) and any(
            k in inner
            for k in _KB_TOPIC_ALIASES + _KB_DEFS_ALIASES + _KB_EXAMPLES_ALIASES
        ):
            raw = inner
            break

    topic = _pick_field(raw, _KB_TOPIC_ALIASES)
    if not isinstance(topic, str) or not topic.strip():
        topic = (fallback_topic or "").strip() or None

    out: dict[str, Any] = {}
    if topic:
        out["main_topic"] = topic.strip()[:200]

    subtopics = _as_string_list(_pick_field(raw, _KB_SUBTOPICS_ALIASES))
    if subtopics:
        out["subtopics"] = subtopics[:30]

    defs = _coerce_atom_list(
        _pick_field(raw, _KB_DEFS_ALIASES), _DEF_FIELD_ALIASES, prefix="def"
    )
    if defs:
        out["definitions"] = defs

    exs = _coerce_atom_list(
        _pick_field(raw, _KB_EXAMPLES_ALIASES), _EX_FIELD_ALIASES, prefix="ex"
    )
    if exs:
        out["examples"] = exs

    fcs = _coerce_atom_list(
        _pick_field(raw, _KB_FC_ALIASES), _FC_FIELD_ALIASES, prefix="fc"
    )
    # Los FC necesitan `kind` obligatoriamente; si no viene, inferimos.
    for fc in fcs:
        if "kind" not in fc or fc.get("kind") not in {"formula", "code"}:
            content = str(fc.get("content", ""))
            fc["kind"] = _infer_fc_kind(content)
    if fcs:
        out["formulas_code"] = fcs

    dts = _coerce_atom_list(
        _pick_field(raw, _KB_DATA_ALIASES), _DT_FIELD_ALIASES, prefix="dt"
    )
    if dts:
        out["numeric_data"] = dts

    rels = _coerce_atom_list(
        _pick_field(raw, _KB_RELATIONS_ALIASES), _REL_FIELD_ALIASES, prefix="rel"
    )
    if rels:
        out["relations"] = rels

    conclusions = _as_string_list(_pick_field(raw, _KB_CONCLUSIONS_ALIASES))
    if conclusions:
        out["conclusions"] = conclusions[:10]

    # Si tras la coerción no hay ni topic ni un solo átomo, es irrecuperable.
    has_content = any(
        out.get(k)
        for k in ("definitions", "examples", "formulas_code", "numeric_data", "relations")
    )
    if "main_topic" not in out and not has_content:
        return None

    # El schema exige main_topic: si aún falta, usar un placeholder mínimo
    # razonable basado en subtopics o en el primer átomo.
    if "main_topic" not in out:
        if subtopics:
            out["main_topic"] = subtopics[0][:200]
        elif defs:
            out["main_topic"] = str(defs[0].get("term", "Documento"))[:200]
        else:
            out["main_topic"] = "Documento"

    return out
