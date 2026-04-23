"""Pipeline Map-Reduce para consolidar un documento largo.

La salida canónica es una `KnowledgeBase` estructurada (JSON), producida
por `build_knowledge_base`. Adicionalmente se mantiene la función legacy
`consolidate_document`, que devuelve el mismo contenido renderizado como
Markdown y actúa como shim de compatibilidad para el código que todavía
espera un resumen textual.

El chunker intenta respetar los límites naturales del Markdown (cabeceras,
párrafos) para no partir una tabla o un concepto por la mitad.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from .exceptions import GenerationError
from .extractors import LiteralHints, extract_literal_hints
from .knowledge_base import (
    Definition,
    Example,
    FormulaOrCode,
    KnowledgeBase,
    Relation,
    coerce_kb_payload,
    slugify_id,
)
from .ollama_client import OllamaClient
from .prompts import (
    MAP_SUMMARY_PROMPT,
    REDUCE_CONSOLIDATION_PROMPT,
    REDUCE_TO_KB_PROMPT,
    SYSTEM_EXPERT_ES,
)

logger = logging.getLogger(__name__)

# Callback tipado: (fase, paso_actual, total_pasos, mensaje) -> None
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class Chunk:
    """Fragmento lógico del documento."""

    index: int
    text: str


# ---------------------------------------------------------------- chunking --


def _split_on_headings(markdown: str) -> list[str]:
    """Divide por títulos Markdown (#, ##, ###) conservando el encabezado."""
    pattern = re.compile(r"(?m)^(?=#{1,6}\s)")
    parts = pattern.split(markdown)
    return [p for p in (x.strip() for x in parts) if p]


def _pack_sections(
    sections: Iterable[str],
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Agrupa secciones en chunks de ~max_chars. Si una sección excede el
    límite, se parte por párrafos. Añade solapamiento entre chunks para
    preservar contexto."""
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
            buffer = ""

    for section in sections:
        if len(section) <= max_chars:
            if len(buffer) + len(section) + 2 <= max_chars:
                buffer = f"{buffer}\n\n{section}" if buffer else section
            else:
                flush()
                buffer = section
        else:
            # Sección demasiado grande: partir por párrafos.
            flush()
            paragraphs = re.split(r"\n\s*\n", section)
            inner = ""
            for p in paragraphs:
                if len(p) > max_chars:
                    # Párrafo gigantesco (tabla enorme): partir por líneas.
                    if inner:
                        chunks.append(inner.strip())
                        inner = ""
                    start = 0
                    while start < len(p):
                        end = min(start + max_chars, len(p))
                        chunks.append(p[start:end])
                        start = end
                    continue
                if len(inner) + len(p) + 2 <= max_chars:
                    inner = f"{inner}\n\n{p}" if inner else p
                else:
                    if inner:
                        chunks.append(inner.strip())
                    inner = p
            if inner:
                chunks.append(inner.strip())
    flush()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # Añadir solapamiento: cada chunk hereda las últimas `overlap` chars del anterior.
    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(f"{prev_tail}\n\n{chunks[i]}")
    return overlapped


def split_markdown(markdown: str) -> list[Chunk]:
    """API pública: markdown -> lista de chunks ordenados."""
    sections = _split_on_headings(markdown) or [markdown]
    packed = _pack_sections(sections)
    logger.info("Documento dividido en %d chunks.", len(packed))
    return [Chunk(index=i, text=t) for i, t in enumerate(packed)]


# ---------------------------------------------------------------- map-reduce


def _map_phase(
    client: OllamaClient,
    chunks: list[Chunk],
    progress_cb: ProgressCallback | None,
) -> list[str]:
    partials: list[str] = []
    total = len(chunks)
    for c in chunks:
        if progress_cb:
            progress_cb(
                "map",
                c.index + 1,
                total,
                f"Resumiendo fragmento {c.index + 1}/{total}…",
            )
        prompt = MAP_SUMMARY_PROMPT.format(
            index=c.index + 1, total=total, chunk=c.text
        )
        summary = client.generate(prompt, system=SYSTEM_EXPERT_ES)
        partials.append(summary.strip())
    return partials


def _reduce_phase(
    client: OllamaClient,
    partials: list[str],
    progress_cb: ProgressCallback | None,
) -> str:
    """Consolidación legacy a Markdown libre. Se conserva como fallback."""
    if progress_cb:
        progress_cb("reduce", 1, 1, "Consolidando resumen global…")
    joined = _join_partials(partials)
    prompt = REDUCE_CONSOLIDATION_PROMPT.format(partials=joined)
    return client.generate(prompt, system=SYSTEM_EXPERT_ES).strip()


def _join_partials(partials: list[str]) -> str:
    return "\n\n---\n\n".join(
        f"[Resumen parcial {i + 1}]\n{p}" for i, p in enumerate(partials)
    )


def _try_build_kb(
    raw: Any, fallback_topic: str | None
) -> tuple[KnowledgeBase | None, str | None]:
    """Coerciona y valida una respuesta cruda del LLM. Nunca lanza."""
    data = coerce_kb_payload(raw, fallback_topic=fallback_topic)
    if data is None:
        return None, "estructura irrecuperable tras coerción"
    try:
        return KnowledgeBase(**data), None
    except ValidationError as exc:
        return None, f"ValidationError: {exc.errors()[:3]}"


# --------- Fallback determinístico de KB desde hints + markdown ----------

_MD_H1_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
_MD_H2_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
_MD_H3_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")


def _infer_main_topic(markdown: str, hints: LiteralHints) -> str:
    """Heurística robusta para adivinar el tema central del documento."""
    m = _MD_H1_RE.search(markdown or "")
    if m:
        candidate = m.group(1).strip()
        if 2 <= len(candidate) <= 200:
            return candidate
    if hints.key_terms:
        return hints.key_terms[0][:200]
    return "Documento"


def _infer_subtopics(markdown: str, hints: LiteralHints) -> list[str]:
    """Encabezados H2 (o H3 si no hay H2). Dedup y límite 30."""
    h2 = [h.strip() for h in _MD_H2_RE.findall(markdown or "")]
    if not h2:
        h2 = [h.strip() for h in _MD_H3_RE.findall(markdown or "")]
    if not h2 and hints.key_terms:
        h2 = list(hints.key_terms[:15])
    seen: set[str] = set()
    out: list[str] = []
    for s in h2:
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s[:120])
    return out[:30]


def _fc_kind_of(content: str) -> str:
    """Heurística para diferenciar fórmula matemática de fragmento de código."""
    if re.search(r"\b(def|class|function|public|private|return|var|let|const)\b", content):
        return "code"
    if re.search(r"\b(import|from)\b\s+\w+", content):
        return "code"
    if re.search(r"[=<>]|\b\d+\s*[+\-*/]\s*\d+\b", content) and not re.search(
        r"[{};]", content
    ):
        return "formula"
    return "code"


def build_fallback_kb(markdown: str, hints: LiteralHints) -> KnowledgeBase:
    """Construye una `KnowledgeBase` mínima sin llamar al LLM.

    Se usa cuando el modelo devuelve JSON inválido o vacío. Usa los hints
    literales extraídos determinísticamente (v1.3+) y estructura del
    Markdown original (H1/H2) para recuperar contenido verificable.
    """
    main_topic = _infer_main_topic(markdown, hints)
    subtopics = _infer_subtopics(markdown, hints)

    definitions: list[Definition] = []
    used_ids: set[str] = set()
    for i, d in enumerate(hints.definitions[:30]):
        term = d.term.strip()
        definition = d.definition.strip()
        if not term or len(definition) < 5:
            continue
        base_id = slugify_id("def", term) or f"def:item_{i}"
        atom_id = base_id
        suffix = 2
        while atom_id in used_ids:
            atom_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(atom_id)
        try:
            definitions.append(
                Definition(
                    id=atom_id,
                    term=term[:120],
                    definition=definition[:600],
                    verbatim=True,
                )
            )
        except ValidationError as exc:
            logger.debug("Fallback KB: definición '%s' descartada (%s)", term, exc)

    formulas_code: list[FormulaOrCode] = []
    used_ids = set()
    for i, c in enumerate(hints.code_blocks[:20]):
        atom_id = f"fc:code_{i + 1}"
        if atom_id in used_ids:
            continue
        used_ids.add(atom_id)
        try:
            formulas_code.append(
                FormulaOrCode(
                    id=atom_id,
                    kind="code",
                    content=c.content[:2000],
                    language=c.language,
                )
            )
        except ValidationError as exc:
            logger.debug("Fallback KB: code block descartado (%s)", exc)
    for i, f in enumerate(hints.formulas[:20]):
        atom_id = f"fc:formula_{i + 1}"
        if atom_id in used_ids:
            continue
        used_ids.add(atom_id)
        content = f.content.strip()
        try:
            formulas_code.append(
                FormulaOrCode(
                    id=atom_id,
                    kind=_fc_kind_of(content),  # type: ignore[arg-type]
                    content=content[:2000],
                )
            )
        except ValidationError as exc:
            logger.debug("Fallback KB: fórmula descartada (%s)", exc)

    kb = KnowledgeBase(
        main_topic=main_topic,
        subtopics=subtopics,
        definitions=definitions,
        examples=[],
        formulas_code=formulas_code,
        numeric_data=[],
        relations=[],
        conclusions=[],
    )
    logger.info(
        "Fallback KB construida sin LLM: topic='%s', %d subtopics, %d átomos.",
        kb.main_topic,
        len(kb.subtopics),
        kb.atom_count,
    )
    return kb


# ---------- Verificación literal de relaciones contra el markdown --------
#
# El LLM puede alucinar relaciones entre entidades que NO aparecen en el
# documento (p. ej. inventar una jerarquía `Bicicleta → Vehículos` cuando
# "Vehículos" ni siquiera se menciona). Estas alucinaciones son muy dañinas
# porque contaminan los bullets de las slides y los distractores del quiz.
#
# Estrategia: tras construir la KB, verificamos que tanto `source` como
# `target` de cada `Relation` aparezcan literalmente en el markdown del
# PDF. La comparación es tolerante a acentos, mayúsculas, espacios y
# conectores habituales del camelcase (`BicicletaDeMontaña` se comprueba
# como `bicicleta de montana` y como `bicicletademontana`). No comprueba
# las definiciones (que sí pasan por extractores literales en la fase MAP)
# ni los ejemplos (validados aparte).

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_entity(text: str) -> str:
    """Minúsculas, sin acentos, solo alfanumérico y espacios (compactados)."""
    n = unicodedata.normalize("NFKD", text or "")
    flat = "".join(c for c in n if not unicodedata.combining(c)).lower()
    flat = flat.replace("ñ", "n")
    flat = _NON_ALNUM_RE.sub(" ", flat)
    return " ".join(flat.split())


def _split_camel_case(text: str) -> str:
    """`BicicletaDeMontaña` -> `Bicicleta De Montaña` (añade espacios)."""
    return re.sub(r"(?<=[a-záéíóúñ])(?=[A-ZÁÉÍÓÚÑ])", " ", text or "")


def _entity_variants(entity: str) -> list[str]:
    """Variantes normalizadas para buscar la entidad en el markdown.

    Devuelve siempre la versión "con espacios" (tras `_split_camel_case`)
    y, como respaldo para IDs pegados estilo `bicicletademontana`, la
    versión sin espacios.
    """
    if not entity:
        return []
    spaced = _normalize_entity(_split_camel_case(entity))
    no_space = spaced.replace(" ", "")
    out: list[str] = []
    if spaced:
        out.append(spaced)
    if no_space and no_space != spaced:
        out.append(no_space)
    return out


def _entity_in_markdown(entity: str, md_spaced: str, md_compact: str) -> bool:
    """True si alguna variante normalizada de `entity` aparece en el MD.

    `md_spaced` es el markdown normalizado con espacios; `md_compact` es
    el mismo sin espacios. Pasamos los dos pre-normalizados para evitar
    recomputarlos por cada relación.
    """
    variants = _entity_variants(entity)
    if not variants:
        return False
    for v in variants:
        if " " in v:
            if v in md_spaced:
                return True
        elif v in md_compact:
            return True
    return False


def _prune_ungrounded_examples(
    kb: KnowledgeBase, markdown: str
) -> KnowledgeBase:
    """Descarta `Example` cuyo `name` no aparezca en el MD; filtra atributos
    y métodos que tampoco aparezcan.

    Mismo contrato que `_prune_ungrounded_relations` pero para la categoría
    `Example`. Es crítica porque los modelos grandes (`qwen2.5:14b`,
    `gemma3:12b`) tienden a inventar ejemplos enteros ("clase Vehículos",
    "clase Automóvil") con atributos/métodos inexistentes
    (`tipo_combustible`, `adaptarSuspension`, `calcularVelocidadMaxima`).

    Cuando el `name` sí aparece pero algunos `attributes`/`methods` no, se
    filtran individualmente. Si el Example se queda sin atributos ni
    métodos, se conserva (su `description` puede ser útil para bullets
    genéricos) salvo que la `description` tampoco aparezca: decisión
    conservadora, preferimos preservar si hay dudas.
    """
    if not kb.examples:
        return kb
    md_spaced = _normalize_entity(markdown or "")
    md_compact = md_spaced.replace(" ", "")
    if not md_spaced:
        return kb

    kept: list[Example] = []
    dropped: list[tuple[str, str]] = []
    attr_reports: list[tuple[str, list[str], list[str]]] = []
    for ex in kb.examples:
        if not _entity_in_markdown(ex.name, md_spaced, md_compact):
            dropped.append((ex.id, ex.name[:60]))
            continue
        new_attrs = [
            a for a in ex.attributes
            if _entity_in_markdown(a, md_spaced, md_compact)
        ]
        new_methods = [
            m for m in ex.methods
            if _entity_in_markdown(m, md_spaced, md_compact)
        ]
        removed_attrs = [a for a in ex.attributes if a not in new_attrs]
        removed_methods = [m for m in ex.methods if m not in new_methods]
        if removed_attrs or removed_methods:
            attr_reports.append((ex.name, removed_attrs, removed_methods))
        kept.append(
            ex.model_copy(update={"attributes": new_attrs, "methods": new_methods})
        )

    if dropped:
        logger.warning(
            "KB: %d ejemplos descartados por no aparecer en el PDF: %s",
            len(dropped),
            dropped[:5],
        )
    if attr_reports:
        for name, attrs, methods in attr_reports[:5]:
            logger.info(
                "KB: Example '%s' — filtrados %d atributos (%s) y %d métodos (%s) "
                "ausentes del PDF.",
                name,
                len(attrs),
                attrs[:5],
                len(methods),
                methods[:5],
            )
    if dropped or attr_reports:
        return kb.model_copy(update={"examples": kept})
    return kb


def _prune_ungrounded_relations(
    kb: KnowledgeBase, markdown: str
) -> KnowledgeBase:
    """Descarta `Relation` cuyo `source` o `target` no aparecen en el MD.

    No toca el resto de la KB. Si todas las relaciones se descartan, la
    KB resultante simplemente no tendrá relaciones (las slides y el quiz
    seguirán funcionando con las otras categorías de átomos).
    """
    if not kb.relations:
        return kb
    md_spaced = _normalize_entity(markdown or "")
    md_compact = md_spaced.replace(" ", "")
    if not md_spaced:
        return kb

    kept: list[Relation] = []
    dropped: list[tuple[str, str]] = []
    for rel in kb.relations:
        src_ok = _entity_in_markdown(rel.source, md_spaced, md_compact)
        tgt_ok = _entity_in_markdown(rel.target, md_spaced, md_compact)
        if src_ok and tgt_ok:
            kept.append(rel)
        else:
            missing = []
            if not src_ok:
                missing.append(f"source='{rel.source}'")
            if not tgt_ok:
                missing.append(f"target='{rel.target}'")
            dropped.append((rel.id, ", ".join(missing)))

    if dropped:
        logger.warning(
            "KB: %d relaciones descartadas por no aparecer literalmente en "
            "el PDF: %s",
            len(dropped),
            dropped[:5],
        )
        return kb.model_copy(update={"relations": kept})
    return kb


def _reduce_to_kb(
    client: OllamaClient,
    partials: list[str],
    literal_hints: LiteralHints,
    progress_cb: ProgressCallback | None,
    source_markdown: str,
) -> KnowledgeBase:
    """Consolida los resúmenes parciales + hints literales en una KB JSON.

    Tres capas de resiliencia, mismo patrón que `plan_slides`:
    1. Coerción defensiva de claves (ES/EN, anidamientos habituales).
    2. Reintento único con temperatura más baja si la 1ª no es válida.
    3. Fallback determinístico desde `literal_hints` + Markdown si el LLM
       sigue sin entregar un JSON utilizable. Garantiza que la generación
       nunca se detenga por un modelo pequeño poco cooperativo.
    """
    if progress_cb:
        progress_cb("reduce", 1, 1, "Construyendo Base de Conocimiento…")

    prompt = REDUCE_TO_KB_PROMPT.format(
        partials=_join_partials(partials),
        literal_hints=(literal_hints.to_prompt_block() or "(sin material literal)"),
    )
    fallback_topic = _infer_main_topic(source_markdown, literal_hints)

    raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2)
    kb, reason = _try_build_kb(raw, fallback_topic)

    if kb is None:
        logger.warning(
            "KB inicial inválida (%s). Reintentando con temperatura 0.1…",
            reason,
        )
        raw_retry = client.generate_json(
            prompt, system=SYSTEM_EXPERT_ES, temperature=0.1
        )
        kb, reason = _try_build_kb(raw_retry, fallback_topic)

    if kb is None:
        logger.warning(
            "KB tampoco válida tras reintento (%s). "
            "Se usa fallback determinístico desde hints literales.",
            reason,
        )
        kb = build_fallback_kb(source_markdown, literal_hints)

    if kb.atom_count == 0:
        # La KB es válida pero sin átomos: el LLM devolvió algo vacío y
        # aceptable por schema. Complementamos con el fallback.
        logger.warning(
            "KB válida pero sin átomos. Complementando con fallback determinístico."
        )
        fb = build_fallback_kb(source_markdown, literal_hints)
        kb = KnowledgeBase(
            main_topic=kb.main_topic or fb.main_topic,
            subtopics=kb.subtopics or fb.subtopics,
            definitions=fb.definitions,
            examples=kb.examples,
            formulas_code=fb.formulas_code,
            numeric_data=kb.numeric_data,
            relations=kb.relations,
            conclusions=kb.conclusions,
        )

    # Verificación literal: descarta relaciones y ejemplos cuyas entidades
    # o nombres no aparecen en el PDF (anti-hallucination de la fase REDUCE).
    # Se filtran también atributos/métodos concretos de los Examples.
    kb = _prune_ungrounded_relations(kb, source_markdown)
    kb = _prune_ungrounded_examples(kb, source_markdown)

    logger.info(
        "KB construida: topic='%s', %d subtopics, %d átomos "
        "(defs=%d, ex=%d, fc=%d, dt=%d, rel=%d)",
        kb.main_topic,
        len(kb.subtopics),
        kb.atom_count,
        len(kb.definitions),
        len(kb.examples),
        len(kb.formulas_code),
        len(kb.numeric_data),
        len(kb.relations),
    )
    return kb


def build_knowledge_base(
    client: OllamaClient,
    markdown: str,
    progress_cb: ProgressCallback | None = None,
) -> KnowledgeBase:
    """Pipeline Map-Reduce con salida estructurada (JSON).

    Además del LLM, aplica extractores deterministas para inyectar material
    literal del PDF (definiciones, código, fórmulas) como anclaje del REDUCE.
    """
    if not markdown.strip():
        raise GenerationError("El documento está vacío; no se puede construir la KB.")

    chunks = split_markdown(markdown)
    if not chunks:
        raise GenerationError("No se pudo dividir el documento en chunks.")

    literal_hints = extract_literal_hints(markdown)
    partials = _map_phase(client, chunks, progress_cb)
    return _reduce_to_kb(
        client, partials, literal_hints, progress_cb, source_markdown=markdown
    )


def consolidate_document(
    client: OllamaClient,
    markdown: str,
    progress_cb: ProgressCallback | None = None,
) -> str:
    """Shim de compatibilidad: devuelve la KB renderizada como Markdown.

    El resto del pipeline (quiz, presentación) aún espera texto plano como
    `summary`. Esta función construye la KB estructurada y la serializa
    a Markdown legible para mantener compatibilidad hasta que esos
    consumidores migren a `KnowledgeBase`.
    """
    kb = build_knowledge_base(client, markdown, progress_cb=progress_cb)
    return kb.to_markdown()
