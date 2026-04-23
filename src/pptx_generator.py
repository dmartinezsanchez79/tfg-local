"""Generación de presentaciones PPTX desde la KnowledgeBase.

Pipeline (v2, por átomos):
1. `plan_slides`     : el LLM asigna átomos de la KB a slides tipadas.
2. `_render_slide_bullets`: una llamada por slide, prompt especializado
   según el `kind` de la slide (definition / example / comparison / …).
3. `_render_conclusion`: conclusión global desde la KB.
4. `render_pptx`     : montaje determinista usando la plantilla.

El contrato público (`PresentationPlan`, `generate_presentation`) se
mantiene estable para no romper la UI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.util import Pt
from pydantic import BaseModel, Field, ValidationError

from .config import (
    DEFAULT_NUM_SLIDES_MAX,
    DEFAULT_NUM_SLIDES_MIN,
    LAYOUT_CONTENT,
    LAYOUT_TITLE,
    MAX_BULLETS_PER_SLIDE,
    MAX_CHARS_PER_BULLET,
    MAX_CHARS_SLIDE_TITLE,
    TEMPLATE_PATH,
)
from .exceptions import GenerationError, TemplateError
from .knowledge_base import (
    Definition,
    Example,
    FormulaOrCode,
    KnowledgeBase,
    NumericDatum,
    Relation,
    relation_to_natural,
)
from .map_reduce import ProgressCallback
from .ollama_client import OllamaClient
from .plans import (
    PlannedSlide,
    SlidePlan,
    build_fallback_slide_plan,
    coerce_slide_plan_payload,
    sanitize_slide_plan,
)
from .prompts import (
    CONCLUSION_FROM_KB_PROMPT,
    SLIDE_BULLETS_FROM_ATOMS_PROMPT,
    SLIDE_PLAN_PROMPT,
    SYSTEM_EXPERT_ES,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------- modelos de salida ---


class SlideBullets(BaseModel):
    bullets: list[str] = Field(min_length=1, max_length=MAX_BULLETS_PER_SLIDE + 2)


@dataclass
class BuiltSlide:
    """Resultado de una slide ya redactada, listo para el renderer."""

    title: str
    bullets: list[str]
    kind: str = "outlook"


@dataclass
class PresentationPlan:
    title: str
    outline: list[str]
    slides: list[BuiltSlide] = field(default_factory=list)
    conclusion: list[str] = field(default_factory=list)


# -------------------------------------------------- utilidades de texto --


# Prefijos de átomo que a veces se cuelan porque el LLM copia el bloque de
# contexto en lugar de redactar. Si un bullet empieza por uno de estos, lo
# descartamos en la limpieza.
_ATOM_ID_PREFIX_RE = re.compile(r"^\s*(def|ex|fc|rel|dt):", re.IGNORECASE)

# Patrón típico del bloque de átomos que pasamos como contexto, p.ej.
# "… · definición · …" o "… · ejemplo · …". También es señal de copia.
_ATOM_BLOCK_TOKEN_RE = re.compile(
    r"·\s*(definición|definicion|ejemplo|dato|relación|relacion|fórmula|formula|code|código|codigo)\s*·",
    re.IGNORECASE,
)

# Aperturas meta que anulan el valor del bullet.
_META_OPENINGS = (
    "en esta diapositiva",
    "en este apartado",
    "en esta presentacion",
    "en esta presentación",
    "a continuacion",
    "a continuación",
    "se habla de",
    "se trata de",
    "es importante destacar",
    "visión panorámica",
    "vision panoramica",
)

# Fragmentos meta que pueden aparecer en CUALQUIER posición del bullet y que
# indican contenido didáctico vacío (no aportan hecho). Case-insensitive.
_META_EMBEDDED_PHRASES = (
    "sirve de ejemplo para ilustrar",
    "sirve para ilustrar",
    "facilita la creación y gestión",
    "facilita la creacion y gestion",
    "facilita la comprensión de",
    "facilita la comprension de",
    "ayuda a comprender",
    "permite entender",
    "permite comprender",
    "es clave para entender",
    "es clave para comprender",
    "ilustra cómo",
    "ilustra como",
)

# Anglicismos innecesarios y neologismos raros que degradan la calidad del
# bullet. Si aparecen, descartamos el bullet (el crítico lo marcará como
# deficiente y se regenerará).
_ANGLICISM_RE = re.compile(
    r"\b(blueprint|inheritance|overriding|concretada|concretado)\b",
    re.IGNORECASE,
)

# Notación INTERNA del KB para relaciones (tripleta con flecha y corchetes)
# que nunca debe llegar al bullet. Cubre varias variantes tipográficas.
_RELATION_ARROW_RE = re.compile(
    r"(?:—|--|-|–)\s*\[[^\]]+\]\s*(?:→|->|—>|–>)",
)


def _bullet_echoes_title(bullet: str, slide_title: str) -> bool:
    """Detecta cuando un bullet arranca repitiendo el título de la slide.

    Ejemplo:
      title  = "Polimorfismo"
      bullet = "Polimorfismo: capacidad de un objeto comportarse…"
      → True (el bullet debe descartarse o reescribirse).

    El matching es permisivo: normaliza espacios y mayúsculas, y acepta
    separadores típicos (`:`, `.`, `—`, `-`, `,`).
    """
    if not bullet or not slide_title:
        return False
    title_norm = " ".join(slide_title.split()).strip().lower()
    bullet_norm = " ".join(bullet.split()).strip().lower()
    if not title_norm or not bullet_norm:
        return False
    # Exige que el título tenga al menos 2 palabras o ≥6 caracteres para
    # evitar falsos positivos con términos muy cortos.
    if len(title_norm) < 6 and len(title_norm.split()) < 2:
        return False
    if not bullet_norm.startswith(title_norm):
        return False
    tail = bullet_norm[len(title_norm) :].lstrip()
    # Si el título es TODO el bullet, es ruido.
    if not tail:
        return True
    # Si tras el título hay un separador típico, es eco.
    return tail[:1] in {":", ".", "—", "–", "-", ",", "·"}


def _truncate_title(text: str, max_len: int) -> str:
    """Recorte duro (con `…`) solo para títulos de slide."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _trim_by_words(text: str, max_len: int) -> str:
    """Recorta por palabras SIN puntos suspensivos.

    El bullet queda como frase cerrada aunque implique perder las últimas
    palabras. Es preferible a mostrar `…` cortando mitad de palabra.
    """
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.5:
        cut = cut[:last_space]
    cut = cut.rstrip(",;:·-—(").rstrip()
    # Asegurar que termina en signo de cierre natural; si no, añadimos "."
    if cut and cut[-1] not in ".!?)":
        cut += "."
    return cut


def _clean_bullet(
    text: str, max_len: int, slide_title: str | None = None
) -> str | None:
    """Normaliza y valida un bullet generado por el LLM.

    Devuelve la versión limpia, o `None` si el bullet debe descartarse
    (copia literal del contexto, lenguaje meta, vacío…). El revisor se
    encargará de marcar la slide como deficiente si quedan pocos bullets.

    Si se pasa `slide_title`, también se descartan bullets que repiten el
    título de la slide como prefijo (`Polimorfismo: capacidad de…`).
    """
    if not isinstance(text, str):
        return None
    t = " ".join(text.split()).strip()
    if not t:
        return None

    if "…" in t or "..." in t:
        return None

    if _ATOM_ID_PREFIX_RE.search(t):
        return None
    if _ATOM_BLOCK_TOKEN_RE.search(t):
        return None

    if _RELATION_ARROW_RE.search(t):
        return None

    if _ANGLICISM_RE.search(t):
        return None

    low = t.lower()
    if any(low.startswith(m) for m in _META_OPENINGS):
        return None
    if any(p in low for p in _META_EMBEDDED_PHRASES):
        return None

    if slide_title and _bullet_echoes_title(t, slide_title):
        return None

    if len(t) < 25:
        return None

    return _trim_by_words(t, max_len)


def _atom_block(
    atom: Definition | Example | FormulaOrCode | NumericDatum | Relation,
) -> str:
    """Renderiza un átomo como bloque Markdown compacto para prompts."""
    if isinstance(atom, Definition):
        tag = " (literal)" if atom.verbatim else ""
        return (
            f"- `{atom.id}` · definición{tag} · **{atom.term}**: {atom.definition}"
        )
    if isinstance(atom, Example):
        attrs = ", ".join(atom.attributes) or "—"
        methods = ", ".join(atom.methods) or "—"
        return (
            f"- `{atom.id}` · ejemplo · **{atom.name}**: {atom.description}\n"
            f"  - atributos: {attrs}\n"
            f"  - métodos: {methods}"
        )
    if isinstance(atom, FormulaOrCode):
        caption = atom.caption or "—"
        lang = atom.language or ""
        return (
            f"- `{atom.id}` · {atom.kind} · {caption}\n"
            f"  ```{lang}\n  {atom.content}\n  ```"
        )
    if isinstance(atom, NumericDatum):
        return f"- `{atom.id}` · dato · **{atom.value}**: {atom.description}"
    if isinstance(atom, Relation):
        # Presentamos la relación en ESPAÑOL NATURAL para que el LLM la
        # reescriba sin copiar la notación técnica `X —[kind]→ Y`.
        return f"- `{atom.id}` · relación · {relation_to_natural(atom)}"
    return f"- `{getattr(atom, 'id', '?')}` · (tipo desconocido)"


def _atoms_for_slide(kb: KnowledgeBase, slide: PlannedSlide) -> str:
    if not slide.atom_ids:
        return "(sin átomos asignados; genera bullets panorámicos coherentes con el tipo)"
    parts: list[str] = []
    for aid in slide.atom_ids:
        atom = kb.get_atom(aid)
        if atom is None:
            logger.warning("Átomo '%s' no existe en KB durante render.", aid)
            continue
        parts.append(_atom_block(atom))
    return "\n".join(parts) or "(sin átomos válidos)"


def _outline_lines(plan: SlidePlan) -> str:
    return "\n".join(f"- {s.title}" for s in plan.slides)


# ------------------------------------------------- planificación slides --


def _try_build_slide_plan(
    raw: Any, kb: KnowledgeBase
) -> tuple[SlidePlan | None, str | None]:
    """Coerciona y valida una respuesta cruda del LLM. Nunca lanza.

    Devuelve `(plan, None)` si la construcción tiene éxito, o
    `(None, motivo)` si falla (para logging/telemetría).
    """
    data = coerce_slide_plan_payload(raw, kb)
    if data is None:
        return None, "estructura irrecuperable tras coerción"
    try:
        return SlidePlan(**data), None
    except ValidationError as exc:
        return None, f"ValidationError: {exc.errors()[:2]}"


def plan_slides(client: OllamaClient, kb: KnowledgeBase) -> SlidePlan:
    """Pide al LLM el `SlidePlan` con tres capas de resiliencia.

    1. **Coerción defensiva** de claves y nombres (español, mayúsculas…).
    2. **Reintento único** con temperatura más baja si la 1ª respuesta no
       produce un plan válido (el coste extra es ≤1 llamada).
    3. **Fallback determinístico** construido directamente desde la KB si
       el LLM sigue sin entregar un JSON utilizable. Así, la generación
       nunca se detiene por un modelo pequeño poco cooperativo.
    """
    if kb.atom_count == 0:
        raise GenerationError(
            "La KB no contiene átomos; no se puede planificar la presentación."
        )

    prompt = SLIDE_PLAN_PROMPT.format(kb_context=kb.to_prompt_context(max_chars=6000))

    raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.2)
    plan, reason = _try_build_slide_plan(raw, kb)

    if plan is None:
        logger.warning(
            "SlidePlan inicial inválido (%s). Reintentando con temperatura 0.1…",
            reason,
        )
        raw_retry = client.generate_json(
            prompt, system=SYSTEM_EXPERT_ES, temperature=0.1
        )
        plan, reason = _try_build_slide_plan(raw_retry, kb)

    if plan is None:
        logger.warning(
            "SlidePlan tampoco válido tras reintento (%s). "
            "Se usa fallback determinístico desde la KB.",
            reason,
        )
        plan = build_fallback_slide_plan(kb)

    plan = sanitize_slide_plan(plan, kb)
    if len(plan.slides) < DEFAULT_NUM_SLIDES_MIN:
        # El saneado pudo dejar el plan por debajo del mínimo (e.g. todos
        # los atom_ids eran alucinaciones). Fusionamos con el fallback.
        logger.warning(
            "SlidePlan quedó con %d slides tras saneado (<%d). "
            "Complementando con fallback.",
            len(plan.slides),
            DEFAULT_NUM_SLIDES_MIN,
        )
        fb = build_fallback_slide_plan(kb)
        existing_titles = {s.title.lower() for s in plan.slides}
        merged = list(plan.slides)
        for s in fb.slides:
            if s.title.lower() not in existing_titles and len(merged) < 20:
                merged.append(s)
                existing_titles.add(s.title.lower())
        plan = SlidePlan(
            presentation_title=plan.presentation_title, slides=merged
        )
    # Recortar a rango razonable.
    if len(plan.slides) > DEFAULT_NUM_SLIDES_MAX:
        plan = SlidePlan(
            presentation_title=plan.presentation_title,
            slides=plan.slides[:DEFAULT_NUM_SLIDES_MAX],
        )
    if len(plan.slides) < DEFAULT_NUM_SLIDES_MIN:
        logger.warning(
            "SlidePlan con %d slides (< mínimo %d). Continuando.",
            len(plan.slides),
            DEFAULT_NUM_SLIDES_MIN,
        )
    plan.presentation_title = _truncate_title(plan.presentation_title, 120)
    for s in plan.slides:
        s.title = _truncate_title(s.title, MAX_CHARS_SLIDE_TITLE)
    logger.info(
        "SlidePlan: título='%s', %d slides", plan.presentation_title, len(plan.slides)
    )
    return plan


# ------------------------------------------------ generación de bullets --


def render_slide_bullets(
    client: OllamaClient,
    kb: KnowledgeBase,
    slide: PlannedSlide,
    plan: SlidePlan,
    index: int,
    total: int,
) -> list[str]:
    prompt = SLIDE_BULLETS_FROM_ATOMS_PROMPT.format(
        presentation_title=plan.presentation_title,
        index=index,
        total=total,
        slide_title=slide.title,
        kind=slide.kind,
        focus=(slide.focus or "—"),
        atom_details=_atoms_for_slide(kb, slide),
        outline=_outline_lines(plan),
    )
    raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.3)
    if not isinstance(raw, dict):
        raise GenerationError(
            f"Bullets inválidos para slide '{slide.title}' (no es JSON objeto)."
        )
    try:
        sb = SlideBullets(**raw)
    except ValidationError as exc:
        raise GenerationError(
            f"Bullets con estructura inválida ({slide.title}): {exc.errors()[:1]}"
        ) from exc

    out: list[str] = []
    for b in sb.bullets:
        cleaned = _clean_bullet(b, MAX_CHARS_PER_BULLET, slide_title=slide.title)
        if cleaned:
            out.append(cleaned)
    if not out:
        # Si todos los bullets fueron descartados (copias del contexto,
        # meta, puntos suspensivos…), forzamos un fallo controlado para
        # que el revisor lo marque como crítico y se regenere.
        raise GenerationError(
            f"Bullets descartados por validación de calidad en slide '{slide.title}'."
        )
    return out[:MAX_BULLETS_PER_SLIDE]


def _render_conclusion(
    client: OllamaClient, kb: KnowledgeBase, presentation_title: str
) -> list[str]:
    prompt = CONCLUSION_FROM_KB_PROMPT.format(
        presentation_title=presentation_title,
        kb_context=kb.to_prompt_context(max_chars=4000),
    )
    raw = client.generate_json(prompt, system=SYSTEM_EXPERT_ES, temperature=0.3)
    if not isinstance(raw, dict):
        raise GenerationError("Conclusión con formato inválido (no es JSON objeto).")
    try:
        sb = SlideBullets(**raw)
    except ValidationError as exc:
        raise GenerationError(f"Conclusión inválida: {exc.errors()[:1]}") from exc
    cleaned = [
        _clean_bullet(b, MAX_CHARS_PER_BULLET, slide_title="Conclusiones")
        for b in sb.bullets
    ]
    return [c for c in cleaned if c][:MAX_BULLETS_PER_SLIDE]


# --------------------------------------- construcción determinista PPTX --


def _set_title(shape: Any, text: str) -> None:
    if shape is None or not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.clear()
    tf.text = text


def _replace_text_preserving_format(shape: Any, new_text: str) -> bool:
    """Sustituye el texto de un shape conservando el formato del primer run.

    `python-pptx` no ofrece API directa para esto en shapes que NO son
    placeholders (shapes de texto libre, típicos de plantillas importadas
    desde Google Slides). Si llamamos a `tf.clear()` + `tf.text = "..."`,
    perdemos fuente/tamaño/color que la plantilla ya había definido.

    Estrategia:
    1. Tomamos el primer run del primer paragraph como "run maestro"
       (lleva el formato deseado: font.name, size, bold, color…).
    2. Sustituimos su texto por `new_text`.
    3. Vaciamos el resto de runs y paragraphs del shape para que no
       arrastren fragmentos de texto antiguo.

    Devuelve True si la sustitución tuvo éxito.
    """
    if shape is None or not shape.has_text_frame:
        return False
    tf = shape.text_frame
    if not tf.paragraphs:
        tf.text = new_text
        return True

    first_paragraph = tf.paragraphs[0]
    runs = list(first_paragraph.runs)
    if runs:
        runs[0].text = new_text
        for extra in runs[1:]:
            extra.text = ""
    else:
        first_paragraph.text = new_text

    for extra_p in list(tf.paragraphs[1:]):
        # Vaciar sus runs sin borrar el paragraph para no tocar estructura XML.
        for r in list(extra_p.runs):
            r.text = ""
    return True


def _iter_non_placeholder_text_shapes(slide: Any) -> list[Any]:
    """Lista shapes de la slide que tienen texto y NO son placeholders."""
    return [
        sh
        for sh in slide.shapes
        if getattr(sh, "has_text_frame", False) and not sh.is_placeholder
    ]


def _normalized_text(shape: Any) -> str:
    """Texto del shape en minúsculas y sin espacios redundantes."""
    if not shape.has_text_frame:
        return ""
    return " ".join(shape.text_frame.text.split()).strip().lower()


def _shape_font_size_pt(shape: Any) -> float:
    """Aproximación del tamaño de fuente máximo de un shape (en pt).

    Útil como heurística para distinguir "el shape del título" (fuente
    grande) de "el shape del subtítulo" (fuente pequeña) cuando no hay
    etiquetas o placeholders identificables.
    """
    if not shape.has_text_frame:
        return 0.0
    max_pt = 0.0
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if r.font.size is not None:
                max_pt = max(max_pt, r.font.size.pt)
    return max_pt


def _overwrite_template_cover(
    slide: Any, title: str, subtitle: str | None = None
) -> None:
    """Sustituye el texto de la portada de plantilla conservando su formato.

    La portada que viene con la plantilla universitaria contiene shapes de
    texto libre con marcadores como "TÍTULO" y "SUBTÍTULO". Este helper
    los identifica y reemplaza por el título real + subtítulo, preservando
    tipografía, tamaño y color definidos en la plantilla.

    Identificación en dos pasos:
    1. Por texto existente (`título`, `titulo`, `title` → título;
       `subtítulo`, `subtitulo`, `subtitle` → subtítulo).
    2. Fallback por tamaño de fuente: el shape con la fuente más grande
       se trata como título, el segundo más grande como subtítulo.
    """
    shapes = _iter_non_placeholder_text_shapes(slide)
    if not shapes:
        logger.warning("Portada de plantilla sin shapes de texto editables.")
        return

    title_markers = {"titulo", "título", "title"}
    subtitle_markers = {"subtitulo", "subtítulo", "subtitle"}

    title_shape: Any | None = None
    subtitle_shape: Any | None = None

    for sh in shapes:
        text = _normalized_text(sh)
        if not title_shape and text in title_markers:
            title_shape = sh
        elif not subtitle_shape and text in subtitle_markers:
            subtitle_shape = sh

    if title_shape is None or (subtitle and subtitle_shape is None):
        by_size = sorted(shapes, key=_shape_font_size_pt, reverse=True)
        if title_shape is None and by_size:
            title_shape = by_size[0]
        if subtitle and subtitle_shape is None:
            for sh in by_size:
                if sh is not title_shape:
                    subtitle_shape = sh
                    break

    if title_shape is not None:
        _replace_text_preserving_format(title_shape, title)
    else:
        logger.warning("No se pudo localizar el shape del título en la portada.")

    if subtitle and subtitle_shape is not None:
        _replace_text_preserving_format(subtitle_shape, subtitle)


def _overwrite_template_body_slide(
    slide: Any, title: str, bullets: list[str]
) -> bool:
    """Rellena una slide precargada con layout TITLE_AND_BODY.

    Devuelve True si logró escribir título + bullets. Si el layout no
    tiene los placeholders esperados, devuelve False y el caller deberá
    caer al flujo normal (añadir una slide nueva con el layout de
    contenido).
    """
    title_ph = slide.shapes.title
    body_ph = _find_placeholder(slide, (1, 2, 13, 14))
    if title_ph is None or body_ph is None:
        return False
    _set_title(title_ph, title)
    _set_bullets(body_ph, bullets)
    return True


def _set_bullets(shape: Any, bullets: list[str]) -> None:
    if shape is None or not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    try:
        tf.auto_size = 1  # MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    except Exception:  # noqa: BLE001 — algunas plantillas no lo admiten
        pass

    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        for run in p.runs:
            if run.font.size is None:
                run.font.size = Pt(20)


def _find_placeholder(slide: Any, idx_candidates: tuple[int, ...]) -> Any | None:
    by_idx: dict[int, Any] = {p.placeholder_format.idx: p for p in slide.placeholders}
    for idx in idx_candidates:
        if idx in by_idx:
            return by_idx[idx]
    for p in slide.placeholders:
        if p.placeholder_format.idx != 0:
            return p
    return None


def _add_title_slide(prs: Any, title: str, subtitle: str | None = None) -> None:
    if LAYOUT_TITLE >= len(prs.slide_layouts):
        raise TemplateError(
            f"La plantilla no tiene el layout {LAYOUT_TITLE} (Portada)."
        )
    layout = prs.slide_layouts[LAYOUT_TITLE]
    slide = prs.slides.add_slide(layout)
    _set_title(slide.shapes.title, title)
    if subtitle:
        placeholder = _find_placeholder(slide, (1, 2, 10, 11))
        if placeholder is not None:
            _set_bullets(placeholder, [subtitle])


def _add_content_slide(prs: Any, title: str, bullets: list[str]) -> None:
    if LAYOUT_CONTENT >= len(prs.slide_layouts):
        raise TemplateError(
            f"La plantilla no tiene el layout {LAYOUT_CONTENT} (Contenido)."
        )
    layout = prs.slide_layouts[LAYOUT_CONTENT]
    slide = prs.slides.add_slide(layout)
    _set_title(slide.shapes.title, title)
    body = _find_placeholder(slide, (1, 2, 13, 14))
    if body is None:
        raise TemplateError(
            "El layout de contenido no tiene un placeholder para el cuerpo."
        )
    _set_bullets(body, bullets)


def _load_template() -> Any:
    if not TEMPLATE_PATH.exists():
        raise TemplateError(
            f"No se encuentra la plantilla en {TEMPLATE_PATH}. "
            "Debe existir 'plantilla_universidad.pptx' en la raíz del proyecto."
        )
    try:
        return Presentation(str(TEMPLATE_PATH))
    except Exception as exc:
        raise TemplateError(f"No se pudo abrir la plantilla PPTX: {exc}") from exc


# --------------------------------------------------------- API pública ---


def build_plan(
    client: OllamaClient,
    kb: KnowledgeBase,
    progress_cb: ProgressCallback | None = None,
    *,
    refine: bool = True,
    max_refine_iterations: int = 1,
    use_llm_critic: bool = False,
    source_markdown: str | None = None,
) -> PresentationPlan:
    """Planifica slides con átomos, genera bullets por slide, refina y concluye.

    Si `refine=True`, al terminar la generación se ejecuta un revisor
    (determinista + LLM) que **regenera solo las slides problemáticas**
    (duplicados, lenguaje meta, bullets poco densos o no anclados a la KB),
    conservando las demás. Coste extra: nº de slides críticas + 1 llamada
    de revisor.
    """
    if progress_cb:
        progress_cb("slides", 0, 0, "Planificando la presentación…")
    plan = plan_slides(client, kb)

    total = len(plan.slides)
    built: list[BuiltSlide] = []
    for i, planned_slide in enumerate(plan.slides, start=1):
        if progress_cb:
            progress_cb(
                "slides",
                i,
                total,
                f"Redactando slide {i}/{total} ({planned_slide.kind}): {planned_slide.title}",
            )
        try:
            bullets = render_slide_bullets(
                client, kb, planned_slide, plan, index=i, total=total
            )
        except GenerationError as exc:
            logger.warning("Fallo al generar slide '%s': %s", planned_slide.title, exc)
            bullets = []
        built.append(
            BuiltSlide(
                title=planned_slide.title,
                bullets=bullets,
                kind=planned_slide.kind,
            )
        )

    if refine:
        if progress_cb:
            progress_cb("slides", total, total, "Revisando calidad y puliendo slides…")
        # Import diferido para evitar ciclo con `critics`.
        from .critics import refine_slides as _refine_slides

        built, review = _refine_slides(
            client,
            kb,
            built,
            plan,
            max_iterations=max_refine_iterations,
            use_llm=use_llm_critic,
            source_markdown=source_markdown,
        )
        if review.issues:
            logger.info(
                "Revisor slides: %d issues tras refinamiento (críticos: %d).",
                len(review.issues),
                len(review.critical_indices()),
            )

    if progress_cb:
        progress_cb("slides", total, total, "Generando conclusiones…")
    conclusion = _render_conclusion(client, kb, plan.presentation_title)

    return PresentationPlan(
        title=plan.presentation_title,
        outline=[s.title for s in plan.slides],
        slides=built,
        conclusion=conclusion,
    )


def render_pptx(plan: PresentationPlan, output_path: Path | None = None) -> bytes:
    """Monta el PPTX a partir del plan ya redactado.

    Reusa las slides precargadas de la plantilla (portada + primera slide
    de contenido) en lugar de añadir slides detrás: así respetamos el
    diseño institucional de la plantilla (tipografías, colores, logos) y
    evitamos arrastrar slides de demostración en blanco.

    Contrato con la plantilla:
    - Slide 0 → portada. Contiene shapes de texto libre con marcadores
      ("TÍTULO", "SUBTÍTULO"). Se sobreescriben preservando el formato.
    - Slide 1 (si existe y tiene TITLE + BODY placeholders) → índice.
    - Resto → se añade detrás con los layouts `LAYOUT_CONTENT`.
    """
    prs = _load_template()

    slides_ok = [s for s in plan.slides if s.bullets]
    for omitted in (s for s in plan.slides if not s.bullets):
        logger.warning("Slide '%s' sin bullets; se omite en el PPTX.", omitted.title)

    index_bullets = [
        f"{i}. {s.title}" for i, s in enumerate(slides_ok, start=1)
    ][: MAX_BULLETS_PER_SLIDE * 2]

    preloaded = list(prs.slides)

    # ----- Portada (slide 0 de la plantilla) ------------------------------
    if preloaded:
        _overwrite_template_cover(
            preloaded[0],
            plan.title,
            subtitle="Presentación generada automáticamente",
        )
    else:
        _add_title_slide(
            prs, plan.title, subtitle="Presentación generada automáticamente"
        )

    # ----- Índice (slide 1 de la plantilla si es TITLE_AND_BODY) ----------
    index_written = False
    if len(preloaded) >= 2:
        index_written = _overwrite_template_body_slide(
            preloaded[1], "Índice", index_bullets
        )
        if not index_written:
            logger.warning(
                "La slide 2 de la plantilla no tiene placeholders TITLE+BODY; "
                "se añade el índice como slide nueva."
            )
    if not index_written:
        _add_content_slide(prs, "Índice", index_bullets)

    # ----- Contenido + conclusiones (siempre añadidas) --------------------
    for slide in slides_ok:
        _add_content_slide(prs, slide.title, slide.bullets)

    if plan.conclusion:
        _add_content_slide(prs, "Conclusiones", plan.conclusion)

    buffer = BytesIO()
    prs.save(buffer)
    data = buffer.getvalue()
    if output_path is not None:
        output_path.write_bytes(data)
    return data


def generate_presentation(
    client: OllamaClient,
    kb: KnowledgeBase,
    progress_cb: ProgressCallback | None = None,
    output_path: Path | None = None,
    source_markdown: str | None = None,
) -> tuple[bytes, PresentationPlan]:
    """Pipeline completo: KB -> plan -> bullets -> PPTX bytes."""
    plan = build_plan(
        client,
        kb,
        progress_cb=progress_cb,
        source_markdown=source_markdown,
    )
    data = render_pptx(plan, output_path=output_path)
    return data, plan
