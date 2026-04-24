"""Aplicación Streamlit — Quiz + Presentación 100% offline desde un PDF.

Uso:
    streamlit run app.py

Requisitos:
    - Ollama corriendo localmente (`ollama serve`).
    - El modelo seleccionado descargado (`ollama pull qwen2.5`, etc.).
    - Fichero `plantilla_universidad.pptx` en la raíz.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import streamlit as st

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_NUM_QUESTIONS_RANGE,
    MIN_NUM_QUESTIONS,
    MAX_INPUT_PAGES,
    MAX_NUM_QUESTIONS,
    TEMPLATE_PATH,
)
from src.exceptions import (
    GenerationError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    PDFError,
    PDFTooLargeError,
    ScannedPDFError,
    TemplateError,
)
from src.knowledge_base import KnowledgeBase
from src.map_reduce import build_knowledge_base
from src.ollama_client import OllamaClient
from src.pdf_processor import ProcessedPDF, process_pdf_from_upload
from src.pptx_generator import generate_presentation
from src.quiz_generator import Quiz, generate_quiz
from src.quiz_pdf_exporter import quiz_to_pdf_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="Generador offline Quiz + PPTX",
    page_icon=":mortar_board:",
    layout="wide",
)


# --------------------------------------------------------------- helpers ---


def _init_state() -> None:
    defaults = {
        "processed_pdf": None,
        "kb": None,
        "quiz": None,
        "quiz_bytes_json": None,
        "quiz_bytes_pdf": None,
        "quiz_model": None,
        "pptx_bytes": None,
        "pptx_plan": None,
        "pptx_model": None,
        "source_filename": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _reset_generated() -> None:
    for k in (
        "kb",
        "quiz",
        "quiz_bytes_json",
        "quiz_bytes_pdf",
        "quiz_model",
        "pptx_bytes",
        "pptx_plan",
        "pptx_model",
    ):
        st.session_state[k] = None


# Nombres de archivo válidos en Windows/macOS/Linux: sin `<>:"/\\|?*` ni
# caracteres de control. El modelo Ollama trae `:` (p. ej. `qwen2.5:7b`),
# inválido en Windows. Sanitizamos con `-`.
_FILENAME_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _sanitize_for_filename(text: str, *, fallback: str = "salida") -> str:
    """Devuelve una versión segura del texto para usar como nombre de archivo.

    - Reemplaza caracteres inválidos (incluyendo `:` de los tags Ollama)
      por guiones.
    - Colapsa espacios y guiones repetidos.
    - Recorta a 80 caracteres para que el nombre final siga siendo
      razonable incluso con sufijos añadidos.
    """
    cleaned = _FILENAME_UNSAFE_RE.sub("-", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    if not cleaned:
        return fallback
    return cleaned[:80]


def _build_download_filename(
    kind: str, extension: str, *, model: str | None = None
) -> str:
    """Compone `<nombre_pdf>_<kind>[_<modelo>].<ext>`.

    Usa `st.session_state.source_filename` como base. Si no hay PDF
    cargado (caso teórico), usa `"documento"`. Si se pasa `model`, se
    incluye como sufijo tras sanitizar `:`/`/` y otros caracteres
    inválidos.

    Ejemplos:
        _build_download_filename("presentacion", "pptx", model="gemma3:4b")
        → "ProgOrientadaObjetos_presentacion_gemma3-4b.pptx"
    """
    src = st.session_state.get("source_filename") or "documento.pdf"
    stem = _sanitize_for_filename(Path(src).stem, fallback="documento")
    parts = [stem, kind]
    if model:
        parts.append(_sanitize_for_filename(model, fallback="modelo"))
    return f"{'_'.join(parts)}.{extension.lstrip('.')}"


def _format_progress_label(phase: str, step: int, total: int, msg: str) -> str:
    phase_label = {
        "map": "Map-Reduce",
        "reduce": "Consolidación",
        "slides": "Presentación",
    }.get(phase, phase)
    if total:
        return f"[{phase_label} {step}/{total}] {msg}"
    return f"[{phase_label}] {msg}"


def _preflight_ollama(model: str) -> OllamaClient | None:
    client = OllamaClient(model=model)
    try:
        client.preflight()
    except OllamaUnavailableError as exc:
        st.error(f":x: Ollama no está disponible. {exc}")
        return None
    except OllamaModelNotFoundError as exc:
        st.error(f":x: Modelo no disponible. {exc}")
        return None
    return client


def _ensure_kb(client: OllamaClient, processed: ProcessedPDF) -> KnowledgeBase | None:
    """Construye (o recupera) la Base de Conocimiento estructurada del PDF."""
    if st.session_state.kb is not None:
        return st.session_state.kb
    progress = st.progress(0.0)
    status = st.empty()

    def cb(phase: str, step: int, total: int, msg: str) -> None:
        if total:
            progress.progress(min(step / total, 1.0))
        status.info(_format_progress_label(phase, step, total, msg))

    try:
        kb = build_knowledge_base(client, processed.markdown, progress_cb=cb)
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        st.error(f":x: Error de Ollama durante el análisis: {exc}")
        return None
    except GenerationError as exc:
        st.error(f":x: No se pudo construir la Base de Conocimiento: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo inesperado en Map-Reduce")
        st.error(f":x: Fallo inesperado durante el análisis: {exc}")
        return None
    finally:
        progress.empty()
        status.empty()

    if kb.atom_count == 0:
        st.error(":x: La Base de Conocimiento resultó vacía.")
        return None
    st.session_state.kb = kb
    return kb


# --------------------------------------------------------------- sidebar --


def render_sidebar() -> tuple[str, tuple[int, int]]:
    with st.sidebar:
        st.header(":gear: Configuración")
        model = st.selectbox(
            "Modelo de Ollama",
            options=list(AVAILABLE_MODELS),
            index=list(AVAILABLE_MODELS).index(DEFAULT_MODEL),
            help=(
                "Debe estar descargado en tu máquina. "
                "Ejecuta `ollama pull <modelo>` si no lo tienes."
            ),
        )
        min_q, max_q = DEFAULT_NUM_QUESTIONS_RANGE
        st.caption(
            f"Rango fijo de Quiz: **{min_q}–{max_q}** preguntas."
        )
        st.markdown("---")
        st.caption(
            "100% offline. Todos los datos permanecen en tu máquina. "
            "El modelo se ejecuta mediante Ollama local."
        )
        if not TEMPLATE_PATH.exists():
            st.warning(
                f":warning: No se encuentra la plantilla en `{TEMPLATE_PATH.name}`."
            )
        return model, (min_q, max_q)


# --------------------------------------------------------------- main UI --


def render_upload_section() -> None:
    st.subheader(":page_facing_up: 1. Sube tu PDF")
    uploaded = st.file_uploader(
        "Arrastra o selecciona un PDF",
        type=["pdf"],
        help=f"Máximo {MAX_INPUT_PAGES} páginas / 50.000 caracteres.",
    )
    if uploaded is None:
        return

    if st.session_state.source_filename != uploaded.name:
        _reset_generated()
        st.session_state.processed_pdf = None
        st.session_state.source_filename = uploaded.name

    if st.session_state.processed_pdf is None:
        with st.spinner("Extrayendo texto y estructura del PDF…"):
            try:
                processed = process_pdf_from_upload(uploaded)
            except ScannedPDFError as exc:
                st.error(
                    f":x: {exc}\n\nSugerencia: usa una herramienta OCR local "
                    "(p. ej. `ocrmypdf`) antes de subir el archivo."
                )
                return
            except PDFTooLargeError as exc:
                st.error(f":x: {exc}")
                return
            except PDFError as exc:
                st.error(f":x: No se pudo procesar el PDF: {exc}")
                return
        st.session_state.processed_pdf = processed

    processed: ProcessedPDF = st.session_state.processed_pdf
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Páginas", processed.num_pages)
    col2.metric("Caracteres", f"{processed.num_chars:,}")
    col3.metric("Imágenes", processed.num_images)
    col4.metric("Tablas detectadas", "Sí" if processed.has_tables else "No")

    with st.expander("Ver Markdown extraído (primeros 2.000 caracteres)"):
        st.code(processed.markdown[:2000], language="markdown")


def render_generate_section(model: str, num_questions_range: tuple[int, int]) -> None:
    processed: ProcessedPDF | None = st.session_state.processed_pdf
    if processed is None:
        return

    st.subheader(":rocket: 2. Generar contenidos")
    col_quiz, col_pptx = st.columns(2)

    # ----- Quiz -----
    with col_quiz:
        st.markdown("### :question: Quiz")
        if st.button("Generar Quiz", type="primary", use_container_width=True):
            client = _preflight_ollama(model)
            if client is None:
                return
            kb = _ensure_kb(client, processed)
            if kb is None:
                return
            min_q, max_q = num_questions_range
            with st.spinner("Planificando y redactando preguntas (Bloom)…"):
                try:
                    quiz = generate_quiz(
                        client, kb,
                        min_questions=min_q,
                        max_questions=max_q,
                    )
                except (OllamaUnavailableError, OllamaModelNotFoundError, GenerationError) as exc:
                    st.error(f":x: {exc}")
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Fallo generando quiz")
                    st.error(f":x: Fallo inesperado: {exc}")
                    return
            st.session_state.quiz = quiz
            st.session_state.quiz_model = model
            st.session_state.quiz_bytes_json = json.dumps(
                quiz.to_dict(), ensure_ascii=False, indent=2
            ).encode("utf-8")
            try:
                st.session_state.quiz_bytes_pdf = quiz_to_pdf_bytes(
                    quiz,
                    document_title=f"Quiz — {Path(st.session_state.source_filename).stem}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fallo generando PDF del quiz")
                st.warning(f"No se pudo exportar el PDF del quiz: {exc}")
                st.session_state.quiz_bytes_pdf = None
            generated = len(quiz.quiz)
            if generated < min_q:
                st.warning(
                    f":warning: Solo {generated} preguntas superaron el filtro "
                    f"de calidad (mínimo configurado: {min_q}). La Base de "
                    "Conocimiento del PDF es corta o muy repetitiva."
                )
            elif generated < max_q:
                st.info(
                    f":information_source: Se generaron {generated} preguntas "
                    f"de calidad (dentro del rango {min_q}–{max_q}). "
                    "El sistema descartó las que no pasaron el filtro."
                )
            st.success(f":white_check_mark: Quiz generado ({generated} preguntas).")

    # ----- PPTX -----
    with col_pptx:
        st.markdown("### :bar_chart: Presentación")
        if st.button("Generar Presentación", type="primary", use_container_width=True):
            if not TEMPLATE_PATH.exists():
                st.error(
                    f":x: No se encuentra la plantilla `{TEMPLATE_PATH.name}`."
                )
                return
            client = _preflight_ollama(model)
            if client is None:
                return
            kb = _ensure_kb(client, processed)
            if kb is None:
                return
            progress = st.progress(0.0)
            status = st.empty()

            def cb(phase: str, step: int, total: int, msg: str) -> None:
                if total:
                    progress.progress(min(step / total, 1.0))
                status.info(_format_progress_label(phase, step, total, msg))

            try:
                pptx_bytes, plan = generate_presentation(
                    client,
                    kb,
                    progress_cb=cb,
                )
            except TemplateError as exc:
                st.error(f":x: Plantilla inválida: {exc}")
                return
            except (OllamaUnavailableError, OllamaModelNotFoundError, GenerationError) as exc:
                st.error(f":x: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fallo generando presentación")
                st.error(f":x: Fallo inesperado: {exc}")
                return
            finally:
                progress.empty()
                status.empty()

            st.session_state.pptx_bytes = pptx_bytes
            st.session_state.pptx_plan = plan
            st.session_state.pptx_model = model
            st.success(
                f":white_check_mark: Presentación generada "
                f"({len(plan.slides)} slides de contenido)."
            )


def render_quiz_results() -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None:
        return
    st.subheader(":memo: Quiz interactivo")
    with st.form("quiz_form"):
        user_answers: dict[int, str] = {}
        for q in quiz.quiz:
            st.markdown(f"**{q.id}.** {q.question}")
            st.caption(f"Nivel Bloom: `{q.bloom_level}`")
            user_answers[q.id] = st.radio(
                label="Selecciona una opción",
                options=["A", "B", "C", "D"],
                format_func=lambda letter, q=q: f"{letter}) {getattr(q.options, letter)}",
                key=f"q_{q.id}",
                index=None,
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Corregir respuestas")

    if submitted:
        correct = 0
        for q in quiz.quiz:
            chosen = user_answers.get(q.id)
            if chosen == q.correct_answer:
                correct += 1
                st.success(
                    f"**{q.id}.** Correcto ({q.correct_answer}). {q.justification}"
                )
            else:
                st.error(
                    f"**{q.id}.** Incorrecto. Tu respuesta: {chosen or '—'}. "
                    f"Correcta: **{q.correct_answer}**. {q.justification}"
                )
        st.info(f"Puntuación: **{correct} / {len(quiz.quiz)}**")

    col_json, col_pdf = st.columns(2)
    quiz_model = st.session_state.get("quiz_model")
    if st.session_state.quiz_bytes_json:
        col_json.download_button(
            "Descargar Quiz (JSON)",
            data=st.session_state.quiz_bytes_json,
            file_name=_build_download_filename("quiz", "json", model=quiz_model),
            mime="application/json",
            use_container_width=True,
        )
    if st.session_state.quiz_bytes_pdf:
        col_pdf.download_button(
            "Descargar Quiz (PDF)",
            data=st.session_state.quiz_bytes_pdf,
            file_name=_build_download_filename("quiz", "pdf", model=quiz_model),
            mime="application/pdf",
            use_container_width=True,
        )


def render_pptx_results() -> None:
    if st.session_state.pptx_bytes is None:
        return
    st.subheader(":bar_chart: Presentación generada")
    plan = st.session_state.pptx_plan
    if plan is not None:
        st.caption(f"**Título:** {plan.title}")
        with st.expander("Ver índice y bullets generados"):
            st.markdown(f"**Índice ({len(plan.outline)} slides):**")
            for i, t in enumerate(plan.outline, start=1):
                st.markdown(f"{i}. {t}")
            st.markdown("---")
            for s in plan.slides:
                st.markdown(f"**{s.title}**")
                for b in s.bullets:
                    st.markdown(f"- {b}")
            st.markdown("**Conclusiones**")
            for b in plan.conclusion:
                st.markdown(f"- {b}")
    st.download_button(
        "Descargar presentación (.pptx)",
        data=st.session_state.pptx_bytes,
        file_name=_build_download_filename(
            "presentacion", "pptx", model=st.session_state.get("pptx_model")
        ),
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        type="primary",
    )


# --------------------------------------------------------------- entry ----


def main() -> None:
    _init_state()
    st.title(":mortar_board: Generador offline — Quiz + Presentación")
    st.caption(
        "Sube un PDF y genera un Quiz (JSON/PDF) y una presentación PPTX, "
        "todo con inferencia local vía Ollama. Cero cloud."
    )

    model, num_questions_range = render_sidebar()
    render_upload_section()
    render_generate_section(model, num_questions_range)

    tabs = st.tabs([":memo: Quiz", ":bar_chart: Presentación"])
    with tabs[0]:
        render_quiz_results()
    with tabs[1]:
        render_pptx_results()


if __name__ == "__main__":
    main()
