"""Conversión de PDF a Markdown estructurado usando pymupdf4llm.

El Markdown resultante preserva:
- Títulos y subtítulos (#, ##, ###).
- Listas y énfasis.
- Tablas (sintaxis GFM), crítico para que el LLM comprenda datos tabulares.
- Contexto asociado a imágenes (pie de foto / texto adyacente) cuando está disponible.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import BinaryIO

import pymupdf  # type: ignore[import-untyped]
import pymupdf4llm  # type: ignore[import-untyped]

from .config import MAX_INPUT_CHARS, MAX_INPUT_PAGES
from .exceptions import PDFError, PDFTooLargeError, ScannedPDFError

logger = logging.getLogger(__name__)

# Umbral mínimo de caracteres de texto por página para considerar que el PDF
# no es un escaneo. Páginas por debajo de este umbral suelen ser imágenes.
_MIN_CHARS_PER_PAGE: int = 40
# Fracción mínima de páginas con texto real para no considerarlo escaneado.
_MIN_TEXT_PAGE_FRACTION: float = 0.3
# Umbral para descartar iconos/logos pequeños en el bloque de contexto visual.
_MIN_IMAGE_AREA_RATIO: float = 0.02


@dataclass(frozen=True)
class ProcessedPDF:
    """Resultado de la extracción del PDF."""

    markdown: str
    num_pages: int
    num_chars: int
    num_images: int
    has_tables: bool


def _detect_scanned(doc: pymupdf.Document) -> bool:
    """Heurística: si muy pocas páginas contienen texto, asumimos escaneado."""
    num_pages = doc.page_count
    if num_pages == 0:
        return True
    pages_with_text = 0
    for page in doc:
        text = page.get_text("text") or ""
        if len(text.strip()) >= _MIN_CHARS_PER_PAGE:
            pages_with_text += 1
    fraction = pages_with_text / num_pages
    logger.debug("Fracción de páginas con texto: %.2f", fraction)
    return fraction < _MIN_TEXT_PAGE_FRACTION


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _normalize_markdown_tables(markdown: str) -> str:
    """Mejora la legibilidad de tablas GFM sin alterar el resto del texto.

    - Une líneas partidas dentro de una tabla.
    - Normaliza `<br>` a ` / ` para evitar celdas visualmente rotas.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    in_table = False
    for raw in lines:
        line = raw.rstrip()
        if _is_table_row(line):
            in_table = True
            normalized = re.sub(r"\s*<br>\s*", " / ", line, flags=re.IGNORECASE)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            out.append(normalized)
            continue
        if in_table and line.strip():
            # Continuación de celda (pymupdf4llm a veces parte filas en dos líneas).
            if out:
                continuation = re.sub(r"\s*<br>\s*", " / ", line.strip(), flags=re.IGNORECASE)
                out[-1] = f"{out[-1]} {continuation}".strip()
            continue
        in_table = False
        out.append(line)
    return "\n".join(out)


def _extract_image_context(doc: pymupdf.Document) -> list[str]:
    """Extrae fragmentos breves de texto cercano a cada imagen como pie de foto.

    Estrategia simple y offline: para cada imagen, toma el bloque de texto más
    próximo verticalmente en la misma página. No es perfecto, pero da contexto
    útil al LLM sin depender de OCR ni de modelos externos.
    """
    captions: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            images = page.get_images(full=True)
            if not images:
                continue
            page_area = max(1.0, page.rect.width * page.rect.height)
            blocks = page.get_text("blocks") or []
            # blocks: (x0, y0, x1, y1, text, block_no, block_type, ...)
            text_blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
            if not text_blocks:
                continue
            for img in images:
                try:
                    xref = img[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    img_rect = rects[0]
                    area_ratio = (img_rect.width * img_rect.height) / page_area
                    if area_ratio < _MIN_IMAGE_AREA_RATIO:
                        continue
                    # Bloque más cercano por distancia vertical a la imagen.
                    closest = min(
                        text_blocks,
                        key=lambda b: min(
                            abs(b[1] - img_rect.y1),  # distancia al borde inferior
                            abs(b[3] - img_rect.y0),  # distancia al borde superior
                        ),
                    )
                    caption = closest[4].strip().replace("\n", " ")
                    caption = re.sub(r"\s+", " ", caption)
                    if 10 <= len(caption) <= 300:
                        w = int(img_rect.width)
                        h = int(img_rect.height)
                        captions.append(f"[p.{page_num}, {w}x{h}] {caption}")
                except Exception:  # noqa: BLE001 — imagen individual no crítica
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("No se pudo procesar imágenes de una página: %s", exc)
            continue
    # Deduplicar manteniendo el orden
    seen: set[str] = set()
    unique: list[str] = []
    for c in captions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _count_images(doc: pymupdf.Document) -> int:
    total = 0
    for page in doc:
        try:
            total += len(page.get_images(full=True))
        except Exception:  # noqa: BLE001
            continue
    return total


def _looks_like_tables(markdown: str) -> bool:
    # Las tablas GFM tienen líneas con múltiples '|' y una línea separadora '---'.
    return bool(re.search(r"\n\s*\|?\s*-{3,}\s*\|", markdown))


def process_pdf(pdf_bytes: bytes | BinaryIO) -> ProcessedPDF:
    """Convierte un PDF (bytes o stream) a Markdown enriquecido.

    Lanza:
        ScannedPDFError: si parece un PDF escaneado sin capa de texto.
        PDFTooLargeError: si excede los límites configurados.
        PDFError: para cualquier otro fallo de parseo.
    """
    if hasattr(pdf_bytes, "read"):
        data = pdf_bytes.read()  # type: ignore[union-attr]
    else:
        data = pdf_bytes

    if not data:
        raise PDFError("El archivo PDF está vacío.")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFError(f"No se pudo abrir el PDF: {exc}") from exc

    try:
        num_pages = doc.page_count
        if num_pages == 0:
            raise PDFError("El PDF no contiene páginas.")
        if num_pages > MAX_INPUT_PAGES:
            raise PDFTooLargeError(
                f"El PDF tiene {num_pages} páginas; el máximo admitido es {MAX_INPUT_PAGES}."
            )

        if _detect_scanned(doc):
            raise ScannedPDFError(
                "El PDF parece un documento escaneado sin texto extraíble. "
                "Procesa el fichero con OCR antes de subirlo."
            )

        try:
            markdown: str = pymupdf4llm.to_markdown(doc, show_progress=False)
        except Exception as exc:
            raise PDFError(f"pymupdf4llm falló al convertir a Markdown: {exc}") from exc

        if not markdown or not markdown.strip():
            raise PDFError("No se pudo extraer texto del PDF.")

        markdown = _normalize_markdown_tables(markdown)
        captions = _extract_image_context(doc)
        num_images = _count_images(doc)
        if captions:
            markdown += "\n\n## Contexto visual detectado\n"
            for cap in captions:
                markdown += f"- {cap}\n"

        if len(markdown) > MAX_INPUT_CHARS:
            logger.warning(
                "Markdown truncado de %d a %d caracteres.", len(markdown), MAX_INPUT_CHARS
            )
            markdown = markdown[:MAX_INPUT_CHARS] + "\n\n[...documento truncado...]"

        return ProcessedPDF(
            markdown=markdown,
            num_pages=num_pages,
            num_chars=len(markdown),
            num_images=num_images,
            has_tables=_looks_like_tables(markdown),
        )
    finally:
        doc.close()


def process_pdf_from_upload(uploaded: io.BytesIO | bytes) -> ProcessedPDF:
    """Wrapper para el objeto `UploadedFile` de Streamlit."""
    if hasattr(uploaded, "getvalue"):
        data = uploaded.getvalue()  # type: ignore[union-attr]
    else:
        data = uploaded
    return process_pdf(data)
