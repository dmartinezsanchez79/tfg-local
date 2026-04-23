"""Exportación del Quiz a PDF imprimible usando ReportLab.

Salida en dos secciones:
1. Hoja de examen (preguntas + opciones).
2. Hoja de soluciones (respuesta correcta + justificación + nivel Bloom).
"""
from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .quiz_generator import Quiz


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=18,
            textColor=colors.HexColor("#1f4e79"),
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#1f4e79"),
            spaceAfter=6,
        ),
        "q": ParagraphStyle(
            "q",
            parent=base["BodyText"],
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "opt": ParagraphStyle(
            "opt",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=13,
            leftIndent=16,
        ),
        "meta": ParagraphStyle(
            "meta",
            parent=base["BodyText"],
            fontSize=9,
            textColor=colors.grey,
            leading=11,
        ),
        "just": ParagraphStyle(
            "just",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=13,
            leftIndent=12,
            alignment=TA_JUSTIFY,
        ),
    }


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def quiz_to_pdf_bytes(quiz: Quiz, document_title: str = "Quiz generado") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=document_title,
    )
    styles = _styles()
    story: list = []

    # --- Hoja de examen ---
    story.append(Paragraph(_escape(document_title), styles["title"]))
    story.append(Paragraph("Examen de opción múltiple", styles["h2"]))
    story.append(Spacer(1, 8))

    for q in quiz.quiz:
        story.append(
            Paragraph(f"<b>{q.id}.</b> {_escape(q.question)}", styles["q"])
        )
        for letter in ("A", "B", "C", "D"):
            option_text = getattr(q.options, letter)
            story.append(
                Paragraph(f"<b>{letter})</b> {_escape(option_text)}", styles["opt"])
            )

    # --- Hoja de soluciones ---
    story.append(PageBreak())
    story.append(Paragraph("Soluciones y justificaciones", styles["title"]))

    for q in quiz.quiz:
        story.append(
            Paragraph(
                f"<b>{q.id}. Respuesta correcta:</b> {q.correct_answer}"
                f" &nbsp;&nbsp;<i>[{_escape(q.bloom_level)}]</i>",
                styles["q"],
            )
        )
        story.append(
            Paragraph(_escape(q.justification), styles["just"])
        )

    doc.build(story)
    return buffer.getvalue()
