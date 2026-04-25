"""Agregación de resultados del benchmark en CSV y plantillas.

Genera cuatro ficheros en `benchmark/reports/`:

* ``benchmark_summary.csv``: una fila por ejecución (pdf × modelo) con
  métricas planas. Es el fichero principal para analizar/filtrar.
* ``model_averages.csv``: media por modelo de las métricas clave.
* ``manual_evaluation_template.csv``: plantilla de puntuación humana
  siguiendo la misma rúbrica 1-5.
* ``rubric_reference.txt``: texto descriptivo de la rúbrica (ver
  ``judge_prompts.rubric_reference_text``).

Sin pandas: solo `csv` y `statistics` de la stdlib.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import REPORTS_DIR
from .judge_prompts import PPTX_CRITERIA, QUIZ_CRITERIA, rubric_reference_text

# Columnas "planas" que vuelcan a CSV. El resto (bloom distribution,
# ejemplos de frases prohibidas, etc.) se deja solo en el JSON completo.
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "pdf_id", "pdf_title", "model",
    "status", "error",
    "pdf_num_chars_catalog", "pdf_num_chars_extracted", "pdf_truncated",
    "pdf_num_images", "pdf_has_tables_detected",
    "time_total_s", "time_kb_s", "time_quiz_s", "time_pptx_s",
    "kb_atoms", "kb_subtopics",
    # quiz
    "quiz_num_questions", "quiz_bloom_diversity",
    "quiz_duplicate_pairs", "quiz_unbalanced_options",
    "quiz_banned_phrases", "quiz_pct_explanation",
    "quiz_kb_term_coverage", "quiz_score",
    # pptx
    "pptx_num_slides_total", "pptx_num_content_slides",
    "pptx_avg_bullets", "pptx_bullets_total",
    "pptx_bullets_too_long", "pptx_bullets_too_short",
    "pptx_bullets_truncated", "pptx_cross_slide_repetition",
    "pptx_index_coherence", "pptx_score",
)

_CSV_DELIMITER = ";"
_CSV_ENCODING = "utf-8-sig"  # BOM para que Excel (ES) abra columnas correctamente


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Aplana un dict de métricas al esquema de ``_SUMMARY_COLUMNS``."""
    quiz = record.get("quiz_metrics") or {}
    pptx = record.get("pptx_metrics") or {}
    timings = record.get("timings") or {}
    pdf_info = record.get("pdf_info") or {}
    kb_info = record.get("kb_info") or {}

    return {
        "pdf_id": record.get("pdf_id", ""),
        "pdf_title": record.get("pdf_title", ""),
        "model": record.get("model", ""),
        "status": record.get("status", ""),
        "error": (record.get("error") or "").replace("\n", " ")[:300],
        "pdf_num_chars_catalog": pdf_info.get("catalog_num_chars", ""),
        "pdf_num_chars_extracted": pdf_info.get("extracted_num_chars", ""),
        "pdf_truncated": pdf_info.get("truncated", ""),
        "pdf_num_images": pdf_info.get("num_images", ""),
        "pdf_has_tables_detected": pdf_info.get("has_tables_detected", ""),
        "time_total_s": timings.get("total_s", ""),
        "time_kb_s": timings.get("kb_s", ""),
        "time_quiz_s": timings.get("quiz_s", ""),
        "time_pptx_s": timings.get("pptx_s", ""),
        "kb_atoms": kb_info.get("atoms", ""),
        "kb_subtopics": kb_info.get("subtopics", ""),
        # quiz
        "quiz_num_questions": quiz.get("num_questions", ""),
        "quiz_bloom_diversity": quiz.get("bloom_diversity", ""),
        "quiz_duplicate_pairs": quiz.get("duplicate_pairs", ""),
        "quiz_unbalanced_options": quiz.get("unbalanced_options_count", ""),
        "quiz_banned_phrases": quiz.get("banned_phrases_count", ""),
        "quiz_pct_explanation": quiz.get("pct_with_explanation", ""),
        "quiz_kb_term_coverage": quiz.get("kb_term_coverage", ""),
        "quiz_score": quiz.get("score_quiz", ""),
        # pptx
        "pptx_num_slides_total": pptx.get("num_slides_total", ""),
        "pptx_num_content_slides": pptx.get("num_content_slides", ""),
        "pptx_avg_bullets": pptx.get("avg_bullets_per_slide", ""),
        "pptx_bullets_total": pptx.get("bullets_total", ""),
        "pptx_bullets_too_long": pptx.get("bullets_too_long", ""),
        "pptx_bullets_too_short": pptx.get("bullets_too_short", ""),
        "pptx_bullets_truncated": pptx.get("bullets_possibly_truncated", ""),
        "pptx_cross_slide_repetition": pptx.get("cross_slide_repetition_pairs", ""),
        "pptx_index_coherence": pptx.get("index_coherence", ""),
        "pptx_score": pptx.get("score_pptx", ""),
    }


def write_summary_csv(records: Iterable[dict[str, Any]], out_path: Path | None = None) -> Path:
    """Actualiza `benchmark_summary.csv` con política UPSERT por (pdf_id, model).

    - Si una combinación `(pdf_id, model)` ya existe, se reemplaza su fila.
    - Si no existe, se añade una fila nueva.
    """
    path = out_path or (REPORTS_DIR / "benchmark_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_summary(path)
    merged: dict[tuple[str, str], dict[str, Any]] = {
        _summary_key(row): row for row in existing
    }
    for r in records:
        row = _flatten_record(r)
        merged[_summary_key(row)] = row

    ordered_rows = sorted(
        merged.values(),
        key=lambda row: (str(row.get("pdf_id", "")), str(row.get("model", ""))),
    )
    with path.open("w", newline="", encoding=_CSV_ENCODING) as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_COLUMNS, delimiter=_CSV_DELIMITER)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(row)
    return path


# --- medias por modelo ----------------------------------------------------

_NUMERIC_AVG_COLUMNS: tuple[str, ...] = (
    "pdf_num_chars_catalog", "pdf_num_chars_extracted",
    "time_total_s", "time_kb_s", "time_quiz_s", "time_pptx_s",
    "quiz_num_questions", "quiz_bloom_diversity",
    "quiz_duplicate_pairs", "quiz_unbalanced_options",
    "quiz_banned_phrases", "quiz_pct_explanation",
    "quiz_kb_term_coverage", "quiz_score",
    "pptx_num_content_slides", "pptx_avg_bullets",
    "pptx_bullets_total", "pptx_bullets_too_long",
    "pptx_bullets_too_short", "pptx_bullets_truncated",
    "pptx_cross_slide_repetition", "pptx_index_coherence",
    "pptx_score",
)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_model_averages(
    records: Iterable[dict[str, Any]],
    out_path: Path | None = None,
) -> Path:
    """Media por modelo usando TODO el histórico de `benchmark_summary.csv`."""
    path = out_path or (REPORTS_DIR / "model_averages.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "benchmark_summary.csv"
    _ = list(records)  # compatibilidad con la firma pública actual
    flat_ok = [
        row for row in _read_existing_summary(summary_path)
        if (row.get("status") or "") == "ok"
    ]

    # Agrupar por modelo y calcular media
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in flat_ok:
        by_model.setdefault(str(row.get("model", "?")), []).append(row)

    columns = ("model", "runs_ok", *_NUMERIC_AVG_COLUMNS)
    with path.open("w", newline="", encoding=_CSV_ENCODING) as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter=_CSV_DELIMITER)
        writer.writeheader()
        for model, rows in sorted(by_model.items()):
            out: dict[str, Any] = {"model": model, "runs_ok": len(rows)}
            for col in _NUMERIC_AVG_COLUMNS:
                vals = [_safe_float(row.get(col)) for row in rows]
                vals = [v for v in vals if v is not None]
                out[col] = round(mean(vals), 3) if vals else ""
            writer.writerow(out)
    return path


# --- plantilla de evaluación manual --------------------------------------

def write_manual_template(
    records: Iterable[dict[str, Any]],
    out_path: Path | None = None,
) -> Path:
    """Genera plantilla manual desde TODO el histórico del summary."""
    path = out_path or (REPORTS_DIR / "manual_evaluation_template.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "benchmark_summary.csv"
    _ = list(records)  # compatibilidad con la firma pública actual
    rows = _read_existing_summary(summary_path)

    columns = ("execution_id", "pdf_id", "model", "artifact_type", "criterion", "score_1_5", "notes")
    with path.open("w", newline="", encoding=_CSV_ENCODING) as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter=_CSV_DELIMITER)
        writer.writeheader()
        for r in rows:
            if (r.get("status") or "") != "ok":
                continue
            pdf_id = r.get("pdf_id", "")
            model = r.get("model", "")
            exec_id = f"{pdf_id}__{_sanitize_model(model)}"
            has_quiz = str(r.get("quiz_score", "")) != ""
            has_pptx = str(r.get("pptx_score", "")) != ""
            if has_quiz:
                for key, _ in QUIZ_CRITERIA:
                    writer.writerow({
                        "execution_id": exec_id,
                        "pdf_id": pdf_id,
                        "model": model,
                        "artifact_type": "quiz",
                        "criterion": key,
                        "score_1_5": "",
                        "notes": "",
                    })
            if has_pptx:
                for key, _ in PPTX_CRITERIA:
                    writer.writerow({
                        "execution_id": exec_id,
                        "pdf_id": pdf_id,
                        "model": model,
                        "artifact_type": "pptx",
                        "criterion": key,
                        "score_1_5": "",
                        "notes": "",
                    })
    return path


def _sanitize_model(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


def _summary_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("pdf_id", "")), str(row.get("model", ""))


def _read_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding=_CSV_ENCODING) as fh:
        # Compatibilidad: soporta históricos antiguos con ',' y nuevos con ';'
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = _CSV_DELIMITER if sample.count(";") >= sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        rows: list[dict[str, Any]] = []
        for row in reader:
            normalized = {col: row.get(col, "") for col in _SUMMARY_COLUMNS}
            rows.append(normalized)
        return rows


def write_rubric_reference(out_path: Path | None = None) -> Path:
    """Escribe la referencia de la rúbrica en texto plano."""
    path = out_path or (REPORTS_DIR / "rubric_reference.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rubric_reference_text(), encoding="utf-8")
    return path


def write_all_reports(records: list[dict[str, Any]]) -> dict[str, Path]:
    """Genera los cuatro ficheros de reporte y devuelve sus rutas."""
    return {
        "summary": write_summary_csv(records),
        "model_averages": write_model_averages(records),
        "manual_template": write_manual_template(records),
        "rubric": write_rubric_reference(),
    }
