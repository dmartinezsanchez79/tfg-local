#!/usr/bin/env python3
"""Análisis benchmark automático + evaluación Gemini (rúbrica).

Lee:
  - benchmark/reports/benchmark_summary.csv  (métricas automáticas por corrida)
  - benchmark/reports/gemini_evaluation.csv (formato largo de consolidate_gemini.py)

Genera:
  - benchmark/reports/gemini_model_ranking.csv
      Media por modelo de `overall` (quiz / pptx / combinado por PDF).
  - benchmark/reports/merged_summary_gemini.csv
      benchmark_summary + columnas gemini_quiz_overall, gemini_pptx_overall,
      gemini_mean_overall.

Separador CSV: `;` · encoding UTF-8 con BOM (Excel ES).

Uso:
    python scripts/analyze_benchmark_gemini.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmark" / "reports"
SUMMARY_CSV = REPORTS / "benchmark_summary.csv"
GEMINI_LONG = REPORTS / "gemini_evaluation.csv"
OUT_RANKING = REPORTS / "gemini_model_ranking.csv"
OUT_MERGED = REPORTS / "merged_summary_gemini.csv"

DELIM = ";"
ENCODING = "utf-8-sig"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding=ENCODING) as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = DELIM if sample.count(";") >= sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _safe_float(v: str) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return None


def load_gemini_overalls(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    """Clave (pdf_id, model, artifact_type) -> score overall."""
    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if (row.get("criterion") or "").strip().lower() != "overall":
            continue
        pdf = (row.get("pdf_id") or "").strip()
        model = (row.get("model") or "").strip()
        art = (row.get("artifact_type") or "").strip().lower()
        if art not in ("quiz", "pptx"):
            continue
        s = _safe_float(row.get("score_1_5") or "")
        if s is None:
            continue
        out[(pdf, model, art)] = s
    return out


def write_ranking(overalls: dict[tuple[str, str, str], float]) -> None:
    """Una fila por modelo: medias de overall quiz/pptx y combinado por PDF."""
    by_model: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"quiz": [], "pptx": []}
    )
    keys_seen = set(overalls.keys())
    for (pdf, model, art), score in overalls.items():
        by_model[model][art].append(score)

    merged_rows: list[dict[str, object]] = []
    for model in sorted(by_model.keys()):
        q = by_model[model]["quiz"]
        p = by_model[model]["pptx"]
        mean_q = round(mean(q), 3) if q else ""
        mean_p = round(mean(p), 3) if p else ""
        combined_per_pdf: list[float] = []
        pdfs = {k[0] for k in keys_seen if k[1] == model}
        for pdf in sorted(pdfs):
            qv = overalls.get((pdf, model, "quiz"))
            pv = overalls.get((pdf, model, "pptx"))
            if qv is not None and pv is not None:
                combined_per_pdf.append((qv + pv) / 2.0)
        mean_balanced = round(mean(combined_per_pdf), 3) if combined_per_pdf else ""

        merged_rows.append({
            "model": model,
            "n_quiz_scores": len(q),
            "n_pptx_scores": len(p),
            "n_pdf_balanced": len(combined_per_pdf),
            "mean_overall_quiz": mean_q,
            "mean_overall_pptx": mean_p,
            "mean_balanced_by_pdf": mean_balanced,
        })

    ranked = sorted(
        merged_rows,
        key=lambda r: (
            -(r["mean_balanced_by_pdf"] if isinstance(r["mean_balanced_by_pdf"], float) else -1),
            str(r["model"]),
        ),
    )
    for i, row in enumerate(ranked, start=1):
        row["rank_balanced_by_pdf"] = i if row["mean_balanced_by_pdf"] != "" else ""

    cols = (
        "rank_balanced_by_pdf",
        "model",
        "mean_balanced_by_pdf",
        "mean_overall_quiz",
        "mean_overall_pptx",
        "n_pdf_balanced",
        "n_quiz_scores",
        "n_pptx_scores",
    )
    OUT_RANKING.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RANKING.open("w", newline="", encoding=ENCODING) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=DELIM, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)


def write_merged(
    summary_rows: list[dict[str, str]],
    summary_cols: list[str],
    overalls: dict[tuple[str, str, str], float],
) -> None:
    extra = (
        "gemini_quiz_overall",
        "gemini_pptx_overall",
        "gemini_mean_overall",
    )
    out_cols = list(summary_cols) + list(extra)
    merged_out: list[dict[str, str]] = []
    for row in summary_rows:
        pdf = (row.get("pdf_id") or "").strip()
        model = (row.get("model") or "").strip()
        gq = overalls.get((pdf, model, "quiz"))
        gp = overalls.get((pdf, model, "pptx"))
        vals = [x for x in (gq, gp) if x is not None]
        mean_cell = round(mean(vals), 3) if vals else ""
        new_row = {**row}
        new_row["gemini_quiz_overall"] = str(gq) if gq is not None else ""
        new_row["gemini_pptx_overall"] = str(gp) if gp is not None else ""
        new_row["gemini_mean_overall"] = str(mean_cell) if mean_cell != "" else ""
        merged_out.append(new_row)

    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MERGED.open("w", newline="", encoding=ENCODING) as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, delimiter=DELIM, extrasaction="ignore")
        w.writeheader()
        for r in merged_out:
            w.writerow({k: r.get(k, "") for k in out_cols})


def main() -> int:
    _, gemini_rows = _read_csv(GEMINI_LONG)
    if not gemini_rows:
        print(f"No hay datos en {GEMINI_LONG}", file=sys.stderr)
        return 1

    overalls = load_gemini_overalls(gemini_rows)
    if not overalls:
        print("No hay filas criterion=overall en gemini_evaluation.csv", file=sys.stderr)
        return 1

    write_ranking(overalls)
    print(f"OK  {OUT_RANKING.relative_to(ROOT)}")

    sum_cols, sum_rows = _read_csv(SUMMARY_CSV)
    if not sum_rows:
        print(
            f"Aviso: no existe o está vacío {SUMMARY_CSV}; "
            "solo se generó el ranking.",
            file=sys.stderr,
        )
        return 0

    write_merged(sum_rows, sum_cols, overalls)
    print(f"OK  {OUT_MERGED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
