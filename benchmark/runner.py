"""CLI del benchmark: ejecuta combinaciones PDF × modelo y volca métricas.

Uso típico (desde la raíz del proyecto):

    python -m benchmark.runner
    python -m benchmark.runner --models qwen2.5:7b mistral:7b
    python -m benchmark.runner --pdfs poo bbdd --only-pptx
    python -m benchmark.runner --dry-run            # lista las ejecuciones

El runner reutiliza íntegramente el pipeline de `src/`:
    PDF → Markdown (pdf_processor)
    Markdown → KnowledgeBase (map_reduce.build_knowledge_base)
    KB → Quiz (quiz_generator.generate_quiz)
    KB → PPTX (pptx_generator.generate_presentation)

Por cada ejecución guarda en ``benchmark/results/<pdf>__<model>/``:

* ``metrics.json``      – métricas automáticas completas + timings.
* ``quiz.json``         – quiz serializado (si se generó).
* ``presentation.pptx`` – binario del .pptx (si se generó).
* ``plan.json``         – plan de la presentación serializado.
* ``eval_prompts/``     – prompts para evaluación externa con IA.
* ``error.log``         – traza si la ejecución falló (no detiene el resto).

Al final agrega todo en ``benchmark/reports/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import reports as reports_mod
from .config import (
    CATALOG_PATH,
    DATASET_PDFS_DIR,
    DEFAULT_MODELS,
    PROJECT_PDFS_DIR,
    QUIZ_MAX_QUESTIONS,
    QUIZ_MIN_QUESTIONS,
    REFINE_QUIZ,
    REFINE_SLIDES,
    RESULTS_DIR,
    ensure_directories,
)
from .judge_prompts import build_pptx_eval_prompt, build_quiz_eval_prompt
from .metrics import (
    compute_pptx_metrics,
    compute_quiz_metrics,
    presentation_plan_to_dict,
)

logger = logging.getLogger("benchmark.runner")


# =========================================================================
# Carga del catálogo y resolución de rutas
# =========================================================================

@dataclass(frozen=True)
class PdfEntry:
    id: str
    filename: str
    title: str
    path: Path
    meta: dict[str, Any]


def load_catalog(path: Path = CATALOG_PATH) -> list[PdfEntry]:
    """Lee `catalog.json` y resuelve rutas reales de los PDFs.

    Busca primero en ``benchmark/dataset/pdfs/`` y, si no está, cae al
    ``PDF/`` del proyecto. Omite entradas cuyo PDF no exista en ninguno.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el catálogo: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[PdfEntry] = []
    for raw in data.get("pdfs", []):
        pdf_id = str(raw.get("id") or "").strip()
        filename = str(raw.get("filename") or "").strip()
        if not pdf_id or not filename:
            continue
        candidates = [DATASET_PDFS_DIR / filename, PROJECT_PDFS_DIR / filename]
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            logger.warning("PDF '%s' no encontrado en %s ni %s; omitiendo.",
                           filename, DATASET_PDFS_DIR, PROJECT_PDFS_DIR)
            continue
        entries.append(PdfEntry(
            id=pdf_id,
            filename=filename,
            title=str(raw.get("title") or pdf_id),
            path=resolved,
            meta={k: v for k, v in raw.items() if k not in ("id", "filename", "title")},
        ))
    if not entries:
        raise RuntimeError(
            "Catálogo vacío: ningún PDF del catálogo se ha encontrado en "
            f"{DATASET_PDFS_DIR} ni en {PROJECT_PDFS_DIR}."
        )
    return entries


# =========================================================================
# Ejecución de una combinación PDF × modelo
# =========================================================================

def _sanitize(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


def _execution_dir(pdf: PdfEntry, model: str) -> Path:
    d = RESULTS_DIR / f"{pdf.id}__{_sanitize(model)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_prompts").mkdir(exist_ok=True)
    return d


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_single(
    pdf: PdfEntry,
    model: str,
    *,
    do_quiz: bool,
    do_pptx: bool,
) -> dict[str, Any]:
    """Ejecuta una combinación (pdf, modelo).

    Nunca lanza excepciones no capturadas: los errores se anotan en el
    record y se devuelven como ``status="error"`` para que el loop
    principal siga adelante.
    """
    # Imports diferidos: así `python -m benchmark.runner --help` no exige
    # que Ollama/pymupdf/etc. estén instalados.
    from src.map_reduce import build_knowledge_base
    from src.ollama_client import OllamaClient
    from src.pdf_processor import process_pdf
    from src.pptx_generator import generate_presentation
    from src.quiz_generator import generate_quiz

    out_dir = _execution_dir(pdf, model)
    error_log = out_dir / "error.log"
    timings: dict[str, float] = {}
    record: dict[str, Any] = {
        "pdf_id": pdf.id,
        "pdf_title": pdf.title,
        "pdf_filename": pdf.filename,
        "model": model,
        "status": "pending",
        "error": "",
        "timings": timings,
        "kb_info": {},
        "quiz_metrics": None,
        "pptx_metrics": None,
        "artifacts": {},
    }
    t0 = time.perf_counter()

    try:
        # 1) PDF → Markdown
        logger.info("[%s × %s] procesando PDF…", pdf.id, model)
        processed = process_pdf(pdf.path.read_bytes())

        # 2) Ollama preflight
        client = OllamaClient(model=model)
        client.preflight()

        # 3) Markdown → KB (compartido por quiz y pptx)
        t_kb = time.perf_counter()
        kb = build_knowledge_base(client, processed.markdown)
        timings["kb_s"] = round(time.perf_counter() - t_kb, 2)
        record["kb_info"] = {
            "atoms": kb.atom_count,
            "subtopics": len(kb.subtopics),
            "main_topic": kb.main_topic,
            "definitions": len(kb.definitions),
            "examples": len(kb.examples),
            "formulas_code": len(kb.formulas_code),
            "numeric_data": len(kb.numeric_data),
            "relations": len(kb.relations),
        }
        _save_json(out_dir / "kb.json", json.loads(kb.model_dump_json()))

        # 4) Quiz (opcional)
        if do_quiz:
            t_q = time.perf_counter()
            try:
                quiz = generate_quiz(
                    client, kb,
                    min_questions=QUIZ_MIN_QUESTIONS,
                    max_questions=QUIZ_MAX_QUESTIONS,
                    refine=REFINE_QUIZ,
                )
                timings["quiz_s"] = round(time.perf_counter() - t_q, 2)
                quiz_dict = quiz.to_dict()
                _save_json(out_dir / "quiz.json", quiz_dict)
                record["quiz_metrics"] = compute_quiz_metrics(quiz_dict, kb=kb)
                record["artifacts"]["quiz"] = str(out_dir / "quiz.json")
                (out_dir / "eval_prompts" / "quiz_eval.txt").write_text(
                    build_quiz_eval_prompt(
                        pdf_id=pdf.id, pdf_title=pdf.title,
                        model=model, quiz_json=quiz_dict,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001 — aislamos el quiz del pptx
                timings["quiz_s"] = round(time.perf_counter() - t_q, 2)
                logger.exception("[%s × %s] Quiz falló: %s", pdf.id, model, exc)
                error_log.write_text(
                    f"Quiz error: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                record["quiz_metrics"] = {"error": str(exc)}

        # 5) PPTX (opcional)
        if do_pptx:
            t_p = time.perf_counter()
            try:
                pptx_bytes, plan = generate_presentation(
                    client, kb,
                    output_path=out_dir / "presentation.pptx",
                )
                timings["pptx_s"] = round(time.perf_counter() - t_p, 2)
                plan_dict = presentation_plan_to_dict(plan)
                _save_json(out_dir / "plan.json", plan_dict)
                record["pptx_metrics"] = compute_pptx_metrics(plan_dict)
                record["artifacts"]["pptx"] = str(out_dir / "presentation.pptx")
                record["artifacts"]["plan"] = str(out_dir / "plan.json")
                (out_dir / "eval_prompts" / "pptx_eval.txt").write_text(
                    build_pptx_eval_prompt(
                        pdf_id=pdf.id, pdf_title=pdf.title,
                        model=model, plan_json=plan_dict,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001 — aislamos el pptx del quiz
                timings["pptx_s"] = round(time.perf_counter() - t_p, 2)
                logger.exception("[%s × %s] PPTX falló: %s", pdf.id, model, exc)
                existing = error_log.read_text(encoding="utf-8") if error_log.exists() else ""
                error_log.write_text(
                    existing + f"\nPPTX error: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                record["pptx_metrics"] = {"error": str(exc)}

        # status global
        had_artifact = any(
            isinstance(record[k], dict) and "error" not in record[k]
            for k in ("quiz_metrics", "pptx_metrics")
            if record[k] is not None
        )
        record["status"] = "ok" if had_artifact else "error"
        if record["status"] == "error":
            record["error"] = "Ningún artefacto se generó con éxito."
    except Exception as exc:  # noqa: BLE001 — infra fatal (Ollama, PDF, KB…)
        logger.exception("[%s × %s] fallo fatal: %s", pdf.id, model, exc)
        error_log.write_text(
            f"Fatal error: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        record["status"] = "error"
        record["error"] = str(exc)
    finally:
        timings["total_s"] = round(time.perf_counter() - t0, 2)
        _save_json(out_dir / "metrics.json", record)

    return record


# =========================================================================
# CLI
# =========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.runner",
        description=(
            "Ejecuta el benchmark: una ejecución por combinación "
            "(PDF × modelo), volca artefactos + métricas y agrega CSVs."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Modelos Ollama a evaluar (por defecto: los de config.DEFAULT_MODELS).",
    )
    parser.add_argument(
        "--pdfs", nargs="+", default=None,
        help="IDs de PDFs del catálogo a ejecutar (por defecto: todos).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-quiz", action="store_true", help="Generar solo quiz.")
    mode.add_argument("--only-pptx", action="store_true", help="Generar solo PPTX.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lista las ejecuciones previstas sin ejecutar nada.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _select_pdfs(catalog: list[PdfEntry], requested: list[str] | None) -> list[PdfEntry]:
    if not requested:
        return catalog
    by_id = {p.id: p for p in catalog}
    missing = [r for r in requested if r not in by_id]
    if missing:
        raise SystemExit(f"IDs de PDF no encontrados en el catálogo: {missing}")
    return [by_id[r] for r in requested]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    ensure_directories()

    catalog = load_catalog()
    pdfs = _select_pdfs(catalog, args.pdfs)
    models = tuple(args.models) if args.models else DEFAULT_MODELS

    do_quiz = not args.only_pptx
    do_pptx = not args.only_quiz

    print(f"\n=== Benchmark: {len(pdfs)} PDFs × {len(models)} modelos "
          f"= {len(pdfs) * len(models)} ejecuciones ===\n")
    for p in pdfs:
        print(f"  · {p.id:<10s} {p.title}")
    print("Modelos:", ", ".join(models))
    print(f"Artefactos: quiz={do_quiz}, pptx={do_pptx}")
    print(f"Resultados: {RESULTS_DIR}\n")

    if args.dry_run:
        print("(dry-run) No se ejecuta nada.")
        return 0

    records: list[dict[str, Any]] = []
    for pdf in pdfs:
        for model in models:
            print(f"\n--- {pdf.id} × {model} ---")
            record = run_single(pdf, model, do_quiz=do_quiz, do_pptx=do_pptx)
            records.append(record)
            print(
                f"  status={record['status']} "
                f"total={record['timings'].get('total_s', '-')}s "
                f"quiz_score={(record.get('quiz_metrics') or {}).get('score_quiz', '-')} "
                f"pptx_score={(record.get('pptx_metrics') or {}).get('score_pptx', '-')}"
            )

    print("\n=== Generando reportes agregados ===")
    paths = reports_mod.write_all_reports(records)
    for name, path in paths.items():
        print(f"  {name:<18s} -> {path.relative_to(RESULTS_DIR.parent.parent)}")

    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_err = len(records) - n_ok
    print(f"\nTerminado: {n_ok} ok, {n_err} error (total {len(records)}).")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
