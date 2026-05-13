#!/usr/bin/env python3
"""Consolida los JSON de evaluación externa (Gemini) en un único CSV largo.

Lee `benchmark/external_eval/*.json` y escribe
`benchmark/reports/gemini_evaluation.csv`.

Convenciones de nombre de fichero:
    <pdf>_<artifact>_<modelsize>.json
    p. ej.: futbol_quiz_14b.json, Ciberseguridad_pptx_4b.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "benchmark" / "external_eval"
OUT_CSV = ROOT / "benchmark" / "reports" / "gemini_evaluation.csv"

# Mapa tamaño-de-modelo -> identificador real en Ollama (DEFAULT_MODELS).
MODEL_BY_SIZE: dict[str, str] = {
    "4b":  "gemma3:4b",
    "7b":  "qwen2.5:7b",
    "9b":  "gemma2:9b",
    "12b": "gemma3:12b",
    "14b": "qwen2.5:14b",
}

# Mapa explícito del prefijo del archivo al `pdf_id` del catálogo.
# Cualquier prefijo no listado se intenta convertir CamelCase -> snake_case.
PDF_ID_OVERRIDES: dict[str, str] = {
    "energiarenovable":       "energia_renovable",
    "ciberseguridad":         "ciberseguridad",
    "sistemassaludpublica":   "sistemas_salud_publica",
    "futbol":                 "futbol",
}


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def parse_filename(stem: str) -> tuple[str, str, str] | None:
    """Devuelve `(pdf_id, artifact, model)` o `None` si el nombre no encaja."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    model_size = parts[-1].lower()
    artifact = parts[-2].lower()
    pdf_raw = "_".join(parts[:-2])
    if artifact not in ("quiz", "pptx"):
        return None
    if model_size not in MODEL_BY_SIZE:
        return None
    key = pdf_raw.lower().replace("_", "")
    pdf_id = PDF_ID_OVERRIDES.get(key) or _camel_to_snake(pdf_raw)
    return pdf_id, artifact, MODEL_BY_SIZE[model_size]


def _clean_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _looks_like_criteria_block(d: dict) -> bool:
    """True si el dict parece contener criterios con `score`/`comment`."""
    if not isinstance(d, dict) or not d:
        return False
    score_subdicts = sum(
        1 for v in d.values()
        if isinstance(v, dict) and ("score" in v or "comment" in v)
    )
    return score_subdicts >= 2


def _unwrap_evaluation(data: dict) -> dict:
    """Si los criterios están envueltos en una sola clave wrapper, los aplana.

    Maneja casos típicos de Gemini como `evaluation_presentation`,
    `presentation_evaluation`, `evaluation_quiz`, `quiz_evaluation`.
    """
    non_meta = {k: v for k, v in data.items() if k != "metadata"}
    if len(non_meta) != 1:
        return data
    only_key, only_val = next(iter(non_meta.items()))
    if not isinstance(only_val, dict):
        return data
    if not _looks_like_criteria_block(only_val):
        return data
    merged = {"metadata": data.get("metadata", {})}
    merged.update(only_val)
    return merged


def extract_rows(
    data: dict,
    *,
    pdf_id: str,
    model: str,
    artifact: str,
    source_file: str,
) -> list[dict]:
    """Extrae filas en formato largo: una fila por criterio + fila `summary`."""
    data = _unwrap_evaluation(data)
    rows: list[dict] = []
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    model_in_json = _clean_text(meta.get("model_used"))
    pdf_in_json = _clean_text(meta.get("pdf_source"))

    for key, value in data.items():
        if key in ("metadata", "summary_comment"):
            continue
        if not isinstance(value, dict):
            continue
        score = value.get("score", "")
        if isinstance(score, bool):
            score = ""
        rows.append({
            "pdf_id": pdf_id,
            "model": model,
            "artifact_type": artifact,
            "criterion": key,
            "score_1_5": score if isinstance(score, (int, float)) else "",
            "comment": _clean_text(value.get("comment")),
            "is_overall": key == "overall",
            "model_in_json": model_in_json,
            "pdf_source_in_json": pdf_in_json,
            "source_file": source_file,
        })

    summary_text = _clean_text(data.get("summary_comment"))
    if summary_text:
        rows.append({
            "pdf_id": pdf_id,
            "model": model,
            "artifact_type": artifact,
            "criterion": "summary",
            "score_1_5": "",
            "comment": summary_text,
            "is_overall": False,
            "model_in_json": model_in_json,
            "pdf_source_in_json": pdf_in_json,
            "source_file": source_file,
        })
    return rows


def main() -> int:
    if not EVAL_DIR.is_dir():
        print(f"No existe la carpeta {EVAL_DIR}", file=sys.stderr)
        return 1

    files = sorted(EVAL_DIR.glob("*.json"))
    if not files:
        print(f"No hay .json en {EVAL_DIR}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    skipped: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()
    inconsistencies: list[str] = []

    for f in files:
        parsed = parse_filename(f.stem)
        if not parsed:
            skipped.append(f"{f.name} (nombre no encaja con <pdf>_<artifact>_<size>)")
            continue
        pdf_id, artifact, model = parsed
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped.append(f"{f.name} (JSON inválido: {exc})")
            continue
        if not isinstance(data, dict):
            skipped.append(f"{f.name} (JSON no es un objeto)")
            continue

        rows = extract_rows(
            data,
            pdf_id=pdf_id,
            model=model,
            artifact=artifact,
            source_file=f.name,
        )
        if not rows:
            skipped.append(f"{f.name} (sin criterios)")
            continue

        # Aviso si el metadata del JSON discrepa del nombre del fichero.
        sample = rows[0]
        if sample["model_in_json"] and sample["model_in_json"] != model:
            inconsistencies.append(
                f"{f.name}: model en JSON='{sample['model_in_json']}' "
                f"vs derivado del nombre='{model}'"
            )

        seen_keys.add((pdf_id, model, artifact))
        all_rows.extend(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pdf_id", "model", "artifact_type", "criterion",
        "score_1_5", "comment", "is_overall",
        "model_in_json", "pdf_source_in_json", "source_file",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        writer.writeheader()
        writer.writerows(all_rows)

    expected = 4 * 5 * 2  # 4 PDFs × 5 modelos × 2 artefactos
    print(f"OK  {OUT_CSV.relative_to(ROOT)}")
    print(f"    Archivos procesados : {len(files) - len(skipped)} / {len(files)}")
    print(f"    Ejecuciones únicas  : {len(seen_keys)} (esperadas {expected})")
    print(f"    Filas en CSV (largo): {len(all_rows)}")
    if inconsistencies:
        print("    Avisos de metadata:")
        for w in inconsistencies:
            print(f"      ! {w}")
    if skipped:
        print(f"    Omitidos: {len(skipped)}")
        for s in skipped:
            print(f"      - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
