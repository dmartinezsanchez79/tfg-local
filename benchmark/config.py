"""Configuración del módulo de benchmarking.

Separada de `src/config.py` porque solo concierne al benchmark (rutas,
modelos por defecto a comparar, umbrales de métricas). El resto del
pipeline sigue usando su propia configuración intacta.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# --- Rutas -----------------------------------------------------------------

BENCH_DIR: Final[Path] = Path(__file__).resolve().parent
DATASET_DIR: Final[Path] = BENCH_DIR / "dataset"
DATASET_PDFS_DIR: Final[Path] = DATASET_DIR / "pdfs"
CATALOG_PATH: Final[Path] = DATASET_DIR / "catalog.json"
RESULTS_DIR: Final[Path] = BENCH_DIR / "results"
REPORTS_DIR: Final[Path] = BENCH_DIR / "reports"

# Carpeta del proyecto con los PDFs originales: se usa como segundo lugar
# de búsqueda si el PDF no está en `dataset/pdfs/`.
PROJECT_PDFS_DIR: Final[Path] = BENCH_DIR.parent / "PDF"

# --- Modelos por defecto ---------------------------------------------------
# Conjunto representativo que cubre distintos tamaños (14B → 3B) y
# familias (qwen, gemma, mistral, llama). Suficiente para el TFG.
DEFAULT_MODELS: Final[tuple[str, ...]] = (
    "qwen2.5:14b",
    "gemma3:12b",
    "qwen2.5:7b",
    "gemma2:9b",
    "gemma3:4b",
    "mistral:7b",
    "llama3.2:3b",
)

# --- Parámetros de ejecución ----------------------------------------------

# Rango de preguntas del quiz: se fija para comparabilidad. Coincide con
# el rango "fijo" del front-end (src/config.DEFAULT_NUM_QUESTIONS_RANGE).
QUIZ_MIN_QUESTIONS: Final[int] = 7
QUIZ_MAX_QUESTIONS: Final[int] = 15

# Si True, aplica el refinado determinístico (src.critics.refine_*).
# Lo mantenemos activo para medir la calidad real del pipeline completo.
REFINE_QUIZ: Final[bool] = True
REFINE_SLIDES: Final[bool] = True

# --- Umbrales para métricas automáticas -----------------------------------
# No pretenden ser "verdad absoluta": son umbrales simples, defendibles
# y explicables en la memoria. Cambiarlos es legítimo si se justifica.

QUIZ_DUPLICATE_JACCARD: Final[float] = 0.70      # similitud léxica de stems
QUIZ_OPTION_LEN_IMBALANCE: Final[float] = 2.2     # ratio max/min de longitud
QUIZ_BANNED_PHRASES: Final[tuple[str, ...]] = (
    "todas las anteriores", "ninguna de las anteriores",
    "todas son correctas", "ninguna es correcta",
    "a y b", "b y c", "a y c",
)

PPTX_BULLET_SHORT_CHARS: Final[int] = 25          # bullets demasiado cortos
PPTX_BULLET_LONG_CHARS: Final[int] = 210          # bullets demasiado largos
PPTX_MIN_BULLETS_PER_SLIDE: Final[int] = 3
PPTX_MAX_BULLETS_PER_SLIDE: Final[int] = 5
PPTX_REPEAT_JACCARD: Final[float] = 0.70          # repetición entre slides


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen. Idempotente."""
    for d in (DATASET_PDFS_DIR, RESULTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
