"""Configuración global del sistema."""
from __future__ import annotations

from pathlib import Path
from typing import Final

# --- Rutas ---
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
TEMPLATE_PATH: Final[Path] = ROOT_DIR / "plantilla_universidad.pptx"

# --- Ollama ---
OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
AVAILABLE_MODELS: Final[tuple[str, ...]] = (
    "qwen2.5:14b",
    "gemma3:12b",
    "qwen2.5:7b",
    "gemma2:9b",
    "gemma3:4b",
    "mistral:7b",
    "llama3.2:3b",
)
DEFAULT_MODEL: Final[str] = "qwen2.5:14b"

# --- Inferencia ---
LLM_TEMPERATURE: Final[float] = 0.2
LLM_TIMEOUT_S: Final[int] = 600
NUM_CTX: Final[int] = 8192  # ventana de contexto típica para 8-12GB VRAM

# --- Chunking / Map-Reduce ---
# Caracteres por chunk: ~3000 caracteres ≈ ~750 tokens, deja margen para prompt + respuesta
CHUNK_SIZE_CHARS: Final[int] = 3000
CHUNK_OVERLAP_CHARS: Final[int] = 250
MAX_INPUT_CHARS: Final[int] = 50_000  # límite duro de entrada
MAX_INPUT_PAGES: Final[int] = 50

# --- Quiz ---
# Rango fijo de preguntas: simplifica la interfaz y mantiene consistencia
# entre ejecuciones y comparativas de modelos.
MIN_NUM_QUESTIONS: Final[int] = 7
MAX_NUM_QUESTIONS: Final[int] = 15
DEFAULT_NUM_QUESTIONS_RANGE: Final[tuple[int, int]] = (7, 15)

# --- PPTX ---
# Mapping verificado contra `plantilla_universidad.pptx`:
#   layout[0] = TITLE              (TITLE + SUBTITLE)        -> portada
#   layout[1] = SECTION_HEADER     (solo TITLE, sin cuerpo)  -> NO sirve para contenido
#   layout[2] = TITLE_AND_BODY     (TITLE + BODY)            -> diapositivas de contenido
LAYOUT_TITLE: Final[int] = 0
LAYOUT_CONTENT: Final[int] = 2
MAX_BULLETS_PER_SLIDE: Final[int] = 5
# Objetivo de ~2 líneas a fuente 20pt con la plantilla actual. No es un
# tope duro visual (auto_size ajusta), pero es el límite que se comunica
# al LLM y el umbral para recortar bullets demasiado largos.
MAX_CHARS_PER_BULLET: Final[int] = 180
MAX_CHARS_SLIDE_TITLE: Final[int] = 80
DEFAULT_NUM_SLIDES_MIN: Final[int] = 6
DEFAULT_NUM_SLIDES_MAX: Final[int] = 14
