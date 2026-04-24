"""Módulo de benchmarking offline para el TFG.

Ejecuta combinaciones (PDF × modelo) reutilizando el pipeline de `src/`
(PDF → Markdown → KnowledgeBase → Quiz/PPTX), calcula métricas automáticas
simples y genera prompts de evaluación externa con IA asistida.

Diseño intencionalmente pequeño: sin dependencias nuevas, sin Streamlit,
CLI única en `benchmark.runner`.
"""
from __future__ import annotations

__all__ = [
    "config",
    "metrics",
    "judge_prompts",
    "reports",
]
