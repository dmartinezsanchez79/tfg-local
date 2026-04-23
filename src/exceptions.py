"""Excepciones de dominio para diferenciar errores de usuario, infraestructura y datos."""
from __future__ import annotations


class AppError(Exception):
    """Error base de la aplicación."""


class PDFError(AppError):
    """El PDF no se puede procesar (corrupto, escaneado sin OCR, vacío…)."""


class ScannedPDFError(PDFError):
    """El PDF parece ser una imagen escaneada sin capa de texto."""


class PDFTooLargeError(PDFError):
    """El PDF supera los límites de páginas o caracteres permitidos."""


class OllamaError(AppError):
    """Error genérico al comunicarse con Ollama."""


class OllamaUnavailableError(OllamaError):
    """No hay servidor de Ollama escuchando en el endpoint configurado."""


class OllamaModelNotFoundError(OllamaError):
    """El modelo seleccionado no está descargado localmente."""


class GenerationError(AppError):
    """Fallo durante la generación de Quiz o PPTX."""


class TemplateError(AppError):
    """La plantilla PPTX no es válida o falta."""
