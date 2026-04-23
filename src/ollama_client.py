"""Cliente Ollama local con detección robusta de errores de infraestructura."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLM_TEMPERATURE, LLM_TIMEOUT_S, NUM_CTX, OLLAMA_BASE_URL
from .exceptions import OllamaError, OllamaModelNotFoundError, OllamaUnavailableError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Envoltorio sobre `ollama-python` con errores de dominio y reintentos."""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        num_ctx: int = NUM_CTX,
        timeout_s: int = LLM_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self._client = ollama.Client(host=base_url, timeout=timeout_s)

    # ---------------------------------------------------------------- health

    def check_server(self) -> None:
        """Verifica que Ollama esté corriendo. Lanza OllamaUnavailableError si no."""
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=5.0).raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailableError(
                f"No se pudo conectar con Ollama en {self.base_url}. "
                "Asegúrate de que el servicio está corriendo (`ollama serve`)."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"Ollama respondió con error: {exc}") from exc

    def check_model_available(self) -> None:
        """Verifica que el modelo esté descargado localmente.

        Ollama exige el nombre EXACTO (con tag) en /api/generate. Por eso solo
        aceptamos match por nombre completo; un match por "base name" pasaría
        el preflight pero fallaría después con 404 en la primera generación.
        """
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            names_full = {m.get("name", "") for m in data.get("models", [])}
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"No se pudo consultar los modelos: {exc}") from exc

        if self.model in names_full:
            return
        raise OllamaModelNotFoundError(
            f"El modelo '{self.model}' no está instalado localmente. "
            f"Modelos disponibles: {sorted(names_full) or 'ninguno'}. "
            f"Descárgalo con: `ollama pull {self.model}`"
        )

    def preflight(self) -> None:
        """Chequeo completo: servidor + modelo."""
        self.check_server()
        self.check_model_available()

    # ---------------------------------------------------------------- generate

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, ollama.ResponseError)),
    )
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Genera una respuesta del modelo. Si json_mode=True fuerza salida JSON."""
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            kwargs["system"] = system
        if json_mode:
            kwargs["format"] = "json"

        try:
            resp = self._client.generate(**kwargs)
        except ollama.ResponseError as exc:
            if "model" in str(exc).lower() and "not found" in str(exc).lower():
                raise OllamaModelNotFoundError(str(exc)) from exc
            raise OllamaError(f"Ollama rechazó la petición: {exc}") from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Se perdió la conexión con Ollama durante la generación."
            ) from exc

        content = resp.get("response", "") if isinstance(resp, dict) else getattr(resp, "response", "")
        if not content:
            raise OllamaError("Ollama devolvió una respuesta vacía.")
        return content

    # ---------------------------------------------------------------- json helper

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Genera JSON y lo parsea. Aplica una limpieza defensiva de code fences."""
        raw = self.generate(
            prompt, system=system, json_mode=True, temperature=temperature
        )
        return _parse_json_loose(raw)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_json_loose(text: str) -> Any:
    """Intenta parsear JSON, removiendo ```json fences y ruido común."""
    text = text.strip()
    if not text:
        raise OllamaError("Respuesta JSON vacía.")
    # Code fences
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Cortar desde el primer { o [ y hasta el último } o ]
    first = min(
        (i for i in (text.find("{"), text.find("[")) if i >= 0),
        default=-1,
    )
    last = max(text.rfind("}"), text.rfind("]"))
    if first >= 0 and last > first:
        text = text[first : last + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"El LLM no devolvió JSON válido: {exc}. Respuesta: {text[:400]}") from exc
