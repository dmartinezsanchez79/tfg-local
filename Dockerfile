# syntax=docker/dockerfile:1.7

# Imagen ligera de Python 3.11. Empaqueta la app Streamlit y todas las
# dependencias de procesamiento (PyMuPDF, python-pptx, reportlab, etc.).
# Ollama queda FUERA del contenedor: la app se conecta al servicio del
# host mediante OLLAMA_BASE_URL (ver docker-compose.yml).
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="tfg-local" \
      org.opencontainers.image.description="Generador offline de Quiz + Presentación desde PDF (Streamlit + Ollama)." \
      org.opencontainers.image.source="https://github.com/dmartinezsanchez79/tfg-local"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instalación de dependencias en una capa propia para aprovechar la
# caché de Docker entre builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia del código de la aplicación. .dockerignore excluye dataset,
# resultados de benchmark, virtualenv, caches, etc.
COPY app.py plantilla_universidad.pptx ./
COPY src ./src
COPY .streamlit ./.streamlit

# Ejecutar como usuario no privilegiado.
RUN useradd --create-home --uid 1001 app \
 && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
