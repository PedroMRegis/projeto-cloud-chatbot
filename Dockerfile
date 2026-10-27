# Multi-stage: as dependências são instaladas no builder e só o venv vai para
# a imagem final, que não tem compilador nem cache de pip.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SERVICE_NAME=chatbot

# curl é usado pelo HEALTHCHECK do container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10003 appuser

COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app /srv/app
COPY scripts /srv/scripts

# Não-root: o container não precisa de privilégio para servir HTTP.
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 2 workers: a carga é I/O-bound (espera de LLM e do backend), então o
# paralelismo vem de workers, não de CPU.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--no-access-log", "--timeout-graceful-shutdown", "25"]
