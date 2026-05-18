FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

FROM base AS builder

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM base AS runtime

COPY --from=builder /install /usr/local

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

COPY --chown=appuser:appgroup orchestrator/ orchestrator/

USER appuser

EXPOSE 8000

CMD uvicorn orchestrator.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
