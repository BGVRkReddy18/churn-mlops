# ---------- build stage: install dependencies ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements/serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r serve.txt

# ---------- runtime stage: small image, non-root user ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    PORT=8000

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/churn/__init__.py src/churn/schema.py ./src/churn/
COPY src/churn/api ./src/churn/api
COPY models/model.joblib models/metadata.json ./models/
RUN mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec uvicorn churn.api.main:app --host 0.0.0.0 --port ${PORT}"]
