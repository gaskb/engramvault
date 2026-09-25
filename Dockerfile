FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEMORIA_DATA_DIR=/app/data \
    MEMORIA_DB_PATH=/app/data/vault.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY memoria/ ./memoria/
RUN pip install --no-cache-dir .

RUN groupadd -g 1000 appuser \
    && useradd -u 1000 -g 1000 -m -s /bin/sh appuser \
    && mkdir -p /app/data /app/data/incoming \
    && chown -R 1000:1000 /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/stats >/dev/null || exit 1

CMD ["python", "-m", "memoria.cli.main", "serve", "--host", "0.0.0.0", "--port", "8080", "--db", "/app/data/vault.db"]
