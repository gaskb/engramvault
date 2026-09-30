FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEMORIA_DATA_DIR=/app/data \
    MEMORIA_DB_PATH=/app/data/vault.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY engramvault/ ./engramvault/

RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python", "-m", "engramvault.cli.main", "serve", "--host", "0.0.0.0", "--port", "8080", "--db", "/app/data/vault.db"]
