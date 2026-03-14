FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOGURU_LEVEL=INFO
ENV LOG_PATH=/app/logs

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('/app/data/trades.db').execute('SELECT 1')" || exit 1

ENTRYPOINT ["python", "main.py"]
