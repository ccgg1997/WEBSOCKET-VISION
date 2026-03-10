FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_WS_MODEL_PATH=/app/runtime/model.pt \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /tmp/Ultralytics && \
    chown -R appuser:appuser /tmp/Ultralytics

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 CMD \
    python -c "import os, urllib.request; port = os.environ.get('YOLO_WS_PORT', os.environ.get('PORT', '8000')); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=3).read()"

CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host ${YOLO_WS_HOST:-0.0.0.0} --port ${YOLO_WS_PORT:-${PORT:-8000}}"]
