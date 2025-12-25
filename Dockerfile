# Timestamp (UTC): 2025-12-22T17:35:36Z
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# lightgbm wheels require OpenMP runtime on Debian.
# dydx-v4-client deps may require compiling wheels (e.g. ed25519-blake2b).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/data/live

# Default to safe paper mode; override via docker run/compose when ready.
CMD ["python3", "scripts/live_paper_trade_2025-12-20T15-01-14Z.py", "--mode", "live", "--trade-mode", "paper"]
