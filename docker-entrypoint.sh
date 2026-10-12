#!/bin/sh
set -e

PORT="${PORT:-7860}"
FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-*}"

exec /app/.venv/bin/uvicorn agenticrag.api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --proxy-headers \
    --forwarded-allow-ips="$FORWARDED_ALLOW_IPS"
