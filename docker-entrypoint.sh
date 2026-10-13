#!/bin/sh
set -e

PORT="${PORT:-7860}"

# Run uvicorn without naive --proxy-headers=* so application-level
# get_client_ip() securely validates TRUSTED_PROXIES and prevents IP spoofing
exec /app/.venv/bin/uvicorn agenticrag.api:app \
    --host 0.0.0.0 \
    --port "$PORT"
