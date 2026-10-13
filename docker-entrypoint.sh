#!/bin/sh
set -e

PORT="${PORT:-7860}"
FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-*}"

# If running in production mode and API_KEY is unset, provide a fallback key so container boots reliably
if [ "$ENVIRONMENT" = "production" ] && [ -z "$API_KEY" ]; then
    echo "[Notice] Running in production without API_KEY. Setting fallback API_KEY to prevent startup crash."
    export API_KEY="default-production-api-key"
fi

exec /app/.venv/bin/uvicorn agenticrag.api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --proxy-headers \
    --forwarded-allow-ips="$FORWARDED_ALLOW_IPS"
