FROM python:3.14-slim

# Install uv from pinned official image
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (without installing the project code yet to cache layers)
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY README.md ./
COPY src/ ./src/

# Install the project itself
RUN uv sync --frozen --no-dev

# Set explicit cache directory and pre-download FastEmbed model during build
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.cache/fastembed')"

# Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for FastAPI (Hugging Face Spaces requires 7860)
EXPOSE 7860

# Health check on /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request, os; p = os.environ.get('PORT', '7860'); urllib.request.urlopen(f'http://localhost:{p}/health')" || exit 1

# Run the FastAPI app with proxy headers and configurable forwarded-allow-ips
CMD /app/.venv/bin/uvicorn agenticrag.api:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-*}"
