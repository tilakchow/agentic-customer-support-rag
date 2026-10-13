# ==========================================
# Stage 1: Build & Dependencies
# ==========================================
FROM python:3.12-slim AS builder

# Install uv from pinned official image
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install virtual environment dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and documentation
COPY README.md ./
COPY src/ ./src/

# Install the project into the virtual environment
RUN uv sync --frozen --no-dev

# Pre-download FastEmbed model during image build to avoid runtime cold starts
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.cache/fastembed')"

# ==========================================
# Stage 2: Final Minimal Runtime Image
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Create unprivileged application user
RUN useradd -m -u 1000 appuser

# Copy virtualenv, cached embeddings, and application files from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/README.md
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Set ownership and permissions
RUN chmod +x /app/docker-entrypoint.sh && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV CHECKPOINT_DB=/app/checkpoints.sqlite
ENV FORWARDED_ALLOW_IPS=*

# Expose default port (Hugging Face Spaces default 7860)
EXPOSE 7860

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request, os; p = os.environ.get('PORT', '7860'); urllib.request.urlopen(f'http://localhost:{p}/health')" || exit 1

# Launch uvicorn via exec-form entrypoint as PID 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
