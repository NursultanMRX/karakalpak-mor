# Multi-stage build with frontend support
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ ./

# Build the application
RUN npm run build

# Production Stage
FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install CPU-only PyTorch first (saves ~4 GB vs CUDA version)
# Separate layer so it's cached independently of app code changes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files (owned by appuser from the start)
COPY --chown=appuser:appuser . .

# Copy built frontend from builder stage
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist ./static

USER appuser

# Use $PORT for Railway/VPS compatibility, default 8000
ENV PORT=8000
EXPOSE ${PORT}

# Health check: model takes ~60-90s to load, so start period is generous
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
