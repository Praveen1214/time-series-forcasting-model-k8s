# ============================================================
# Stage 1: Builder — install dependencies in a virtual env
# ============================================================
FROM python:3.10-slim-bullseye AS builder

WORKDIR /build

# Build toolchain (only in builder)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual env so we can copy it cleanly to runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install Python dependencies (production requirements)
COPY api/requirements.prod.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ============================================================
# Stage 2: Runtime — lean production image
# ============================================================
FROM python:3.10-slim-bullseye AS runtime

# Labels
LABEL maintainer="K8s Proactive Scaler" \
      description="BiLSTM inference API for predictive Kubernetes autoscaling" \
      version="1.0.0"

# Minimal runtime libraries:
# - curl: for healthcheck
# - libstdc++6/libgcc-s1: very common TF runtime deps on slim
# - libgomp1: common numeric/TF dep
# - libhdf5-310: for Keras HDF5/backends (may be needed depending on your stack)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libstdc++6 \
        libgcc-s1 \
        libgomp1 \
        libhdf5-103-1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy the pre-built virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Create directories explicitly (safer) and set ownership
RUN mkdir -p /app/api /app/models /app/data && \
    chown -R appuser:appuser /app

# Copy application code (owned by non-root user)
COPY --chown=appuser:appuser api/ /app/api/

# Copy only the model files we need (owned by non-root user)
COPY --chown=appuser:appuser notebooks/models/bilstm-pods-20260104_012142.keras /app/models/bilstm-pods-20260104_012142.keras
COPY --chown=appuser:appuser notebooks/models/bilstm-scalers-20260104_012142.pkl  /app/models/bilstm-scalers-20260104_012142.pkl

# Copy data (owned by non-root user)
COPY --chown=appuser:appuser data/data.csv /app/data/data.csv

# Switch to non-root user
USER appuser

# Environment variables (paths match where we copied files)
ENV MODEL_PATH=/app/models/bilstm-pods-20260104_012142.keras \
    SCALERS_PATH=/app/models/bilstm-scalers-20260104_012142.pkl

# Expose port
EXPOSE 8000

# Health check — make sure your FastAPI app has GET /health returning 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run with Gunicorn + Uvicorn worker (1 worker is safer for TF/Keras)
# NOTE: ensure gunicorn + uvicorn are in requirements.prod.txt
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]