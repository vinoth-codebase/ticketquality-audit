# syntax=docker/dockerfile:1
#
# TicketQuality-Audit — production image
# --------------------------------------
# Multi-stage build:
#   1. `builder`  installs the package and its dependencies into a venv.
#   2. `runtime`  copies only the venv and the source, then runs as a
#      non-root user.
#
# libgomp1 is a hard runtime dependency of LightGBM's native library; it is
# installed explicitly so the image does not depend on a transitive wheel.

# --------------------------------------------------------------------------- #
# Stage 1 — builder
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# --------------------------------------------------------------------------- #
# Stage 2 — runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="ticketquality-audit" \
      org.opencontainers.image.description="Premature closure & re-open risk forecaster for software support" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Vinothkumar N <vinothofficial14@gmail.com>"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TICKETQUALITY_CONFIG=/app/config/config.yaml

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 auditor

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=auditor:auditor src ./src
COPY --chown=auditor:auditor config ./config
COPY --chown=auditor:auditor scripts ./scripts
COPY --chown=auditor:auditor docs ./docs
COPY --chown=auditor:auditor examples ./examples
COPY --chown=auditor:auditor README.md ./

RUN mkdir -p /app/data /app/artifacts && chown -R auditor:auditor /app

USER auditor

# Train-on-build so the container is immediately useful, then keep the
# artefact. Failures here are non-fatal: the CLI/dashboard retrain on demand.
RUN python scripts/generate_audit_dataset.py --samples 1000 \
        --output data/synthetic_tickets.csv \
    && ticketquality train --data data/synthetic_tickets.csv \
        --model-out artifacts/reopen_risk_model.joblib \
    || echo "warm-up training skipped"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import ticketquality, lightgbm, shap; print(ticketquality.__version__)" || exit 1

# Default: serve the visual-management dashboard.
CMD ["streamlit", "run", "src/ticketquality/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
