# Container image for the scoring API. Deployed on Render as a Docker web service; it
# also runs unchanged on Hugging Face Spaces.
#
# Key decisions:
#   * Serves the COMMITTED model artefacts (models/*.pkl + metadata.json): the same files
#     the Streamlit dashboard loads and the model card describes. Retraining at build
#     time would let the live API drift from the documented model, because XGBoost's
#     training arithmetic differs slightly across platforms.
#   * Installs from requirements-lock.txt so the build resolves to the exact versions the
#     artefacts were produced and tested with; unpickling needs them.
#   * Listens on $PORT when the platform sets one (Render does), otherwise on 7860.
#   * Runs as non-root uid 1000.
#   * No GPU: XGBoost + SHAP over 15 tabular features runs comfortably on a small CPU.
#   * To retrain inside the image instead, add `RUN python run.py` after the COPY lines,
#     and publish the metrics that image produces, because they will differ slightly.

FROM python:3.13-slim

# libgomp1 is required by XGBoost's OpenMP runtime; curl is used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR $HOME/app

# Dependencies first so the layer caches across source edits.
COPY --chown=user requirements-lock.txt ./
RUN pip install --no-cache-dir --user -r requirements-lock.txt

# Source.
COPY --chown=user config.py run.py sample_order.json ./
COPY --chown=user data/ ./data/
COPY --chown=user models/ ./models/
COPY --chown=user evaluate/ ./evaluate/
COPY --chown=user api/ ./api/
COPY --chown=user dashboard/ ./dashboard/

# No training step: the image serves the committed artefacts copied above, so the live
# API, the dashboard and the model card all describe the same model.

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/health" || exit 1

# One container serves the API. The Streamlit dashboard runs as its own service (on
# Render: a Python web service started with
#   streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port $PORT).
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
