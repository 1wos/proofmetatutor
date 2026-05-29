# Serving image for the ProofMetaTutor verifier API (FastAPI + flax mBERT).
# Loads the TPU-trained verifier_v2 artifact from a Cloud Storage volume mount.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY services ./services
RUN pip install --no-cache-dir -e ".[serving]"

ENV PORT=8080
# Default to the GCS-mounted servable artifact; override with VERIFIER_MODEL_DIR.
# (The old outputs/verifier/ is the untrained-save artifact and must not be served.)
ENV VERIFIER_MODEL_DIR=/gcs/YOUR_GCS_BUCKET/outputs/verifier_v2/flax_model

CMD ["sh", "-c", "uvicorn services.verifier_api.app.main:app --host 0.0.0.0 --port ${PORT}"]
