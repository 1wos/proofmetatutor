#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-asia-northeast3}"
REPOSITORY="${REPOSITORY:-prooftutor}"
SERVICE="${SERVICE:-prooftutor-verifier}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE}:latest"
# Bucket holding the servable TPU-trained artifact (outputs/verifier_v2/).
MODEL_BUCKET="${MODEL_BUCKET:-YOUR_GCS_BUCKET}"

gcloud config set project "${PROJECT_ID}"
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="ProofMetaTutor containers" || true

# Build the verifier *API* serving image (FastAPI + flax), not the trainer.
# A Cloud Build config is required because `--tag` cannot target a non-root
# Dockerfile path.
gcloud builds submit \
  --config infra/cloudbuild_serve.yaml \
  --substitutions "_IMAGE=${IMAGE}" \
  .

# Mount the model bucket at /gcs/<bucket> so model.py finds verifier_v2/flax_model.
# 678 MiB flax model + jax needs more than the 512Mi default memory.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --add-volume "name=models,type=cloud-storage,bucket=${MODEL_BUCKET}" \
  --add-volume-mount "volume=models,mount-path=/gcs/${MODEL_BUCKET}"

