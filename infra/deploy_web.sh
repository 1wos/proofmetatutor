#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-asia-northeast3}"
REPOSITORY="${REPOSITORY:-prooftutor}"
SERVICE="${SERVICE:-prooftutor-web}"
API_URL="${API_URL:?Set API_URL to the verifier Cloud Run URL}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE}:latest"

gcloud config set project "${PROJECT_ID}"
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker --location="${REGION}" \
  --description="ProofMetaTutor containers" || true

# Build the Next.js UI; bake the API URL into the client bundle.
gcloud builds submit \
  --config infra/cloudbuild_web.yaml \
  --substitutions "_IMAGE=${IMAGE},_API_URL=${API_URL}" \
  .

gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080
