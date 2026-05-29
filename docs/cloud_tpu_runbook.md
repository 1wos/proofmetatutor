# ProofMetaTutor — Cloud TPU Runbook (reproducible)

Every command needed to reproduce the project on Google Cloud: data → **TPU
training** → serving. Two training paths are documented:

1. **Encoder verifier** (mBERT, JAX/Flax) — the shipped baseline.
2. **Gemma tutor** (Keras 3 / JAX + LoRA) — the decoder-LM upgrade that
   genuinely needs a TPU.

> Project: `YOUR_GCP_PROJECT` · Bucket: `gs://YOUR_GCS_BUCKET` ·
> TPU: `v6e-1` (`us-east5-a`). Adjust to your project/zone.

---

## 0. One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT
gcloud auth application-default login

# APIs
gcloud services enable \
  tpu.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com

# Training bucket
gsutil mb -l us-east5 gs://YOUR_GCS_BUCKET || true
```

---

## 1. Data

```bash
# Public HF datasets (no approval needed)
pip install -e ".[training]"
python scripts/download_hf_datasets.py          # KMMLU math, gsm8k

# AIHub No.30 Korean math -> step-native jsonl
python scripts/prepare_aihub_math.py            # -> data/aihub/math_train.jsonl (16,246)

# Misconception negatives via self-hosted Gemma (Vertex Model Garden)
python scripts/deploy_gemma.py                  # deploy endpoint
python scripts/gen_gemma_negatives.py           # -> data/synthetic/negatives_train.jsonl
python scripts/teardown_gemma.py                # ALWAYS: GPU endpoints don't scale to zero

# Assemble the verifier training set
python scripts/assemble_training.py             # -> data/training/verifier_train.jsonl (20,127)

# Push derived data to GCS
gsutil -m cp -r data gs://YOUR_GCS_BUCKET/
```

> Cost lesson learned: a Vertex GPU endpoint left up idle ~14h before we ran
> `teardown_gemma.py`. Deploy right before generation, tear down right after.

---

## 2A. Train the encoder verifier on Cloud TPU (mBERT, JAX/Flax)

```bash
# Create a v6e-1 TPU VM (sweep zones if capacity is out -> error code 8)
ZONE=us-east5-a
gcloud compute tpus tpu-vm create proof-v6e \
  --zone=$ZONE --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e

# Ship code + install
gcloud compute tpus tpu-vm scp --zone=$ZONE --recurse . proof-v6e:~/tpubuilders
gcloud compute tpus tpu-vm ssh proof-v6e --zone=$ZONE --command '
  cd tpubuilders &&
  pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
  pip install -e ".[training]"'

# Train (GCS is fuse-mounted at /gcs on the TPU VM)
gcloud compute tpus tpu-vm ssh proof-v6e --zone=$ZONE --command '
  cd tpubuilders &&
  python training/verifier/train_jax_tpu.py \
    --train /gcs/YOUR_GCS_BUCKET/data/training/verifier_train.jsonl \
    --output-dir /gcs/YOUR_GCS_BUCKET/outputs/verifier_v2 \
    --backbone google-bert/bert-base-multilingual-cased \
    --epochs 3 --batch-size 32'

# ALWAYS delete the VM when done (TPU VMs bill while up)
gcloud compute tpus tpu-vm delete proof-v6e --zone=$ZONE --quiet
```

Result (verified): `val_accuracy 0.880`, error-class P/R/F1 `0.909/0.760/0.828`,
and `reload_val_accuracy == val_accuracy` (save+reload self-check proves the
artifact holds the trained weights). Servable artifact:
`gs://YOUR_GCS_BUCKET/outputs/verifier_v2/flax_model`.

> Why a save+reload check: an earlier run saved `model.params` (untrained init)
> instead of `state.params`, so the GCS artifact scored near chance while the
> in-memory eval looked fine. The trainer now reloads from disk and re-evals.

---

## 2B. Fine-tune Gemma on Cloud TPU (Keras 3 / JAX + LoRA) — the TPU-justified upgrade

A 178M encoder does not need a TPU; a multi-billion-param **Gemma** decoder does.
KerasHub shards Gemma across the TPU and LoRA keeps the trainable set small.

```bash
# SFT data from the same step-level set
python training/gemma_tutor/prepare_gemma_sft.py \
  --in data/training/verifier_train.jsonl --out data/training/gemma_sft.jsonl
gsutil cp data/training/gemma_sft.jsonl gs://YOUR_GCS_BUCKET/data/training/

# TPU VM (v6e-1 fine for 2B+LoRA; use a v6e-8 pod slice to scale up)
ZONE=us-east5-a
gcloud compute tpus tpu-vm create gemma-v6e \
  --zone=$ZONE --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e

gcloud compute tpus tpu-vm scp --zone=$ZONE --recurse . gemma-v6e:~/tpubuilders
gcloud compute tpus tpu-vm ssh gemma-v6e --zone=$ZONE --command '
  cd tpubuilders &&
  pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
  pip install -e ".[gemma]"'

# Gated Gemma weights: set Kaggle creds (or `kagglehub login`)
gcloud compute tpus tpu-vm ssh gemma-v6e --zone=$ZONE --command '
  cd tpubuilders &&
  export KAGGLE_USERNAME=YOUR_USER KAGGLE_KEY=YOUR_KEY KERAS_BACKEND=jax &&
  python training/gemma_tutor/train_gemma_lora_tpu.py \
    --train /gcs/YOUR_GCS_BUCKET/data/training/gemma_sft.jsonl \
    --preset gemma2_2b_en \
    --output-dir /gcs/YOUR_GCS_BUCKET/outputs/gemma_tutor \
    --epochs 1 --batch-size 8 --lora-rank 4 --max-len 512'

# (smoke test first: add `--limit 64` for a 1-minute sanity run)
gcloud compute tpus tpu-vm delete gemma-v6e --zone=$ZONE --quiet
```

Artifact: `gs://YOUR_GCS_BUCKET/outputs/gemma_tutor/gemma_tutor_preset`
plus `sample_generation.txt` and `train_meta.json` (records the JAX TPU devices used).

---

## 3. Serve the verifier on Cloud Run

```bash
PROJECT_ID=YOUR_GCP_PROJECT REGION=asia-northeast3 bash infra/deploy_cloud_run.sh
```

This builds `services/verifier_api/Dockerfile` (FastAPI + flax), mounts the model
bucket at `/gcs/YOUR_GCS_BUCKET` as a Cloud Storage volume, and deploys
with 4Gi/2cpu. Verify:

```bash
URL=$(gcloud run services describe prooftutor-verifier \
  --project=YOUR_GCP_PROJECT --region=asia-northeast3 --format='value(status.url)')
curl "$URL/health"
curl -X POST "$URL/api/verifier/run" -H 'content-type: application/json' \
  -d '{"problem_text":"x + 5 = 9","step_text":"x = 4"}'
# logs should show: [verifier] loaded flax encoder from .../verifier_v2/flax_model
```

> Serving pin gotcha: `transformers` v5 removed Flax. The serving image stays on
> `transformers<5` + explicit `jax[cpu]` + `flax`, or `FlaxAutoModelForSequenceClassification`
> is unimportable and the API silently falls back to a keyword heuristic.

---

## 4. Web UI

```bash
cd apps/web
cp .env.example .env          # NEXT_PUBLIC_VERIFIER_API is prefilled to the live URL
npm install && npm run dev    # http://localhost:3000
```

---

## Cost guardrails (always)
- TPU VMs bill while **up** — `tpu-vm delete` the moment training ends.
- Vertex GPU endpoints **don't** scale to zero — `teardown_gemma.py` after generation.
- Cloud Run **does** scale to zero — safe to leave deployed.
