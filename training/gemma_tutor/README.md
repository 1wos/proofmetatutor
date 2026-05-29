# Gemma tutor — LoRA fine-tune on Cloud TPU (the generative verifier)

The shipped mBERT verifier scores step *plausibility* but cannot reason about
arithmetic (a deterministic sympy checker covers that today — see
`services/verifier_api/app/arithmetic.py`). The **generative** upgrade is a
LoRA-fine-tuned **Gemma** that can both judge a step *and explain why it is
wrong* — and, being a multi-billion-parameter decoder, it genuinely needs a TPU
to fine-tune. That is the "why TPU" closing argument for the project.

## Pipeline (this folder)
- `prepare_gemma_sft.py` — turns the step-level dataset into instruction/response
  SFT records (CORRECT / INCORRECT + reason).
- `train_gemma_lora_tpu.py` — Keras 3 (JAX backend) + KerasHub `GemmaCausalLM`,
  LoRA, sharded across the TPU. Saves a servable preset + a sample generation.

## Run it (one TPU VM)
Full commands are in [`docs/cloud_tpu_runbook.md`](../../docs/cloud_tpu_runbook.md) §2B. In short:

```bash
python training/gemma_tutor/prepare_gemma_sft.py \
  --in data/training/verifier_train.jsonl --out data/training/gemma_sft.jsonl

# on a v6e TPU VM, with Gemma weights authorised:
export KERAS_BACKEND=jax KAGGLE_USERNAME=... KAGGLE_KEY=...
python training/gemma_tutor/train_gemma_lora_tpu.py \
  --train /gcs/YOUR_GCS_BUCKET/data/training/gemma_sft.jsonl \
  --preset gemma2_2b_en \
  --output-dir /gcs/YOUR_GCS_BUCKET/outputs/gemma_tutor \
  --epochs 1 --lora-rank 4
```

## Run status — DONE (2026-05-29)
Executed on a **Cloud TPU v6e-1** (us-east5-a), Keras 3 / JAX backend:
`gemma-2-2b-it`, LoRA rank 4, 1 epoch over 1,500 step-level SFT rows.

Artifact: `gs://YOUR_GCS_BUCKET/outputs/gemma_tutor/gemma_out/`
- `gemma_tutor_preset/` (model.weights.h5 + tokenizer + config) — servable KerasHub preset
- `train_meta.json` (records `jax_devices: TPU_0`), `sample_generation.txt`

The fine-tuned model emits the instruction-tuned verdict format (e.g.
`Verdict: CORRECT`). This is a proof-of-pipeline finetune (small/1-epoch), not a
tuned-for-accuracy release — the point is a reproducible Gemma-on-TPU workflow.

Notes from the run (all fixed in this repo / runbook):
- HF→KerasHub Gemma load needs `sentencepiece` + `safetensors`.
- The script calls `keras.config.disable_flash_attention()` — Gemma2's fused
  TPU attention kernel otherwise throws a `ConcretizationTypeError` in jit.

To re-run / scale up: bump `--preset` (e.g. `hf://google/gemma-3-1b-it`) or epochs
(see [`docs/cloud_tpu_runbook.md`](../../docs/cloud_tpu_runbook.md) §2B).
