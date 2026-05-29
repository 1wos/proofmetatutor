# TPU Run Report

## Status

Smoke test PASSED on Cloud TPU v6e (2026-05-25).
Real verifier training COMPLETED on Cloud TPU v6e (2026-05-27), val_accuracy 0.880. See "Real verifier training run" below.
Servable re-run COMPLETED on Cloud TPU v6e (2026-05-27): save bug fixed, reload self-check passed (reload_val_accuracy 0.880 == val_accuracy 0.880), error-class P/R/F1 logged. Servable artifact at gs://YOUR_GCS_BUCKET/outputs/verifier_v2/.

## Environment

- Account: <your-account>
- Project: YOUR_GCP_PROJECT (925792993653)
- Accelerator: TPU v6e-1 (Trillium), single chip
- Zone: us-east5-b
- Runtime image: v2-alpha-tpuv6e
- Framework: JAX 0.6.2, backend = tpu
- Access path: direct Cloud TPU API (`tpu.googleapis.com`) via `gcloud compute tpus tpu-vm create`.
  NOT Vertex AI Custom Training, that quota path was auto denied for this new project.

## How TPU access actually worked

The Vertex AI Custom Training TPU quota (`aiplatform.googleapis.com`,
`CustomModelTrainingV6ETPUPerProjectPerRegion`) was granted 0 and denied.
The direct Cloud TPU API quota is separate and was open on this project:

| Chip | us-central1 quota (per zone) |
|---|---|
| v6e | 16 |
| v2 / v3 | 16 / 16 |
| v2 / v3 preemptible | 48 |
| v5e-litepod | 16 |

So the blocker was never "no TPU access". The blocker was (a) using the
most gated Vertex wrapper, and (b) hardware capacity per zone.

## Capacity sweep (the real constraint)

| Attempt | Result |
|---|---|
| v2-8 preemptible @ us-central1-f | code 8, insufficient capacity |
| v5litepod-1 @ us-central1-a | code 8, no capacity in zone |
| v6e-1 @ us-central1-a | code 8, no capacity |
| v5litepod-1 @ us-east5-a | code 5, reservation not found |
| v6e-1 @ us-east5-b | READY |

Quota check passed every time (no permission or quota error). Failures were
pure capacity stockouts. v6e-1 in us-east5-b had a free chip.

## Smoke test command

```bash
gcloud compute tpus tpu-vm create proof-smoke \
  --zone=us-east5-b --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e --project=YOUR_GCP_PROJECT
```

Device check:

```
JAX_VERSION 0.6.2
JAX_BACKEND tpu
JAX_DEVICES [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
MATMUL_SUM 134217728.0   # 512^3, matmul executed on TPU
```

## Verifier training command

```bash
python3 train_jax_tpu.py --train train.jsonl \
  --output-dir ./out --epochs 150 --learning-rate 0.3
```

## Results (metrics_jax.json)

| Metric | Value |
|---|---|
| status | ok |
| jax_backend | tpu |
| jax_devices | TPU_0(process=0,(0,0,0,0)) |
| examples | 8 |
| accuracy | 1.0 |

Note: accuracy 1.0 is on an 8 sample linearly separable toy set. This is a
TPU plumbing smoke test, not a real model metric. Real metrics come after the
AIHub pipeline and the embedding based verifier land.

## Real verifier training run (2026-05-27)

First real model training, not a toy. Same direct Cloud TPU API path, v6e-1 in us-east5-b.

### Data
- `gs://YOUR_GCS_BUCKET/data/training/verifier_train.jsonl`, 20127 step-level rows.
- 12263 gold steps (label 1, AIHub No.30), 7864 Gemma synthetic wrong steps (label 0).
- Pulled to the VM from GCS, so no GitHub auth needed on the VM.

### VM setup
The `v2-alpha-tpuv6e` image ships libtpu but no JAX. Installed in order, TPU backend re-verified after each step:

```bash
pip3 install -U "jax[tpu]"                      # JAX 0.6.2, backend=tpu, 1 TpuDevice
pip3 install flax optax "transformers<4.50"     # flax 0.10.7, optax 0.2.8, transformers 4.49.0
```

### Command

```bash
python3 train_jax_tpu.py \
  --train verifier_train.jsonl --output-dir ./out_full \
  --backbone google-bert/bert-base-multilingual-cased \
  --epochs 3 --batch-size 32
```

### Two bugs fixed in train_jax_tpu.py before it would run
1. Backbone id was `google/bert-base-multilingual-cased`. The canonical HF org for the original BERT checkpoints is `google-bert`, so the Hub returned 401 RepositoryNotFound. Fixed to `google-bert/bert-base-multilingual-cased`.
2. With `train=True` the Flax BERT dropout needs a PRNG, but `train_step` passed none, so it raised `InvalidRngError`. Added a `dropout_rng` split threaded through `train_step`.

### Results (metrics_jax.json)

| Metric | Value |
|---|---|
| status | ok |
| jax_backend | tpu |
| jax_devices | TPU_0(process=0,(0,0,0,0)) |
| backbone | google-bert/bert-base-multilingual-cased |
| examples / n_train / n_val | 20127 / 18115 / 2012 |
| epochs | 3 |
| val_accuracy | 0.880 |
| train loss | 0.340 -> 0.305 -> 0.283 |
| epoch time | 109s (epoch 1, incl XLA compile), then 27s |

Artifacts in `gs://YOUR_GCS_BUCKET/outputs/verifier/`: metrics_jax.json + flax_model (678 MiB msgpack, tokenizer, vocab).

> **Save bug (found and fixed 2026-05-27, verified by re-run).**
> The first run's `train_jax_tpu.py` called `model.save_pretrained()`, which serializes `model.params`. The trained weights lived in the optax `TrainState` (`state.params`) and were never written back, so that run's persisted msgpack (in `outputs/verifier/`) is base mBERT plus a random classification head. The 0.880 eval used `state.params`, so the number was real, but the v1 artifact is not servable.
> Fixed by passing `params=state.params` to `save_pretrained`, plus a reload self-check: the trainer reloads the saved model from disk and re-evals it (`reload_val_accuracy`). The 2026-05-27 re-run produced `reload_val_accuracy` 0.880, exactly equal to `val_accuracy` 0.880, which proves the saved artifact holds the trained weights (an untrained reload would sit near chance). The servable model is `outputs/verifier_v2/`; the old `outputs/verifier/` is left untouched and should not be served.

### Honest reading of 0.880
- Above the 61 percent majority baseline (positive class share), so the model learned real signal, not a coin flip.
- The val negatives are all Gemma synthetic corruptions, so this measures detection of Gemma style wrong steps, in distribution. It is not a measurement on real student errors.
- About 5 percent of the negatives are false negatives (mathematically fine but labeled 0). That label noise caps the achievable accuracy.
- The re-run reports precision / recall / F1 on the error class (label 0 = wrong step), the metric that matters most for an error catcher: precision 0.909, recall 0.760, F1 0.828. Precision is high (few false alarms when it flags a step as wrong); recall 0.760 means it still misses about one wrong step in four. Both are measured on in-distribution Gemma negatives, not real student errors.

## Servable re-run (v2, 2026-05-27)

Same direct Cloud TPU API path, fixed trainer (`save_pretrained(params=state.params)` + reload self-check + error-class P/R/F1). us-east5-b was capacity-dry (code 8) at run time, so a zone sweep landed v6e-1 in us-east5-a. Same data, same hyperparams (mBERT, 20127 rows, 3 epochs, batch 32) as the first run, so the numbers compare directly.

| Metric | Value |
|---|---|
| val_accuracy | 0.880 |
| reload_val_accuracy | 0.880 (== val_accuracy, save verified) |
| error_precision (label 0) | 0.909 |
| error_recall (label 0) | 0.760 |
| error_f1 (label 0) | 0.828 |
| train loss | 0.340 -> 0.305 -> 0.283 |

Servable artifact: `gs://YOUR_GCS_BUCKET/outputs/verifier_v2/` (flax_model.msgpack 678 MiB + tokenizer + metrics_jax.json). The v1 `outputs/verifier/` is the untrained-save artifact and is left in place but must not be served.

## Cost and cleanup

- Smoke test (2026-05-25): v6e-1 about 10 minutes, under 1 USD.
- Real run (2026-05-27): create, install deps, train, upload, delete in one session. Roughly half an hour of v6e-1 time, on the order of 1 to 2 USD. `tpu-vm list` confirmed empty after delete, no idle TPU left.
- Servable re-run (2026-05-27): create in us-east5-a, install, train, reload-check, upload, delete in one session. About 4 minutes of compute on the VM (start 21:43 UTC, done 21:47), well under 1 USD. `tpu-vm list` confirmed empty in us-east5-a and us-east5-b after delete.

## Next

- Done: the servable re-run. The fixed trainer produced `outputs/verifier_v2/` with `reload_val_accuracy` 0.880 (== `val_accuracy`), so the saved artifact is the trained model, and error-class P/R/F1 (0.909 / 0.760 / 0.828) is logged.
- Wire serving to `verifier_v2/`. The verifier API is step-level (`services/verifier_api/app/model.py`): `_build_pair` mirrors `train_jax_tpu.build_text_pair`, so serving and training tokenize the same way. Pull `gs://YOUR_GCS_BUCKET/outputs/verifier_v2/flax_model` to the serving host to load it. The loader is still unrun on the desktop (no JAX stack), but its compute path matches the trainer reload-eval, which ran on the VM.
- Optional: a rule based arithmetic checker to strip the residual 5 percent false-negative labels before a second training pass, which should lift recall.
