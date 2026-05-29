---
name: cloud-tpu-training
description: >-
  Provisions Cloud TPU VMs and runs JAX/Flax or Keras 3 training jobs on them.
  Use when you need to train or fine-tune a model on Google Cloud TPUs (v5e/v6e),
  install the TPU JAX stack, read/write artifacts via a GCS fuse mount, run a
  LoRA fine-tune of Gemma with KerasHub, or tear TPU resources down to control cost.
---

# Cloud TPU Training

Cloud TPUs are Google's custom accelerators for matrix-heavy ML workloads. The
**TPU VM** architecture gives you SSH access to the host attached to the TPU
chips, so you run training directly on the VM. This skill covers the v5e/v6e
single-host and pod-slice path with JAX/Flax and Keras 3 (JAX backend).

## When a TPU is the right call
- Decoder-LM pre-training / fine-tuning (Gemma, Llama-class) — memory- and
  matmul-bound; LoRA still benefits from TPU memory bandwidth.
- Large-batch encoder training, diffusion, or anything XLA compiles well.
- **Not** worth it for a <200M encoder fine-tune that a single GPU handles — pick
  the accelerator the workload actually needs.

## Prerequisites

```bash
gcloud services enable tpu.googleapis.com storage.googleapis.com
gcloud auth application-default login
```

Required roles: `roles/tpu.admin` (or `compute.admin`) and `roles/storage.objectAdmin`
on the training bucket.

## 1. Create a TPU VM

```bash
ZONE=us-east5-a                 # sweep zones if capacity is out (error code 8)
gcloud compute tpus tpu-vm create my-tpu \
  --zone=$ZONE \
  --accelerator-type=v6e-1 \    # v6e-8, v5litepod-16, ... for pod slices
  --version=v2-alpha-tpuv6e     # runtime must match the generation
```

List / describe / sweep zones:

```bash
gcloud compute tpus tpu-vm list --zone=$ZONE
for z in us-east5-a us-east5-b us-central1-a; do
  gcloud compute tpus tpu-vm create my-tpu --zone=$z \
    --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e && break
done
```

## 2. Install the JAX TPU stack

```bash
gcloud compute tpus tpu-vm ssh my-tpu --zone=$ZONE --command '
  pip install -U "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
# sanity: all chips visible
gcloud compute tpus tpu-vm ssh my-tpu --zone=$ZONE --command \
  'python -c "import jax; print(jax.devices())"'
```

For Keras 3: `pip install keras keras-hub` and set `KERAS_BACKEND=jax`.

## 3. Read/write data via the GCS fuse mount

A TPU VM auto-mounts buckets at `/gcs/<bucket>`. Read training data and write
checkpoints straight to GCS — no manual copy:

```bash
gcloud compute tpus tpu-vm ssh my-tpu --zone=$ZONE --command '
  python train.py \
    --train /gcs/my-bucket/data/train.jsonl \
    --output-dir /gcs/my-bucket/outputs/run1'
```

## 4. Multi-chip sharding

- **JAX/Flax:** use `jax.sharding` / `jax.experimental.mesh_utils`; data-parallel
  replicates params and splits the batch across `jax.devices()`.
- **Keras 3:** `keras.distribution.set_distribution(keras.distribution.DataParallel(devices=jax.devices()))`
  before building the model.

## 5. Gemma LoRA fine-tune (KerasHub)

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras, keras_hub
gemma = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en")
gemma.backbone.enable_lora(rank=4)          # small trainable set
gemma.compile(optimizer=keras.optimizers.AdamW(5e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
gemma.fit(texts, batch_size=8, epochs=1)
gemma.save_to_preset("/gcs/my-bucket/outputs/gemma_lora")
```

Gated weights: set `KAGGLE_USERNAME` / `KAGGLE_KEY` (or `kagglehub login`).

## 6. Save + reload self-check (don't trust a checkpoint blindly)

After saving, reload from disk and re-evaluate. A common bug saves the untrained
init (e.g. Flax `save_pretrained(params=model.params)` instead of `state.params`),
so the in-memory metric looks fine while the artifact scores near chance. If
`reload_accuracy != train_accuracy`, the save is wrong.

## 7. Tear down (cost control)

**TPU VMs bill for as long as they exist** — delete immediately when training ends.

```bash
gcloud compute tpus tpu-vm delete my-tpu --zone=$ZONE --quiet
gcloud compute tpus tpu-vm list --zone=$ZONE     # confirm 0
```

## Common pitfalls
- **Capacity (error code 8):** sweep zones; v6e is scarce in single regions.
- **Runtime mismatch:** `--version` must match the TPU generation, or JAX won't
  see the chips and silently falls back to CPU.
- **Leaving the VM up:** the #1 surprise bill. Automate `delete` in your script's
  exit trap.
- **GPU endpoints ≠ TPU VMs:** Vertex GPU endpoints don't scale to zero either;
  tear them down too.
