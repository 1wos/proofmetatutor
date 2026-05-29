"""LoRA fine-tune Gemma on Cloud TPU with Keras 3 (JAX backend) + KerasHub.

Why this is a real TPU workload (unlike the mBERT encoder):
    Gemma is a multi-billion-parameter decoder LM. Fine-tuning it — even with
    LoRA — is matrix-heavy and memory-bound, and KerasHub shards the model
    across the TPU chip(s) via keras.distribution. On a single CPU/GPU this is
    impractical; on a Cloud TPU v6e it is a few minutes. That is the
    justification for TPU that a 178M encoder did not have.

Task: instruction tuning to verify Korean math reasoning steps (see
    prepare_gemma_sft.py) — the tutor outputs CORRECT / INCORRECT (+ reason).

Run on a Cloud TPU v6e VM (see docs/cloud_tpu_runbook.md):
    JAX_PLATFORMS=tpu KERAS_BACKEND=jax python training/gemma_tutor/train_gemma_lora_tpu.py \
        --train data/training/gemma_sft.jsonl \
        --preset gemma2_2b_en \
        --output-dir /gcs/YOUR_GCS_BUCKET/outputs/gemma_tutor \
        --epochs 1 --batch-size 8 --lora-rank 4 --max-len 512

Auth for gated Gemma weights (one of):
    export KAGGLE_USERNAME=... KAGGLE_KEY=...        # KerasHub default source
    # or `pip install kagglehub` and `kagglehub login`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Keras 3 must use the JAX backend on TPU; set before importing keras.
os.environ.setdefault("KERAS_BACKEND", "jax")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        with p.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--preset", default="gemma2_2b_en")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int, default=0, help="cap rows for a smoke run")
    args = ap.parse_args()

    import jax
    import keras
    import keras_hub

    # Gemma2's fused flash/splash attention kernel triggers a
    # ConcretizationTypeError on TPU during train_step jit; use the standard
    # attention path so tracing succeeds.
    try:
        keras.config.disable_flash_attention()
    except Exception:
        pass

    print(f"[gemma] JAX devices: {jax.devices()}")

    # Shard the model across all TPU chips (data-parallel by default; this is
    # where the TPU actually earns its keep on a multi-billion-param model).
    devices = jax.devices()
    if len(devices) > 1:
        keras.distribution.set_distribution(
            keras.distribution.DataParallel(devices=devices)
        )

    rows = read_jsonl(Path(args.train))
    if args.limit:
        rows = rows[: args.limit]
    prompts = [r["prompt"] for r in rows]
    responses = [r["response"] for r in rows]
    print(f"[gemma] {len(rows)} SFT rows, preset={args.preset}")

    gemma = keras_hub.models.GemmaCausalLM.from_preset(args.preset)
    gemma.backbone.enable_lora(rank=args.lora_rank)
    gemma.preprocessor.sequence_length = args.max_len

    gemma.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # KerasHub GemmaCausalLM trains on full strings; concatenate prompt+response
    # so the model learns to produce the verdict after the prompt.
    texts = [p + r for p, r in zip(prompts, responses)]
    gemma.fit(x=texts, batch_size=args.batch_size, epochs=args.epochs)

    # Save LoRA-merged weights + a tiny eval sample so the run is inspectable.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gemma.save_to_preset(str(out / "gemma_tutor_preset"))

    sample = prompts[0]
    generated = gemma.generate(sample, max_length=args.max_len)
    (out / "sample_generation.txt").write_text(
        f"PROMPT:\n{sample}\n\nGENERATED:\n{generated}\n", encoding="utf-8"
    )
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "rows": len(rows),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lora_rank": args.lora_rank,
                "max_len": args.max_len,
                "jax_devices": [str(d) for d in devices],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[gemma] saved preset + sample generation to {out}")


if __name__ == "__main__":
    main()
