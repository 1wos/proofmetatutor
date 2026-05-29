"""Train a multilingual encoder-based verifier with JAX/Flax on Cloud TPU.

Backbone: google-bert/bert-base-multilingual-cased (mBERT) via Flax.
Task: binary classification, is this reasoning step (or full explanation) correct?
Input pairs: (problem + context [SEP] step_or_explanation) -> label (1=correct, 0=incorrect)

TPU note:
    On Cloud TPU v6e-1 this runs with XLA device=TPU:0.
    Falls back to CPU if JAX TPU backend is unavailable.

Usage:
    python training/verifier/train_jax_tpu.py \
        --train data/aihub/math_problems_sample.jsonl \
        --output-dir /gcs/YOUR_GCS_BUCKET/outputs/verifier \
        --backbone google-bert/bert-base-multilingual-cased \
        --epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# ─── Model selection rationale ───────────────────────────────────────────────
# Evaluated SOTA encoders for Korean math solution verification:
#   - google-bert/bert-base-multilingual-cased (mBERT): Flax-native, 178M, TPU-ready
#   - microsoft/mdeberta-v3-base: XNLI SOTA, but no official Flax weights
#   - klue/roberta-large: Korean SOTA, PyTorch-only (see train_pytorch_xla.py)
#   - snunlp/KR-ELECTRA-discriminator: Korean ELECTRA, lightweight
# Winner for JAX path: mBERT (Flax weights available, fastest TPU startup)
# Winner for PyTorch-XLA path: mDeBERTa-v3 (train_pytorch_xla.py)
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_DEFAULT = "google-bert/bert-base-multilingual-cased"
MAX_SEQ_LEN = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        with p.open("r", encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def build_text_pair(record: dict[str, Any]) -> tuple[str, str, int]:
    """Return (sentence_a, sentence_b, label).

    두 스키마 다 받는다.
    - step-level (gen_gemma_negatives + assemble_training): A=문제+이전 step, B=검증 대상 step.
    - explanation-level (generate_synthetic_negatives): A=문제, B=설명 전체.
    """
    problem = str(record.get("problem_text", record.get("question", "")))
    label = int(record.get("label", 1))

    step_text = record.get("step_text")
    if step_text:  # step-level: judge one step given problem + prior steps as context
        prior = record.get("prior_steps") or []
        context = " ".join([problem, *(str(s) for s in prior)]).strip()
        return context, str(step_text), label

    # explanation-level: real AIHub explanation (label=1) vs Gemma synthetic wrong (label=0)
    explanation = str(record.get("student_explanation", record.get("explanation", "")))
    return problem, explanation, label


def tokenize_batch(
    tokenizer: Any,
    texts_a: list[str],
    texts_b: list[str],
    max_len: int,
) -> dict[str, Any]:
    import numpy as np

    encoded = tokenizer(
        texts_a,
        texts_b,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return {k: np.array(v) for k, v in encoded.items()}


def train_encoder(
    records: list[dict[str, Any]],
    backbone: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from flax.training import train_state
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
    except ImportError as exc:
        return {
            "status": "missing_deps",
            "missing": str(exc),
            "hint": "pip install transformers[flax] optax",
            "examples": len(records),
        }

    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX backend: {backend}, devices: {devices}")

    # ── Prepare data ─────────────────────────────────────────────────────────
    pairs = [build_text_pair(r) for r in records]
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    labels_all = np.array([p[2] for p in pairs], dtype=np.int32)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    inputs = tokenize_batch(tokenizer, texts_a, texts_b, MAX_SEQ_LEN)

    # ── Model + optimizer ────────────────────────────────────────────────────
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=2
    )
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.01)

    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
    )

    @jax.jit
    def train_step(
        state: TrainState,
        batch_inputs: dict[str, Any],
        batch_labels: Any,
        dropout_rng: Any,
    ) -> tuple[TrainState, Any, Any]:
        dropout_rng, next_rng = jax.random.split(dropout_rng)

        def loss_fn(params: Any) -> Any:
            outputs = state.apply_fn(
                **batch_inputs, params=params, train=True, dropout_rng=dropout_rng
            )
            logits = outputs.logits
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch_labels
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, next_rng

    # ── Train/val split (90/10) for an honest held-out number ───────────────
    n = len(records)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    steps_per_epoch = max(1, len(train_idx) // batch_size)
    history: list[dict[str, Any]] = []
    dropout_rng = jax.random.PRNGKey(0)

    for epoch in range(epochs):
        perm = rng.permutation(train_idx)
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_inputs = {k: jnp.array(v[idx]) for k, v in inputs.items()}
            batch_labels = jnp.array(labels_all[idx])
            state, loss, dropout_rng = train_step(
                state, batch_inputs, batch_labels, dropout_rng
            )
            epoch_loss += float(loss)
        elapsed = time.time() - t0
        avg_loss = epoch_loss / steps_per_epoch
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  ({elapsed:.1f}s)")

    # ── Evaluate on held-out val, batched to avoid OOM on large data ─────────
    # Error class = label 0 (a wrong step). Report P/R/F1 on it; catching
    # wrong steps is the job, and accuracy hides that on an imbalanced set.
    eval_bs = 256
    correct = tp = fp = fn = 0
    for start in range(0, len(val_idx), eval_bs):
        vi = val_idx[start : start + eval_bs]
        batch_inputs = {k: jnp.array(v[vi]) for k, v in inputs.items()}
        logits = model(**batch_inputs, params=state.params, train=False).logits
        preds = np.array(jnp.argmax(logits, axis=-1))
        gold = labels_all[vi]
        correct += int((preds == gold).sum())
        tp += int(((preds == 0) & (gold == 0)).sum())
        fp += int(((preds == 0) & (gold == 1)).sum())
        fn += int(((preds == 1) & (gold == 0)).sum())
    accuracy = correct / len(val_idx)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0

    # ── Save artifact ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    # trained weights live in state.params, not model.params; pass them
    # explicitly or save_pretrained persists the untrained init instead
    model.save_pretrained(
        str(output_dir / "flax_model"), params=state.params
    )
    tokenizer.save_pretrained(str(output_dir / "flax_model"))

    # ── Reload from disk and re-eval: proves the saved artifact holds the
    #    trained weights. An untrained reload would score near chance. ────────
    reloaded = FlaxAutoModelForSequenceClassification.from_pretrained(
        str(output_dir / "flax_model")
    )
    r_correct = 0
    for start in range(0, len(val_idx), eval_bs):
        vi = val_idx[start : start + eval_bs]
        batch_inputs = {k: jnp.array(v[vi]) for k, v in inputs.items()}
        logits = reloaded(**batch_inputs, train=False).logits
        rp = np.array(jnp.argmax(logits, axis=-1))
        r_correct += int((rp == labels_all[vi]).sum())
    reload_val_accuracy = r_correct / len(val_idx)

    metrics = {
        "status": "ok",
        "backbone": backbone,
        "examples": n,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "epochs": epochs,
        "val_accuracy": accuracy,  # held-out 10% split, not train acc
        "error_precision": precision,  # label 0 = wrong step
        "error_recall": recall,
        "error_f1": f1,
        "reload_val_accuracy": reload_val_accuracy,  # saved artifact re-eval
        "jax_backend": backend,
        "jax_devices": [str(d) for d in devices],
        "training_history": history,
    }
    return metrics


def train_logreg_fallback(
    records: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Logistic regression over handcrafted features — smoke test only."""
    import jax
    import jax.numpy as jnp

    features: list[list[float]] = []
    labels: list[int] = []
    for r in records:
        expl = str(r.get("step_text") or r.get("student_explanation") or r.get("explanation", ""))
        conf = float(r.get("verifier_confidence", 0.5))
        gap = float(r.get("metacognitive_gap", 0.0))
        step_hit = any(t in expl.lower() for t in ("because", "then", "so", "따라서", "그러므로", "왜냐하면"))
        features.append([min(len(expl) / 200.0, 1.0), conf, gap, float(step_hit)])
        labels.append(int(r.get("label", 1 if conf >= 0.5 else 0)))

    x = jnp.array(features)
    y = jnp.array(labels)
    w = jnp.zeros((x.shape[1],))
    b = jnp.array(0.0)

    grad_fn = jax.grad(
        lambda w_, b_: jnp.mean(
            -(y * jnp.log(jax.nn.sigmoid(x @ w_ + b_) + 1e-7)
              + (1 - y) * jnp.log(1 - jax.nn.sigmoid(x @ w_ + b_) + 1e-7))
        ),
        argnums=(0, 1),
    )
    for _ in range(epochs):
        gw, gb = grad_fn(w, b)
        w = w - learning_rate * gw
        b = b - learning_rate * gb

    preds = (jax.nn.sigmoid(x @ w + b) >= 0.5).astype(jnp.int32)
    accuracy = float(jnp.mean(preds == y))
    return {
        "status": "ok_logreg_fallback",
        "examples": len(labels),
        "accuracy": accuracy,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path,
                        help="JSONL file or directory of JSONL files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", default=BACKBONE_DEFAULT,
                        help="HuggingFace model ID for Flax encoder")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--logreg-only", action="store_true",
                        help="Skip encoder, run logistic regression smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.train)
    print(f"Loaded {len(records)} records from {args.train}")

    if args.logreg_only:
        metrics = train_logreg_fallback(records, args.epochs, args.learning_rate)
    else:
        metrics = train_encoder(
            records=records,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
        )
        if metrics.get("status") == "missing_deps":
            print(f"[WARN] Falling back to logistic regression: {metrics['missing']}")
            metrics = train_logreg_fallback(records, args.epochs, args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_jax.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
