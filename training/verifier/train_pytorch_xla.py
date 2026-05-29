"""Train verifier with PyTorch/XLA on Cloud TPU — mDeBERTa-v3 backbone.

Backbone: microsoft/mdeberta-v3-base (SOTA XNLI, 86M params).
Task: binary sequence pair classification.
  sentence_A = problem_text
  sentence_B = student_explanation
  label: 1 = correct reasoning, 0 = incorrect / missing steps

SOTA benchmark context (Korean math solution verification):
  Model                                   | XNLI | #params | Korean?
  ----------------------------------------|------|---------|--------
  microsoft/mdeberta-v3-base              | 79.8 |  86M   | via mT |  ← chosen
  klue/roberta-large                      |  --  | 355M   | native |  train_jax
  snunlp/KR-ELECTRA-discriminator         |  --  |  14M   | native |  fast
  google/bert-base-multilingual-cased     | 74.5 | 178M   | via mT |  JAX path
  intfloat/multilingual-e5-large          |  --  | 560M   | RAG    |  retrieval

mDeBERTa-v3 wins on multilingual classification accuracy with smallest footprint.
KR-ELECTRA is used as fast inference backbone in model.py.

Usage:
    python training/verifier/train_pytorch_xla.py \
        --train data/aihub/math_problems_sample.jsonl \
        --output-dir /gcs/YOUR_GCS_BUCKET/outputs/verifier_deberta \
        --backbone microsoft/mdeberta-v3-base \
        --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

BACKBONE_DEFAULT = "microsoft/mdeberta-v3-base"
MAX_SEQ_LEN = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        with p.open("r", encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def build_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        problem = str(r.get("problem_text", r.get("question", "")))
        explanation = str(
            r.get("student_explanation", r.get("explanation", ""))
        )
        label = int(r.get("label", 1))
        if problem and explanation:
            out.append({"problem": problem, "explanation": explanation, "label": label})
    return out


def train_with_torch(
    records: list[dict[str, Any]],
    backbone: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        return {
            "status": "missing_deps",
            "missing": str(exc),
            "hint": "pip install transformers torch",
        }

    # ── XLA device setup ─────────────────────────────────────────────────────
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_name = str(device)
        use_xla = True
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = str(device)
        use_xla = False

    print(f"Device: {device_name}  XLA={use_xla}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    examples = build_examples(records)

    class VerifierDataset(Dataset):
        def __init__(self, items: list[dict[str, Any]], tok: Any) -> None:
            self.items = items
            self.tok = tok

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.items[idx]
            enc = self.tok(
                item["problem"],
                item["explanation"],
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc.get(
                    "token_type_ids",
                    torch.zeros_like(enc["input_ids"]),
                ).squeeze(0),
                "label": torch.tensor(item["label"], dtype=torch.long),
            }

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    dataset = VerifierDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            if use_xla:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(len(loader), 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  ({elapsed:.1f}s)")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)

    # ── Save ────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "pytorch_model"))
    tokenizer.save_pretrained(str(output_dir / "pytorch_model"))
    print(f"Model saved to {output_dir}/pytorch_model")

    return {
        "status": "ok",
        "backbone": backbone,
        "examples": len(examples),
        "epochs": epochs,
        "accuracy": accuracy,
        "device": device_name,
        "use_xla": use_xla,
        "training_history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", default=BACKBONE_DEFAULT)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.train)
    print(f"Loaded {len(records)} records from {args.train}")

    metrics = train_with_torch(
        records=records,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "metrics_pytorch_xla.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
