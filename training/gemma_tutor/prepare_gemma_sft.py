"""Convert the step-level verifier dataset into Gemma SFT (instruction) records.

The encoder verifier (train_jax_tpu.py) treats this as binary classification.
For the Gemma path we reframe the SAME data as instruction tuning: given a
problem and a reasoning step, the tutor must output a short verdict and, when
the step is wrong, a misconception tag — a generative behaviour that genuinely
needs a decoder LM (and so genuinely needs a TPU to fine-tune at Gemma scale).

Input:  verifier_train.jsonl rows (problem_text/question, step_text, prior_steps,
        explanation, label, optional misconception tag).
Output: {"prompt": "...", "response": "..."} jsonl for Gemma SFT.

Usage:
    python training/gemma_tutor/prepare_gemma_sft.py \
        --in data/training/verifier_train.jsonl \
        --out data/training/gemma_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "You are a Korean math tutor. Decide whether the student's reasoning step "
    "is correct for the given problem. Reply with 'CORRECT' or 'INCORRECT', and "
    "if incorrect, add a short reason."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        with p.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def to_prompt(rec: dict[str, Any]) -> tuple[str, str]:
    problem = str(rec.get("problem_text", rec.get("question", ""))).strip()
    prior = rec.get("prior_steps") or []
    context = "\n".join(f"- {s}" for s in prior)
    step = str(rec.get("step_text") or rec.get("explanation", "")).strip()
    label = int(rec.get("label", 1))
    tag = rec.get("misconception_tag") or rec.get("error_type")

    prompt = (
        f"{INSTRUCTION}\n\n"
        f"Problem: {problem}\n"
        f"{f'Prior steps:{chr(10)}{context}{chr(10)}' if context else ''}"
        f"Step to check: {step}\n"
        f"Verdict:"
    )
    if label == 1:
        response = " CORRECT"
    else:
        reason = f" ({tag})" if tag else ""
        response = f" INCORRECT{reason}"
    return prompt, response


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.inp))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            prompt, response = to_prompt(rec)
            if not prompt or not response:
                continue
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} SFT records to {out}")


if __name__ == "__main__":
    main()
