"""Download HuggingFace datasets for verifier training (no AIHub access required).

Downloads:
  - HAERAE-HUB/KMMLU  (Korean MMLU — math subset)
  - openai/gsm8k      (English step-by-step math solutions)

Both are public and require no registration.

Output: data/hf/kmmlu_math.jsonl, data/hf/gsm8k.jsonl
Schema matches AIHub schema so both can feed the same training pipeline.

Usage:
    python scripts/download_hf_datasets.py
    python scripts/download_hf_datasets.py --output-dir data/hf --splits train,test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KMMLU_SUBJECT = "Math"
GSM8K_SPLIT_MAP = {"train": "train", "test": "test"}


def _item_to_schema(
    problem_text: str,
    explanation: str,
    answer: str,
    problem_id: str,
    source: str,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "answer": answer,
        "student_explanation": explanation,
        "explanation": explanation,
        "label": 1,
        "error_type": "none",
        "school_level": "미상",
        "grade": "미상",
        "curriculum_standard": "",
        "difficulty": "",
        "source": source,
    }


def download_kmmlu(output_dir: Path, splits: list[str]) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] pip install datasets")
        return 0

    total = 0
    for split in splits:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split=split)
        except Exception as exc:
            print(f"  [warn] KMMLU {split} failed: {exc}")
            continue

        records: list[dict[str, Any]] = []
        for i, item in enumerate(ds):
            choices = [item.get(f"A", ""), item.get(f"B", ""), item.get(f"C", ""), item.get(f"D", "")]
            answer_idx = item.get("answer", 1)
            answer_str = str(choices[answer_idx - 1]) if 1 <= answer_idx <= 4 else str(answer_idx)
            question = str(item.get("question", ""))
            explanation = f"정답: {answer_str}. 주어진 선택지 중 {answer_idx}번이 정답입니다."
            records.append(
                _item_to_schema(
                    problem_text=question,
                    explanation=explanation,
                    answer=answer_str,
                    problem_id=f"kmmlu_math_{split}_{i:05d}",
                    source="HAERAE-HUB/KMMLU",
                )
            )

        out_path = output_dir / f"kmmlu_math_{split}.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  KMMLU {split}: {len(records)} records → {out_path}")
        total += len(records)

    return total


def download_gsm8k(output_dir: Path, splits: list[str]) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] pip install datasets")
        return 0

    total = 0
    for split in splits:
        hf_split = GSM8K_SPLIT_MAP.get(split, split)
        try:
            ds = load_dataset("openai/gsm8k", "main", split=hf_split)
        except Exception as exc:
            print(f"  [warn] GSM8K {split} failed: {exc}")
            continue

        records: list[dict[str, Any]] = []
        for i, item in enumerate(ds):
            question = str(item.get("question", ""))
            full_answer = str(item.get("answer", ""))
            # GSM8K answers are "explanation #### final_number"
            parts = full_answer.split("####")
            explanation = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else full_answer
            records.append(
                _item_to_schema(
                    problem_text=question,
                    explanation=explanation,
                    answer=answer,
                    problem_id=f"gsm8k_{split}_{i:05d}",
                    source="openai/gsm8k",
                )
            )

        out_path = output_dir / f"gsm8k_{split}.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  GSM8K {split}: {len(records)} records → {out_path}")
        total += len(records)

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--splits", default="train,test",
                        help="Comma-separated splits to download")
    parser.add_argument("--skip-kmmlu", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",")]
    total = 0

    if not args.skip_kmmlu:
        print("Downloading KMMLU (Korean MMLU — Math)...")
        total += download_kmmlu(args.output_dir, splits)

    if not args.skip_gsm8k:
        print("Downloading GSM8K...")
        total += download_gsm8k(args.output_dir, splits)

    print(f"\nTotal: {total} records written to {args.output_dir}")


if __name__ == "__main__":
    main()
