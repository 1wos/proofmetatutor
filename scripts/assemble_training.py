"""positive(정답 step) + negative(Gemma 손상 step)을 단일 step-level 학습셋으로 합친다.

정답 풀이의 모든 step 은 valid(label=1), Gemma 가 손상한 step 은 invalid(label=0).
verifier 는 (문제 + 직전 step 들 + 판정 대상 step) 을 보고 타당성과 오개념을 예측한다.
이 스키마가 toy train_jax_tpu.py 의 4-feature 스키마를 대체한다.

balance: 특정 태그(ALG-EQ-ONESIDE 등) 과대표집을 막으려 --max-per-tag 로 상한.

Usage:
    python scripts/assemble_training.py \
        --positives data/aihub/math_train.jsonl \
        --negatives data/synthetic/negatives_train.jsonl \
        --output data/training/verifier_train.jsonl \
        --split train --max-per-tag 400
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def positive_rows(rec: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """정답 레코드의 각 step 을 valid 학습행으로. 단일 step 도 1행 포함."""
    steps = rec.get("steps", [])
    if not steps:
        return []
    rows = []
    for i, s in enumerate(steps):
        rows.append(
            {
                "uid": f"{rec.get('problem_id','')}#s{i}#pos",
                "split": split,
                "problem_id": rec.get("problem_id", ""),
                "school_level": rec.get("school_level", ""),
                "grade": rec.get("grade", ""),
                "problem_text": rec.get("problem_text", ""),
                "prior_steps": steps[:i],
                "step_text": s,
                "label": 1,
                "misconception_id": None,
                "misconception_category": None,
                "misconception_domain": None,
                "source": "gold",
            }
        )
    return rows


def negative_row(neg: dict[str, Any], split: str) -> dict[str, Any]:
    i = neg.get("corrupted_step_index", 0)
    steps = neg.get("steps", [])
    return {
        "uid": f"{neg.get('problem_id','')}#s{i}#neg#{neg.get('misconception_id','')}",
        "split": split,
        "problem_id": neg.get("problem_id", ""),
        "school_level": neg.get("school_level", ""),
        "grade": neg.get("grade", ""),
        "problem_text": neg.get("problem_text", ""),
        "prior_steps": steps[:i],
        "step_text": neg.get("corrupted_step", ""),
        "label": 0,
        "misconception_id": neg.get("misconception_id"),
        "misconception_category": neg.get("misconception_category"),
        "misconception_domain": neg.get("misconception_domain"),
        "source": neg.get("source", "gemma"),
    }


def balance_negatives(
    negs: list[dict[str, Any]], max_per_tag: int | None, rng: random.Random
) -> list[dict[str, Any]]:
    if not max_per_tag:
        return negs
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for n in negs:
        by_tag.setdefault(n.get("misconception_id", "?"), []).append(n)
    out = []
    for tag, items in by_tag.items():
        rng.shuffle(items)
        out.extend(items[:max_per_tag])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--positives", required=True, type=Path)
    p.add_argument("--negatives", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", default="train")
    p.add_argument("--max-per-tag", default=None, type=int, help="태그별 음성 상한(균형)")
    p.add_argument("--only-multistep", action="store_true", help="positive 도 n_steps>=2만")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    pos_recs = read_jsonl(args.positives)
    if args.only_multistep:
        pos_recs = [r for r in pos_recs if len(r.get("steps", [])) >= 2]
    pos_rows: list[dict[str, Any]] = []
    for r in pos_recs:
        pos_rows.extend(positive_rows(r, args.split))

    neg_recs = read_jsonl(args.negatives)
    neg_recs = balance_negatives(neg_recs, args.max_per_tag, rng)
    neg_rows = [negative_row(n, args.split) for n in neg_recs if n.get("corrupted_step")]

    all_rows = pos_rows + neg_rows
    rng.shuffle(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)
    by_tag = Counter(r["misconception_id"] for r in neg_rows)
    print(f"split={args.split}  rows={len(all_rows)}  pos={n_pos}  neg={n_neg}  ratio=1:{n_neg/max(n_pos,1):.2f}")
    print(f"wrote {args.output}")
    print("neg by tag (top10):", dict(by_tag.most_common(10)))


if __name__ == "__main__":
    main()
