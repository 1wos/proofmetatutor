"""Evaluate verifier predictions stored as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores: list[float] = []
    for target in sorted(set(labels) | set(predictions)):
        tp_count = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label == target and prediction == target
        )
        fp_count = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label != target and prediction == target
        )
        fn_count = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label == target and prediction != target
        )
        precision = tp_count / (tp_count + fp_count or 1)
        recall = tp_count / (tp_count + fn_count or 1)
        scores.append(2 * precision * recall / (precision + recall or 1))
    return sum(scores) / (len(scores) or 1)


def evaluate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    labels = [int(record["label"]) for record in records]
    predictions = [int(record["prediction"]) for record in records]
    correct = sum(
        1
        for label, prediction in zip(labels, predictions)
        if label == prediction
    )
    return {
        "examples": len(records),
        "accuracy": correct / (len(records) or 1),
        "macro_f1": macro_f1(labels, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(read_jsonl(args.predictions))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

