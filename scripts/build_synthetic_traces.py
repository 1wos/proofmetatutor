"""Build synthetic tutor traces from public sample problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_trace(problem: dict[str, Any], index: int) -> dict[str, Any]:
    explanation = str(problem.get("explanation", ""))
    confidence = 0.88 if explanation else 0.2
    self_rating = 0.75
    return {
        "trace_id": f"trace-{index:03d}",
        "student_alias": f"student-{index:03d}",
        "problem_id": problem["problem_id"],
        "student_explanation": explanation,
        "self_rating": self_rating,
        "verifier_confidence": confidence,
        "metacognitive_gap": abs(self_rating - confidence),
        "misconception_tags": [],
        "teacher_action_status": "pending",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = read_jsonl(args.problems)
    traces = [
        build_trace(problem, index)
        for index, problem in enumerate(problems, start=1)
    ]
    write_jsonl(args.output, traces)
    print(f"Wrote {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()

