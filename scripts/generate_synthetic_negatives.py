"""Generate synthetic wrong explanations using Gemma 4 on Vertex AI.

For each math problem in the input JSONL, Gemma 4 generates 3 plausible-but-wrong
student explanations covering different error types:
  - missing_step: correct start, cuts off before key step
  - wrong_concept: uses a plausible but incorrect math rule
  - calculation_error: right approach, arithmetic mistake

Output JSONL has schema:
  {problem_id, problem_text, answer, explanation, label, error_type}
  label=1 for correct (original), label=0 for negatives (synthetic)

Usage:
    python scripts/generate_synthetic_negatives.py \
        --input data/aihub/math_problems_sample.jsonl \
        --output data/synthetic/negatives_gemma4.jsonl \
        --project YOUR_GCP_PROJECT \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

GEMMA4_MODEL = "gemma-4-27b-it"
VERTEX_LOCATION = "us-central1"

NEGATIVE_PROMPT = """\
You are generating training data for an AI tutoring system.

A Korean student solved the following math problem. Their correct explanation is given.
Generate exactly 3 WRONG student explanations for the same problem. Each wrong explanation
should be plausible (a real student might write it) but contain a specific error.

Problem: {problem_text}
Correct answer: {answer}
Correct explanation: {explanation}

Generate exactly this JSON (no extra text):
{{
  "missing_step": "a student explanation that starts correctly but omits a critical step",
  "wrong_concept": "a student explanation using an incorrect math rule or concept",
  "calculation_error": "a student explanation with correct approach but arithmetic mistake"
}}

Rules:
- Write in Korean if the problem is in Korean, English otherwise
- Each wrong explanation must be 1-3 sentences
- Do NOT reveal the correct answer explicitly in the wrong explanations
- The errors should be subtle, not obviously wrong
"""


def call_gemma4(prompt: str, project: str) -> str | None:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        return None

    vertexai.init(project=project, location=VERTEX_LOCATION)
    model = GenerativeModel(GEMMA4_MODEL)
    config = GenerationConfig(
        temperature=0.8,
        max_output_tokens=512,
        response_mime_type="application/json",
    )
    try:
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except Exception as exc:
        print(f"  [warn] Gemma4 call failed: {exc}")
        return None


def call_gemini_flash_fallback(prompt: str, project: str) -> str | None:
    """Fallback to Gemini 2.5 Flash if Gemma 4 is unavailable."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        return None

    vertexai.init(project=project, location=VERTEX_LOCATION)
    model = GenerativeModel("gemini-2.5-flash")
    config = GenerationConfig(
        temperature=0.8,
        max_output_tokens=512,
        response_mime_type="application/json",
    )
    try:
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except Exception as exc:
        print(f"  [warn] Gemini Flash fallback failed: {exc}")
        return None


def generate_negatives_for_record(
    record: dict[str, Any],
    project: str,
) -> list[dict[str, Any]]:
    problem_text = record.get("problem_text", "")
    answer = record.get("answer", "")
    explanation = record.get("explanation", "")
    problem_id = record.get("problem_id", "unknown")

    if not problem_text or not explanation:
        return []

    prompt = NEGATIVE_PROMPT.format(
        problem_text=problem_text,
        answer=answer,
        explanation=explanation,
    )

    raw = call_gemma4(prompt, project) or call_gemini_flash_fallback(prompt, project)
    if raw is None:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    negatives: list[dict[str, Any]] = []
    for error_type in ("missing_step", "wrong_concept", "calculation_error"):
        neg_expl = parsed.get(error_type, "")
        if neg_expl:
            negatives.append({
                "problem_id": f"{problem_id}_neg_{error_type}",
                "problem_text": problem_text,
                "answer": answer,
                "student_explanation": neg_expl,
                "label": 0,
                "error_type": error_type,
                "school_level": record.get("school_level", ""),
                "grade": record.get("grade", ""),
                "curriculum_standard": record.get("curriculum_standard", ""),
                "difficulty": record.get("difficulty", ""),
            })
    return negatives


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", default="YOUR_GCP_PROJECT")
    parser.add_argument("--limit", default=None, type=int,
                        help="Max number of source records to process")
    parser.add_argument("--delay", default=1.0, type=float,
                        help="Seconds between API calls (rate limiting)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    print(f"Processing {len(records)} source records → 3 negatives each")

    all_negatives: list[dict[str, Any]] = []
    positives: list[dict[str, Any]] = []

    for i, record in enumerate(records):
        # Also include original as positive example
        positives.append({
            "problem_id": record.get("problem_id", f"pos_{i}"),
            "problem_text": record.get("problem_text", ""),
            "answer": record.get("answer", ""),
            "student_explanation": record.get("explanation", ""),
            "label": 1,
            "error_type": "none",
            "school_level": record.get("school_level", ""),
            "grade": record.get("grade", ""),
            "curriculum_standard": record.get("curriculum_standard", ""),
            "difficulty": record.get("difficulty", ""),
        })

        negatives = generate_negatives_for_record(record, args.project)
        all_negatives.extend(negatives)
        print(f"  [{i+1}/{len(records)}] {record.get('problem_id', '?')} → {len(negatives)} negatives")

        if args.delay > 0:
            time.sleep(args.delay)

    all_examples = positives + all_negatives
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_examples)} examples to {args.output}")
    print(f"  Positives: {len(positives)}, Negatives: {len(all_negatives)}")
    label_balance = len(all_negatives) / max(len(positives), 1)
    print(f"  Negative/positive ratio: {label_balance:.1f}x")


if __name__ == "__main__":
    main()
