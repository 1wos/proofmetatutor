"""Local tool implementations for the tutor agent skeleton."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerifierResult:
    correctness_confidence: float
    missing_step_score: float
    misconception_tags: list[str]
    difficulty_match: str


@dataclass(frozen=True)
class EvidenceEvent:
    trace_id: str
    event_type: str
    payload: dict[str, Any]


def verify_explanation(
    problem_text: str,
    answer: str,
    explanation: str,
) -> dict[str, Any]:
    normalized = explanation.lower()
    answer_hit = answer.lower() in normalized
    has_step = any(token in normalized for token in ("because", "then", "so"))
    confidence = 0.85 if answer_hit and has_step else 0.35
    tags = [] if confidence >= 0.6 else ["missing_reasoning"]
    result = VerifierResult(
        correctness_confidence=confidence,
        missing_step_score=0.1 if has_step else 0.8,
        misconception_tags=tags,
        difficulty_match="unknown",
    )
    return asdict(result)


def write_evidence_event(
    trace_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    event = EvidenceEvent(
        trace_id=trace_id,
        event_type=event_type,
        payload=payload,
    )
    return asdict(event)


def create_teacher_intervention(
    trace_id: str,
    recommendation: str,
) -> dict[str, str]:
    return {
        "trace_id": trace_id,
        "recommendation": recommendation,
        "teacher_action_status": "pending",
    }

