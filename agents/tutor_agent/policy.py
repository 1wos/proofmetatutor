"""Policy checks for explanation-first tutoring."""

from __future__ import annotations

ANSWER_LEAK_MARKERS = (
    "the answer is",
    "final answer",
    "정답은",
    "답은",
)

PERSONAL_LABELS = (
    "lazy",
    "bad at math",
    "low ability",
    "수학을 못",
)


def contains_answer_leakage(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in ANSWER_LEAK_MARKERS)


def contains_personal_label(text: str) -> bool:
    normalized = text.lower()
    return any(label in normalized for label in PERSONAL_LABELS)


def should_ask_clarification(
    explanation: str,
    verifier_confidence: float | None = None,
) -> bool:
    """Return True if the tutor should request more detail before giving feedback.

    Triggers when:
    - Explanation is shorter than 40 chars (likely incomplete), OR
    - Verifier confidence is very low (< 0.35) AND explanation is short (< 80 chars)
      — low confidence on a short text usually means there's not enough to verify

    Does NOT trigger solely on low confidence: if the explanation is detailed,
    the verifier should generate specific misconception feedback instead.
    """
    text = explanation.strip()
    if len(text) < 40:
        return True
    if verifier_confidence is not None and verifier_confidence < 0.35 and len(text) < 80:
        return True
    return False


def can_create_intervention(
    verifier_confidence: float,
    teacher_action_status: str,
) -> bool:
    return verifier_confidence < 0.6 and teacher_action_status == "pending"

