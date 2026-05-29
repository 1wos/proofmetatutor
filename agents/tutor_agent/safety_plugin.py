"""Safety callbacks for tutor turns."""

from __future__ import annotations

from dataclasses import dataclass

from agents.tutor_agent.policy import (
    contains_answer_leakage,
    contains_personal_label,
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str


class SafetyPlugin:
    def before_model_callback(self, user_input: str) -> SafetyDecision:
        lowered = user_input.lower()
        if "ignore previous instructions" in lowered:
            return SafetyDecision(False, "prompt_injection")
        if "주민등록번호" in user_input:
            return SafetyDecision(False, "possible_personal_data")
        return SafetyDecision(True, "ok")

    def after_model_callback(
        self,
        model_output: str,
        requires_reasoning_first: bool,
    ) -> SafetyDecision:
        if contains_personal_label(model_output):
            return SafetyDecision(False, "personal_label")
        if requires_reasoning_first and contains_answer_leakage(model_output):
            return SafetyDecision(False, "answer_leakage")
        return SafetyDecision(True, "ok")

