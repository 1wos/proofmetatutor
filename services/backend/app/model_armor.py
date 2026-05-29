"""Local guardrail adapter shaped like a Model Armor boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str


def screen_text(text: str) -> dict[str, bool | str]:
    lowered = text.lower()
    if "ignore previous instructions" in lowered:
        return asdict(GuardrailResult(False, "prompt_injection"))
    if "주민등록번호" in text:
        return asdict(GuardrailResult(False, "possible_personal_data"))
    return asdict(GuardrailResult(True, "ok"))

