"""ADK agent entry point with a local fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.tutor_agent.policy import should_ask_clarification
from agents.tutor_agent.prompts import CLARIFICATION_PROMPT, SYSTEM_PROMPT
from agents.tutor_agent.safety_plugin import SafetyPlugin
from agents.tutor_agent.tools import verify_explanation, write_evidence_event


@dataclass
class LocalTutorAgent:
    name: str = "prooftutor"

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        safety = SafetyPlugin()
        explanation = str(request.get("explanation", ""))
        before = safety.before_model_callback(explanation)
        if not before.allowed:
            return {"status": "blocked", "reason": before.reason}

        verifier_result = verify_explanation(
            problem_text=str(request.get("problem_text", "")),
            answer=str(request.get("answer", "")),
            explanation=explanation,
        )
        needs_clarification = should_ask_clarification(
            explanation,
            verifier_confidence=verifier_result.get("correctness_confidence"),
        )
        trace_id = str(request.get("trace_id", "local-trace"))
        evidence_event = write_evidence_event(
            trace_id=trace_id,
            event_type="verifier_run",
            payload=verifier_result,
        )
        message = (
            CLARIFICATION_PROMPT
            if needs_clarification
            else "I checked your reasoning and stored the evidence trace."
        )
        after = safety.after_model_callback(
            model_output=message,
            requires_reasoning_first=needs_clarification,
        )
        if not after.allowed:
            return {"status": "blocked", "reason": after.reason}

        return {
            "status": "ok",
            "message": message,
            "verifier_result": verifier_result,
            "evidence_event": evidence_event,
        }


def build_agent(prefer_local: bool = False) -> Any:
    if prefer_local:
        return LocalTutorAgent()

    try:
        from google.adk.agents import LlmAgent
    except ImportError:
        return LocalTutorAgent()

    return LlmAgent(
        name="prooftutor",
        model="gemma-4-27b-it",
        instruction=SYSTEM_PROMPT,
        tools=[verify_explanation, write_evidence_event],
    )


root_agent = build_agent()
