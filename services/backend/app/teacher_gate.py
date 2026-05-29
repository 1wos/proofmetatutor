"""Teacher approval gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TeacherAction:
    trace_id: str
    action: str
    intervention_text: str
    reason: str = ""


class TeacherGate:
    def __init__(self) -> None:
        self._actions: dict[str, TeacherAction] = {}

    def approve(self, trace_id: str, intervention_text: str) -> dict[str, str]:
        action = TeacherAction(
            trace_id=trace_id,
            action="approved",
            intervention_text=intervention_text,
        )
        self._actions[trace_id] = action
        return asdict(action)

    def edit(
        self,
        trace_id: str,
        intervention_text: str,
        reason: str,
    ) -> dict[str, str]:
        action = TeacherAction(
            trace_id=trace_id,
            action="edited_and_approved",
            intervention_text=intervention_text,
            reason=reason,
        )
        self._actions[trace_id] = action
        return asdict(action)

    def reject(self, trace_id: str, reason: str) -> dict[str, str]:
        action = TeacherAction(
            trace_id=trace_id,
            action="rejected",
            intervention_text="",
            reason=reason,
        )
        self._actions[trace_id] = action
        return asdict(action)


teacher_gate = TeacherGate()

