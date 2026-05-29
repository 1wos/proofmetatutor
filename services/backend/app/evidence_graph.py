"""In-memory evidence graph adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceTrace:
    trace_id: str
    events: list[dict[str, Any]] = field(default_factory=list)


class EvidenceGraphStore:
    def __init__(self) -> None:
        self._traces: dict[str, EvidenceTrace] = {}

    def add_event(
        self,
        trace_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        trace = self._traces.setdefault(trace_id, EvidenceTrace(trace_id))
        event = {"event_type": event_type, "payload": payload}
        trace.events.append(event)
        return {"trace_id": trace_id, "event": event}

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        trace = self._traces.get(trace_id)
        if trace is None:
            return {"trace_id": trace_id, "events": []}
        return asdict(trace)

    def search(self, query: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for trace in self._traces.values():
            text = str(trace.events).lower()
            if query.lower() in text:
                matches.append(asdict(trace))
        return matches


graph_store = EvidenceGraphStore()

