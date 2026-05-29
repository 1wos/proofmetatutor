"""ProofMetaTutor backend API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.tutor_agent.agent import build_agent
from services.backend.app.evidence_graph import graph_store
from services.backend.app.model_armor import screen_text
from services.backend.app.teacher_gate import teacher_gate


class TutorRequest(BaseModel):
    trace_id: str = Field(default="local-trace")
    problem_text: str
    answer: str
    explanation: str


class EvidenceEventRequest(BaseModel):
    trace_id: str
    event_type: str
    payload: dict[str, Any]


class SearchRequest(BaseModel):
    query: str


class TeacherActionRequest(BaseModel):
    intervention_text: str = ""
    reason: str = ""


app = FastAPI(title="ProofMetaTutor Backend")
agent = build_agent(prefer_local=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tutor/explain")
def tutor_explain(request: TutorRequest) -> dict[str, Any]:
    guardrail = screen_text(request.explanation)
    if not guardrail["allowed"]:
        raise HTTPException(status_code=400, detail=guardrail["reason"])
    if not hasattr(agent, "run"):
        raise HTTPException(status_code=501, detail="adk_runtime_required")
    return agent.run(request.model_dump())


@app.post("/api/evidence/events")
def create_evidence_event(request: EvidenceEventRequest) -> dict[str, Any]:
    return graph_store.add_event(
        trace_id=request.trace_id,
        event_type=request.event_type,
        payload=request.payload,
    )


@app.get("/api/evidence/traces/{trace_id}")
def get_evidence_trace(trace_id: str) -> dict[str, Any]:
    return graph_store.get_trace(trace_id)


@app.post("/api/evidence/search")
def search_evidence(request: SearchRequest) -> list[dict[str, Any]]:
    return graph_store.search(request.query)


@app.post("/api/teacher/interventions/{trace_id}/approve")
def approve_intervention(
    trace_id: str,
    request: TeacherActionRequest,
) -> dict[str, str]:
    return teacher_gate.approve(trace_id, request.intervention_text)


@app.post("/api/teacher/interventions/{trace_id}/edit")
def edit_intervention(
    trace_id: str,
    request: TeacherActionRequest,
) -> dict[str, str]:
    return teacher_gate.edit(
        trace_id=trace_id,
        intervention_text=request.intervention_text,
        reason=request.reason,
    )


@app.post("/api/teacher/interventions/{trace_id}/reject")
def reject_intervention(
    trace_id: str,
    request: TeacherActionRequest,
) -> dict[str, str]:
    return teacher_gate.reject(trace_id, request.reason)
