"""ProofMetaTutor verifier API.

A small, documented HTTP surface over the Cloud TPU-trained verifier. The
OpenAPI schema (served at ``/docs`` and ``/redoc``) is the source of truth for
client developers — every field and endpoint is described inline so the Swagger
UI is self-explanatory.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.verifier_api.app.arithmetic import check_solution
from services.verifier_api.app.model import verifier_model

API_DESCRIPTION = """
Verifies Korean math reasoning, step by step.

- **`POST /api/verifier/run`** — score a *single* step with the Cloud TPU-trained
  mBERT verifier (plausibility confidence + misconception tags).
- **`POST /api/check-solution`** — verify a *multi-step* solution: each step gets
  the model's confidence **plus** a deterministic `sympy` arithmetic check, and the
  first wrong step is pinpointed.

The model scores *plausibility*; the arithmetic checker catches actual
computation errors the model can't. Treat the output as a teacher-facing signal,
not an authoritative grade.
"""

TAGS_METADATA = [
    {"name": "health", "description": "Liveness probe."},
    {"name": "verify", "description": "Single-step and multi-step verification."},
]

app = FastAPI(
    title="ProofMetaTutor Verifier API",
    description=API_DESCRIPTION,
    version="0.2.0",
    contact={"name": "storeops-ai / ProofMetaTutor",
             "url": "https://github.com/ideation-lab/tpubuilders"},
    license_info={"name": "MIT"},
    openapi_tags=TAGS_METADATA,
)

# Allow the static web UI (served from a different origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────────────────────────
class VerifierRequest(BaseModel):
    problem_text: str = Field(
        ...,
        description="The math problem the student is solving.",
        examples=["2x + 3 = 11"],
    )
    answer: str = Field(
        "", description="Optional final answer, if the flow provides one.")
    explanation: str = Field(
        "",
        description="The student's reasoning. Used as the single-step fallback "
        "when `step_text` is not given.",
        examples=["양변에서 3을 빼면 2x = 8, 그래서 x = 4"],
    )
    prior_steps: list[str] = Field(
        default_factory=list,
        description="Earlier steps that provide context for the step being "
        "checked (matches how the model was trained: problem + prior steps).",
        examples=[["양변에서 3을 빼면 2x = 8"]],
    )
    step_text: str | None = Field(
        None,
        description="The single reasoning step to judge. If omitted, "
        "`explanation` is judged instead.",
        examples=["2x = 8, 그래서 x = 4"],
    )


class SolutionRequest(BaseModel):
    problem_text: str = Field(
        ..., description="The math problem.", examples=["2x + 6 = 10"])
    steps: list[str] = Field(
        ...,
        description="The full solution as an ordered list of steps (one reasoning "
        "step per item). Each step is checked against the problem and all prior steps.",
        examples=[["양변에서 6을 빼면 2x = 4", "양변을 2로 나누면 x = 4"]],
    )


# ── Response models ─────────────────────────────────────────────────────────
class VerifierResponse(BaseModel):
    correctness_confidence: float = Field(
        ..., description="Model probability that the step is correct (0–1).",
        examples=[0.91])
    missing_step_score: float = Field(
        ..., description="Score for likely missing/incomplete reasoning (0–1).",
        examples=[0.09])
    misconception_tags: list[str] = Field(
        ..., description="Misconception labels the model associates with the step.",
        examples=[[]])
    difficulty_match: str = Field(
        ..., description="Difficulty alignment signal.", examples=["unknown"])


class StepResult(BaseModel):
    step: str = Field(..., description="The step text that was checked.")
    confidence: float | None = Field(
        None, description="Model plausibility confidence for this step (0–1).",
        examples=[0.92])
    misconception_tags: list[str] = Field(
        default_factory=list, description="Misconception labels for this step.")
    arithmetic: str | None = Field(
        None,
        description="Deterministic arithmetic check: `ok`, `error`, or `null` "
        "when the step has no checkable equation.",
        examples=["error"])
    reason: str = Field(
        ..., description="Human-readable reason, e.g. `x = 4 (정답은 2)`.",
        examples=["x = 4 (정답은 2)"])


class SolutionResponse(BaseModel):
    expected: list[str] | None = Field(
        None, description="The correct answer(s) the checker solved for, if any.",
        examples=[["2"]])
    first_error_index: int = Field(
        ..., description="0-based index of the first wrong step, or `-1` if none.",
        examples=[1])
    steps: list[StepResult] = Field(..., description="Per-step results, in order.")


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"], summary="Liveness check")
def health() -> dict[str, str]:
    """Returns `{"status": "ok"}` once the service is up."""
    return {"status": "ok"}


@app.post(
    "/api/verifier/run",
    tags=["verify"],
    summary="Verify a single reasoning step",
    response_model=VerifierResponse,
)
def run_verifier(request: VerifierRequest) -> dict:
    """Score one reasoning step with the Cloud TPU-trained mBERT verifier.

    The model judges `step_text` (or `explanation`) given `problem_text` and any
    `prior_steps`, returning a correctness-confidence and misconception tags.
    """
    return verifier_model.predict(
        problem_text=request.problem_text,
        answer=request.answer,
        explanation=request.explanation,
        prior_steps=request.prior_steps,
        step_text=request.step_text,
    )


@app.post(
    "/api/check-solution",
    tags=["verify"],
    summary="Verify a multi-step solution and pinpoint the first wrong step",
    response_model=SolutionResponse,
)
def check_full_solution(request: SolutionRequest) -> dict:
    """Verify a full solution end to end.

    For every step we combine two signals:

    1. **TPU model** — the mBERT verifier's plausibility confidence.
    2. **Arithmetic (`sympy`)** — a deterministic check that actually solves the
       problem and verifies each step's equations.

    `first_error_index` points to the first step the arithmetic check rejects, so
    the UI can show *"stuck at step N"* with the reason.
    """
    arith = check_solution(request.problem_text, request.steps)
    out = []
    for i, step in enumerate(request.steps):
        model = verifier_model.predict(
            problem_text=request.problem_text,
            answer="",
            explanation=step,
            prior_steps=request.steps[:i],
            step_text=step,
        )
        a = arith["steps"][i]
        out.append(
            {
                "step": step,
                "confidence": model.get("correctness_confidence"),
                "misconception_tags": model.get("misconception_tags", []),
                "arithmetic": a["verdict"],   # 'ok' | 'error' | None
                "reason": a["reason"],
            }
        )
    return {
        "expected": arith["expected"],
        "first_error_index": arith["first_error_index"],
        "steps": out,
    }
