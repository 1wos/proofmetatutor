"""Verifier model interface.

Loading priority (first that loads wins):
1. PyTorch encoder (mDeBERTa-v3) from a SavedModel dir
2. Flax encoder (mBERT, the Cloud TPU-trained verifier) from flax_model/
3. Keyword-heuristic fallback (always available, no ML deps)

Trained by train_pytorch_xla.py (pytorch_model/) and
train_jax_tpu.py (flax_model/).

Both encoders pair-tokenize via _build_pair, mirroring
train_jax_tpu.build_text_pair so serving matches training: step-level
(problem + prior_steps, step_text) when step_text is given, else
explanation-level (problem, explanation) as the single-step fallback.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerifierPrediction:
    correctness_confidence: float
    missing_step_score: float
    misconception_tags: list[str]
    difficulty_match: str


def _prediction_from_confidence(
    confidence: float,
) -> dict[str, float | str | list[str]]:
    """Shared post-processing: model P(correct) -> response shape.

    두 인코더(torch/flax)가 같은 규칙을 쓰도록 한 곳에 모은다.
    """
    missing_step_score = max(0.0, 1.0 - confidence - 0.1)
    if confidence >= 0.6:
        tags: list[str] = []
    elif confidence < 0.4:
        tags = ["missing_reasoning"]
    else:
        tags = ["partial_reasoning"]
    prediction = VerifierPrediction(
        correctness_confidence=round(confidence, 4),
        missing_step_score=round(missing_step_score, 4),
        misconception_tags=tags,
        difficulty_match="unknown",
    )
    return asdict(prediction)


def _build_pair(
    problem_text: str,
    prior_steps: list[str] | None,
    step_text: str | None,
    explanation: str,
) -> tuple[str, str]:
    """Mirror train_jax_tpu.build_text_pair so serving matches training.

    학습과 같은 규칙이어야 분포가 안 깨진다.
    step_text 있으면 step-level (A=문제+이전 step, B=대상 step),
    없으면 explanation-level (A=문제, B=설명 전체).
    """
    if step_text:
        prior = prior_steps or []
        context = " ".join([problem_text, *(str(s) for s in prior)])
        return context.strip(), step_text
    return problem_text, explanation


# ── Encoder-backed model ──────────────────────────────────────────────────────

class EncoderVerifierModel:
    """Fine-tuned encoder model loaded from a local SavedModel directory."""

    def __init__(self, model_dir: str | Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.eval()
        self._torch = torch

    def predict(
        self,
        problem_text: str,
        answer: str,
        explanation: str = "",
        prior_steps: list[str] | None = None,
        step_text: str | None = None,
    ) -> dict[str, float | str | list[str]]:
        import torch

        sentence_a, sentence_b = _build_pair(
            problem_text, prior_steps, step_text, explanation
        )
        enc = self._tok(
            sentence_a,
            sentence_b,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs[0, 1])

        return _prediction_from_confidence(confidence)


# ── Flax encoder (Cloud TPU-trained mBERT) ──

class FlaxEncoderVerifierModel:
    """TPU-trained Flax mBERT verifier, loaded from a flax_model/ dir.

    train_jax_tpu.py 와 동일한 pair 토크나이즈(max_len 256)로 추론한다.
    sentence_a = problem, sentence_b = explanation(=judged step).
    """

    def __init__(self, model_dir: str | Path) -> None:
        from transformers import (
            AutoTokenizer,
            FlaxAutoModelForSequenceClassification,
        )

        self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = FlaxAutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

    def predict(
        self,
        problem_text: str,
        answer: str,
        explanation: str = "",
        prior_steps: list[str] | None = None,
        step_text: str | None = None,
    ) -> dict[str, float | str | list[str]]:
        import jax

        sentence_a, sentence_b = _build_pair(
            problem_text, prior_steps, step_text, explanation
        )
        enc = self._tok(
            sentence_a,
            sentence_b,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="np",
        )
        logits = self._model(**enc).logits
        probs = jax.nn.softmax(logits, axis=-1)
        confidence = float(probs[0, 1])
        return _prediction_from_confidence(confidence)


# ── Keyword-heuristic fallback ────────────────────────────────────────────────

_KO_STEP_TOKENS = ("따라서", "그러므로", "왜냐하면", "이므로", "즉", "결론적으로")
_EN_STEP_TOKENS = ("because", "therefore", "so", "since", "thus", "hence")


class LocalVerifierModel:
    """Keyword-heuristic verifier — no ML deps required."""

    def predict(
        self,
        problem_text: str,
        answer: str,
        explanation: str = "",
        prior_steps: list[str] | None = None,
        step_text: str | None = None,
    ) -> dict[str, float | str | list[str]]:
        target = step_text or explanation
        normalized = target.lower()
        answer_hit = bool(answer) and answer.lower() in normalized
        step_hit = any(
            t in normalized for t in _KO_STEP_TOKENS + _EN_STEP_TOKENS
        )
        has_numbers = any(c.isdigit() for c in target)

        # Weighted heuristic: answer mention + causal connector + numeric work
        score = (0.4 * answer_hit) + (0.35 * step_hit) + (0.25 * has_numbers)
        confidence = min(0.95, max(0.15, score))

        tags: list[str] = []
        if not step_hit:
            tags.append("missing_reasoning")
        if not has_numbers:
            tags.append("missing_calculation")

        prediction = VerifierPrediction(
            correctness_confidence=round(confidence, 4),
            missing_step_score=round(1.0 - confidence, 4),
            misconception_tags=tags,
            difficulty_match="unknown",
        )
        return asdict(prediction)


# ── Factory ───────────────────────────────────────────────────────────────────

def _load_encoder_model() -> (
    EncoderVerifierModel | FlaxEncoderVerifierModel | None
):
    candidates = [
        os.environ.get("VERIFIER_MODEL_DIR", ""),
        "training/verifier/artifacts/pytorch_model",
        "/gcs/YOUR_GCS_BUCKET/outputs/verifier_deberta/pytorch_model",
        # Servable TPU-trained artifact (2026-05-27 re-run, reload_val_accuracy 0.880).
        # The old outputs/verifier/ is the untrained-save artifact and must not be served.
        "/gcs/YOUR_GCS_BUCKET/outputs/verifier_v2/flax_model",
    ]
    for path_str in candidates:
        if not path_str:
            continue
        path = Path(path_str)
        if not (path.exists() and (path / "config.json").exists()):
            continue
        is_flax = (path / "flax_model.msgpack").exists()
        try:
            if is_flax:
                model: Any = FlaxEncoderVerifierModel(path)
            else:
                model = EncoderVerifierModel(path)
            kind = "flax" if is_flax else "torch"
            print(f"[verifier] loaded {kind} encoder from {path}")
            return model
        except Exception as exc:
            print(f"[verifier] failed to load {path}: {exc}")
    return None


def _build_verifier() -> (
    EncoderVerifierModel | FlaxEncoderVerifierModel | LocalVerifierModel
):
    encoder = _load_encoder_model()
    if encoder is not None:
        return encoder
    print("[verifier] keyword fallback (no trained model found)")
    return LocalVerifierModel()


verifier_model: (
    EncoderVerifierModel | FlaxEncoderVerifierModel | LocalVerifierModel
) = _build_verifier()
