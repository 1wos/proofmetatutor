# ProofMetaTutor PRD

## Product Summary

ProofMetaTutor is a GCP-native Korean math learning prototype that combines:

- Cloud TPU-trained solution verification
- Gemma tutor behavior
- ADK agent orchestration
- ontology-style evidence tracing
- teacher human-in-the-loop approval
- Google Cloud safety and evaluation services

## One-line Pitch

ProofMetaTutor verifies Korean math solution explanations with a
Cloud TPU-trained model and turns each tutor recommendation into an
inspectable evidence trace.

## Problem

AI tutors often provide answers before students produce evidence of
understanding. Teachers then struggle to inspect why the system thinks a
student has a misconception.

## Hypothesis

If students explain solutions before receiving help, and those
explanations are verified by a Cloud TPU-trained model, then the system
can produce more inspectable learning evidence than a chat-only tutor.

## Non-goals

- No real student efficacy claim.
- No school deployment.
- No collection of new minor student data.
- No diagnosis of learning disabilities.
- No autonomous student-facing intervention.

## MVP Flow

1. Load a math problem.
2. Ask the student to explain the solution.
3. Ask one clarification question.
4. Run the verifier.
5. Store an evidence bundle.
6. Show a teacher-facing recommendation.
7. Require approve, edit, or reject.

## Success Metrics

- verifier macro F1
- calibration error
- tool-call accuracy
- direct-answer leakage rate
- evidence bundle coverage
- teacher gate bypass rate

## Public Positioning

ProofMetaTutor is not a generic tutor chatbot. It is a reference project for
connecting Cloud TPU training to an inspectable agent product on Google
Cloud.

