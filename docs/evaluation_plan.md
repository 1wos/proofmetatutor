# Evaluation Plan

## Verifier Metrics

- accuracy
- macro F1
- calibration error
- confusion matrix by difficulty
- missing-step detection F1

## Agent Metrics

- tool-call accuracy
- invalid tool-call rate
- direct-answer leakage rate
- evidence bundle coverage
- teacher gate bypass rate

## Evidence Metrics

- trace completeness
- concept coverage
- source turn coverage
- recommendation evidence coverage

## Benchmark File

Use `data/synthetic/agent_eval_cases.jsonl` for early local checks.

