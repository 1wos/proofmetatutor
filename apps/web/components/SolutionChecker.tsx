"use client";

import { useState } from "react";
import { checkSolution, type SolutionResult } from "../lib/verifier";

const DEFAULT_PROBLEM = "2x + 6 = 10";
const DEFAULT_STEPS = "양변에서 6을 빼면 2x = 4\n양변을 2로 나누면 x = 4";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
);
const Alert = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M12 8v5m0 3h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /></svg>
);

export function SolutionChecker() {
  const [problem, setProblem] = useState(DEFAULT_PROBLEM);
  const [stepsText, setStepsText] = useState(DEFAULT_STEPS);
  const [res, setRes] = useState<SolutionResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function onCheck() {
    const steps = stepsText.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!steps.length) return;
    setLoading(true);
    setRes(null);
    try {
      setRes(await checkSolution(problem, steps));
    } finally {
      setLoading(false);
    }
  }

  const err = res?.first_error_index ?? -1;

  return (
    <article className="panel solution">
      <p className="eyebrow"><Alert /> 풀이 전체 검사 — 어디서 막혔을까요?</p>
      <div className="sc-inputs">
        <label className="field">
          <span>문제 · Problem</span>
          <input value={problem} onChange={(e) => setProblem(e.target.value)} />
        </label>
        <label className="field">
          <span>풀이 단계 (한 줄에 하나씩)</span>
          <textarea rows={4} value={stepsText} onChange={(e) => setStepsText(e.target.value)} />
        </label>
      </div>
      <button type="button" onClick={onCheck} disabled={loading}>
        {loading ? "검사 중…" : "풀이 검사"}
      </button>

      {res && (
        <div className="sc-result">
          <p className={`sc-summary ${err >= 0 ? "bad" : "ok"}`}>
            {err >= 0
              ? `${err + 1}단계에서 막혔어요`
              : "모든 단계가 검증을 통과했어요"}
            {res.expected && <span className="sc-ans"> · 정답 x = {res.expected.join(", ")}</span>}
          </p>
          <ol className="sc-steps">
            {res.steps.map((s, i) => {
              const bad = s.arithmetic === "error";
              const ok = s.arithmetic === "ok";
              return (
                <li
                  key={i}
                  className={`sc-step ${bad ? "bad" : ok ? "ok" : "unknown"} ${i === err ? "first-error" : ""}`}
                  style={{ animationDelay: `${i * 120}ms` }}
                >
                  <span className="sc-icon">{bad ? <Alert /> : ok ? <Check /> : null}</span>
                  <span className="sc-text">{s.step}</span>
                  {s.confidence != null && (
                    <span className="sc-conf">{Math.round(s.confidence * 100)}%</span>
                  )}
                  {bad && <span className="sc-reason">{s.reason}</span>}
                </li>
              );
            })}
          </ol>
          {res.source === "tpu+cas" && (
            <p className="sc-note">TPU 모델 신뢰도 + 규칙 기반 산술 검증 결합</p>
          )}
        </div>
      )}
    </article>
  );
}
