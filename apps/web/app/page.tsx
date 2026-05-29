"use client";

import { useEffect, useRef, useState } from "react";
import { EvidencePanel } from "../components/EvidencePanel";
import { SolutionChecker } from "../components/SolutionChecker";
import { GemmaShowcase } from "../components/GemmaShowcase";
import { runVerifier, type VerifierResult } from "../lib/verifier";

const SAMPLES = [
  { label: "일차방정식", problem: "2x + 3 = 11", step: "양변에서 3을 빼면 2x = 8, 그래서 x = 4" },
  { label: "이차방정식", problem: "x² = 9", step: "x = 3 이다" },
  { label: "분수 계산", problem: "3/4 + 1/4 을 계산하시오", step: "분자와 분모를 각각 더하면 4/8 = 1/2" },
];

const THRESHOLD = 0.6;

// SVG icons (no emoji)
const IconSpark = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1m0-12.8l-2.1 2.1M7.7 16.3l-2.1 2.1" />
  </svg>
);
const IconCheck = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
);
const IconAlert = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M12 8v5m0 3h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /></svg>
);
const IconSpinner = () => (
  <svg className="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-6.2-8.5" /></svg>
);

function useCountUp(target: number, on: boolean) {
  const [v, setV] = useState(0);
  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!on) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setV(target); return; }
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / 700);
      const eased = 1 - Math.pow(1 - t, 3);
      setV(from + (target - from) * eased);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [target, on]);
  return v;
}

export default function Page() {
  const [problem, setProblem] = useState(SAMPLES[0].problem);
  const [explanation, setExplanation] = useState(SAMPLES[0].step);
  const [result, setResult] = useState<VerifierResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(0);
  const [runId, setRunId] = useState(0);

  function loadSample(s: (typeof SAMPLES)[number], i: number) {
    setProblem(s.problem);
    setExplanation(s.step);
    setSelected(i);
    setResult(null);
  }

  async function onRun() {
    setLoading(true);
    try {
      const r = await runVerifier({ problem_text: problem, explanation, step_text: explanation });
      setResult(r);
      setRunId((n) => n + 1);
    } finally {
      setLoading(false);
    }
  }

  const conf = result?.correctness_confidence ?? 0;
  const isCorrect = conf >= THRESHOLD;
  const shown = useCountUp(conf, !!result);

  const metrics: [string, string][] = [
    ["검증 신뢰도", result ? shown.toFixed(2) : "—"],
    ["누락 단계 점수", result ? result.missing_step_score.toFixed(2) : "—"],
    ["오개념 태그", result ? String(result.misconception_tags.length) : "—"],
    ["교사 상태", result ? (isCorrect ? "검토 대기" : "개입 제안") : "대기"],
  ];

  return (
    <main className="shell">
      <section className="hero">
        <div className="reveal" style={{ animationDelay: "0.05s" }}>
          <p className="eyebrow"><IconSpark /> AI 수학 풀이 검증</p>
          <h1>ProofMetaTutor</h1>
          <p className="summary">
            <strong>정답이 아니라 풀이 과정</strong>을 봅니다. AI가 단계별로
            검증하고, 그 근거를 교사가 한눈에 확인할 수 있어요.
            <span className="summary-en">
              Verifies the reasoning, not just the answer — with the evidence a
              teacher can check at a glance.
            </span>
          </p>
        </div>
        <div className="metricGrid reveal" style={{ animationDelay: "0.18s" }}>
          {metrics.map(([label, value]) => (
            <div className="metric" key={label}>
              <span>{label}</span>
              <strong key={runId} className={result ? "pop" : ""}>{value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="workspace">
        <article className="panel reveal" style={{ animationDelay: "0.28s" }}>
          <p className="eyebrow"><IconSpark /> 학생 풀이 — 이 단계, 맞을까요?</p>

          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button
                type="button"
                className={`chip ${selected === i ? "active" : ""}`}
                aria-pressed={selected === i}
                key={s.label}
                onClick={() => loadSample(s, i)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <label className="field">
            <span>문제 · Problem</span>
            <input value={problem} onChange={(e) => setProblem(e.target.value)} />
          </label>
          <label className="field">
            <span>풀이 단계 · Step</span>
            <textarea rows={3} value={explanation} onChange={(e) => setExplanation(e.target.value)} />
          </label>

          <div className="actions">
            <button type="button" onClick={onRun} disabled={loading}>
              {loading ? <><IconSpinner /> 검증 중…</> : "검증 실행"}
            </button>
            {result && (
              <span className={`badge ${result.source === "tpu-model" ? "live" : "mock"}`}>
                {result.source === "tpu-model" ? "실시간 검증" : "오프라인 데모"}
              </span>
            )}
          </div>

          {result && (
            <div className="verdict" key={`${conf}-${result.source}`}>
              <span className={`verdict-chip ${isCorrect ? "ok" : "bad"}`}>
                {isCorrect ? <IconCheck /> : <IconAlert />}
                {isCorrect ? "CORRECT" : "INCORRECT"}
              </span>
              <div className="bar">
                <span className="bar-tick" style={{ left: `${THRESHOLD * 100}%` }} title="기준 0.6" />
                <div
                  className={`bar-fill ${isCorrect ? "ok" : "bad"}`}
                  style={{ width: `${Math.round(shown * 100)}%` }}
                />
              </div>
              <span className="bar-num">{(shown * 100).toFixed(0)}%</span>
            </div>
          )}

        </article>

        <div className="reveal" style={{ animationDelay: "0.38s" }}>
          <EvidencePanel problemId="pt-math-001" concept="linear equation" result={result} />
        </div>
      </section>

      <section className="reveal" style={{ animationDelay: "0.46s" }}>
        <SolutionChecker />
      </section>

      <section className="reveal" style={{ animationDelay: "0.52s" }}>
        <GemmaShowcase />
      </section>

      <footer className="foot reveal" style={{ animationDelay: "0.56s" }}>
        ProofMetaTutor — 근거 기반 수학 튜터 프로토타입
      </footer>
    </main>
  );
}
