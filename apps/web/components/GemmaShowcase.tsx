// Showcases the Cloud TPU-trained Gemma generative verifier using a real
// generation captured from the training run (outputs/gemma_tutor/sample_generation.txt).
// This is a static, honest sample — the model itself is heavy (2B) and not served
// on the CPU-only demo host; the artifact lives in GCS (outputs/gemma_tutor/).

const SAMPLE = {
  problem: "a = 1/4 일 때, 다음 중 그 값이 가장 작은 것은?",
  priorSteps: [
    "① √(1/a) = √4 = 2",
    "② 1/a = 4",
    "③ a = √(1/4) = 1/2",
    "④ a² = (1/4)² = 1/16",
  ],
  stepToCheck: "⑤ a = 1/4",
  verdict: "CORRECT",
};

export function GemmaShowcase() {
  return (
    <article className="panel gemma">
      <p className="eyebrow">
        Gemma 생성형 검증기 · Cloud TPU 학습
        <span className="gemma-tag">gemma-2-2b-it · LoRA · v6e</span>
      </p>
      <p className="gemma-desc">
        단계 검증을 <strong>생성형(Gemma)</strong>으로도 학습했어요. 아래는 Cloud
        TPU에서 LoRA 파인튜닝한 모델이 실제로 내놓은 판정이에요.
      </p>

      <div className="gemma-card">
        <div className="gemma-row">
          <span className="gemma-k">문제</span>
          <span className="gemma-v">{SAMPLE.problem}</span>
        </div>
        <div className="gemma-row">
          <span className="gemma-k">이전 단계</span>
          <span className="gemma-v gemma-steps">
            {SAMPLE.priorSteps.map((s) => (
              <span key={s}>{s}</span>
            ))}
          </span>
        </div>
        <div className="gemma-row">
          <span className="gemma-k">검증 단계</span>
          <span className="gemma-v">{SAMPLE.stepToCheck}</span>
        </div>
        <div className="gemma-verdict">
          <span className="gemma-k">Gemma 판정</span>
          <span className="verdict-chip ok">{SAMPLE.verdict}</span>
        </div>
      </div>

      <p className="gemma-note">
        실제 학습 산출물:{" "}
        <code>gs://YOUR_GCS_BUCKET/outputs/gemma_tutor/</code>
        {" "}· 재현 가이드는 <code>docs/cloud_tpu_runbook.md</code> §2B
      </p>
    </article>
  );
}
