// Client for the ProofMetaTutor verifier API (the Cloud TPU-trained mBERT model
// served on Cloud Run). Falls back to a deterministic offline mock so the demo
// always renders, even before the API URL is configured.

export type VerifierRequest = {
  problem_text: string;
  answer?: string;
  explanation?: string;
  prior_steps?: string[];
  step_text?: string | null;
};

export type VerifierResult = {
  correctness_confidence: number;
  missing_step_score: number;
  misconception_tags: string[];
  difficulty_match: string;
  source: "tpu-model" | "offline-mock";
};

const API_BASE = process.env.NEXT_PUBLIC_VERIFIER_API?.replace(/\/$/, "") ?? "";

export function isLiveApiConfigured(): boolean {
  return API_BASE.length > 0;
}

// Deterministic offline heuristic — clearly labelled, used only when the live
// API is not reachable. It is NOT the model; it just keeps the UI demoable.
function offlineMock(req: VerifierRequest): VerifierResult {
  const text = `${req.explanation ?? ""} ${req.step_text ?? ""}`.toLowerCase();
  const hasNumber = /\d/.test(text);
  const hasReason = /(because|so|since|move|subtract|divide|따라서|이므로|므로)/.test(text);
  const len = text.trim().length;
  let conf = 0.45;
  if (hasNumber) conf += 0.2;
  if (hasReason) conf += 0.2;
  if (len > 40) conf += 0.1;
  conf = Math.min(0.97, Math.round(conf * 100) / 100);
  return {
    correctness_confidence: conf,
    missing_step_score: Math.round((1 - conf) * 100) / 100,
    misconception_tags: conf < 0.6 ? ["ALG-EQ-ONESIDE"] : [],
    difficulty_match: "unknown",
    source: "offline-mock",
  };
}

export type StepResult = {
  step: string;
  confidence: number | null;
  misconception_tags: string[];
  arithmetic: "ok" | "error" | null;
  reason: string;
};

export type SolutionResult = {
  expected: string[] | null;
  first_error_index: number;
  steps: StepResult[];
  source: "tpu+cas" | "offline-mock";
};

export async function checkSolution(
  problem: string,
  steps: string[],
): Promise<SolutionResult> {
  if (API_BASE) {
    try {
      const res = await fetch(`${API_BASE}/api/check-solution`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ problem_text: problem, steps }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      return { ...data, source: "tpu+cas" } as SolutionResult;
    } catch {
      /* fall through to offline mock */
    }
  }
  // Offline mock: flag a step that claims "x = N" disagreeing with a trivial solve.
  const out = steps.map((s) => ({
    step: s,
    confidence: 0.7,
    misconception_tags: [] as string[],
    arithmetic: null as "ok" | "error" | null,
    reason: "오프라인 데모",
  }));
  return { expected: null, first_error_index: -1, steps: out, source: "offline-mock" };
}

export async function runVerifier(req: VerifierRequest): Promise<VerifierResult> {
  if (!API_BASE) return offlineMock(req);
  try {
    const res = await fetch(`${API_BASE}/api/verifier/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`API ${res.status}`);
    const data = await res.json();
    return { ...data, source: "tpu-model" } as VerifierResult;
  } catch {
    return offlineMock(req);
  }
}
