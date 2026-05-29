import type { VerifierResult } from "../lib/verifier";

type Props = {
  problemId: string;
  concept: string;
  result: VerifierResult | null;
};

export function EvidencePanel({ problemId, concept, result }: Props) {
  const action = result
    ? result.correctness_confidence >= 0.6
      ? "teacher review pending"
      : "intervention suggested"
    : "awaiting verifier";

  const rows: [string, string][] = [
    ["Problem", problemId],
    ["Concept", concept],
    ["Verifier", result ? `correctness ${result.correctness_confidence.toFixed(2)}` : "—"],
    [
      "Misconceptions",
      result && result.misconception_tags.length
        ? result.misconception_tags.join(", ")
        : "none flagged",
    ],
    ["Action", action],
  ];

  return (
    <article className="panel evidence">
      <p className="eyebrow">Evidence trace</p>
      <h2>trace-001</h2>
      <dl key={result ? result.correctness_confidence : "empty"}>
        {rows.map(([label, value], i) => (
          <div
            key={label}
            className={result ? "ev-row" : ""}
            style={result ? { animationDelay: `${i * 70}ms` } : undefined}
          >
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}
