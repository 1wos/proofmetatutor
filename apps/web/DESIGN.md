# ProofMetaTutor — DESIGN.md

A DESIGN.md analysis (getdesign.md-style) for the ProofMetaTutor web UI. The tone is
**calm, academic, evidence-first** — a teacher's review surface, not a flashy app.

## Philosophy
- **Evidence over verdicts.** Show *why* before *what*. The trace panel is a peer
  of the input, not an afterthought.
- **Calm confidence.** Muted sage paper, one teal accent, generous whitespace.
- **Honest signals.** Correct = green, incorrect = amber (never alarming red),
  uncertain = neutral. Confidence is a bar, not just a number.
- **Bilingual, Korean-first.** The product verifies Korean math; copy leads in
  Korean with a quiet English subtitle.

## Color tokens
| Token | Value | Use |
|---|---|---|
| `--bg` | `#f7f8f4` | sage paper background |
| `--ink` | `#17201c` | primary text |
| `--muted` | `#5c6761` | secondary text, labels |
| `--line` | `#dce2d8` | hairline borders |
| `--surface` | `#ffffff` | panels, cards |
| `--teal` | `#0f766e` | primary accent, buttons, links |
| `--amber` | `#b45309` | caution / incorrect |
| `--green` | `#15803d` | correct verdict |
| `--violet` | `#6d5bd0` | reserved (metacognition) |

## Type
- Family: Inter / system sans.
- Scale: hero `56px/1.0`, section `24px/1.2`, body `18px/1.6`, label `13px` uppercase teal.
- Weight: 800 for eyebrows & numbers, 700 for emphasis, 400 body.

## Shape & space
- Radius: 8px panels, 999px chips. 1px `--line` borders, no heavy shadows.
- Rhythm: 24px panel padding, 16px grid gaps, 48px page padding.
- Layout: 1160px max width; hero `1.2fr / 1fr`; workspace `1fr / 1fr`, stacks on mobile.

## Components
- **Eyebrow** — uppercase teal micro-label above every section.
- **Metric** — label + big number; the live verifier fills these.
- **Verdict chip** — green `CORRECT` / amber `INCORRECT` pill from the model output.
- **Confidence bar** — teal fill on a `--line` track; visualizes correctness probability.
- **Sample chips** — one-tap Korean math examples so a reviewer never has to type.
- **Source badge** — `TPU-trained mBERT` (teal) vs `offline demo` (amber) so the
  demo never lies about whether the live model answered.

## Premium layer ("Calm Glass")
- **Depth:** glass panels (`backdrop-filter: blur(14px) saturate(140%)`, 72% white),
  16px radius, layered soft shadows (sm/md/lg) — not flat cards.
- **Aurora:** two slow-drifting blurred teal/mint radial blobs behind everything
  (`body::before/::after`), low opacity. Frozen under reduced-motion.
- **Display type:** Fraunces (optical serif) for the wordmark, metrics, and section
  titles; Inter for body. The h1 uses a teal→ink gradient text fill.

## Motion
- Easing `cubic-bezier(0.22, 1, 0.36, 1)`.
- Micro-interactions 150–250ms (hover lift, button press, focus ring).
- Larger reveals 500–900ms: staggered `fadeUp` entrance, verdict `popIn`,
  confidence bar count-up + width, shimmer sweep on the bar fill.
- All transforms/opacity (GPU). `prefers-reduced-motion` disables every animation.

## Voice
- Plain, respectful, teacherly. "이 단계, 맞을까요?" not "Analyze!". English stays a
  quiet subtitle.
