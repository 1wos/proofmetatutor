# Verifier Model Selection — SOTA Analysis

Last updated: 2026-05-25

## Task Definition

Binary sequence-pair classification:
- Input: (problem_text, student_explanation)
- Output: label ∈ {correct, incorrect} + confidence score

This is structurally similar to NLI (Natural Language Inference) but domain-specific
to Korean K-12 math, requiring:
1. Mathematical reasoning understanding
2. Korean language comprehension
3. Step-by-step logic verification

---

## Candidate Models Evaluated

| Model | Params | XNLI acc | Korean | Flax | PyTorch | Verdict |
|-------|--------|----------|--------|------|---------|---------|
| `google/bert-base-multilingual-cased` (mBERT) | 178M | 74.5% | via multilingual | ✅ | ✅ | **JAX path** |
| `microsoft/mdeberta-v3-base` | 86M | **79.8%** | via multilingual | ❌ | ✅ | **PyTorch-XLA path** |
| `klue/roberta-large` | 355M | Korean SOTA | ✅ native | ❌ | ✅ | too large for smoke test |
| `snunlp/KR-ELECTRA-discriminator` | 14M | — | ✅ native | ❌ | ✅ | fast inference candidate |
| `intfloat/multilingual-e5-large` | 560M | — | ✅ | ❌ | ✅ | retrieval only |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | — | ✅ | ❌ | ✅ | baseline |

---

## Selected Architecture

### Primary (TPU training target): mDeBERTa-v3-base + Classification Head

```
Input: [problem_text] [SEP] [student_explanation]
       ↓
mDeBERTa-v3-base encoder (86M params)
  - Disentangled attention (relative position + content)
  - Superior cross-lingual transfer vs mBERT
       ↓
[CLS] pooled representation (768-dim)
       ↓
Dropout(0.1) → Linear(768→2) → Softmax
       ↓
P(correct), P(incorrect)
```

**Why mDeBERTa-v3 over mBERT:**
- +5.3% on XNLI (79.8% vs 74.5%) with 52% fewer parameters
- Disentangled attention better captures step-by-step reasoning structure
- Trains well on small datasets (< 10K examples) due to stronger pre-training

**Why not klue/roberta-large:**
- 355M params → expensive on TPU v6e-1 with 32 batch size
- Korean-only → poor transfer if we add English GSM8K data
- No Flax weights available

### JAX/Flax Path: mBERT + Flax Classification Head

Used for the TPU smoke test and Vertex AI Custom Training job because:
- `FlaxBertForSequenceClassification` is available natively in HuggingFace Transformers
- No transpilation needed (unlike mDeBERTa)
- Lower memory footprint for Cloud TPU v6e-1 (1 chip)

---

## Training Data Strategy

### Phase 1 (immediate): AIHub 30번 + synthetic negatives from Gemma 4

| Split | Source | Size | Label |
|-------|--------|------|-------|
| Positive | AIHub 수학 풀이과정 (No. 30) | ~20K | 1 (correct) |
| Negative — missing_step | Gemma 4 generated | ~60K | 0 |
| Negative — wrong_concept | Gemma 4 generated | ~60K | 0 |
| Negative — calculation_error | Gemma 4 generated | ~60K | 0 |

Script: `scripts/generate_synthetic_negatives.py`

### Phase 2: KMMLU + GSM8K multilingual augmentation

| Dataset | Language | Size |
|---------|----------|------|
| HAERAE-HUB/KMMLU (Math) | Korean | ~2K |
| openai/gsm8k | English | ~8.5K |

Script: `scripts/download_hf_datasets.py`

---

## Expected Performance

| Model | Dataset | Expected F1 |
|-------|---------|------------|
| Keyword heuristic (current baseline) | AIHub test | ~0.45 |
| mBERT fine-tuned (Flax, 5 epochs) | AIHub test | ~0.72–0.78 |
| mDeBERTa-v3 fine-tuned (PyTorch-XLA, 5 epochs) | AIHub test | ~0.78–0.84 |
| Gemma 4 zero-shot (reference) | AIHub test | ~0.68–0.75 |

Note: estimates based on similar Korean classification benchmarks. Actual numbers
will be in `docs/tpu_run_report.md` after training runs.

---

## Serving Architecture

```
Request → verifier_api/app/model.py
    ↓
_load_encoder_model()
    ↓
Priority 1: $VERIFIER_MODEL_DIR (env var, production)
Priority 2: training/verifier/artifacts/pytorch_model (local test)
Priority 3: /gcs/YOUR_GCS_BUCKET/outputs/verifier_deberta/pytorch_model
Priority 4: Keyword-heuristic fallback (always available)
```

Model is singleton — loaded once at startup, reused across requests.
