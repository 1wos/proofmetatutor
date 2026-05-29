"""Gemma 기반 step-level 음성표본(negative) 생성기.

정답 풀이(positive)의 한 step을 misconception 태그에 따라 손상시켜 음성표본을
만든다. 손상 규칙은 taxonomy(misconceptions_kr_math.json)의 gemma_injection을
그대로 프롬프트로 사용한다. 무작위 노이즈가 아니라 '실제 오개념'을 주입하므로
verifier가 답 일치만으로 통과시키지 않도록 학습된다.

설계 요점:
- applicability: 태그별 적용 가능 step만 후보로 삼아 헛손상을 막는다
  (예: NUM-ROOT-ADD 는 근호 합이 있는 step 에만).
- anchor: 레코드에 실제 객관식 오답(wrong_answer)이 있으면 프롬프트에 넣어
  Gemma 가 '그 오답에 도달하는' 그럴듯한 틀린 풀이를 만들게 유도한다.
- backend: dryrun(API 무비용, 배관 검증) / aistudio / vertex 를 교체 가능.

Usage:
    python scripts/gen_gemma_negatives.py \
        --input data/aihub/math_train.jsonl \
        --taxonomy data/taxonomy/misconceptions_kr_math.json \
        --output data/synthetic/negatives_train.jsonl \
        --backend dryrun --limit 50 --per-record 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# applicability: step 텍스트(+문제 텍스트)에 태그 오류를 주입할 수 있는지 판정.
# 보수적으로 잡아 헛손상을 줄인다. universal 태그는 항상 True.
# ---------------------------------------------------------------------------
def _has(*subs: str) -> Callable[[str, str], bool]:
    return lambda step, prob: any(s in step for s in subs)


def _re(pattern: str) -> Callable[[str, str], bool]:
    rx = re.compile(pattern)
    return lambda step, prob: bool(rx.search(step))


def _prob_has(*subs: str) -> Callable[[str, str], bool]:
    return lambda step, prob: any(s in prob for s in subs)


def _always(step: str, prob: str) -> bool:
    return True


_FRAC = r"\\frac|\\dfrac|/"
_VAR = re.compile(r"[a-zA-Z]")  # Latin 변수: 한국어 본문/단위 오탐을 거른다


def _exp_on_nonunit(step: str, prob: str) -> bool:
    """거듭제곱이 단위(cm^2 등)가 아닌 실제 밑에 붙고 곱셈/나눗셈이 있을 때만."""
    if not re.search(r"\\times|×|\\div|÷", step):
        return False
    powers = re.findall(r"(.{0,10})\^", step)
    return any(not re.search(r"cm|mm|km|mathrm|\bm\b|\bL\b", seg) for seg in powers)


APPLICABILITY: dict[str, Callable[[str, str], bool]] = {
    "NUM-FRAC-DENOM": lambda s, p: bool(re.search(_FRAC, s)) and ("+" in s or "-" in s),
    "NUM-ROOT-ADD": lambda s, p: "\\sqrt" in s and ("+" in s or "-" in s),
    "NUM-MIXED-BORROW": lambda s, p: bool(re.search(r"\d\s*\\d?frac", s)) and "-" in s,
    "NUM-DIV-SMALLER": lambda s, p: bool(re.search(r"\\div|÷", s))
    and any(k in s + p for k in (">", "<", "큰", "작", "크")),
    "NUM-ORDER-OPS": lambda s, p: bool(re.search(r"\\times|×|\\div|÷", s))
    and ("+" in s or "-" in s),
    "NUM-GCD-LCM": _has("공배수", "공약수", "최소공배수", "최대공약수"),
    "NUM-SIGN": _re(r"-\s*\d|음수|-\s*\\"),
    "NUM-DECIMAL-ALIGN": lambda s, p: bool(re.search(r"\d+\.\d+", s))
    and bool(re.search(r"\\times|×|\+|-", s)),
    # 분배 오류: Latin 변수 필수(한국어 본문의 ')(' 오탐 차단).
    "ALG-DIST-NEG": lambda s, p: bool(_VAR.search(s))
    and bool(re.search(r"[-+]?\s*\(|\)\s*\(", s))
    and ("+" in s or "-" in s),
    "ALG-EXP-LAW": _exp_on_nonunit,
    "ALG-LIKE-TERMS": _re(r"[a-zA-Z]\s*[\+\-]\s*\d|\d\s*[a-zA-Z]"),
    "ALG-TRANSPOSE-SIGN": lambda s, p: "=" in s
    and bool(_VAR.search(s))
    and ("+" in s or "-" in s),
    "ALG-EQ-ONESIDE": lambda s, p: "=" in s and bool(_VAR.search(s)),
    # 평행이동: '평행이동' 만. '그래프를'은 그림그래프(초등)에도 매칭돼 제외.
    "ALG-SHIFT-SIGN": lambda s, p: "평행이동" in s + p
    or ("축의 방향으로" in s + p),
    "ALG-PRODUCT-FORMULA": lambda s, p: any(k in s + p for k in ("인수분해", "전개", "곱셈공식"))
    or (bool(re.search(r"\)\s*\(", s)) and bool(_VAR.search(s))),
    # 비례/선형성: 변수의 제곱된 합 또는 비례 맥락만.
    "ALG-PROPORTION": lambda s, p: ("비례" in s + p)
    or (bool(re.search(r"\)\s*\^?\s*\{?\s*2", s)) and bool(_VAR.search(s))),
    "GEO-PERIM-AREA": _has("둘레", "넓이"),
    "GEO-AREA-VOLUME-SCALE": _has("닮음", "넓이비", "부피비", "닮은"),
    "GEO-UNIT-CONVERT": lambda s, p: bool(re.search(r"cm|mm|km|\bm\b", s))
    and any(k in s for k in ("제곱", "세제곱", "²", "³", "^2", "^3", "환산")),
    # 도형 성질: 명시적 도형어만. 일반 '각' 단독은 너무 넓어 제외.
    "GEO-ANGLE-PROP": lambda s, p: any(
        k in s + p for k in ("삼각형", "사각형", "오각형", "육각형", "정삼각", "합동", "닮음", "다각형", "내각", "외각")
    ),
    "DATA-FREQ-RELFREQ": _has("도수", "상대도수"),
    "DATA-MEAN-MISUSE": _has("평균", "도수분포", "계급"),
    "DATA-PROB-DEF": _has("확률", "경우의 수", "경우의수"),
    "LOG-LEAP": _always,
    "LOG-CONDITION-IGNORE": _prob_has("단,", "자연수", "정수", "범위", "이상", "이하", "양수", "음이 아닌"),
    "LOG-CIRCULAR": lambda s, p: any(k in s + p for k in ("증명", "보이", "성립함을")),
    "LOG-NO-CHECK": lambda s, p: any(k in s + p for k in ("무리", "제곱근", "\\sqrt", "방정식", "근")),
    "LOG-CORRECT-ANS-WRONG-PROCESS": _always,
    "LOG-OVERGENERALIZE": lambda s, p: any(k in s + p for k in ("모든", "항상", "일반", "임의의"))
    or "증명" in p,
    "CALC-ARITHMETIC": lambda s, p: bool(re.search(r"\d", s)),
    "CALC-COPY-ERROR": lambda s, p: bool(re.search(r"[a-zA-Z0-9]", s)),
}

# 항상 적용 가능한 fallback 태그(특정 태그가 안 걸려도 음성 1개는 확보).
UNIVERSAL = {"CALC-ARITHMETIC", "CALC-COPY-ERROR", "LOG-LEAP", "LOG-CORRECT-ANS-WRONG-PROCESS"}


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in data["tags"]}


def candidate_tags(
    step: str, problem: str, taxonomy: dict[str, dict[str, Any]]
) -> list[str]:
    out = []
    for tid in taxonomy:
        pred = APPLICABILITY.get(tid, _always)
        if pred(step, problem):
            out.append(tid)
    return out


def build_prompt(
    problem: str, steps: list[str], idx: int, tag: dict[str, Any], anchor: str
) -> str:
    """Gemma 에 보낼 한국어 손상 지시문. 라벨형 평문으로만 답하도록 강제(LaTeX 백슬래시 보존)."""
    numbered = "\n".join(f"  ({i + 1}) {s}" for i, s in enumerate(steps))
    anchor_line = (
        f"\n학생이 실제로 고른 오답: {anchor}\n가능하면 손상된 풀이가 이 오답에 도달하도록 하라."
        if anchor
        else ""
    )
    return (
        "너는 한국 수학 교사다. 아래 정답 풀이에서 지정한 한 단계만 흔한 오개념으로 "
        "틀리게 바꿔라. 나머지 단계는 그대로 두고, 수식/LaTeX 형식과 말투를 유지하라.\n\n"
        f"[문제]\n{problem}\n\n"
        f"[정답 풀이 단계]\n{numbered}\n\n"
        f"[손상할 단계 번호] ({idx + 1})\n"
        f"[주입할 오개념] {tag['name_ko']} ({tag['id']})\n"
        f"[손상 규칙] {tag['gemma_injection']}{anchor_line}\n\n"
        "먼저 이 오개념을 이 단계에 자연스럽게 주입할 수 있는지 판단하라. "
        "단계 내용과 오개념이 안 맞으면 'APPLICABLE: no' 한 줄로만 답하라.\n"
        "제약 1: CORRUPTED 에는 지정한 그 단계 하나만 다시 써라. "
        "다른 단계 번호 (1)(2)(3) 를 포함하거나 여러 단계를 합치지 마라.\n"
        "제약 2: 바꾼 단계는 수학적으로 반드시 틀려야 한다. 정답과 같은 값이나 같은 결론이 "
        "나오면 안 된다. 분배법칙 전개, 반복덧셈, 나눗셈 정의(a=b*q+r) 같은 '올바른 변형'은 "
        "손상이 아니므로 금지한다. 실제로 계산 결과나 결론이 틀린 단계를 만들어라.\n"
        "아래 형식 그대로만 출력하라(JSON 금지, 코드블록 금지, 군더더기 설명 금지). "
        "수식은 원본 LaTeX 표기(예: \\times, \\frac)를 그대로 유지하라:\n"
        "APPLICABLE: yes\n"
        "CORRUPTED: <바뀐 단계>\n"
        "WHY: <왜 틀렸는지 한 문장>"
    )


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def gen_dryrun(prompt: str, tag: dict[str, Any], step: str) -> dict[str, Any]:
    """API 무비용 배관 검증용. LLM 게이트가 없으니 applicable=True 로 둔다."""
    return {
        "applicable": True,
        "corrupted_step": f"[DRYRUN:{tag['id']}] {step}",
        "why_wrong": f"(dryrun) {tag['name_ko']} 주입 예정: {tag['gemma_injection'][:60]}",
    }


def _norm(s: str) -> str:
    """공백 제거 비교용. Gemma 가 원본 단계를 그대로 베낀 no-op 손상 탐지."""
    return re.sub(r"\s+", "", s)


def _safe_call(backend: Callable[[str], str], prompt: str, retries: int = 3) -> str:
    """대량 실행 중 transient 에러(503/timeout)에 죽지 않게 재시도 후 빈 문자열로 강등.

    빈 문자열은 _parse_reply 에서 PARSE_FAIL -> 게이트에서 drop 되므로 전체 run 은
    멈추지 않고 해당 후보 하나만 버린다. ex.map 의 first-exception abort 방지.
    """
    for attempt in range(retries):
        try:
            return backend(prompt)
        except Exception:
            if attempt == retries - 1:
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


def _parse_reply(text: str) -> dict[str, Any]:
    """Gemma 라벨형 평문(APPLICABLE/CORRUPTED/WHY) 파싱.

    JSON 을 안 쓰는 이유: corrupted_step 안의 LaTeX 백슬래시(\\times, \\frac 등)가
    json.loads 에서 \\t/\\r/\\f 로 디코딩되어 라벨이 망가지거나 invalid-escape 로
    통째로 드롭되던 버그(label leakage) 회피. regex 라 백슬래시를 원형 그대로 보존.
    """
    m_app = re.search(r"APPLICABLE\s*[:：]\s*(yes|no|true|false|y|n)", text, re.I)
    if not m_app:
        return {"applicable": False, "corrupted_step": "", "why_wrong": "PARSE_FAIL"}
    applicable = m_app.group(1).lower() in ("yes", "true", "y")

    m_corr = re.search(r"CORRUPTED\s*[:：]\s*(.+)", text, re.S | re.I)
    corrupted, why = "", ""
    if m_corr:
        # CORRUPTED 본문과 WHY 를 분리 (WHY 라벨이 같은 줄/다음 줄 어디든)
        parts = re.split(r"\n?\s*WHY\s*[:：]\s*", m_corr.group(1), maxsplit=1, flags=re.I)
        corrupted = parts[0].strip().strip('"').strip()
        if len(parts) > 1:
            why = parts[1].strip().strip('"').strip()
    if not why:
        m_why = re.search(r"WHY\s*[:：]\s*(.+)", text, re.S | re.I)
        why = m_why.group(1).strip().strip('"').strip() if m_why else ""

    if applicable and not corrupted:
        return {"applicable": False, "corrupted_step": "", "why_wrong": "PARSE_FAIL"}
    return {"applicable": applicable, "corrupted_step": corrupted, "why_wrong": why}


def make_aistudio_backend(model: str) -> Callable[[str], str]:
    import google.generativeai as genai  # API 키: GOOGLE_API_KEY 환경변수

    gm = genai.GenerativeModel(model)

    def call(prompt: str) -> str:
        resp = gm.generate_content(prompt)
        return resp.text or ""

    return call


def _extract_chat_text(pred: Any) -> str:
    """vLLM-MaaS chatCompletions 응답에서 본문 텍스트만 추출.

    컨테이너/버전마다 prediction 모양이 달라 방어적으로 여러 형태를 훑는다:
    OpenAI ChatCompletion({choices:[{message:{content}}]}) / {content} / 순수 문자열.
    """
    if pred is None:
        return ""
    if isinstance(pred, str):
        return pred.split("Output:\n", 1)[-1].strip()
    if isinstance(pred, dict):
        choices = pred.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):  # 멀티모달 파트 리스트
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                return str(content).strip()
            if choices[0].get("text"):  # completions 모양
                return str(choices[0]["text"]).strip()
        for k in ("content", "text", "generated_text", "output"):
            if pred.get(k):
                return str(pred[k]).strip()
    return str(pred).strip()


def make_vertex_backend(
    model: str, project: str, location: str, endpoint_id: str
) -> Callable[[str], str]:
    """Model Garden 자가호스팅 Gemma 엔드포인트 호출.

    GenerativeModel(=Gemini 관리형)이 아니라 배포된 Endpoint.predict 를 쓴다.
    gemma-3-12b-it vLLM-MaaS 컨테이너 계약: @requestFormat=chatCompletions, messages 배열.
    (deploy_metadata.sample_request 에서 확인. {"prompt":...} 평문 포맷 아님.)
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    ep = aiplatform.Endpoint(endpoint_id)

    def call(prompt: str) -> str:
        resp = ep.predict(
            instances=[
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.6,
                }
            ]
        )
        preds = resp.predictions
        pred = preds[0] if isinstance(preds, (list, tuple)) and preds else preds
        return _extract_chat_text(pred)

    return call


def pick_tags(cands: list[str], per_record: int, rng: random.Random) -> list[str]:
    """특정(specific) 태그를 universal fallback 보다 우선해 다양하게 K개 고른다."""
    specific = [c for c in cands if c not in UNIVERSAL]
    universal = [c for c in cands if c in UNIVERSAL]
    rng.shuffle(specific)
    rng.shuffle(universal)
    ordered = specific + universal
    return ordered[:per_record]


def process_record(
    rec: dict[str, Any],
    taxonomy: dict[str, dict[str, Any]],
    backend: Callable[[str], str] | None,
    per_record: int,
    rng: random.Random,
    keep_prompt: bool,
) -> list[dict[str, Any]]:
    steps = rec.get("steps", [])
    if len(steps) < 2:
        return []
    problem = rec.get("problem_text", "")
    anchor = rec.get("wrong_answer", "") or ""
    negatives: list[dict[str, Any]] = []
    used: set[tuple[int, str]] = set()

    # step 별 후보 태그를 모아, (step,tag) 풀에서 K개 선택.
    pool: list[tuple[int, str]] = []
    for i, s in enumerate(steps):
        for tid in candidate_tags(s, problem, taxonomy):
            pool.append((i, tid))
    if not pool:
        return []

    # 다양성: 서로 다른 step/tag 우선. specific 우선 정렬 후 K개.
    rng.shuffle(pool)
    pool.sort(key=lambda it: 0 if it[1] not in UNIVERSAL else 1)
    for i, tid in pool:
        if len([n for n in negatives]) >= per_record:
            break
        if (i, tid) in used:
            continue
        used.add((i, tid))
        tag = taxonomy[tid]
        prompt = build_prompt(problem, steps, i, tag, anchor)
        if backend is None:
            gen = gen_dryrun(prompt, tag, steps[i])
        else:
            gen = _parse_reply(_safe_call(backend, prompt))
        # LLM 게이트: 오개념이 단계와 안 맞으면 Gemma 가 거부 -> 다음 후보로.
        if not gen.get("applicable", True) or not gen["corrupted_step"]:
            continue
        # no-op 가드: 공백만 다르고 내용이 같으면 손상 실패 -> drop.
        if _norm(gen["corrupted_step"]) == _norm(steps[i]):
            continue
        corrupted_steps = list(steps)
        corrupted_steps[i] = gen["corrupted_step"]
        out = {
            "problem_id": rec.get("problem_id", ""),
            "school_level": rec.get("school_level", ""),
            "grade": rec.get("grade", ""),
            "problem_text": problem,
            "label": "negative",
            "misconception_id": tid,
            "misconception_category": tag["category"],
            "misconception_domain": tag["domain"],
            "corrupted_step_index": i,
            "original_step": steps[i],
            "corrupted_step": gen["corrupted_step"],
            "why_wrong": gen["why_wrong"],
            "steps": corrupted_steps,
            "anchor_distractor": anchor,
            "source": "gemma-dryrun" if backend is None else "gemma",
        }
        if keep_prompt:
            out["_prompt"] = prompt
        negatives.append(out)
    return negatives


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="positive step-native JSONL")
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", choices=["dryrun", "aistudio", "vertex"], default="dryrun")
    p.add_argument("--model", default="gemma-3-12b-it")
    p.add_argument("--project", default="YOUR_GCP_PROJECT")
    p.add_argument("--location", default="us-central1")
    p.add_argument("--endpoint-id", default="", help="Vertex 배포 Gemma 엔드포인트 ID")
    p.add_argument("--limit", default=None, type=int, help="처리할 positive 레코드 수")
    p.add_argument("--per-record", default=2, type=int, help="레코드당 음성표본 수")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--keep-prompt", action="store_true", help="레코드에 _prompt 보존(검수용)")
    p.add_argument("--concurrency", default=1, type=int, help="레코드 병렬 처리 수(vLLM 동시요청)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = load_taxonomy(args.taxonomy)

    backend: Callable[[str], str] | None = None
    if args.backend == "aistudio":
        backend = make_aistudio_backend(args.model)
    elif args.backend == "vertex":
        if not args.endpoint_id:
            raise SystemExit("vertex 백엔드는 --endpoint-id 가 필요합니다 (Gemma 엔드포인트 배포 후 지정).")
        backend = make_vertex_backend(args.model, args.project, args.location, args.endpoint_id)

    records = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        records = [r for r in records if len(r.get("steps", [])) >= 2][: args.limit]

    multistep_recs = [r for r in records if len(r.get("steps", [])) >= 2]
    multistep = len(multistep_recs)

    def work(idx_rec: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        idx, rec = idx_rec
        # 레코드별 독립 rng: thread-safe + concurrency 무관 결정성.
        rec_rng = random.Random(args.seed + idx)
        try:
            return process_record(rec, taxonomy, backend, args.per_record, rec_rng, args.keep_prompt)
        except Exception as e:  # 한 레코드 실패가 전체 run 을 죽이지 않게.
            print(f"  [warn] record {idx} failed: {e}", flush=True)
            return []

    from collections import Counter

    args.output.parent.mkdir(parents=True, exist_ok=True)
    by_tag: Counter[str] = Counter()
    total = 0

    # 증분 기록(flush): 장시간 유료 run 이 중간에 죽어도 부분 결과는 유효 JSONL 로 남는다.
    with args.output.open("w", encoding="utf-8") as f:

        def emit(res: list[dict[str, Any]]) -> None:
            nonlocal total
            for n in res:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
                by_tag[n["misconception_id"]] += 1
                total += 1
            f.flush()

        if args.concurrency > 1 and backend is not None:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = [ex.submit(work, item) for item in enumerate(multistep_recs)]
                done = 0
                for fut in as_completed(futs):
                    emit(fut.result())
                    done += 1
                    if done % 200 == 0:
                        print(f"  progress {done}/{multistep} records, negatives={total}", flush=True)
        else:
            for item in enumerate(multistep_recs):
                emit(work(item))

    print(f"backend={args.backend} multistep_records={multistep} negatives={total}")
    print(f"wrote {args.output}")
    print("top tags:", dict(by_tag.most_common(10)))


if __name__ == "__main__":
    main()
