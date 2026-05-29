"""Convert AIHub math dataset (zip-of-JSONs) into ProofMetaTutor step-native JSONL.

해설(텍스트)에 HTML 표 마크업이 섞여 있어 정제가 필수다. 정제 후 1~4개
step으로 경량 분절한다. 데이터가 짧아(초등 1문장, 중고 2~4문장) PRM800K식
깊은 단계 분해는 하지 않는다. step-native 출력이라 whole-explanation
베이스라인과 step-level 모델을 같은 데이터로 둘 다 커버한다.

Usage:
    python scripts/prepare_aihub_math.py \
        --input /d/tmp/aihub30/train --split train \
        --output data/aihub/math_train.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from pathlib import Path
from typing import Any


def extract_class_text(learning_data: list[dict], class_name: str) -> str:
    for item in learning_data:
        if item.get("class_name") == class_name:
            for info in item.get("class_info_list", []):
                text = info.get("text_description", "")
                if text:
                    return text.strip()
    return ""


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# 수식 보호: \[..\] \(..\) $$..$$ \begin{}..\end{} $..$ 를 마스킹한 뒤 분절한다.
# (LaTeX 내부의 \\ 행구분이나 OCR 동그라미숫자 ①을 step 경계로 오인하지 않도록)
_MATH_PATTERNS = [
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\\(.*?\\\)", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\\begin\{(\w+)\}.*?\\end\{\1\}", re.S),
    re.compile(r"\$[^$]*\$"),
]
# step 경계: 한국어 종결어미+공백, 결론 접속사, 보기 라벨(수식 마스킹 후 남은 것만 진짜 라벨)
_BOUNDARY = re.compile(
    r"(?<=다\.)\s+|(?<=이다\.)\s+|(?<=된다\.)\s+|(?<=한다\.)\s+|(?<=구한다\.)\s+"
    r"|(?=따라서)|(?=그러므로)|(?=[㉠㉡㉢㉣㉤])|(?=[①②③④⑤])"
)


def _mask_math(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def repl(m: re.Match) -> str:
        store.append(m.group(0))
        return f"\x00{len(store) - 1}\x00"

    for pat in _MATH_PATTERNS:
        text = pat.sub(repl, text)
    return text, store


def _unmask(text: str, store: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)


def clean_text(text: str) -> str:
    """HTML 태그 제거(수식 $...$ 는 span 제거 후 그대로 보존), 엔티티 복원, 공백 정규화."""
    if not text:
        return ""
    t = _TAG.sub(" ", text)  # 태그 제거, 표 셀 텍스트는 공백으로 분리됨
    t = html.unescape(t)  # &nbsp; &lt; 등 복원
    t = t.replace("\xa0", " ")
    return _WS.sub(" ", t).strip()


def segment_steps(explanation: str) -> list[str]:
    """수식을 마스킹해 LaTeX 내부를 보호한 뒤 경량 분절. 깊은 분해는 안 한다."""
    if not explanation:
        return []
    masked, store = _mask_math(explanation)
    steps: list[str] = []
    seen: set[str] = set()
    for c in _BOUNDARY.split(masked):
        if not c:
            continue
        s = _unmask(c, store).strip()
        if len(s) < 4:
            continue
        if not re.search(r"[가-힣0-9]", s):  # "$\\$" 같은 빈 토막 제거
            continue
        if s in seen:  # 해설 통째 중복 아티팩트 제거
            continue
        seen.add(s)
        steps.append(s)
    return steps


def normalize_aihub_record(
    data: dict[str, Any], source_file: str, split: str = ""
) -> dict[str, Any] | None:
    raw = data.get("raw_data_info", {})
    source = data.get("source_data_info", {})
    learning = data.get("learning_data_info", [])

    # 문항 = 지시문(텍스트) + 수식(이미지). 둘 다 합쳐야 문제가 완성됨
    problem_instruction = extract_class_text(learning, "문항(텍스트)")
    problem_expr = extract_class_text(learning, "문항(이미지)")
    problem_text = clean_text(" ".join(t for t in (problem_instruction, problem_expr) if t))
    answer = clean_text(
        extract_class_text(learning, "정답(텍스트)")
        or extract_class_text(learning, "정답(이미지)")
    )
    # 풀이과정은 해설(텍스트), 일부 레코드는 해설(이미지) LaTeX로만 존재
    explanation = clean_text(
        extract_class_text(learning, "해설(텍스트)")
        or extract_class_text(learning, "해설(이미지)")
    )
    wrong_answer = clean_text(
        extract_class_text(learning, "오답(텍스트)")
        or extract_class_text(learning, "오답(이미지)")
    )
    steps = segment_steps(explanation)

    if not problem_text:
        return None

    standards = source.get("2022_achievement_standard", [])
    curriculum_standard = standards[0].strip() if standards else ""

    return {
        "problem_id": source.get("source_data_name", Path(source_file).stem),
        "split": split,
        "school_level": raw.get("school", ""),
        "grade": raw.get("grade", ""),
        "semester": raw.get("semester", ""),
        "curriculum_standard": curriculum_standard,
        "difficulty": source.get("level_of_difficulty", ""),
        "problem_type": source.get("types_of_problems", ""),
        "problem_text": problem_text,
        "answer": answer,
        "explanation": explanation,
        "steps": steps,
        "n_steps": len(steps),
        "wrong_answer": wrong_answer,
    }


def process_zip(zip_path: Path, limit: int | None, split: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        for json_file in json_files:
            if limit and len(records) >= limit:
                break
            try:
                with zf.open(json_file) as f:
                    data = json.load(f)
                record = normalize_aihub_record(data, json_file, split)
                if record:
                    records.append(record)
            except Exception:
                continue
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="라벨링데이터 폴더 경로")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=None, type=int, help="총 최대 레코드 수 (None=전체)")
    parser.add_argument("--split", default="", help="train/val 태그")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records: list[dict[str, Any]] = []
    zip_files = sorted(Path(args.input).glob("*.zip"))

    print(f"Found {len(zip_files)} zip files")
    for zip_path in zip_files:
        if args.limit and len(all_records) >= args.limit:
            break
        remaining = (args.limit - len(all_records)) if args.limit else None
        records = process_zip(zip_path, remaining, args.split)
        all_records.extend(records)
        print(f"  {zip_path.name}: {len(records)} records (total: {len(all_records)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with_expl = sum(1 for r in all_records if r["explanation"])
    multistep = sum(1 for r in all_records if r["n_steps"] >= 2)
    print(f"\nWrote {len(all_records)} records to {args.output}")
    print(f"  해설 보유: {with_expl} | step>=2: {multistep}")


if __name__ == "__main__":
    main()
