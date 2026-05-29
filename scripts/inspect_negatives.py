"""생성된 네거티브를 사람이 검수하기 좋게 출력. "이게 진짜 학생 실수처럼 그럴듯한가?" 판정용.
원래 단계 -> 망가진 단계 -> 왜 틀림 -> 앵커 오답을 한 블록으로 보여준다.

Usage:
    python scripts/inspect_negatives.py --neg data/synthetic/negatives_smoke.jsonl
    python scripts/inspect_negatives.py --neg ... --tag ALG-DIST-NEG   # 특정 태그만
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TAXO = Path("data/taxonomy/misconceptions_kr_math.json")


def load_tax_names() -> dict[str, str]:
    if not TAXO.exists():
        return {}
    data = json.loads(TAXO.read_text(encoding="utf-8"))
    tags = data.get("tags") or data.get("misconceptions") or []
    if isinstance(tags, dict):  # {id: {...}} 형태 대비
        return {k: v.get("name_ko", "") for k, v in tags.items()}
    return {t["id"]: t.get("name_ko", "") for t in tags if "id" in t}


def trunc(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", required=True, type=Path)
    ap.add_argument("--tag", default=None, help="이 misconception_id 만 출력")
    ap.add_argument("--limit", default=40, type=int)
    ap.add_argument("--out", default=None, type=Path, help="UTF-8 파일로 저장(Windows 콘솔 mojibake 회피)")
    args = ap.parse_args()

    names = load_tax_names()
    rows = [json.loads(l) for l in args.neg.open(encoding="utf-8") if l.strip()]
    if args.tag:
        rows = [r for r in rows if r.get("misconception_id") == args.tag]

    by_tag = Counter(r.get("misconception_id") for r in rows)
    buf: list[str] = []
    buf.append(f"== {args.neg}  총 {len(rows)}건 ==")
    buf.append(f"태그 분포: {dict(by_tag.most_common())}")
    buf.append("=" * 70)

    for r in rows[: args.limit]:
        tid = r.get("misconception_id", "?")
        nm = names.get(tid, "")
        buf.append(f"[{r.get('problem_id','')}] {r.get('grade','')}  {r.get('school_level','')}")
        buf.append(f"  문제   : {trunc(r.get('problem_text',''), 110)}")
        buf.append(f"  오개념 : {tid}  {nm}  ({r.get('misconception_category','')}/{r.get('misconception_domain','')})")
        buf.append(f"  원래   : {trunc(r.get('original_step',''), 140)}")
        buf.append(f"  망가짐 : {trunc(r.get('corrupted_step',''), 140)}")
        buf.append(f"  왜틀림 : {trunc(r.get('why_wrong',''), 140)}")
        anc = r.get("anchor_distractor")
        if anc:
            buf.append(f"  앵커오답: {trunc(str(anc), 80)}")
        buf.append("-" * 70)

    text = "\n".join(buf)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}  ({len(rows)} rows)")
    else:
        print(text)


if __name__ == "__main__":
    main()
