"""배포된 Gemma 엔드포인트에 단일 호출. 요청/응답 포맷을 1콜로 검증(비용 거의 0).
raw prediction 구조 + 실제 파서(_extract_chat_text) 출력 둘 다 찍어 배치 전 계약 확정.

Usage:
    GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT python scripts/ping_endpoint.py
"""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import aiplatform

from gen_gemma_negatives import _extract_chat_text

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
ENDPOINT_FILE = Path("D:/tmp/gemma_endpoint.txt")

PROMPT = (
    "다음 한국어 수학 풀이 단계가 올바른지 한 문장으로 답하라.\n"
    "단계: 3 + 4 × 2 = 14"
)


def main() -> None:
    lines = ENDPOINT_FILE.read_text(encoding="utf-8").splitlines()
    eid = lines[1].strip() if len(lines) > 1 else lines[0].strip()
    aiplatform.init(project=PROJECT, location=LOCATION)
    ep = aiplatform.Endpoint(eid)
    print(f"[ping] endpoint={eid}", flush=True)

    resp = ep.predict(
        instances=[
            {
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
                "max_tokens": 128,
                "temperature": 0.2,
            }
        ]
    )
    preds = resp.predictions
    print("[ping] RAW predictions repr:", flush=True)
    print(repr(preds)[:2000], flush=True)
    pred = preds[0] if isinstance(preds, (list, tuple)) and preds else preds
    print("\n[ping] PARSED text:", flush=True)
    print(_extract_chat_text(pred), flush=True)


if __name__ == "__main__":
    main()
