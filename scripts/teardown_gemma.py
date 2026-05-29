"""배포된 Gemma 엔드포인트를 내려 과금 중단. undeploy_all 후 endpoint+model 삭제.
ENDPOINT_FILE 에서 id 를 읽는다. 생성 작업 끝나면 즉시 실행.

Usage:
    GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT python scripts/teardown_gemma.py
"""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import aiplatform

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
ENDPOINT_FILE = Path("D:/tmp/gemma_endpoint.txt")


def main() -> None:
    if not ENDPOINT_FILE.exists():
        print(f"[teardown] {ENDPOINT_FILE} 없음. 내릴 엔드포인트 모름.", flush=True)
        return
    lines = ENDPOINT_FILE.read_text(encoding="utf-8").splitlines()
    eid = lines[1].strip() if len(lines) > 1 else lines[0].strip()
    aiplatform.init(project=PROJECT, location=LOCATION)
    ep = aiplatform.Endpoint(eid)
    print(f"[teardown] endpoint={eid} undeploy_all...", flush=True)
    ep.undeploy_all()
    print("[teardown] delete endpoint...", flush=True)
    ep.delete(force=True)
    print("[teardown] done. 과금 중단.", flush=True)
    ENDPOINT_FILE.rename(ENDPOINT_FILE.with_suffix(".torn.txt"))


if __name__ == "__main__":
    main()
