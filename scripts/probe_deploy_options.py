"""Vertex Model Garden Gemma 후보 모델의 deploy 옵션(머신/가속기/리전)을 덤프.
deploy 안 함. 비용 0. 스펙만 수집해서 배포 계획 근거로 쓴다.

Usage:
    GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT \
    python scripts/probe_deploy_options.py
"""

from __future__ import annotations

import json
import os

import vertexai
from vertexai.preview import model_garden

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

CANDIDATES = [
    "google/gemma2@gemma-2-9b-it",
    "google/gemma3@gemma-3-12b-it",
    "google/gemma3@gemma-3-4b-it",
]


def summarize_option(opt) -> dict:
    """list_deploy_options() 항목에서 머신/가속기/컨테이너 핵심만 뽑는다."""
    out: dict = {}
    dms = getattr(opt, "dedicated_resources", None)
    if dms is not None:
        spec = getattr(dms, "machine_spec", None)
        if spec is not None:
            out["machine_type"] = getattr(spec, "machine_type", None)
            out["accelerator_type"] = str(getattr(spec, "accelerator_type", None))
            out["accelerator_count"] = getattr(spec, "accelerator_count", None)
        out["min_replica"] = getattr(dms, "min_replica_count", None)
    csp = getattr(opt, "container_spec", None)
    if csp is not None:
        out["image_uri"] = getattr(csp, "image_uri", None)
    # raw fallback: 위 구조가 비면 문자열로
    if not out:
        out["raw"] = str(opt)[:600]
    return out


def main() -> None:
    vertexai.init(project=PROJECT, location=LOCATION)
    report: dict = {}
    for model_id in CANDIDATES:
        entry: dict = {"deploy_options": [], "error": None}
        try:
            om = model_garden.OpenModel(model_id)
            opts = om.list_deploy_options()
            for o in opts:
                entry["deploy_options"].append(summarize_option(o))
        except Exception as e:  # noqa: BLE001  스펙 수집용, 광범위 포착 의도
            entry["error"] = f"{type(e).__name__}: {e}"
        report[model_id] = entry
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
