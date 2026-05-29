"""Gemma-3-12B-it 를 Vertex Model Garden 으로 배포.
list_deploy_options() 의 검증된 container_spec 을 그대로 넘긴다(auto-resolve 가 500 INTERNAL 냄).
A100 80GB 우선, 실패 시 L4x2 로 자동 폴백(같은 12B 모델이라 품질 동일, 하드웨어만 다름).

비용 발생: 노드가 뜨면 과금. 사용 후 teardown_gemma.py 로 내린다.
성공 시 endpoint id/resource_name 을 ENDPOINT_FILE 에 기록.

Usage:
    GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT python scripts/deploy_gemma.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import vertexai
from vertexai.preview import model_garden

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID = "google/gemma3@gemma-3-12b-it"
ENDPOINT_FILE = Path("D:/tmp/gemma_endpoint.txt")

# 우선순위: 최고품질 12B 를 어떤 가속기든 올린다. A100 80GB -> L4x2 -> H100.
PREFERENCE = ["a2-ultragpu-1g", "g2-standard-24", "a3-highgpu-1g"]


def pick_options(om: model_garden.OpenModel) -> list:
    opts = om.list_deploy_options()
    by_mt = {o.dedicated_resources.machine_spec.machine_type: o for o in opts}
    ordered = [by_mt[mt] for mt in PREFERENCE if mt in by_mt]
    # 선호 외 나머지도 뒤에 붙여 마지막 폴백 확보
    ordered += [o for o in opts if o not in ordered]
    return ordered


def try_deploy(om: model_garden.OpenModel, opt) -> "object | None":
    ms = opt.dedicated_resources.machine_spec
    mt = ms.machine_type
    accel = ms.accelerator_type.name  # enum -> 'NVIDIA_A100_80GB'
    cnt = ms.accelerator_count
    print(f"[deploy] try machine={mt} accel={accel} x{cnt}", flush=True)
    try:
        endpoint = om.deploy(
            accept_eula=True,
            machine_type=mt,
            accelerator_type=accel,
            accelerator_count=cnt,
            min_replica_count=1,
            max_replica_count=1,
            serving_container_spec=opt.container_spec,  # 검증된 spec 그대로
            endpoint_display_name="gemma3-12b-negatives",
            model_display_name="gemma3-12b-it-negatives",
        )
        return endpoint
    except Exception as e:  # noqa: BLE001  폴백 위해 광범위 포착
        print(f"[deploy] {mt} FAILED: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> int:
    vertexai.init(project=PROJECT, location=LOCATION)
    print(f"[deploy] project={PROJECT} location={LOCATION} model={MODEL_ID}", flush=True)
    print("[deploy] node 가 떠야 과금 시작; ~15-25min", flush=True)
    om = model_garden.OpenModel(MODEL_ID)
    options = pick_options(om)
    print(f"[deploy] candidate order: {[o.dedicated_resources.machine_spec.machine_type for o in options]}", flush=True)

    endpoint = None
    for opt in options:
        endpoint = try_deploy(om, opt)
        if endpoint is not None:
            break

    if endpoint is None:
        print("[deploy] ALL OPTIONS FAILED", flush=True)
        traceback.print_exc()
        return 1

    res = endpoint.resource_name
    eid = endpoint.name
    ms = None
    try:
        ms = endpoint.gca_resource.deployed_models[0].dedicated_resources.machine_spec.machine_type
    except Exception:  # noqa: BLE001
        pass
    ENDPOINT_FILE.write_text(f"{res}\n{eid}\n{ms or ''}\n", encoding="utf-8")
    print(f"[deploy] DONE endpoint_resource={res}", flush=True)
    print(f"[deploy] DONE endpoint_id={eid} machine={ms}", flush=True)
    print(f"[deploy] wrote {ENDPOINT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
