"""RTX PRO 6000 을 여러 region 에 동시 배포해서 먼저 뜨는 놈을 잡는다(병렬 폴백).
한 region 씩 순차로 기다리면 stockout 503 한 번에 15-25분씩 날아가서, 쿼터 확인된 4개 region 동시 시도.
첫 성공을 winner 로 쓰고 나머지(그리고 죽인 순차 task 의 orphan endpoint)는 display_name 으로 싹 teardown.

비용: 잠깐 여러 endpoint 가 동시에 뜰 수 있음. 끝나면 winner 외 전부 내려서 과금 끊음.
GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT 로 호출(글로벌 ADC 안 건드림). process 격리로 SDK init 충돌 회피.

Usage:
    GOOGLE_CLOUD_QUOTA_PROJECT=YOUR_GCP_PROJECT python scripts/deploy_gemma_parallel.py
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import vertexai
from google.cloud import aiplatform
from vertexai.preview import model_garden

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT")
MODEL_ID = "google/gemma3@gemma-3-12b-it"
DISPLAY = "gemma3-12b-negatives"
MODEL_DISPLAY = "gemma3-12b-it-negatives"
# RTX PRO 6000 (Blackwell 96GB) 쿼터 2 확인된 region. A100 80GB 보다 위, 멀티 region 으로 재고 stockout 분산.
REGIONS = ["us-central1", "us-east1", "europe-west4", "asia-southeast2"]
TARGET_ACCEL = "NVIDIA_RTX_PRO_6000"
ENDPOINT_FILE = Path("D:/tmp/gemma_endpoint.txt")


def _find_rtx_option(om: model_garden.OpenModel):
    for o in om.list_deploy_options():
        if o.dedicated_resources.machine_spec.accelerator_type.name == TARGET_ACCEL:
            return o
    return None


def worker(region: str, q: "Queue") -> None:
    """한 region 에 RTX PRO 6000 배포 시도. 결과만 큐로 보고(teardown 은 main 이 일괄)."""
    try:
        vertexai.init(project=PROJECT, location=region)
        om = model_garden.OpenModel(MODEL_ID)
        opt = _find_rtx_option(om)
        if opt is None:
            q.put(("nofit", region, "no RTX_PRO_6000 deploy option in region", ""))
            return
        ms = opt.dedicated_resources.machine_spec
        print(f"[{region}] deploy {ms.machine_type} {ms.accelerator_type.name} x{ms.accelerator_count}", flush=True)
        ep = om.deploy(
            accept_eula=True,
            machine_type=ms.machine_type,
            accelerator_type=ms.accelerator_type.name,
            accelerator_count=ms.accelerator_count,
            min_replica_count=1,
            max_replica_count=1,
            serving_container_spec=opt.container_spec,
            endpoint_display_name=DISPLAY,
            model_display_name=MODEL_DISPLAY,
        )
        q.put(("ok", region, ep.resource_name, ep.name))
    except Exception as e:  # noqa: BLE001  region 별 독립 실패 -> 폴백
        q.put(("fail", region, f"{type(e).__name__}: {e}", ""))


def teardown(region: str, resource: str) -> None:
    try:
        aiplatform.init(project=PROJECT, location=region)
        ep = aiplatform.Endpoint(resource)
        ep.undeploy_all()
        ep.delete(force=True)
        print(f"[teardown] {region} {resource} 내림", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[teardown] {region} {resource} FAILED: {e}", flush=True)


def sweep_endpoints() -> list:
    """모든 region 에서 DISPLAY 이름 endpoint 수집. 죽인 순차 task 의 orphan LRO 결과까지 잡는다."""
    found = []  # (region, resource, id, n_deployed)
    for region in REGIONS:
        try:
            aiplatform.init(project=PROJECT, location=region)
            for ep in aiplatform.Endpoint.list(filter=f'display_name="{DISPLAY}"'):
                n = len(ep.gca_resource.deployed_models or [])
                found.append((region, ep.resource_name, ep.name, n))
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] {region} list FAILED: {e}", flush=True)
    return found


def main() -> int:
    print(f"[parallel] regions={REGIONS} model={MODEL_ID}", flush=True)
    q: "Queue" = Queue()
    procs = [Process(target=worker, args=(r, q), name=r) for r in REGIONS]
    for p in procs:
        p.start()

    for _ in REGIONS:
        status, region, info, _eid = q.get()
        print(f"[parallel] {region} -> {status}: {info}", flush=True)
    for p in procs:
        p.join()

    # 이름 기반 sweep: 실제 뜬 endpoint 전부 확인(병렬 worker + orphan).
    found = sweep_endpoints()
    print(f"[parallel] found endpoints: {found}", flush=True)
    ready = [f for f in found if f[3] >= 1]

    if not ready:
        print("[parallel] NO READY ENDPOINT. 전부 실패/재고없음. 아침에 보고.", flush=True)
        return 1

    # winner: region 선호 순서대로 첫 ready 1개.
    pref = {r: i for i, r in enumerate(REGIONS)}
    ready.sort(key=lambda f: pref.get(f[0], 99))
    win_region, win_res, win_id, _ = ready[0]

    # winner 외 ready 전부 + deployed_model 0개짜리(반쯤 뜬) 전부 teardown.
    for region, res, _eid, _n in ready[1:]:
        teardown(region, res)
    for region, res, _eid, n in found:
        if n == 0:
            teardown(region, res)

    ENDPOINT_FILE.write_text(f"{win_res}\n{win_id}\n{win_region}\n", encoding="utf-8")
    print(f"[parallel] WINNER region={win_region} id={win_id}", flush=True)
    print(f"[parallel] resource={win_res}", flush=True)
    print(f"[parallel] wrote {ENDPOINT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
