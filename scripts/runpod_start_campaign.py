#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _extract_ssh_hint(pod: dict) -> str | None:
    runtime = pod.get("runtime") if isinstance(pod, dict) else None
    ports = runtime.get("ports") if isinstance(runtime, dict) else None
    if isinstance(ports, list):
        for p in ports:
            if not isinstance(p, dict):
                continue
            is_public = p.get("isIpPublic")
            ip = p.get("ip")
            public_port = p.get("publicPort")
            private_port = p.get("privatePort")
            if ip and public_port and (private_port == 22 or str(private_port) == "22"):
                user = "root"
                return f"ssh {user}@{ip} -p {public_port}"
            if ip and public_port and is_public:
                # fallback hint
                return f"ssh root@{ip} -p {public_port}"

    machine = runtime.get("machine") if isinstance(runtime, dict) else None
    if isinstance(machine, dict):
        ip = machine.get("podHostId") or machine.get("id")
        if ip:
            return f"machine={ip} (inspect Runpod console for SSH details)"
    return None


def _find_pod_by_id(runpod, pod_id: str) -> dict | None:
    pods = runpod.get_pods() or {}
    items = pods.get("myself", {}).get("pods", [])
    for p in items:
        if p.get("id") == pod_id:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Create/reuse Runpod pod for Parameter Golf campaign")
    ap.add_argument("--template-id", default="y5cejece4j", help="Runpod template id")
    ap.add_argument("--name", default=None, help="Pod name")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--volume-gb", type=int, default=80)
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--cloud-type", default="ALL", choices=["ALL", "SECURE", "COMMUNITY"])
    ap.add_argument("--reuse-id", default=None, help="Reuse an existing pod id instead of creating")
    ap.add_argument("--wait-seconds", type=int, default=600)
    ap.add_argument("--poll-seconds", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="Print final object as JSON")
    args = ap.parse_args()

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")

    try:
        import runpod
    except ImportError as exc:
        raise SystemExit("python package `runpod` is required (pip install runpod)") from exc

    runpod.api_key = api_key

    if args.reuse_id:
        pod_id = args.reuse_id
        created = False
    else:
        name = args.name or f"parameter-golf-{args.gpu_count}x-{_utc_stamp()}"
        created_pod = runpod.create_pod(
            name=name,
            image_name="",
            gpu_count=args.gpu_count,
            volume_in_gb=args.volume_gb,
            container_disk_in_gb=args.container_disk_gb,
            cloud_type=args.cloud_type,
            support_public_ip=True,
            start_ssh=True,
            template_id=args.template_id,
        )
        pod_id = created_pod.get("id") or created_pod.get("podId")
        if not pod_id:
            raise SystemExit(f"Failed to create pod. Response: {created_pod}")
        created = True

    started = time.time()
    last = None
    while True:
        pod = _find_pod_by_id(runpod, pod_id)
        if pod is None:
            state = "NOT_FOUND"
            status = "NOT_FOUND"
        else:
            state = pod.get("desiredStatus") or pod.get("machine", {}).get("podHost") or "unknown"
            status = pod.get("runtime", {}).get("uptimeInSeconds")
        if pod is not None:
            runtime = pod.get("runtime")
            if isinstance(runtime, dict) and runtime.get("uptimeInSeconds") is not None:
                last = pod
                break
        if time.time() - started > args.wait_seconds:
            last = pod
            break
        time.sleep(max(args.poll_seconds, 2))

    pod = last or _find_pod_by_id(runpod, pod_id) or {}
    ssh_hint = _extract_ssh_hint(pod)

    result = {
        "created": created,
        "pod_id": pod_id,
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "cost_per_hr": pod.get("costPerHr"),
        "gpu_count": pod.get("gpuCount"),
        "template_id": args.template_id,
        "ssh_hint": ssh_hint,
        "next": [
            "ssh into pod",
            "cd /workspace/parameter-golf (or clone repo)",
            "git checkout mark/quant-sweep-tooling",
            "bash scripts/run_first_campaign.sh",
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"pod_id: {result['pod_id']}")
        print(f"created: {result['created']}")
        print(f"desired_status: {result['desired_status']}")
        if result["ssh_hint"]:
            print(f"ssh: {result['ssh_hint']}")
        else:
            print("ssh: not yet available; check Runpod console")
        print("next:")
        for s in result["next"]:
            print(f"  - {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
