#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass


GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
PROJECT = os.environ.get("PROJECT", "gcp-research-credits-489020")
CONTROL_PLANE = os.environ.get(
    "CONTROL_PLANE",
    "gs://gcp-researchcredits-blocklab-europe-west4/coord_v2",
)


@dataclass
class VmType:
    name: str
    zone: str
    accel: str
    runtime: str
    bucket: str
    pkg_tar_gcs: str


def sh(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def gcloud_tpu(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return sh([GCLOUD, "alpha", "compute", "tpus", "tpu-vm", *args, f"--project={PROJECT}"], timeout=timeout)


def wait_ready(vm: str, zone: str, timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        r = gcloud_tpu("describe", vm, f"--zone={zone}", "--format=json(state)", timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            try:
                state = json.loads(r.stdout).get("state", "")
            except Exception:
                state = r.stdout.strip()
            last = state
            if state == "READY":
                return
        time.sleep(15)
    raise RuntimeError(f"{vm} not READY after {timeout_s}s (last_state={last})")


def create_vm(vm: str, t: VmType) -> None:
    args = [
        "create",
        vm,
        f"--zone={t.zone}",
        f"--accelerator-type={t.accel}",
        f"--version={t.runtime}",
        "--spot",
        "--async",
    ]
    # Prefer IAP + internal IPs for no-internet zones; still works with NAT where configured.
    if t.name in ("v4", "v6e_us"):
        args.insert(-1, "--internal-ips")
    r = gcloud_tpu(*args, timeout=60)
    if r.returncode != 0 and "already exists" not in (r.stderr or "").lower():
        raise RuntimeError(f"create failed: {r.stderr[-500:]}")


def delete_vm(vm: str, zone: str) -> None:
    gcloud_tpu("delete", vm, f"--zone={zone}", "--quiet", timeout=120)


def ssh(vm: str, zone: str, cmd: str, timeout_s: int) -> str:
    r = gcloud_tpu(
        "ssh",
        vm,
        f"--zone={zone}",
        "--tunnel-through-iap",
        f"--command={cmd}",
        timeout=timeout_s,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed rc={r.returncode}: {r.stderr[-500:]}")
    return r.stdout


def parse_loss_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("LOSS_RESULT:"):
            payload = line.split("LOSS_RESULT:", 1)[1].strip()
            return json.loads(payload)
    raise ValueError("LOSS_RESULT line not found")


def verify_jsonl(gcs_path: str) -> tuple[bool, str]:
    # Ensure exists and contains strictly increasing steps with >=2 entries.
    r = sh(["gsutil", "cat", gcs_path], timeout=60)
    if r.returncode != 0:
        return False, f"missing_jsonl: {r.stderr[-200:]}"
    steps = []
    for line in r.stdout.strip().splitlines():
        try:
            steps.append(int(json.loads(line).get("step", -1)))
        except Exception:
            continue
    if len(steps) < 2:
        return False, f"too_few_steps: {steps}"
    if any(b <= a for a, b in zip(steps, steps[1:])):
        return False, f"non_monotonic_steps: {steps}"
    return True, f"ok_steps: {steps[:5]}...({len(steps)} lines)"


def unit_test_once(vm: str, t: VmType, run_idx: int) -> dict:
    runstamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_r{run_idx}"
    test_id = f"unit_{runstamp}_{vm}_chip0"
    gcs_out = f"{t.bucket}/checkpoints/{test_id}"
    cmd = (
        "set -euo pipefail; "
        f"PKG={t.pkg_tar_gcs}; "
        "rm -rf /tmp/setup_pkg && mkdir -p /tmp/setup_pkg; "
        "gsutil cp \"$PKG\" /tmp/setup_pkg.tgz; "
        "tar xzf /tmp/setup_pkg.tgz -C /tmp/setup_pkg; "
        "chmod +x /tmp/setup_pkg/run.sh; "
        f"TPU_NAME={vm} ZONE={t.zone} BUCKET={t.bucket} CONTROL_PLANE={CONTROL_PLANE} "
        f"RUNSTAMP={runstamp} TEST_ID={test_id} GCS_OUT={gcs_out} "
        "/tmp/setup_pkg/run.sh"
    )
    out = ssh(vm, t.zone, cmd, timeout_s=7200)
    lr = parse_loss_result(out)
    ok, detail = verify_jsonl(lr.get("gcs_train_loss_jsonl", ""))
    lr["verify_jsonl_ok"] = ok
    lr["verify_jsonl_detail"] = detail
    lr["vm_type"] = t.name
    lr["run_idx"] = run_idx
    return lr


def main() -> None:
    ts = time.strftime("%m%d%H%M", time.gmtime())
    types = [
        VmType(
            name="v6e_us",
            zone="us-east1-d",
            accel="v6e-8",
            runtime="v2-alpha-tpuv6e",
            bucket="gs://gcp-researchcredits-blocklab-us-east1",
            pkg_tar_gcs="gs://gcp-researchcredits-blocklab-us-east1/pull_code_v3/setup_packages/setup_v6e_us_20260315_023654.tar.gz",
        ),
        VmType(
            name="v6e_eu",
            zone="europe-west4-a",
            accel="v6e-8",
            runtime="v2-alpha-tpuv6e",
            bucket="gs://gcp-researchcredits-blocklab-europe-west4",
            pkg_tar_gcs="gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/setup_packages/setup_v6e_eu_20260315_023654.tar.gz",
        ),
        VmType(
            name="v4",
            zone="us-central2-b",
            accel="v4-8",
            runtime="tpu-ubuntu2204-base",
            bucket="gs://gcp-researchcredits-blocklab-1-us-central2",
            pkg_tar_gcs="gs://gcp-researchcredits-blocklab-1-us-central2/pull_code_v3/setup_packages/setup_v4_20260315_023654.tar.gz",
        ),
    ]

    all_results: list[dict] = []
    for t in types:
        # TPU node IDs must be RFC1035-ish: lowercase, digits, hyphens. No underscores.
        vm = f"unit-{t.name.replace('_','')}-{ts}"
        print(f"[runner] creating {vm} ({t.name}) in {t.zone}", flush=True)
        create_vm(vm, t)
        print(f"[runner] waiting READY: {vm}", flush=True)
        wait_ready(vm, t.zone, timeout_s=1800)

        try:
            for i in range(1, 4):
                print(f"[runner] unit test {t.name} run {i}/3 on {vm}", flush=True)
                res = unit_test_once(vm, t, i)
                all_results.append(res)
                print(f"[runner] result {t.name}#{i}: jsonl_ok={res['verify_jsonl_ok']} {res['gcs_train_loss_jsonl']}", flush=True)
                if not res["verify_jsonl_ok"]:
                    raise RuntimeError(f"{t.name} run {i} failed: {res['verify_jsonl_detail']}")
        finally:
            print(f"[runner] deleting {vm}", flush=True)
            delete_vm(vm, t.zone)

    out_path = os.path.expanduser(f"~/distributed_tpu_training/v3/UNIT_TEST_REPORT_{time.strftime('%Y-%m-%d', time.gmtime())}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"[runner] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
