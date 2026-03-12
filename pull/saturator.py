#!/usr/bin/env python3
"""
saturator.py — Queued-resource saturator for TPU fleet.

Floods GCP with queued-resource requests to keep 320c quota fully utilized.
Runs every N minutes. Submits more requests than quota to ensure instant
replacement when VMs are preempted.

Usage:
    python3 -u saturator.py [--once] [--interval 120] [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import time

PROJECT = "gcp-research-credits-489020"
GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")

# ── Zone configurations ──
# Each zone: TPU type, runtime version, quota chips, chips per VM, bucket, xla_cache, model_gcs
ZONES = {
    "europe-west4-a": {
        "type": "v6e-8", "chips_per_vm": 8, "quota": 64,
        "version": "v2-alpha-tpuv6e",
        "bucket": "gs://gcp-researchcredits-blocklab-europe-west4",
        "xla_cache": "gs://gcp-researchcredits-blocklab-europe-west4/xla_cache",
        "model_gcs": "gs://gcp-researchcredits-blocklab-europe-west4/models/SmolLM2-135M",
        "wandb_mode": "online",
        "prefix": "v6e-ew4a",
    },
    "us-east1-d": {
        "type": "v6e-8", "chips_per_vm": 8, "quota": 64,
        "version": "v2-alpha-tpuv6e",
        "bucket": "gs://gcp-researchcredits-blocklab-us-east1",
        "xla_cache": "gs://gcp-researchcredits-blocklab-us-east1/xla_cache",
        "model_gcs": "gs://gcp-researchcredits-blocklab-us-east1/models/SmolLM2-135M",
        "wandb_mode": "disabled",
        "prefix": "v6e-ue1d",
    },
    "us-central2-b": {
        "type": "v4-8", "chips_per_vm": 4, "quota": 64,
        "version": "tpu-ubuntu2204-base",
        "bucket": "gs://gcp-researchcredits-blocklab-1-us-central2",
        "xla_cache": "gs://gcp-researchcredits-blocklab-1-us-central2/xla_cache",
        "model_gcs": "gs://gcp-researchcredits-blocklab-1-us-central2/models/SmolLM2-135M",
        "wandb_mode": "disabled",
        "prefix": "v4-uc2b",
    },
    "europe-west4-b": {
        "type": "v5litepod-4", "chips_per_vm": 4, "quota": 64,
        "version": "v2-alpha-tpuv5litepod",
        "bucket": "gs://gcp-researchcredits-blocklab-europe-west4",
        "xla_cache": "gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e",
        "model_gcs": "gs://gcp-researchcredits-blocklab-europe-west4/models/SmolLM2-135M",
        "wandb_mode": "disabled",
        "prefix": "v5e-ew4b",
    },
    "us-central1-a": {
        "type": "v5litepod-4", "chips_per_vm": 4, "quota": 64,
        "version": "v2-alpha-tpuv5litepod",
        "bucket": "gs://gcp-researchcredits-blocklab-europe-west4",
        "xla_cache": "gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e",
        "model_gcs": "gs://gcp-researchcredits-blocklab-europe-west4/models/SmolLM2-135M",
        "wandb_mode": "disabled",
        "prefix": "v5e-uc1a",
    },
}

CONTROL_PLANE = "gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"
EXP_DIR = "exp13_smollm2_smoltalk"
TRAIN_SCRIPT = "exp13_tpu/train_tpu.py"

# ── GCP helpers ──

def list_vms(zone):
    """List existing TPU VMs in a zone. Returns list of (name, state)."""
    try:
        r = subprocess.run(
            [GCLOUD, "alpha", "compute", "tpus", "tpu-vm", "list",
             f"--zone={zone}", f"--project={PROJECT}",
             "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return []
        vms = json.loads(r.stdout) if r.stdout.strip() else []
        result = []
        for vm in vms:
            name = vm.get("name", "").split("/")[-1]  # extract short name
            state = vm.get("state", "UNKNOWN")
            result.append((name, state))
        return result
    except Exception as e:
        print(f"  [list] {zone}: error: {e}")
        return []


def list_queued_resources(zone):
    """List queued resources in a zone. Returns list of (name, state)."""
    try:
        r = subprocess.run(
            [GCLOUD, "alpha", "compute", "tpus", "queued-resources", "list",
             f"--zone={zone}", f"--project={PROJECT}",
             "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return []
        qrs = json.loads(r.stdout) if r.stdout.strip() else []
        result = []
        for qr in qrs:
            name = qr.get("name", "").split("/")[-1]
            state_info = qr.get("state", {})
            state = state_info.get("state", "UNKNOWN") if isinstance(state_info, dict) else str(state_info)
            result.append((name, state))
        return result
    except Exception as e:
        print(f"  [qr-list] {zone}: error: {e}")
        return []


def delete_preempted_vm(name, zone):
    """Delete a preempted VM to free quota."""
    print(f"  [delete] {name} in {zone} (PREEMPTED)")
    try:
        subprocess.run(
            [GCLOUD, "alpha", "compute", "tpus", "tpu-vm", "delete", name,
             f"--zone={zone}", f"--project={PROJECT}", "--quiet"],
            capture_output=True, text=True, timeout=120
        )
    except Exception as e:
        print(f"  [delete] {name}: error: {e}")


def create_queued_resource(name, zone, zone_cfg, startup_script_path):
    """Create a queued-resource request with startup-script metadata."""
    print(f"  [create-qr] {name} in {zone} ({zone_cfg['type']})")

    # Read startup script
    with open(startup_script_path) as f:
        startup_script = f.read()

    # Build metadata string
    metadata_items = [
        f"startup-script={startup_script}",
    ]

    try:
        cmd = [
            GCLOUD, "alpha", "compute", "tpus", "queued-resources", "create", name,
            f"--zone={zone}",
            f"--project={PROJECT}",
            f"--accelerator-type={zone_cfg['type']}",
            f"--runtime-version={zone_cfg['version']}",
            "--spot",
            "--internal-ips",
            f"--metadata=control_plane={CONTROL_PLANE},"
            f"bucket={zone_cfg['bucket']},"
            f"exp_dir={EXP_DIR},"
            f"train_script={TRAIN_SCRIPT},"
            f"wandb_mode={zone_cfg['wandb_mode']},"
            f"model_gcs={zone_cfg['model_gcs']},"
            f"xla_cache_gcs={zone_cfg['xla_cache']}",
            f"--metadata-from-file=startup-script={startup_script_path}",
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            print(f"  [create-qr] {name}: SUBMITTED")
            return True
        else:
            # Extract short error
            err = r.stderr.strip().split('\n')[-1] if r.stderr else "unknown"
            print(f"  [create-qr] {name}: FAILED — {err}")
            return False
    except Exception as e:
        print(f"  [create-qr] {name}: error: {e}")
        return False


def delete_queued_resource(name, zone):
    """Delete a stale queued resource."""
    try:
        subprocess.run(
            [GCLOUD, "alpha", "compute", "tpus", "queued-resources", "delete", name,
             f"--zone={zone}", f"--project={PROJECT}", "--quiet", "--force"],
            capture_output=True, text=True, timeout=60
        )
    except Exception:
        pass


# ── Main loop ──

def saturate_zone(zone, zone_cfg, startup_script_path, dry_run=False):
    """Ensure a zone is fully saturated with VMs + queued resources."""
    quota_chips = zone_cfg["quota"]
    chips_per_vm = zone_cfg["chips_per_vm"]
    prefix = zone_cfg["prefix"]
    max_vms = quota_chips // chips_per_vm  # e.g. 64/8 = 8 VMs

    # Get current state
    vms = list_vms(zone)
    qrs = list_queued_resources(zone)

    # Categorize VMs
    ready_vms = [(n, s) for n, s in vms if s == "READY"]
    preempted_vms = [(n, s) for n, s in vms if s == "PREEMPTED"]
    creating_vms = [(n, s) for n, s in vms if s == "CREATING"]
    other_vms = [(n, s) for n, s in vms if s not in ("READY", "PREEMPTED", "CREATING")]

    # Categorize QRs
    active_qrs = [(n, s) for n, s in qrs if s in ("WAITING_FOR_RESOURCES", "PROVISIONING", "ACTIVE")]
    failed_qrs = [(n, s) for n, s in qrs if s in ("FAILED", "SUSPENDED")]

    active_chips = len(ready_vms) * chips_per_vm
    pending_chips = (len(creating_vms) + len([q for q in active_qrs if q[1] != "ACTIVE"])) * chips_per_vm

    print(f"\n  [{zone}] {zone_cfg['type']}: "
          f"{len(ready_vms)} READY ({active_chips}c), "
          f"{len(creating_vms)} CREATING, "
          f"{len(preempted_vms)} PREEMPTED, "
          f"{len(active_qrs)} QR active, "
          f"{len(failed_qrs)} QR failed")

    actions = 0

    # 1. Delete preempted VMs to free quota
    for name, _ in preempted_vms:
        if dry_run:
            print(f"  [dry-run] Would delete preempted {name}")
        else:
            delete_preempted_vm(name, zone)
        actions += 1

    # 2. Clean up failed queued resources
    for name, state in failed_qrs:
        if dry_run:
            print(f"  [dry-run] Would delete failed QR {name} ({state})")
        else:
            print(f"  [cleanup] Deleting failed QR {name} ({state})")
            delete_queued_resource(name, zone)
        actions += 1

    # 3. Calculate how many more VMs we need
    # Target: fill quota + overshoot by 25% for instant replacement
    target_total = max_vms + max_vms // 4  # overshoot
    current_total = len(ready_vms) + len(creating_vms) + len([q for q in active_qrs if q[1] != "ACTIVE"])
    deficit = target_total - current_total

    if deficit > 0:
        # Find next available index
        existing_names = {n for n, _ in vms} | {n for n, _ in qrs}
        idx = 1
        created = 0
        while created < deficit:
            name = f"{prefix}-{idx}"
            if name not in existing_names:
                if dry_run:
                    print(f"  [dry-run] Would create QR {name}")
                else:
                    create_queued_resource(name, zone, zone_cfg, startup_script_path)
                created += 1
                actions += 1
            idx += 1
            if idx > 50:  # safety limit
                break
    else:
        print(f"  [{zone}] Already at target ({current_total}/{target_total})")

    return actions


def run_cycle(args):
    """Run one saturation cycle across all zones."""
    startup_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup.sh")
    if not os.path.exists(startup_script):
        print(f"ERROR: startup.sh not found at {startup_script}")
        return

    print(f"\n{'='*60}")
    print(f"[saturator] {time.strftime('%H:%M:%S UTC', time.gmtime())} — Saturation cycle")
    print(f"{'='*60}")

    total_actions = 0
    total_ready = 0
    total_chips = 0

    for zone, zone_cfg in ZONES.items():
        actions = saturate_zone(zone, zone_cfg, startup_script, dry_run=args.dry_run)
        total_actions += actions

        # Quick count for summary
        vms = list_vms(zone)
        ready = sum(1 for _, s in vms if s == "READY")
        total_ready += ready
        total_chips += ready * zone_cfg["chips_per_vm"]

    print(f"\n[saturator] Summary: {total_ready} VMs READY ({total_chips}/320 chips = {total_chips*100//320}%)")
    print(f"[saturator] Actions taken: {total_actions}")
    return total_actions


def main():
    parser = argparse.ArgumentParser(description="TPU fleet saturator")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=120, help="Cycle interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    print(f"[saturator] Starting — interval={args.interval}s, dry_run={args.dry_run}")
    print(f"[saturator] Zones: {list(ZONES.keys())}")

    while True:
        run_cycle(args)

        if args.once:
            break

        print(f"\n[saturator] Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
