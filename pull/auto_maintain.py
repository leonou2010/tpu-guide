#!/usr/bin/env python3
"""
auto_maintain.py — Autonomous fleet maintenance for pull-based coordinator.

Runs every cycle (default 10 min):
1. Check experiment progress (validated counts)
2. Re-queue failed tasks
3. Detect and fix degraded VMs (delete + recreate if SSH fails)
4. Set up new VMs (setup.sh + babysitter deploy)
5. Copy results when experiments complete
6. Update status log

Usage:
    python3 auto_maintain.py --once          # single cycle
    python3 auto_maintain.py --interval 600  # every 10 min
"""

import json
import os
import subprocess
import sys
import time
import glob as globmod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import (gcs_list, gcs_read, gcs_write, gcs_delete, gcs_read_batch,
                 get_queue_counts, reclaim_stale, CONTROL_PLANE)

# ── Config ────────────────────────────────────────────────────────────────

GCLOUD = os.path.expanduser('~/google-cloud-sdk/bin/gcloud')
PROJECT = 'gcp-research-credits-489020'

EXPERIMENTS = {
    'exp13': {
        'target': 120,
        'result_dest': os.path.expanduser(
            '~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/'),
        'validated_dir': os.path.expanduser('~/sf_bema/results/exp13/validated/'),
    },
    'exp12_1': {
        'target': 185,
        'result_dest': os.path.expanduser(
            '~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu/results/'),
        'validated_dir': os.path.expanduser('~/sf_bema/results/exp12_1/validated/'),
    },
}

VM_CONFIGS_DIR = os.path.expanduser('~/distributed_tpu_training/vm_configs/')
STATUS_FILE = os.path.expanduser('~/distributed_tpu_training/pull/OVERNIGHT_STATUS.md')

# XLA cache locations per TPU type
XLA_CACHE = {
    'v6e': 'gs://gcp-researchcredits-blocklab-europe-west4/xla_cache/',
    'v5e': 'gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e/',
    'v4': 'gs://gcp-researchcredits-blocklab-1-us-central2/xla_cache/',  # broken but try anyway
}

# Buckets per zone prefix
ZONE_BUCKETS = {
    'europe-west4': 'gs://gcp-researchcredits-blocklab-europe-west4',
    'us-central2': 'gs://gcp-researchcredits-blocklab-1-us-central2',
    'us-east1': 'gs://gcp-researchcredits-blocklab-us-east1',
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def run_cmd(cmd, timeout=60):
    """Run command, return (success, stdout)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except Exception as e:
        return False, str(e)


def ssh_vm(tpu_name, zone, command, timeout=120):
    """SSH to TPU VM via IAP."""
    cmd = (f'{GCLOUD} alpha compute tpus tpu-vm ssh {tpu_name} '
           f'--zone={zone} --project={PROJECT} --tunnel-through-iap '
           f'--worker=all --command="{command}"')
    return run_cmd(cmd, timeout=timeout)


def get_vm_state(tpu_name, zone):
    """Get VM state (READY, CREATING, etc)."""
    ok, out = run_cmd(
        f'{GCLOUD} alpha compute tpus tpu-vm describe {tpu_name} '
        f'--zone={zone} --project={PROJECT} --format="get(state)"',
        timeout=30)
    return out if ok else None


def create_vm(tpu_name, zone, accel_type, version):
    """Create a new TPU VM (spot, internal IPs only)."""
    cmd = (f'{GCLOUD} alpha compute tpus tpu-vm create {tpu_name} '
           f'--zone={zone} --project={PROJECT} '
           f'--accelerator-type={accel_type} --version={version} --spot --internal-ips')
    log(f"  Creating VM: {tpu_name} in {zone}")
    ok, out = run_cmd(cmd, timeout=300)
    if ok:
        log(f"  VM {tpu_name} created successfully")
    else:
        log(f"  VM {tpu_name} creation failed: {out[:200]}")
    return ok


def delete_vm(tpu_name, zone):
    """Delete a TPU VM."""
    cmd = (f'{GCLOUD} alpha compute tpus tpu-vm delete {tpu_name} '
           f'--zone={zone} --project={PROJECT} --quiet')
    log(f"  Deleting VM: {tpu_name}")
    ok, out = run_cmd(cmd, timeout=120)
    return ok


def bucket_for_zone(zone):
    """Get the GCS bucket for a zone."""
    for prefix, bucket in ZONE_BUCKETS.items():
        if zone.startswith(prefix):
            return bucket
    return ZONE_BUCKETS['europe-west4']


def tpu_type_from_accel(accel):
    """Determine TPU type from accelerator string."""
    if 'v6e' in accel:
        return 'v6e'
    elif 'v5' in accel:
        return 'v5e'
    elif 'v4' in accel:
        return 'v4'
    return 'unknown'


# ── Core maintenance functions ────────────────────────────────────────────

def check_progress():
    """Check validated counts for all experiments."""
    results = {}
    for exp, cfg in EXPERIMENTS.items():
        vdir = cfg['validated_dir']
        if os.path.isdir(vdir):
            count = len([f for f in os.listdir(vdir) if f.endswith('.json')])
        else:
            count = 0
        results[exp] = count
        log(f"  {exp}: {count}/{cfg['target']} validated")
    return results


def requeue_failed():
    """Move all failed tasks back to pending with reset retries."""
    failed = gcs_list(f'{CONTROL_PLANE}/failed')
    if not failed:
        return 0
    moved = 0
    for path in failed:
        raw = gcs_read(path)
        if not raw:
            continue
        try:
            task = json.loads(raw)
        except json.JSONDecodeError:
            gcs_delete(path)
            continue
        task['retries'] = 0
        task.pop('failed_at', None)
        task.pop('last_error', None)
        task.pop('worker_id', None)
        task.pop('claimed_at', None)
        tid = task['task_id']
        gcs_write(f'{CONTROL_PLANE}/pending/{tid}.json', json.dumps(task))
        gcs_delete(path)
        moved += 1
        log(f"  Re-queued: {tid}")
    return moved


def get_fleet_health():
    """Get heartbeat status for all workers."""
    hb_paths = gcs_list(f'{CONTROL_PLANE}/heartbeats')
    if not hb_paths:
        return {}
    raw_data = gcs_read_batch(hb_paths, max_workers=20)
    heartbeats = {}
    for path, raw in raw_data.items():
        if raw:
            try:
                hb = json.loads(raw)
                heartbeats[hb['worker_id']] = hb
            except (json.JSONDecodeError, KeyError):
                pass
    return heartbeats


def detect_degraded_vms(heartbeats):
    """Find VMs with stuck/missing workers."""
    now = time.time()
    vm_health = {}  # vm_name -> {total, training, stuck, compiling}

    # Group by VM
    for wid, hb in heartbeats.items():
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        if vm not in vm_health:
            vm_health[vm] = {'total': 0, 'training': 0, 'stuck': 0, 'compiling': 0, 'idle': 0}
        vm_health[vm]['total'] += 1
        age = now - hb.get('timestamp', 0)
        status = hb.get('status', 'unknown')

        if status == 'training' and age < 600:
            vm_health[vm]['training'] += 1
        elif status in ('xla_compile', 'starting') and age < 2700:
            vm_health[vm]['compiling'] += 1
        elif status == 'idle' and age < 300:
            vm_health[vm]['idle'] += 1
        else:
            vm_health[vm]['stuck'] += 1

    return vm_health


def setup_and_deploy_vm(tpu_name, zone, bucket, accel_type, wandb_mode='online'):
    """Full setup + babysitter deploy on a VM."""
    tpu_type = tpu_type_from_accel(accel_type)

    # 1. Run setup.sh
    log(f"  Running setup.sh on {tpu_name}...")
    setup_cmd = (
        f'export BUCKET={bucket} && '
        f'bash <(gcloud storage cat {bucket}/config/setup.sh 2>/dev/null || '
        f'gsutil cat {bucket}/config/setup.sh)'
    )
    ok, out = ssh_vm(tpu_name, zone, setup_cmd, timeout=600)
    if not ok:
        log(f"  setup.sh FAILED on {tpu_name}: {out[:200]}")
        return False

    # 2. Copy pull code
    log(f"  Deploying pull code to {tpu_name}...")
    code_cmd = (
        f'mkdir -p ~/distributed_tpu_training/pull && '
        f'gcloud storage cp {bucket}/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || '
        f'gsutil -m cp {bucket}/code/pull/*.py ~/distributed_tpu_training/pull/'
    )
    ok, _ = ssh_vm(tpu_name, zone, code_cmd, timeout=60)

    # 3. Pre-load XLA cache
    xla_src = XLA_CACHE.get(tpu_type, '')
    if xla_src:
        log(f"  Loading XLA cache ({tpu_type}) on {tpu_name}...")
        xla_cmd = (
            f'mkdir -p /tmp/xla_cache && '
            f'gcloud storage cp "{xla_src}*" /tmp/xla_cache/ 2>/dev/null || '
            f'gsutil -m cp "{xla_src}*" /tmp/xla_cache/ 2>/dev/null || true'
        )
        ssh_vm(tpu_name, zone, xla_cmd, timeout=120)

    # 4. Launch babysitter
    log(f"  Launching babysitter on {tpu_name}...")
    launch_cmd = (
        f'for s in $(tmux list-sessions 2>/dev/null | cut -d: -f1); do '
        f'tmux kill-session -t "$s" 2>/dev/null; done; '
        f'fuser /dev/vfio/[0-9]* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'fuser /dev/vfio/devices/vfio* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'fuser /dev/accel* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'pkill -9 -f train_tpu 2>/dev/null || true; '
        f'sleep 2; rm -f /tmp/tpu_babysitter.lock /tmp/ckpt_*.pt; '
        f'tmux new-session -d -s babysitter '
        f'"export PATH=\\$HOME/miniconda3/bin:\\$HOME/.local/bin:\\$PATH; '
        f'export CONTROL_PLANE={CONTROL_PLANE} '
        f'BUCKET={bucket} TPU_NAME={tpu_name} '
        f'ACCELERATOR_TYPE={accel_type} '
        f'MODEL_PATH=/tmp/SmolLM2-135M '
        f'WANDB_MODE={wandb_mode} '
        f'PJRT_DEVICE=TPU '
        f'XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache '
        f'XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache '
        f'LLVM_NUM_THREADS=32 '
        f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1; '
        f'flock -n /tmp/tpu_babysitter.lock python3 -u ~/distributed_tpu_training/pull/babysitter.py '
        f'2>&1 | tee /tmp/babysitter.log; echo BABYSITTER_EXITED"'
    )
    ok, out = ssh_vm(tpu_name, zone, launch_cmd, timeout=60)
    if ok and 'babysitter' in out:
        log(f"  {tpu_name} deployed successfully")
        return True
    else:
        log(f"  {tpu_name} deploy FAILED: {out[:200]}")
        return False


def deploy_babysitter_only(tpu_name, zone, bucket, accel_type, wandb_mode='online'):
    """Quick deploy — code + babysitter only (skip setup.sh for existing VMs)."""
    # Copy latest pull code
    code_cmd = (
        f'mkdir -p ~/distributed_tpu_training/pull && '
        f'gcloud storage cp {bucket}/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || '
        f'gsutil -m cp {bucket}/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || true'
    )
    ssh_vm(tpu_name, zone, code_cmd, timeout=60)

    # Kill old sessions and launch
    launch_cmd = (
        f'for s in $(tmux list-sessions 2>/dev/null | cut -d: -f1); do '
        f'tmux kill-session -t "$s" 2>/dev/null; done; '
        f'fuser /dev/vfio/[0-9]* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'fuser /dev/vfio/devices/vfio* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'fuser /dev/accel* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; '
        f'pkill -9 -f train_tpu 2>/dev/null || true; '
        f'sleep 2; rm -f /tmp/tpu_babysitter.lock /tmp/ckpt_*.pt; '
        f'tmux new-session -d -s babysitter '
        f'"export PATH=\\$HOME/miniconda3/bin:\\$HOME/.local/bin:\\$PATH; '
        f'export CONTROL_PLANE={CONTROL_PLANE} '
        f'BUCKET={bucket} TPU_NAME={tpu_name} '
        f'ACCELERATOR_TYPE={accel_type} '
        f'MODEL_PATH=/tmp/SmolLM2-135M '
        f'WANDB_MODE={wandb_mode} '
        f'PJRT_DEVICE=TPU '
        f'XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache '
        f'XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache '
        f'LLVM_NUM_THREADS=32 '
        f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1; '
        f'flock -n /tmp/tpu_babysitter.lock python3 -u ~/distributed_tpu_training/pull/babysitter.py '
        f'2>&1 | tee /tmp/babysitter.log; echo BABYSITTER_EXITED"'
    )
    ok, out = ssh_vm(tpu_name, zone, launch_cmd, timeout=60)
    return ok and 'babysitter' in out


def copy_results(exp):
    """Copy validated results to experiment destination folder."""
    cfg = EXPERIMENTS[exp]
    src = cfg['validated_dir']
    dst = cfg['result_dest']
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.endswith('.json')]
    import shutil
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    log(f"  Copied {len(files)} results to {dst}")
    return len(files)


def load_vm_config(tpu_name):
    """Load VM config from file."""
    cfg_path = os.path.join(VM_CONFIGS_DIR, f'{tpu_name}.env')
    if not os.path.isfile(cfg_path):
        return None
    cfg = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                cfg[k] = v
    return cfg


def list_all_vms():
    """List all TPU VMs across zones."""
    vms = []
    zones = ['europe-west4-a', 'europe-west4-b', 'us-central2-b', 'us-east1-d']
    for zone in zones:
        ok, out = run_cmd(
            f'{GCLOUD} alpha compute tpus tpu-vm list --zone={zone} '
            f'--project={PROJECT} --format="csv[no-heading](name,acceleratorType,state)"',
            timeout=30)
        if ok and out:
            for line in out.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 3:
                    vms.append({
                        'name': parts[0], 'accel': parts[1],
                        'state': parts[2], 'zone': zone
                    })
    return vms


# ── Main cycle ────────────────────────────────────────────────────────────

def maintenance_cycle():
    """One full maintenance cycle."""
    log("=" * 60)
    log("MAINTENANCE CYCLE START")

    # 1. Check progress
    log("[1/6] Checking experiment progress...")
    progress = check_progress()
    p, r, c, f = get_queue_counts()
    log(f"  Queue: pending={p} running={r} completed={c} failed={f}")

    # 2. Re-queue failed tasks
    log("[2/6] Re-queuing failed tasks...")
    moved = requeue_failed()
    if moved:
        log(f"  Re-queued {moved} failed tasks")

    # 3. Reclaim stale tasks
    log("[3/6] Reclaiming stale tasks...")
    reclaimed = reclaim_stale(stale_ttl_s=900, startup_grace_s=2700)
    if reclaimed:
        log(f"  Reclaimed {reclaimed} stale tasks")

    # 4. Check fleet and fix degraded VMs
    log("[4/6] Checking fleet health...")
    heartbeats = get_fleet_health()
    vm_health = detect_degraded_vms(heartbeats)

    all_vms = list_all_vms()
    known_vms = {vm['name'] for vm in all_vms}

    for vm in all_vms:
        name = vm['name']
        state = vm['state']
        zone = vm['zone']
        accel = vm['accel']
        cfg = load_vm_config(name)

        if state == 'CREATING':
            log(f"  {name}: CREATING (waiting...)")
            continue

        if state != 'READY':
            log(f"  {name}: {state} (skipping)")
            continue

        health = vm_health.get(name)
        has_heartbeat = health is not None and health['total'] > 0

        if not has_heartbeat:
            # VM is READY but no heartbeat — needs deploy
            log(f"  {name}: READY but no heartbeat — deploying...")
            bucket = bucket_for_zone(zone)
            wandb = 'disabled' if 'us-east1' in zone or 'us-central2' in zone else 'online'

            if cfg:
                bucket = cfg.get('BUCKET', bucket)
                wandb = cfg.get('WANDB_MODE', wandb)
                accel = cfg.get('ACCELERATOR_TYPE', accel)

            # Check if this is a new VM (no code installed)
            ok, out = ssh_vm(name, zone, 'ls ~/distributed_tpu_training/pull/babysitter.py 2>/dev/null && echo EXISTS || echo MISSING', timeout=30)
            if 'MISSING' in out:
                log(f"  {name}: New VM — running full setup...")
                setup_and_deploy_vm(name, zone, bucket, accel, wandb)
            else:
                log(f"  {name}: Existing VM — quick redeploy...")
                deploy_babysitter_only(name, zone, bucket, accel, wandb)

        elif health and health['stuck'] > 0 and health['training'] == 0:
            # All chips stuck, none training — redeploy
            log(f"  {name}: ALL STUCK ({health['stuck']} stuck) — redeploying...")
            bucket = bucket_for_zone(zone)
            wandb = 'disabled' if 'us-east1' in zone or 'us-central2' in zone else 'online'
            if cfg:
                bucket = cfg.get('BUCKET', bucket)
                wandb = cfg.get('WANDB_MODE', wandb)
                accel = cfg.get('ACCELERATOR_TYPE', accel)
            deploy_babysitter_only(name, zone, bucket, accel, wandb)

        elif health:
            status = 'HEALTHY' if health['training'] > 0 else 'WARMING'
            log(f"  {name}: {status} (train={health['training']} compile={health['compiling']} "
                f"stuck={health['stuck']} idle={health['idle']})")

    # 5. Check for completion and copy results
    log("[5/6] Checking for experiment completion...")
    all_done = True
    for exp, cfg in EXPERIMENTS.items():
        count = progress.get(exp, 0)
        target = cfg['target']
        if count >= target:
            log(f"  {exp}: COMPLETE ({count}/{target})! Copying results...")
            copy_results(exp)
        else:
            all_done = False
            log(f"  {exp}: {count}/{target} — in progress")

    # 6. Update status
    log("[6/6] Updating status...")
    append_status(progress, p, r, c, f, vm_health)

    log("MAINTENANCE CYCLE DONE")
    log("=" * 60)
    return all_done


def append_status(progress, pending, running, completed, failed, vm_health):
    """Append status update to OVERNIGHT_STATUS.md."""
    try:
        with open(STATUS_FILE, 'a') as f:
            f.write(f"\n### {time.strftime('%H:%M:%S UTC')} Update\n")
            for exp, count in progress.items():
                target = EXPERIMENTS[exp]['target']
                f.write(f"- {exp}: {count}/{target}\n")
            f.write(f"- Queue: pending={pending} running={running} completed={completed} failed={failed}\n")
            healthy = sum(1 for v in vm_health.values() if v['training'] > 0)
            total = len(vm_health)
            f.write(f"- Fleet: {healthy}/{total} VMs healthy\n")
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=600, help='Seconds between cycles')
    args = parser.parse_args()

    if args.once:
        maintenance_cycle()
        return

    while True:
        try:
            all_done = maintenance_cycle()
            if all_done:
                log("ALL EXPERIMENTS COMPLETE!")
                break
        except Exception as e:
            log(f"ERROR in maintenance cycle: {e}")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
