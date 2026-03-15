#!/usr/bin/env python3
"""
populate.py (v3) — Populate GCS pending/ with experiment configs.

New in v3: --drain mode.
  --drain: write drain flag to GCS, wait for running/ to empty, then return.
           Use before --clear to safely stop babysitters without losing tasks.
  --clear: only safe AFTER --drain completes (or with --force).

Usage:
    EXP=exp13_rerun3 python3 ~/distributed_tpu_training/v3/populate.py [--dry-run]
    EXP=exp13_rerun3 python3 ~/distributed_tpu_training/v3/populate.py --drain
    EXP=exp13_rerun3 python3 ~/distributed_tpu_training/v3/populate.py --clear --force
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import gcs_write, gcs_read, gcs_list, gcs_delete, gcs_delete_prefix, CONTROL_PLANE


DRAIN_FLAG_PATH = f"{CONTROL_PLANE}/flags/drain.json"


def load_exp_config():
    exp = os.environ.get('EXP')
    if not exp:
        print("ERROR: EXP env var required", file=sys.stderr)
        sys.exit(1)
    env_file = os.path.expanduser(f'~/distributed_tpu_training/experiments/{exp}.env')
    if not os.path.exists(env_file):
        print(f"ERROR: {env_file} not found", file=sys.stderr)
        sys.exit(1)
    cfg = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                cfg[key.strip()] = val.strip()
    return cfg


def load_exp_module(cfg):
    import importlib
    module_name = cfg['EXP_MODULE']
    work_dir = os.path.expanduser(f"~/sf_bema/experiments/{cfg['WORK_DIR']}")
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    parent = os.path.expanduser("~/sf_bema/experiments")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(module_name)


def drain_and_wait(timeout_s=600, poll_s=15):
    """Write drain flag. Wait for running/ to empty. Return True if clean."""
    print(f"[drain] Writing drain flag to {DRAIN_FLAG_PATH}...")
    gcs_write(DRAIN_FLAG_PATH, json.dumps({'drained_at': time.time()}))
    print(f"[drain] Waiting for running/ to empty (timeout={timeout_s}s)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        running = gcs_list(f"{CONTROL_PLANE}/running")
        if not running:
            print(f"[drain] running/ is empty — safe to clear")
            return True
        print(f"[drain] {len(running)} tasks still running... (sleeping {poll_s}s)")
        time.sleep(poll_s)
    still_running = len(gcs_list(f"{CONTROL_PLANE}/running"))
    print(f"[drain] WARNING: timeout after {timeout_s}s — {still_running} still running")
    return False


def clear_drain_flag():
    gcs_delete(DRAIN_FLAG_PATH)
    print("[drain] Drain flag cleared")


def main():
    parser = argparse.ArgumentParser(description='Populate pending tasks (v3)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing pending/running/failed before populating')
    parser.add_argument('--force', action='store_true',
                        help='Allow --clear even with active heartbeats')
    parser.add_argument('--drain', action='store_true',
                        help='Write drain flag and wait for running/ to empty, then exit')
    parser.add_argument('--undrain', action='store_true',
                        help='Remove drain flag (re-enable babysitters)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print tasks without uploading')
    parser.add_argument('--skip-completed', action='store_true', default=True)
    args = parser.parse_args()

    # --undrain: just remove the flag and exit
    if args.undrain:
        clear_drain_flag()
        return

    # --drain: write flag, wait for clean state, then exit
    if args.drain:
        drained = drain_and_wait()
        if not drained:
            print("[drain] Did not drain cleanly. Use --clear --force to override.")
            sys.exit(1)
        return

    cfg = load_exp_config()
    exp_name = cfg['EXP_NAME']
    work_dir = cfg['WORK_DIR']

    module = load_exp_module(cfg)

    if hasattr(module, 'SCRIPT'):
        script_abs = module.SCRIPT
        work_dir_abs = os.path.expanduser(f"~/sf_bema/experiments/{work_dir}")
        train_script = os.path.relpath(script_abs, work_dir_abs)
    else:
        mod_parts = cfg['EXP_MODULE'].split('.')
        train_script = f"{mod_parts[0]}/train_tpu.py"

    configs = module.build_configs()
    print(f"Experiment: {exp_name}")
    print(f"Work dir: {work_dir}")
    print(f"Train script: {train_script}")
    print(f"Total configs: {len(configs)}")

    skip_ids = set()
    skip_labels = set()

    if args.skip_completed:
        for p in gcs_list(f"{CONTROL_PLANE}/completed"):
            skip_ids.add(os.path.basename(p).replace('.json', ''))
        if skip_ids:
            print(f"Already completed: {len(skip_ids)}")

    for state in ('pending', 'running', 'failed'):
        for p in gcs_list(f"{CONTROL_PLANE}/{state}"):
            skip_ids.add(os.path.basename(p).replace('.json', ''))

    inflight_count = sum(1 for tid in skip_ids if tid.startswith(f"{exp_name}__"))
    if inflight_count:
        print(f"In-flight (pending/running/failed): {inflight_count}")

    validated_dir = os.path.expanduser(f"~/sf_bema/results/{exp_name}/validated")
    if os.path.isdir(validated_dir):
        for f in os.listdir(validated_dir):
            if f.endswith('.json'):
                skip_labels.add(f.replace('.json', ''))
        if skip_labels:
            print(f"Already validated locally: {len(skip_labels)}")

    if args.clear and not args.dry_run:
        if not args.force:
            # Check drain flag is set (safe) OR no heartbeats
            drain_active = gcs_read(DRAIN_FLAG_PATH) is not None
            if not drain_active:
                heartbeats = gcs_list(f"{CONTROL_PLANE}/heartbeats")
                if heartbeats:
                    print(f"ERROR: {len(heartbeats)} active heartbeat(s) — workers may be in-flight.")
                    print("Run: python3 populate.py --drain   (then --clear --force)")
                    print("Or: python3 populate.py --clear --force   (to bypass)")
                    sys.exit(1)
        print("Clearing pending/running/failed...")
        gcs_delete_prefix(f"{CONTROL_PLANE}/pending")
        gcs_delete_prefix(f"{CONTROL_PLANE}/running")
        gcs_delete_prefix(f"{CONTROL_PLANE}/failed")
        skip_ids.clear()
        skip_labels.clear()
        # Remove drain flag so babysitters can resume
        clear_drain_flag()

    populated = 0
    skipped = 0
    for label, overrides in configs:
        task_id = f"{exp_name}__{label}"
        if task_id in skip_ids or label in skip_labels:
            skipped += 1
            continue
        task = {
            'task_id': task_id,
            'experiment': exp_name,
            'label': label,
            'overrides': overrides,
            'work_dir': work_dir,
            'train_script': train_script,
            'created_at': time.time(),
        }
        if args.dry_run:
            print(f"  [dry-run] {task_id}")
        else:
            path = f"{CONTROL_PLANE}/pending/{task_id}.json"
            content = json.dumps(task, indent=2)
            if not gcs_write(path, content):
                print(f"  ERROR: gcs_write failed for {task_id} — skipping")
                continue
            written = gcs_read(path)
            if not written or len(written) < 10:
                print(f"  ERROR: write verification failed for {task_id} — skipping")
                continue
            populated += 1
            if populated % 10 == 0:
                print(f"  Uploaded {populated} tasks...")

    print(f"\nDone: populated={populated}, skipped={skipped}")
    if not args.dry_run:
        pending = len(gcs_list(f"{CONTROL_PLANE}/pending"))
        completed = len(gcs_list(f"{CONTROL_PLANE}/completed"))
        print(f"Queue: pending={pending} completed={completed}")


if __name__ == '__main__':
    main()
