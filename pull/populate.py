#!/usr/bin/env python3
"""
populate.py — Populate GCS pending/ with experiment configs.

Reads experiment module's build_configs(), writes one JSON per task to
the control-plane bucket's pending/ directory.

Usage:
    EXP=exp13 python3 ~/distributed_tpu_training/pull/populate.py [--clear] [--dry-run]
    EXP=exp12_1 python3 ~/distributed_tpu_training/pull/populate.py
"""

import argparse
import json
import os
import sys
import time

# Add distributed_tpu_training to path for coordinator helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import gcs_write, gcs_list, gcs_delete, gcs_delete_prefix, CONTROL_PLANE


def load_exp_config():
    """Load experiment .env file."""
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
    """Import the experiment module."""
    import importlib
    module_name = cfg['EXP_MODULE']
    work_dir = os.path.expanduser(f"~/sf_bema/experiments/{cfg['WORK_DIR']}")
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    parent = os.path.expanduser("~/sf_bema/experiments")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(module_name)


def main():
    parser = argparse.ArgumentParser(description='Populate pending tasks')
    parser.add_argument('--clear', action='store_true', help='Clear existing pending/running/failed before populating')
    parser.add_argument('--dry-run', action='store_true', help='Print tasks without uploading')
    parser.add_argument('--skip-completed', action='store_true', default=True, help='Skip already-completed tasks (default)')
    args = parser.parse_args()

    cfg = load_exp_config()
    exp_name = cfg['EXP_NAME']
    work_dir = cfg['WORK_DIR']

    # Derive train_script from module
    module = load_exp_module(cfg)

    # Get train script relative path
    if hasattr(module, 'SCRIPT'):
        script_abs = module.SCRIPT
        work_dir_abs = os.path.expanduser(f"~/sf_bema/experiments/{work_dir}")
        train_script = os.path.relpath(script_abs, work_dir_abs)
    else:
        # Fallback: guess from module name (exp13_tpu.run_tpu -> exp13_tpu/train_tpu.py)
        mod_parts = cfg['EXP_MODULE'].split('.')
        train_script = f"{mod_parts[0]}/train_tpu.py"

    # Build configs
    configs = module.build_configs()
    print(f"Experiment: {exp_name}")
    print(f"Work dir: {work_dir}")
    print(f"Train script: {train_script}")
    print(f"Total configs: {len(configs)}")

    # Check completed tasks
    completed_ids = set()
    if args.skip_completed:
        completed_paths = gcs_list(f"{CONTROL_PLANE}/completed")
        for p in completed_paths:
            tid = os.path.basename(p).replace('.json', '')
            completed_ids.add(tid)
        if completed_ids:
            print(f"Already completed: {len(completed_ids)}")

    # Also check validated local results
    validated_dir = os.path.expanduser(f"~/sf_bema/results/{exp_name}/validated")
    validated_labels = set()
    if os.path.isdir(validated_dir):
        for f in os.listdir(validated_dir):
            if f.endswith('.json'):
                validated_labels.add(f.replace('.json', ''))
        if validated_labels:
            print(f"Already validated locally: {len(validated_labels)}")

    # Clear if requested
    if args.clear and not args.dry_run:
        drain_flag = f"{CONTROL_PLANE}/DRAIN"
        print("Setting DRAIN flag — babysitters will stop claiming new tasks...")
        gcs_write(drain_flag, '{"reason":"populate --clear","ts":' + str(int(time.time())) + '}')

        # Wait for running/ to empty (babysitters finish current tasks, don't claim new ones)
        drain_timeout = 300  # 5 min max wait
        drain_start = time.time()
        while time.time() - drain_start < drain_timeout:
            running = gcs_list(f"{CONTROL_PLANE}/running")
            if not running:
                print("running/ is empty — safe to clear")
                break
            print(f"  Waiting for {len(running)} running tasks to finish... ({int(time.time()-drain_start)}s)")
            time.sleep(15)
        else:
            print(f"WARNING: Drain timeout after {drain_timeout}s — {len(running)} tasks still running. Clearing anyway.")

        print("Clearing pending/running/failed...")
        gcs_delete_prefix(f"{CONTROL_PLANE}/pending")
        gcs_delete_prefix(f"{CONTROL_PLANE}/running")
        gcs_delete_prefix(f"{CONTROL_PLANE}/failed")

    # Populate
    populated = 0
    skipped = 0
    for label, overrides in configs:
        task_id = f"{exp_name}__{label}"

        if task_id in completed_ids or label in validated_labels:
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
            gcs_write(path, json.dumps(task, indent=2))
            populated += 1
            if populated % 10 == 0:
                print(f"  Uploaded {populated} tasks...", flush=True)

    # Remove DRAIN flag so babysitters can resume claiming
    if args.clear and not args.dry_run:
        gcs_delete(drain_flag)
        print("DRAIN flag removed — babysitters can claim new tasks")

    print(f"\nDone: populated={populated}, skipped={skipped} (completed/validated)")
    if not args.dry_run:
        pending, running, completed, failed = 0, 0, 0, 0
        pending = len(gcs_list(f"{CONTROL_PLANE}/pending"))
        completed = len(gcs_list(f"{CONTROL_PLANE}/completed"))
        print(f"Queue: pending={pending} completed={completed}")


if __name__ == '__main__':
    main()
