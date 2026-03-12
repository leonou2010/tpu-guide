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
from gcs import gcs_write, gcs_list, gcs_delete_prefix, CONTROL_PLANE


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

    # Build the full set of task IDs to skip (truly idempotent):
    # completed, validated, AND currently in-flight (running/pending/failed).
    # Skipping in-flight tasks prevents populate from overwriting retries/last_error
    # on tasks that are actively being retried after a failure.
    skip_ids = set()   # task_ids (exp__label format)
    skip_labels = set()  # labels only (for validated check)

    if args.skip_completed:
        for p in gcs_list(f"{CONTROL_PLANE}/completed"):
            skip_ids.add(os.path.basename(p).replace('.json', ''))
        if skip_ids:
            print(f"Already completed: {len(skip_ids)}")

    # In-flight: pending + running + failed — do not overwrite these
    for state in ('pending', 'running', 'failed'):
        for p in gcs_list(f"{CONTROL_PLANE}/{state}"):
            skip_ids.add(os.path.basename(p).replace('.json', ''))

    inflight_count = sum(1 for tid in skip_ids if tid.startswith(f"{exp_name}__"))
    if inflight_count:
        print(f"In-flight (pending/running/failed): {inflight_count}")

    # Validated local results
    validated_dir = os.path.expanduser(f"~/sf_bema/results/{exp_name}/validated")
    if os.path.isdir(validated_dir):
        for f in os.listdir(validated_dir):
            if f.endswith('.json'):
                skip_labels.add(f.replace('.json', ''))
        if skip_labels:
            print(f"Already validated locally: {len(skip_labels)}")

    # Clear if requested (--clear bypasses the skip logic above)
    if args.clear and not args.dry_run:
        print("Clearing pending/running/failed...")
        gcs_delete_prefix(f"{CONTROL_PLANE}/pending")
        gcs_delete_prefix(f"{CONTROL_PLANE}/running")
        gcs_delete_prefix(f"{CONTROL_PLANE}/failed")
        skip_ids.clear()
        skip_labels.clear()

    # Populate
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
            gcs_write(path, json.dumps(task, indent=2))
            populated += 1
            if populated % 10 == 0:
                print(f"  Uploaded {populated} tasks...", flush=True)

    print(f"\nDone: populated={populated}, skipped={skipped} (completed/validated)")
    if not args.dry_run:
        pending, running, completed, failed = 0, 0, 0, 0
        pending = len(gcs_list(f"{CONTROL_PLANE}/pending"))
        completed = len(gcs_list(f"{CONTROL_PLANE}/completed"))
        print(f"Queue: pending={pending} completed={completed}")


if __name__ == '__main__':
    main()
