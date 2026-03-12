#!/usr/bin/env python3
"""
monitor.py — Pull-based coordinator monitor.

Runs on blocklab. Reclaims dead tasks, validates results, reports status.
Replaces the push-based coordinator --monitor.

Usage:
    python3 -u ~/distributed_tpu_training/pull/monitor.py --exp exp14:200 [--once] [--interval 60]
    EXPERIMENTS="exp14:200 exp15:300" python3 -u ~/distributed_tpu_training/pull/monitor.py
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import (reclaim_stale, get_queue_counts, gcs_list, gcs_read,
                 gcs_copy, gcs_delete, gcs_move, gcs_write, CONTROL_PLANE)


# ── Result validation ────────────────────────────────────────────────────

def validate_result(local_path, min_steps=400):
    """Validate a result JSON. Returns (is_valid, reason)."""
    try:
        with open(local_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "invalid JSON"

    summary = data.get('summary', {})
    if summary.get('best_val_loss') is None:
        return False, "no best_val_loss"
    if math.isnan(summary.get('best_val_loss', float('nan'))):
        return False, "NaN loss"
    if summary.get('total_steps', 0) < min_steps:
        return False, f"only {summary.get('total_steps')} steps (need >= {min_steps})"
    return True, "ok"


def pull_and_validate(exp_name, results_base):
    """Download completed results, validate, copy to validated dir.

    Returns number of newly validated results.
    """
    completed_paths = gcs_list(f"{CONTROL_PLANE}/completed")
    validated_dir = os.path.join(results_base, exp_name, 'validated')
    tmp_dir = os.path.join(results_base, exp_name, 'tmp_pull')
    os.makedirs(validated_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Already validated locally
    already_validated = set()
    for f in os.listdir(validated_dir):
        if f.endswith('.json'):
            already_validated.add(f.replace('.json', ''))

    newly_validated = 0

    for path in completed_paths:
        task_id = os.path.basename(path).replace('.json', '')

        # Extract label from task_id (format: exp13__adamw_lr0.001)
        parts = task_id.split('__', 1)
        if len(parts) != 2:
            continue
        task_exp, label = parts

        if task_exp != exp_name:
            continue
        if label in already_validated:
            continue

        # Read completion record
        raw = gcs_read(path)
        if not raw:
            continue
        try:
            completion = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Download the actual result from regional bucket
        # The worker uploaded to: $BUCKET/coord/$EXP/results/$LABEL/summary.json
        # We need to find which bucket — check common ones
        local_path = os.path.join(tmp_dir, f'{label}.json')
        found = False
        for bucket in [
            'gs://gcp-researchcredits-blocklab-europe-west4',
            'gs://gcp-researchcredits-blocklab-1-us-central2',
            'gs://gcp-researchcredits-blocklab-us-east1',
        ]:
            gcs_result_path = f"{bucket}/coord/{exp_name}/results/{label}/summary.json"
            if gcs_copy(gcs_result_path, local_path):
                found = True
                break

        if not found:
            # Result might be in the completion record itself
            result = completion.get('result', {})
            if result and result.get('best_val_loss') is not None:
                with open(local_path, 'w') as f:
                    json.dump({'summary': result}, f)
                found = True

        if not found:
            print(f"  [validate] {label}: no result file found, skipping")
            continue

        # Validate
        valid, reason = validate_result(local_path)
        if valid:
            dst = os.path.join(validated_dir, f'{label}.json')
            shutil.move(local_path, dst)
            # Download per-step JSONL if available (so it survives cleanup_gcs.sh)
            jsonl_local = os.path.join(validated_dir, f'{label}_train_loss.jsonl')
            if not os.path.exists(jsonl_local):
                for b in [
                    'gs://gcp-researchcredits-blocklab-europe-west4',
                    'gs://gcp-researchcredits-blocklab-1-us-central2',
                    'gs://gcp-researchcredits-blocklab-us-east1',
                ]:
                    gcs_jsonl = f"{b}/coord/{exp_name}/results/{label}/train_loss.jsonl"
                    if gcs_copy(gcs_jsonl, jsonl_local):
                        break
            newly_validated += 1
            print(f"  [validate] {label}: OK")
        else:
            os.remove(local_path)
            current_retries = completion.get('retries', 0)
            MAX_INVALIDATION_RETRIES = 5
            if current_retries < MAX_INVALIDATION_RETRIES:
                # Requeue — strip result/worker metadata, keep retries so we cap eventually
                requeue = {k: v for k, v in completion.items()
                           if k not in ('worker_id', 'claimed_at', 'completed_at', 'result', 'failed_at')}
                requeue['retries'] = current_retries + 1
                requeue['requeued_reason'] = f'invalid:{reason}'
                if gcs_write(f"{CONTROL_PLANE}/pending/{task_id}.json", json.dumps(requeue)):
                    gcs_delete(path)
                    print(f"  [validate] {label}: INVALID ({reason}) — requeued (retry {requeue['retries']}/{MAX_INVALIDATION_RETRIES})")
                else:
                    print(f"  [validate] {label}: INVALID — requeue failed, leaving in completed/")
            else:
                # Too many retries — move to invalidated/ (terminal)
                print(f"  [validate] {label}: INVALID ({reason}) — terminal after {current_retries} retries, moving to invalidated/")
                invalidated_path = f"{CONTROL_PLANE}/invalidated/{task_id}.json"
                if not gcs_move(path, invalidated_path):
                    gcs_delete(path)
                    print(f"  [validate] {label}: WARNING — gcs_move failed, deleted")

    return newly_validated


# ── Status display ───────────────────────────────────────────────────────

def print_status(experiments):
    """Print summary status for all experiments."""
    pending, running, completed, failed = get_queue_counts()
    invalidated = len(gcs_list(f"{CONTROL_PLANE}/invalidated"))
    total = pending + running + completed + failed + invalidated
    print(f"\n[monitor] {time.strftime('%H:%M:%S')} | "
          f"pending={pending} running={running} completed={completed} "
          f"failed={failed} invalidated={invalidated} total={total}")

    # Per-experiment breakdown
    for exp_name, (target, results_base) in experiments.items():
        validated_dir = os.path.join(results_base, exp_name, 'validated')
        validated = len([f for f in os.listdir(validated_dir) if f.endswith('.json')]) if os.path.isdir(validated_dir) else 0
        pct = validated / target * 100 if target > 0 else 0
        print(f"  {exp_name}: {validated}/{target} validated ({pct:.0f}%)")

    # Show active heartbeats only (much faster than reading all running tasks)
    heartbeat_paths = gcs_list(f"{CONTROL_PLANE}/heartbeats")
    active = 0
    training = 0
    compiling = 0
    idle = 0
    for path in heartbeat_paths:
        raw = gcs_read(path)
        if raw:
            try:
                hb = json.loads(raw)
                age = time.time() - hb.get('timestamp', 0)
                if age < 900:  # active
                    active += 1
                    status = hb.get('status', '')
                    if status == 'training':
                        training += 1
                    elif status == 'xla_compile':
                        compiling += 1
                    elif status == 'idle':
                        idle += 1
            except json.JSONDecodeError:
                pass
    print(f"  Active workers: {active} ({training} training, {compiling} compiling, {idle} idle)")


def copy_final_results(exp_name, results_base, dest_dir):
    """Copy validated results to final experiment directory."""
    src_dir = os.path.join(results_base, exp_name, 'validated')
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        if f.endswith('.json'):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))
            count += 1
    return count


# ── Main loop ────────────────────────────────────────────────────────────

def parse_experiments(exp_specs, results_base):
    """Parse experiment specs like 'exp14:200' into {name: (total, results_base)} dict."""
    experiments = {}
    for spec in exp_specs:
        if ':' in spec:
            name, total = spec.rsplit(':', 1)
            experiments[name] = (int(total), results_base)
        else:
            raise ValueError(f"Invalid --exp spec '{spec}': expected NAME:TOTAL (e.g. exp14:200)")
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='Pull-based sweep monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python3 monitor.py --exp exp14:200\n'
               '  python3 monitor.py --exp exp14:200 exp15:300\n'
               '  EXPERIMENTS="exp14:200 exp15:300" python3 monitor.py',
    )
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    parser.add_argument('--interval', type=int, default=60, help='Poll interval (seconds)')
    parser.add_argument('--stale-ttl', type=int, default=1800, help='Heartbeat stale threshold (seconds)')
    parser.add_argument('--exp', nargs='+', metavar='NAME:TOTAL',
                        help='Experiments to track (e.g. exp14:200 exp15:300). '
                             'Overrides EXPERIMENTS env var.')
    parser.add_argument('--results-base', default=os.path.expanduser('~/sf_bema/results'),
                        help='Local results base dir (default: ~/sf_bema/results)')
    args = parser.parse_args()

    # Resolve experiment list: --exp > EXPERIMENTS env var
    results_base = args.results_base
    exp_specs = args.exp
    if not exp_specs:
        env_val = os.environ.get('EXPERIMENTS', '')
        exp_specs = env_val.split() if env_val.strip() else []
    if not exp_specs:
        parser.error('No experiments specified. Use --exp NAME:TOTAL or EXPERIMENTS env var.')

    experiments = parse_experiments(exp_specs, results_base)

    print(f"[monitor] Control plane: {CONTROL_PLANE}")
    print(f"[monitor] Tracking: { {k: v[0] for k, v in experiments.items()} }")
    print(f"[monitor] Results base: {results_base}")
    print(f"[monitor] Poll interval: {args.interval}s, stale TTL: {args.stale_ttl}s")

    _start_time = time.time()

    while True:
        # Write liveness marker for watchdog stall detection
        with open('/tmp/monitor_last_active', 'w') as f:
            f.write(str(time.time()))

        # 0. Dedup: remove pending tasks that are already completed
        completed_paths = gcs_list(f"{CONTROL_PLANE}/completed")
        completed_ids = {os.path.basename(p).replace('.json', '') for p in completed_paths}
        if completed_ids:
            pending_paths = gcs_list(f"{CONTROL_PLANE}/pending")
            deduped = 0
            for pp in pending_paths:
                tid = os.path.basename(pp).replace('.json', '')
                if tid in completed_ids:
                    gcs_delete(pp)
                    deduped += 1
            if deduped:
                print(f"[monitor] Deduped {deduped} pending tasks (already completed)")

        # 1. Reclaim stale tasks
        reclaimed = reclaim_stale(stale_ttl_s=args.stale_ttl)
        if reclaimed:
            print(f"[monitor] Reclaimed {reclaimed} stale tasks")

        # 2. Pull and validate results for each experiment
        for exp_name, (_, exp_results_base) in experiments.items():
            newly = pull_and_validate(exp_name, exp_results_base)
            if newly:
                print(f"[monitor] Validated {newly} new results for {exp_name}")

        # 3. Print status
        print_status(experiments)

        # 4. Check completion per experiment
        all_done = True
        for exp_name, (target, exp_results_base) in experiments.items():
            validated_dir = os.path.join(exp_results_base, exp_name, 'validated')
            count = len([f for f in os.listdir(validated_dir) if f.endswith('.json')]) if os.path.isdir(validated_dir) else 0

            if count >= target:
                print(f"[monitor] === {exp_name} COMPLETE: {count}/{target} ===")
            else:
                all_done = False

        if all_done:
            _wall_sec = time.time() - _start_time
            _total_tasks = sum(v[0] for v in experiments.values())
            print(f"\n{'='*60}")
            print(f"ALL EXPERIMENTS COMPLETE")
            print(f"Wall time: {_wall_sec/3600:.1f}h | Tasks: {_total_tasks} | Rate: {_total_tasks/(_wall_sec/3600):.1f}/hr")
            print(f"Run: bash ~/distributed_tpu_training/v2/cleanup_gcs.sh --dry-run && bash ~/distributed_tpu_training/v2/cleanup_gcs.sh")
            print(f"{'='*60}\n")
            break

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
