#!/usr/bin/env python3
"""
monitor.py (v3) — Pull-based coordinator monitor with rapid failure detection.

New in v3:
  - check_rapid_failures(): VMs that fail N tasks in T seconds get a redeploy flag
    which vm_manager.py picks up and force-redeploys the VM.

Usage:
    python3 -u ~/distributed_tpu_training/v3/monitor.py --exp exp13_rerun3:120 [--once]
    python3 -u ~/distributed_tpu_training/v3/monitor.py --exp exp13_rerun3:120 --interval 60 --stale-ttl 1800
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import (reclaim_stale, get_queue_counts, gcs_list, gcs_read,
                 gcs_copy, gcs_delete, gcs_move, gcs_write, CONTROL_PLANE)


# ── Rapid failure detection ───────────────────────────────────────────────

def check_rapid_failures(window_secs=600, threshold=5):
    """Flag VMs that fail N tasks in the last T seconds for forced redeploy.

    Writes gs://.../coord_v2/flags/redeploy_{vm_name}.json which vm_manager.py
    picks up and consumes, triggering a force SSH redeploy.
    """
    failed_paths = gcs_list(f"{CONTROL_PLANE}/failed")
    if not failed_paths:
        return

    now = time.time()
    recent_by_vm = defaultdict(list)

    for path in failed_paths:
        raw = gcs_read(path)
        if not raw:
            continue
        try:
            task = json.loads(raw)
        except json.JSONDecodeError:
            continue
        failed_at = task.get('failed_at', 0)
        if failed_at < now - window_secs:
            continue
        # Extract VM name from worker_id (format: v6e-ew4a-1_chip0)
        worker_id = task.get('worker_id', '')
        vm = worker_id.rsplit('_chip', 1)[0] if '_chip' in worker_id else ''
        if vm:
            recent_by_vm[vm].append(task)

    for vm, failures in recent_by_vm.items():
        if len(failures) >= threshold:
            flag_path = f"{CONTROL_PLANE}/flags/redeploy_{vm}.json"
            existing = gcs_read(flag_path)
            if existing:
                continue  # flag already set
            flag = {
                'vm': vm,
                'failure_count': len(failures),
                'window_secs': window_secs,
                'flagged_at': now,
                'task_ids': [t.get('task_id', '') for t in failures[:10]],
            }
            gcs_write(flag_path, json.dumps(flag))
            print(f"[rapid_fail] Flagged {vm} for redeploy: "
                  f"{len(failures)} failures in {window_secs}s", flush=True)


# ── Result validation ─────────────────────────────────────────────────────

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
    """Download completed results, validate, copy to validated dir."""
    completed_paths = gcs_list(f"{CONTROL_PLANE}/completed")
    validated_dir = os.path.join(results_base, exp_name, 'validated')
    tmp_dir = os.path.join(results_base, exp_name, 'tmp_pull')
    os.makedirs(validated_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    already_validated = set()
    for f in os.listdir(validated_dir):
        if f.endswith('.json'):
            already_validated.add(f.replace('.json', ''))

    newly_validated = 0

    for path in completed_paths:
        task_id = os.path.basename(path).replace('.json', '')
        parts = task_id.split('__', 1)
        if len(parts) != 2:
            continue
        task_exp, label = parts
        if task_exp != exp_name:
            continue
        if label in already_validated:
            continue

        raw = gcs_read(path)
        if not raw:
            continue
        try:
            completion = json.loads(raw)
        except json.JSONDecodeError:
            continue

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
            result = completion.get('result', {})
            if result and result.get('best_val_loss') is not None:
                with open(local_path, 'w') as f:
                    json.dump({'summary': result}, f)
                found = True

        if not found:
            print(f"  [validate] {label}: no result file found, skipping", flush=True)
            continue

        valid, reason = validate_result(local_path)
        if valid:
            dst = os.path.join(validated_dir, f'{label}.json')
            shutil.move(local_path, dst)
            # Download per-step JSONL
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
            print(f"  [validate] {label}: OK", flush=True)
        else:
            os.remove(local_path)
            current_retries = completion.get('retries', 0)
            MAX_INVALIDATION_RETRIES = 5
            if current_retries < MAX_INVALIDATION_RETRIES:
                requeue = {k: v for k, v in completion.items()
                           if k not in ('worker_id', 'claimed_at', 'completed_at', 'result', 'failed_at')}
                requeue['retries'] = current_retries + 1
                requeue['requeued_reason'] = f'invalid:{reason}'
                if gcs_write(f"{CONTROL_PLANE}/pending/{task_id}.json", json.dumps(requeue)):
                    gcs_delete(path)
                    print(f"  [validate] {label}: INVALID ({reason}) — requeued "
                          f"(retry {requeue['retries']}/{MAX_INVALIDATION_RETRIES})", flush=True)
                else:
                    print(f"  [validate] {label}: INVALID — requeue failed", flush=True)
            else:
                print(f"  [validate] {label}: INVALID ({reason}) — terminal after "
                      f"{current_retries} retries, moving to invalidated/", flush=True)
                invalidated_path = f"{CONTROL_PLANE}/invalidated/{task_id}.json"
                if not gcs_move(path, invalidated_path):
                    gcs_delete(path)

    return newly_validated


# ── Status display ────────────────────────────────────────────────────────

def print_status(experiments):
    pending, running, completed, failed = get_queue_counts()
    invalidated = len(gcs_list(f"{CONTROL_PLANE}/invalidated"))
    total = pending + running + completed + failed + invalidated
    flags = len(gcs_list(f"{CONTROL_PLANE}/flags"))
    print(f"\n[monitor] {time.strftime('%H:%M:%S')} | "
          f"pending={pending} running={running} completed={completed} "
          f"failed={failed} invalidated={invalidated} flags={flags} total={total}", flush=True)

    for exp_name, (target, results_base) in experiments.items():
        validated_dir = os.path.join(results_base, exp_name, 'validated')
        validated = len([f for f in os.listdir(validated_dir) if f.endswith('.json')]) \
            if os.path.isdir(validated_dir) else 0
        pct = validated / target * 100 if target > 0 else 0
        print(f"  {exp_name}: {validated}/{target} validated ({pct:.0f}%)", flush=True)

    heartbeat_paths = gcs_list(f"{CONTROL_PLANE}/heartbeats")
    active = training = compiling = idle = 0
    for path in heartbeat_paths:
        raw = gcs_read(path)
        if raw:
            try:
                hb = json.loads(raw)
                age = time.time() - hb.get('timestamp', 0)
                if age < 900:
                    active += 1
                    s = hb.get('status', '')
                    if s == 'training':
                        training += 1
                    elif s == 'xla_compile':
                        compiling += 1
                    elif s == 'idle':
                        idle += 1
            except json.JSONDecodeError:
                pass
    print(f"  Active workers: {active} ({training} training, {compiling} compiling, {idle} idle)",
          flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────

def parse_experiments(exp_specs, results_base):
    experiments = {}
    for spec in exp_specs:
        if ':' in spec:
            name, total = spec.rsplit(':', 1)
            experiments[name] = (int(total), results_base)
        else:
            raise ValueError(f"Invalid --exp spec '{spec}': expected NAME:TOTAL (e.g. exp13_rerun3:120)")
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Pull-based sweep monitor (v3)')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--stale-ttl', type=int, default=1800)
    parser.add_argument('--exp', nargs='+', metavar='NAME:TOTAL')
    parser.add_argument('--results-base', default=os.path.expanduser('~/sf_bema/results'))
    # Rapid failure detection params
    parser.add_argument('--rapid-fail-window', type=int, default=600,
                        help='Window (secs) for rapid failure detection (default: 600)')
    parser.add_argument('--rapid-fail-threshold', type=int, default=5,
                        help='Failures in window to trigger redeploy flag (default: 5)')
    args = parser.parse_args()

    results_base = args.results_base
    exp_specs = args.exp
    if not exp_specs:
        env_val = os.environ.get('EXPERIMENTS', '')
        exp_specs = env_val.split() if env_val.strip() else []
    if not exp_specs:
        parser.error('No experiments specified. Use --exp NAME:TOTAL')

    experiments = parse_experiments(exp_specs, results_base)

    print(f"[monitor] Control plane: {CONTROL_PLANE}", flush=True)
    print(f"[monitor] Tracking: { {k: v[0] for k, v in experiments.items()} }", flush=True)
    print(f"[monitor] Poll interval: {args.interval}s, stale TTL: {args.stale_ttl}s", flush=True)
    print(f"[monitor] Rapid fail: threshold={args.rapid_fail_threshold} "
          f"window={args.rapid_fail_window}s", flush=True)

    _start_time = time.time()

    while True:
        with open('/tmp/monitor_last_active', 'w') as f:
            f.write(str(time.time()))

        # 0. Dedup: remove pending tasks already completed
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
                print(f"[monitor] Deduped {deduped} pending tasks (already completed)", flush=True)

        # 1. Reclaim stale tasks
        reclaimed = reclaim_stale(stale_ttl_s=args.stale_ttl)
        if reclaimed:
            print(f"[monitor] Reclaimed {reclaimed} stale tasks", flush=True)

        # 2. Rapid failure detection (new in v3)
        check_rapid_failures(
            window_secs=args.rapid_fail_window,
            threshold=args.rapid_fail_threshold,
        )

        # 3. Pull and validate results
        for exp_name, (_, exp_results_base) in experiments.items():
            newly = pull_and_validate(exp_name, exp_results_base)
            if newly:
                print(f"[monitor] Validated {newly} new results for {exp_name}", flush=True)

        # 4. Print status
        print_status(experiments)

        # 5. Check completion
        all_done = True
        for exp_name, (target, exp_results_base) in experiments.items():
            validated_dir = os.path.join(exp_results_base, exp_name, 'validated')
            count = len([f for f in os.listdir(validated_dir) if f.endswith('.json')]) \
                if os.path.isdir(validated_dir) else 0
            if count >= target:
                print(f"[monitor] === {exp_name} COMPLETE: {count}/{target} ===", flush=True)
            else:
                all_done = False

        if all_done:
            _wall_sec = time.time() - _start_time
            _total = sum(v[0] for v in experiments.values())
            print(f"\n{'='*60}", flush=True)
            print(f"ALL EXPERIMENTS COMPLETE", flush=True)
            print(f"Wall time: {_wall_sec/3600:.1f}h | Tasks: {_total} | "
                  f"Rate: {_total/(_wall_sec/3600):.1f}/hr", flush=True)
            print(f"{'='*60}\n", flush=True)
            break

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
