#!/usr/bin/env python3
"""Quick progress check — reads all heartbeats in parallel."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

CTRL = os.environ.get('CONTROL_PLANE', 'gs://gcp-researchcredits-blocklab-europe-west4/coord_v2')

def gcs_read(path):
    try:
        r = subprocess.run(['gcloud', 'storage', 'cat', path],
                          capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except:
        return None

def gcs_list(prefix):
    try:
        r = subprocess.run(['gcloud', 'storage', 'ls', f'{prefix}/'],
                          capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return [l.strip().rstrip('/') for l in r.stdout.strip().split('\n') if l.strip()]
        return []
    except:
        return []

def main():
    now = time.time()
    print(f"=== Progress @ {time.strftime('%H:%M:%S', time.gmtime()) + ' UTC'} ===\n")

    # Queue counts (parallel)
    states = ['pending', 'running', 'completed', 'failed']
    counts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(gcs_list, f"{CTRL}/{s}"): s for s in states}
        for f in as_completed(futs):
            counts[futs[f]] = len(f.result())
    total = sum(counts.values())
    print(f"Queue: pending={counts['pending']} running={counts['running']} "
          f"completed={counts['completed']} failed={counts['failed']} total={total}")

    # Read all heartbeats in parallel
    hb_files = gcs_list(f"{CTRL}/heartbeats")
    heartbeats = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(gcs_read, f): f for f in hb_files}
        for f in as_completed(futs):
            path = futs[f]
            raw = f.result()
            if raw:
                try:
                    heartbeats[path] = json.loads(raw)
                except:
                    pass

    # Group by VM
    vms = {}
    for path, hb in heartbeats.items():
        wid = hb.get('worker_id', '?')
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        vms.setdefault(vm, []).append(hb)

    print(f"\nHeartbeats: {len(heartbeats)} chips on {len(vms)} VMs\n")

    for vm in sorted(vms.keys()):
        chips = vms[vm]
        steps = [c.get('step', 0) for c in chips]
        statuses = set(c.get('status', '?') for c in chips)
        ages = [now - c.get('timestamp', 0) for c in chips]
        labels = set(c.get('label', '?') for c in chips)
        task_exps = set((c.get('task_id') or '').split('__')[0] for c in chips)
        print(f"  {vm}: {len(chips)} chips, steps={min(steps)}-{max(steps)}, "
              f"status={statuses}, age={min(ages):.0f}-{max(ages):.0f}s, exps={task_exps}")

    all_steps = [hb.get('step', 0) for hb in heartbeats.values()]
    training = sum(1 for s in all_steps if s > 0)
    compiling = sum(1 for s in all_steps if s == 0)
    max_step = max(all_steps) if all_steps else 0

    print(f"\nSummary: max_step={max_step}, training={training}, compiling={compiling}")

    # Validated counts
    for exp in ['exp13', 'exp12_1']:
        vdir = os.path.expanduser(f'~/sf_bema/results/{exp}/validated')
        count = len([f for f in os.listdir(vdir) if f.endswith('.json')]) if os.path.isdir(vdir) else 0
        print(f"  {exp} validated: {count}")

    if max_step >= 100:
        print("\nSUCCESS: Training past step 100!")
    if max_step >= 400:
        print("CRITICAL SUCCESS: Past crash point (step 200-400)!")

if __name__ == '__main__':
    main()
