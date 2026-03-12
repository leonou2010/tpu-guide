#!/usr/bin/env python3
"""
vm_status.py — Rich per-VM status dashboard.

Shows detailed per-VM health: boot phase, step progress, task, errors, chip type.
Reads from GCS: heartbeats/, telemetry/, logs/env_fail_*.log

Usage:
    python3 ~/distributed_tpu_training/pull/vm_status.py             # one-shot
    python3 ~/distributed_tpu_training/pull/vm_status.py --watch 30  # refresh every 30s
    python3 ~/distributed_tpu_training/pull/vm_status.py --zone europe-west4-a  # filter zone
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

# ── ANSI colors ───────────────────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_GREEN  = "\033[32m"
C_YELLOW = "\033[33m"
C_RED    = "\033[31m"
C_CYAN   = "\033[36m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"

CTRL = "gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"


def gcs_ls(prefix):
    """List GCS objects under prefix."""
    r = subprocess.run(["gsutil", "ls", prefix], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]


def gcs_cat(path):
    """Read GCS object as text."""
    r = subprocess.run(["gsutil", "cat", path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def gcs_ls_parallel(paths):
    """Read multiple GCS objects in parallel. Returns dict path->content."""
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(gcs_cat, p): p for p in paths}
        for f in concurrent.futures.as_completed(futures):
            p = futures[f]
            results[p] = f.result()
    return results


def get_queue_stats():
    """Count tasks in each state."""
    stats = {}
    for state in ["pending", "running", "completed", "failed"]:
        r = subprocess.run(["gsutil", "ls", f"{CTRL}/{state}/"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            lines = [l for l in r.stdout.strip().split("\n") if l.strip() and l.endswith(".json")]
            stats[state] = len(lines)
        else:
            stats[state] = 0
    return stats


def get_heartbeats():
    """Read all heartbeats. Returns dict worker_id -> hb_dict."""
    paths = gcs_ls(f"{CTRL}/heartbeats/")
    if not paths:
        return {}
    contents = gcs_ls_parallel(paths)
    hbs = {}
    now = time.time()
    for path, raw in contents.items():
        if not raw:
            continue
        try:
            hb = json.loads(raw)
            wid = hb.get("worker_id", path.split("/")[-1].replace(".json", ""))
            hb["_age"] = now - hb.get("timestamp", now)
            hbs[wid] = hb
        except Exception:
            pass
    return hbs


def get_telemetry():
    """Read boot phase telemetry from GCS. Returns dict tpu_name -> phase_dict."""
    paths = gcs_ls(f"{CTRL}/telemetry/")
    if not paths:
        return {}
    contents = gcs_ls_parallel(paths)
    telem = {}
    for path, raw in contents.items():
        if not raw:
            continue
        try:
            d = json.loads(raw)
            name = d.get("tpu_name", path.split("/")[-1].replace("_boot.json", ""))
            telem[name] = d
        except Exception:
            pass
    return telem


def get_env_fails():
    """Read recent env_fail logs. Returns dict tpu_name -> last_fail_msg."""
    paths = gcs_ls(f"{CTRL}/logs/")
    env_fails = {}
    for p in paths:
        if "env_fail_" not in p:
            continue
        name = p.split("env_fail_")[1].replace(".log", "")
        content = gcs_cat(p)
        if content:
            # Most recent line
            lines = [l for l in content.strip().split("\n") if l.strip()]
            env_fails[name] = lines[-1] if lines else ""
    return env_fails


def infer_vm_type(worker_id, hb):
    """Infer VM type from chips or zone in heartbeat."""
    zone = hb.get("zone", "")
    tpu_name = hb.get("tpu_name", worker_id)
    chips = hb.get("chips", 0)
    # Try from tpu_name
    if "ew4a" in tpu_name or "ue1d" in tpu_name:
        return "v6e-8"
    if "ew4b" in tpu_name or "uc1a" in tpu_name:
        return "v5e-4"
    if "uc2b" in tpu_name:
        return "v4-8"
    # From zone
    if "europe-west4-a" in zone or "us-east1" in zone:
        return "v6e-8"
    if "europe-west4-b" in zone or "us-central1" in zone:
        return "v5e-4"
    if "us-central2" in zone:
        return "v4-8"
    # From chip count
    if chips == 8:
        return "v6e-8"
    if chips == 4:
        return "v4/v5e-4"
    return "unknown"


def status_color(status, age, step):
    """Return colored status string."""
    if status in ("training",) and age < 600:
        return f"{C_GREEN}{status}{C_RESET}"
    if status in ("xla_compile",) and age < 1200:
        return f"{C_YELLOW}{status}{C_RESET}"
    if status in ("idle",) and age < 600:
        return f"{C_CYAN}{status}{C_RESET}"
    if age > 1800:
        return f"{C_RED}STALE({int(age//60)}min){C_RESET}"
    if age > 600:
        return f"{C_YELLOW}{status}({int(age//60)}min){C_RESET}"
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0=one-shot)")
    parser.add_argument("--zone", default="", help="Filter by zone")
    args = parser.parse_args()

    while True:
        print_status(args.zone)
        if args.watch == 0:
            break
        print(f"\n{C_DIM}Refreshing in {args.watch}s... (Ctrl+C to exit){C_RESET}")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break


def print_status(zone_filter=""):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n{C_BOLD}{'='*78}{C_RESET}")
    print(f"{C_BOLD}  TPU VM Status — {now}{C_RESET}")
    print(f"{C_BOLD}{'='*78}{C_RESET}")

    # Gather data in parallel
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        f_hbs    = ex.submit(get_heartbeats)
        f_telem  = ex.submit(get_telemetry)
        f_efails = ex.submit(get_env_fails)
        f_queue  = ex.submit(get_queue_stats)
    hbs    = f_hbs.result()
    telem  = f_telem.result()
    efails = f_efails.result()
    queue  = f_queue.result()

    # Queue summary
    print(f"\n{C_BOLD}Queue:{C_RESET} "
          f"pending={C_YELLOW}{queue.get('pending',0)}{C_RESET}  "
          f"running={C_GREEN}{queue.get('running',0)}{C_RESET}  "
          f"completed={C_GREEN}{queue.get('completed',0)}{C_RESET}  "
          f"failed={C_RED}{queue.get('failed',0)}{C_RESET}")

    # Group heartbeats by VM (tpu_name or worker_id prefix)
    vm_workers = {}
    for wid, hb in hbs.items():
        tpu = hb.get("tpu_name", "") or wid.rsplit("_c", 1)[0]
        if tpu not in vm_workers:
            vm_workers[tpu] = []
        vm_workers[tpu].append(hb)

    if not vm_workers:
        print(f"\n{C_RED}No heartbeats found.{C_RESET}")
        return

    # Group by zone
    zones = {}
    for tpu, workers in vm_workers.items():
        zone = workers[0].get("zone", "unknown") if workers else "unknown"
        if zone_filter and zone_filter not in zone:
            continue
        if zone not in zones:
            zones[zone] = []
        zones[zone].append((tpu, workers))

    total_chips = 0
    total_training = 0

    for zone, vms in sorted(zones.items()):
        print(f"\n{C_BOLD}  Zone: {zone}{C_RESET}")
        print(f"  {'VM':<22} {'Type':<10} {'Chips':<6} {'Steps':<14} {'Status':<20} {'Task':<35} {'Age'}")
        print(f"  {'-'*22} {'-'*10} {'-'*6} {'-'*14} {'-'*20} {'-'*35} {'-'*5}")

        for tpu, workers in sorted(vms):
            chips = sum(1 for _ in workers)
            total_chips += chips

            steps = [w.get("step", 0) for w in workers]
            statuses = list({w.get("status", "?") for w in workers})
            ages = [w.get("_age", 0) for w in workers]
            labels = list({w.get("label", "") for w in workers if w.get("label")})
            age_max = max(ages) if ages else 0

            step_range = f"{min(steps)}-{max(steps)}" if len(set(steps)) > 1 else str(steps[0])
            status_str = "/".join(statuses)
            label_str = labels[0][:34] if labels else "---"

            # Color
            is_training = any("training" in s for s in statuses)
            is_compiling = any("compile" in s for s in statuses)
            is_stale = age_max > 1800

            if is_training and age_max < 600:
                vm_color = C_GREEN
                total_training += chips
            elif is_compiling and age_max < 1200:
                vm_color = C_YELLOW
            elif is_stale:
                vm_color = C_RED
            else:
                vm_color = C_DIM

            vm_type = infer_vm_type(tpu, workers[0])
            age_str = f"{int(age_max//60)}m{int(age_max%60)}s"

            # Check for env_fail
            env_note = ""
            tpu_key = tpu if tpu in efails else None
            if not tpu_key:
                # Try matching internal name
                for k in efails:
                    if k in tpu or tpu in k:
                        tpu_key = k
                        break
            if tpu_key:
                fail_time = efails[tpu_key][:8] if efails[tpu_key] else ""
                env_note = f" {C_RED}[env_fail {fail_time}]{C_RESET}"

            # Boot phase from telemetry
            phase_note = ""
            if tpu in telem:
                phase = telem[tpu].get("phase", "")
                phase_ts = telem[tpu].get("timestamp", 0)
                phase_age = int(time.time() - phase_ts)
                if phase not in ("TRAINING", "IDLE_AWAITING_WORK", ""):
                    phase_note = f" {C_CYAN}[{phase} {phase_age//60}m ago]{C_RESET}"

            print(f"  {vm_color}{tpu:<22}{C_RESET} "
                  f"{vm_type:<10} {chips:<6} {step_range:<14} "
                  f"{status_str:<20} {label_str:<35} {age_str}"
                  f"{env_note}{phase_note}")

    print(f"\n{C_BOLD}Total: {total_chips} chips ({total_training} training){C_RESET}")

    # Warn about env_fail VMs with no recent heartbeat
    active_tpus = set()
    for tpu, workers in vm_workers.items():
        active_tpus.add(tpu)
        for w in workers:
            active_tpus.add(w.get("tpu_name", ""))

    broken_vms = [k for k in efails if k not in active_tpus and k != "unknown"]
    if broken_vms:
        print(f"\n{C_RED}{C_BOLD}VMs with env_fail and no heartbeat:{C_RESET}")
        for v in sorted(broken_vms):
            msg = efails[v][:60] if efails[v] else ""
            print(f"  {C_RED}{v}: {msg}{C_RESET}")


if __name__ == "__main__":
    main()
