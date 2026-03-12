#!/usr/bin/env python3
"""
preemption_log.py — Track preemption events and VM lifecycle.
Records every VM state change to a JSON log file for analysis.
"""
import json
import os
import subprocess
import time

PROJECT = "gcp-research-credits-489020"
GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
LOG_FILE = os.path.expanduser("~/distributed_tpu_training/pull/preemption_events.jsonl")

ZONES = ["europe-west4-a", "us-east1-d", "us-central2-b", "europe-west4-b", "us-central1-a"]

def get_fleet_snapshot():
    """Get current state of all VMs across all zones."""
    snapshot = {}
    for zone in ZONES:
        try:
            r = subprocess.run(
                [GCLOUD, "alpha", "compute", "tpus", "tpu-vm", "list",
                 f"--zone={zone}", f"--project={PROJECT}", "--format=json"],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0 and r.stdout.strip():
                for vm in json.loads(r.stdout):
                    name = vm.get("name", "")
                    state = vm.get("state", "UNKNOWN")
                    snapshot[name] = {"zone": zone, "state": state}
        except Exception:
            pass
    return snapshot

def log_event(event_type, vm_name, zone, details=""):
    """Append event to JSONL log."""
    event = {
        "timestamp": time.time(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event_type,
        "vm": vm_name,
        "zone": zone,
        "details": details,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[preemption] {event['time_utc']} {event_type}: {vm_name} ({zone}) {details}")

def main():
    print(f"[preemption] Tracking preemption events to {LOG_FILE}")
    prev_snapshot = {}
    
    while True:
        current = get_fleet_snapshot()
        
        if prev_snapshot:
            # Detect changes
            all_vms = set(list(prev_snapshot.keys()) + list(current.keys()))
            for vm in all_vms:
                prev = prev_snapshot.get(vm)
                curr = current.get(vm)
                
                if prev and not curr:
                    log_event("DELETED", vm, prev["zone"])
                elif not prev and curr:
                    log_event("CREATED", vm, curr["zone"], f"state={curr['state']}")
                elif prev and curr and prev["state"] != curr["state"]:
                    log_event("STATE_CHANGE", vm, curr["zone"], 
                             f"{prev['state']} -> {curr['state']}")
                    if curr["state"] == "PREEMPTED":
                        log_event("PREEMPTED", vm, curr["zone"])
        else:
            # First run — log initial state
            for vm, info in current.items():
                log_event("INITIAL", vm, info["zone"], f"state={info['state']}")
        
        prev_snapshot = current
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
