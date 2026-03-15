#!/usr/bin/env python3
"""
vm_manager.py — Thread-per-VM TPU manager using QueuedResource API.

Replaces vm_requester.sh. Key improvements:
- Thread-per-VM: no global blocking, one hung SSH can't stall other zones
- QueuedResource API: GCP handles capacity queuing, no "internal error" loops
- FAILED/SUSPENDED → delete+recreate immediately (not after 45 min)
- 5x SSH retry per deploy attempt
- Reads rapid-failure flags from monitor (force-redeploys broken VMs)

Usage:
    nohup python3 -u ~/distributed_tpu_training/v3/vm_manager.py >> /tmp/vm_manager.log 2>&1 &

Pre-launch:
    Upload v3 deploy_babysitter.sh to all 3 GCS buckets:
        gsutil cp ~/distributed_tpu_training/v3/deploy_babysitter.sh \\
            gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/deploy_babysitter.sh
        gsutil cp ~/distributed_tpu_training/v3/babysitter.py \\
            gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/babysitter.py
        (repeat for us-east1 and us-central2 buckets)
    Then upload gcs.py to all 3 as well.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

# Uses gcloud CLI for QR operations (no ADC/google-cloud-tpu required)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import gcs_read, gcs_write, gcs_delete, gcs_list, CONTROL_PLANE, get_queue_counts

# ── Config ─────────────────────────────────────────────────────────────────

PROJECT = os.environ.get('PROJECT', 'gcp-research-credits-489020')
GCLOUD = os.path.expanduser('~/google-cloud-sdk/bin/gcloud')

# Seconds of healthy heartbeat absence before declaring babysitter dead
HEALTHY_TTL = 2700   # 45 min (covers XLA compile time)
# How often each VM thread polls (seconds)
POLL_INTERVAL = 60
# Seconds between SSH deploy attempts (exponential backoff base)
DEPLOY_BACKOFF_BASE = 15
# XLA compile stuck threshold — if status=xla_compile for >120 min, force redeploy
# xla_compile status = step==0, covers startup + XLA compile + first step.
# On ue1d/uc2b without matched XLA cache this takes 45-60 min. Use 120 min to avoid
# killing legitimately-compiling babysitters (which burns task retries).
XLA_STUCK_S = 7200
# Grace period after a successful deploy — don't re-check health for this long.
# Babysitter needs time to install deps + download XLA cache + write first heartbeat.
DEPLOY_GRACE_S = 600  # 10 min
# Scale-down: delete idle VMs when queue is empty and chips are idle for this long.
IDLE_SCALE_DOWN_S = 600  # 10 min of idle before scale-down (when no pending work)

# Fleet definition: (name, zone, accel_type, bucket)
# Edit this list to change the fleet composition.
FLEET = [
    # europe-west4-a — v6e-8, has internet, 8 VMs (64 chip quota)
    ('v6e-ew4a-1', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-2', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-3', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-4', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-5', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-6', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-7', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    ('v6e-ew4a-8', 'europe-west4-a', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-europe-west4'),
    # us-east1-d — v6e-8, no internet, 8 VMs (64 chip quota)
    ('v6e-ue1d-1', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-2', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-3', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-4', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-5', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-6', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-7', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    ('v6e-ue1d-8', 'us-east1-d', 'v6e-8',
     'gs://gcp-researchcredits-blocklab-us-east1'),
    # us-central2-b — DISABLED: zone down (GCP internal error since 2026-03-15)
    # v5e — DISABLED: serving quota limit 4 (needs GCP support ticket)
]


def get_runtime(accel_type):
    """Map accelerator type to GCP runtime version."""
    if 'v6e' in accel_type:
        return 'v2-alpha-tpuv6e'
    elif 'v4' in accel_type:
        return 'tpu-ubuntu2204-base'
    elif 'v5litepod' in accel_type or 'v5e' in accel_type:
        return 'v2-alpha-tpuv5-lite'
    return 'tpu-ubuntu2204-base'


# ── VMSlot ─────────────────────────────────────────────────────────────────

class VMSlot:
    """Manages one TPU VM via QueuedResource API.

    Lifecycle:
        WAITING_FOR_RESOURCES → ACTIVE → (babysitter deployed) → training
        FAILED/SUSPENDED → delete → WAITING_FOR_RESOURCES

    Thread-safe: each VMSlot runs in its own daemon thread.
    """

    def __init__(self, name, zone, accel_type, bucket):
        self.name = name
        self.zone = zone
        self.accel_type = accel_type
        self.bucket = bucket
        self._deploy_lock = threading.Lock()
        # Track status duration for stuck-babysitter detection
        self._last_hb_status = None
        self._last_hb_status_since = 0
        # Grace period: skip health check for DEPLOY_GRACE_S after a successful deploy.
        # Initialize from GCS telemetry so restart doesn't re-deploy recently-deployed VMs.
        self._last_deploy_time = self._init_deploy_time_from_telemetry()
        # Scale-down: set to True to make run() delete this VM and exit
        self._shutdown = False
        self._idle_since = None

    def _read_failed_telemetry_phase(self):
        """Returns phase string if GCS telemetry shows a FAILED_* phase, else None."""
        try:
            ctrl = f"gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"
            raw = gcs_read(f"{ctrl}/telemetry/{self.name}_boot.json")
            if raw:
                data = json.loads(raw)
                phase = data.get('phase', '')
                if phase.startswith('FAILED_'):
                    return phase
        except Exception:
            pass
        return None

    def _init_deploy_time_from_telemetry(self):
        """On startup, read GCS telemetry to restore grace period for recently-deployed VMs."""
        try:
            ctrl = f"gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"
            raw = gcs_read(f"{ctrl}/telemetry/{self.name}_boot.json")
            if raw:
                data = json.loads(raw)
                ts = data.get('timestamp', 0)
                phase = data.get('phase', '')
                age = time.time() - ts
                if phase in ('IDLE_AWAITING_WORK', 'LAUNCHING_BABYSITTER') and age < DEPLOY_GRACE_S:
                    # Recent deploy — pretend we just deployed so grace period applies
                    return time.time() - age
        except Exception:
            pass
        return 0

    def _log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}][{self.name}] {msg}", flush=True)

    # ── VM management (direct tpu-vm create/delete, v2-style confirmed working) ──
    # QueuedResource API requires ADC and --spot is rejected ("STANDARD provisioning
    # model incompatible"). Using direct tpu-vm create --spot --internal-ips instead.

    def _gcloud_tpu(self, *args, timeout=120):
        """Run a gcloud tpu-vm command."""
        cmd = [GCLOUD, 'alpha', 'compute', 'tpus', 'tpu-vm'] + list(args) + [
            f'--zone={self.zone}', f'--project={PROJECT}'
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _get_vm_state(self):
        """Returns VM state string or None if not found."""
        try:
            r = self._gcloud_tpu('describe', self.name, '--format=json(state)')
            if r.returncode != 0:
                if 'not found' in r.stderr.lower() or 'resource not found' in r.stderr.lower():
                    return None
                return 'UNKNOWN'
            data = json.loads(r.stdout)
            return data.get('state', 'UNKNOWN')
        except Exception as e:
            self._log(f"VM describe error: {e}")
            return 'UNKNOWN'

    def _ensure_queued_resource(self):
        """Check VM state; create if missing, delete+recreate if broken.
        Returns pseudo-state for run() loop compatibility.
        """
        try:
            state = self._get_vm_state()
            if state is None:
                self._log("VM not found — creating")
                self._create_vm()
                return 'WAITING_FOR_RESOURCES'
            if state in ('PREEMPTED', 'TERMINATED', 'STOPPING'):
                self._log(f"VM state={state} — deleting and recreating")
                self._delete_vm()
                if self._wait_for_absence():
                    self._create_vm()
                    return 'WAITING_FOR_RESOURCES'
                else:
                    self._log("VM deletion timed out — retrying next cycle")
                    return 'DELETING'
            elif state == 'DELETING':
                return 'DELETING'
            elif state == 'READY':
                return 'ACTIVE'
            else:  # CREATING, RESTARTING, etc.
                return 'WAITING_FOR_RESOURCES'
        except Exception as e:
            self._log(f"VM check error: {e}")
            return 'UNKNOWN'

    def _create_vm(self):
        """Create TPU VM via direct tpu-vm create (async — returns immediately)."""
        runtime = get_runtime(self.accel_type)
        r = self._gcloud_tpu(
            'create', self.name,
            f'--accelerator-type={self.accel_type}',
            f'--version={runtime}',
            '--spot',
            '--internal-ips',
            '--async',
            timeout=60,
        )
        if r.returncode == 0:
            self._log(f"VM created (runtime={runtime})")
        elif 'already exists' in r.stderr.lower():
            self._log("VM already exists")
        elif 'capacity' in r.stderr.lower():
            self._log("No capacity — will retry next cycle")
        else:
            self._log(f"VM creation failed: {r.stderr[-200:]}")

    def _delete_vm(self):
        """Delete TPU VM."""
        r = self._gcloud_tpu('delete', self.name, '--quiet', timeout=120)
        if r.returncode == 0:
            self._log("VM deletion initiated")
        elif 'not found' in r.stderr.lower():
            self._log("VM already gone")
        else:
            self._log(f"VM deletion error: {r.stderr[-100:]}")

    def _wait_for_absence(self, timeout_s=300, poll_s=15):
        """Wait until VM is deleted. Returns True if gone, False on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._get_vm_state() is None:
                return True
            time.sleep(poll_s)
        self._log(f"Warning: VM still present after {timeout_s}s delete wait")
        return False

    def _startup_script(self):
        """Bash startup-script embedded in VM metadata. Runs at first boot as root."""
        # Select per-type deploy script based on VM name
        if 'ew4a' in self.name:
            _script = 'deploy_ew4a.sh'
        elif 'ue1d' in self.name:
            _script = 'deploy_ue1d.sh'
        elif 'uc2b' in self.name:
            _script = 'deploy_uc2b.sh'
        elif 'v5e' in self.name or 'ew4b' in self.name or 'uc1a' in self.name:
            _script = 'deploy_v5e.sh'
        else:
            _script = 'deploy_babysitter.sh'
        # gsutil works on all zones (v4 old gcloud has gsutil, v6e also has it)
        return f"""#!/bin/bash
export HOME=/root
export PATH=$HOME/miniconda3/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
# Download and run per-type v3 deploy script from GCS
gsutil cp '{self.bucket}/pull_code_v3/{_script}' /tmp/deploy_startup.sh 2>/dev/null || \\
    gcloud storage cp '{self.bucket}/pull_code_v3/{_script}' /tmp/deploy_startup.sh 2>/dev/null || \\
    echo "STARTUP: failed to download {_script} from {self.bucket}"
if [ -f /tmp/deploy_startup.sh ]; then
    TPU_NAME={self.name} ZONE={self.zone} WANDB_MODE=disabled FORCE_REDEPLOY=1 bash /tmp/deploy_startup.sh
else
    echo "STARTUP: {_script} not downloaded — VM will be redeployed by vm_manager"
fi
"""

    # ── Heartbeat health check ────────────────────────────────────────────

    def _read_best_heartbeat(self):
        """Read freshest chip heartbeat. Returns (age_s, status) or (9999, 'unknown')."""
        now = time.time()
        best_age = 9999
        best_status = 'unknown'
        for chip_idx in range(8):
            path = f"{CONTROL_PLANE}/heartbeats/{self.name}_chip{chip_idx}.json"
            raw = gcs_read(path)
            if not raw:
                continue
            try:
                hb = json.loads(raw)
                age = now - hb.get('timestamp', 0)
                if 0 <= age < best_age:
                    best_age = age
                    best_status = hb.get('status', 'unknown')
            except Exception:
                pass
        return best_age, best_status

    def _babysitter_healthy(self):
        """Return True if babysitter has fresh, non-stuck heartbeat."""
        age, status = self._read_best_heartbeat()
        if age >= HEALTHY_TTL:
            return False  # stale
        # Track status duration for xla_compile stuck detection
        now = time.time()
        if status != self._last_hb_status:
            self._last_hb_status = status
            self._last_hb_status_since = now
        status_duration = now - self._last_hb_status_since
        if status == 'xla_compile' and status_duration > XLA_STUCK_S:
            self._log(f"Babysitter stuck in xla_compile for {status_duration:.0f}s (>{XLA_STUCK_S}s)")
            return False  # stuck
        return True

    def _all_chips_idle(self):
        """Return True if all chip heartbeats exist and all report status=idle."""
        now = time.time()
        found = 0
        for chip_idx in range(8):
            path = f"{CONTROL_PLANE}/heartbeats/{self.name}_chip{chip_idx}.json"
            raw = gcs_read(path)
            if not raw:
                continue
            try:
                hb = json.loads(raw)
                age = now - hb.get('timestamp', 0)
                status = hb.get('status', 'unknown')
                if age > 300:
                    continue  # stale heartbeat — chip may be dead, ignore
                if status != 'idle':
                    return False  # chip is training/compiling
                found += 1
            except Exception:
                pass
        return found > 0  # True only if we saw fresh heartbeats and all were idle

    def mark_shutdown(self):
        """Signal this slot to delete its VM and stop on next loop iteration."""
        self._shutdown = True

    # ── Rapid-failure flag ────────────────────────────────────────────────

    def _check_redeploy_flag(self):
        """Check and consume monitor's rapid-failure redeploy flag."""
        flag_path = f"{CONTROL_PLANE}/flags/redeploy_{self.name}.json"
        raw = gcs_read(flag_path)
        if raw:
            try:
                flag = json.loads(raw)
                self._log(f"Redeploy flag consumed: {flag}")
            except Exception:
                pass
            gcs_delete(flag_path)
            return True
        return False

    # ── SSH deploy ────────────────────────────────────────────────────────

    def _deploy_babysitter(self, force=True):
        """Deploy babysitter via SSH with 5x retry + exponential backoff."""
        with self._deploy_lock:
            force_env = 'FORCE_REDEPLOY=1 ' if force else ''
            # Pull latest v3 code from GCS, then launch deploy_babysitter.sh in background.
            # Background launch (nohup ... &) means SSH exits immediately — avoids 240s timeout
            # on slow VMs (v4 needs 15-20 min for wheel download + install + XLA cache).
            # vm_manager tracks success via GCS telemetry + heartbeats (grace period).
            # NOTE: env vars must come BEFORE nohup, not between nohup and bash.
            # `nohup VAR=val bash` tries to run VAR=val as command → fails with "No such file".
            # Correct: `env VAR=val nohup bash ...` or `VAR=val nohup bash ...`
            # Select per-type deploy script based on VM name
            if 'ew4a' in self.name:
                _deploy_script = 'deploy_ew4a.sh'
            elif 'ue1d' in self.name:
                _deploy_script = 'deploy_ue1d.sh'
            elif 'uc2b' in self.name:
                _deploy_script = 'deploy_uc2b.sh'
            elif 'v5e' in self.name or 'ew4b' in self.name or 'uc1a' in self.name:
                _deploy_script = 'deploy_v5e.sh'
            else:
                _deploy_script = 'deploy_babysitter.sh'
            cmd_inner = (
                f'mkdir -p ~/pull_code && '
                f'(gcloud storage cp \'{self.bucket}/pull_code_v3/*\' ~/pull_code/ 2>/dev/null || '
                f'gsutil -m cp \'{self.bucket}/pull_code_v3/*\' ~/pull_code/ 2>/dev/null) && '
                f'chmod +x ~/pull_code/{_deploy_script} && '
                f'{force_env}TPU_NAME={self.name} ZONE={self.zone} WANDB_MODE=disabled '
                f'nohup bash ~/pull_code/{_deploy_script} > /tmp/deploy_babysitter.log 2>&1 &'
            )
            ssh_cmd = [
                GCLOUD, 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh', self.name,
                f'--zone={self.zone}',
                f'--project={PROJECT}',
                '--tunnel-through-iap',
                f'--command={cmd_inner}',
            ]
            for attempt in range(1, 6):
                self._log(f"SSH deploy attempt {attempt}/5...")
                try:
                    result = subprocess.run(
                        ssh_cmd, capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        self._log(f"Deploy launched in background on attempt {attempt}")
                        self._last_deploy_time = time.time()
                        return True
                    # Log last 3 lines of stderr for diagnosis
                    err_tail = '\n'.join(result.stderr.strip().splitlines()[-3:])
                    self._log(f"Deploy attempt {attempt} rc={result.returncode}: {err_tail}")
                except subprocess.TimeoutExpired:
                    self._log(f"Deploy attempt {attempt} timed out (60s)")
                except Exception as e:
                    self._log(f"Deploy attempt {attempt} error: {e}")
                if attempt < 5:
                    backoff = DEPLOY_BACKOFF_BASE * attempt
                    self._log(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
            self._log("All 5 SSH deploy attempts failed")
            return False

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self, stop_event):
        """VM management loop. Runs in its own thread."""
        self._log("VM slot manager started")
        while not stop_event.is_set():
            if self._shutdown:
                self._log("Scale-down shutdown — deleting VM to free HEALTH_CHECKS quota")
                self._delete_vm()
                return
            try:
                state = self._ensure_queued_resource()
                if state == 'ACTIVE':
                    force_redeploy = self._check_redeploy_flag()
                    grace_remaining = DEPLOY_GRACE_S - (time.time() - self._last_deploy_time)
                    in_grace = grace_remaining > 0
                    healthy = self._babysitter_healthy()
                    if force_redeploy:
                        self._log("Redeploy flag set — force redeploying")
                        self._deploy_babysitter(force=True)
                    elif in_grace:
                        # During grace period, still check for FAILED_ENV phases — these need
                        # immediate redeploy (not waiting for grace to expire) because the deploy
                        # itself failed (wrong Python path, missing deps, etc.)
                        failed_phase = self._read_failed_telemetry_phase()
                        if failed_phase:
                            self._log(f"FAILED telemetry ({failed_phase}) during grace — force redeploying")
                            self._last_deploy_time = 0  # clear grace period
                            self._deploy_babysitter(force=True)
                        else:
                            self._log(f"Deploy grace period ({grace_remaining:.0f}s remaining) — skipping health check")
                    elif not healthy:
                        age, status = self._read_best_heartbeat()
                        self._log(f"Babysitter unhealthy (age={age:.0f}s status={status}) — deploying")
                        self._deploy_babysitter(force=True)
                    else:
                        age, status = self._read_best_heartbeat()
                        self._log(f"Healthy (age={age:.0f}s status={status})")
                elif state == 'WAITING_FOR_RESOURCES':
                    self._log("Waiting for GCP to provision...")
                elif state in ('DELETING', 'UNKNOWN'):
                    pass  # already logged above
            except Exception as e:
                self._log(f"Unhandled error: {e}")
            stop_event.wait(POLL_INTERVAL)
        self._log("VM slot manager stopped")


# ── Health check quota guard ────────────────────────────────────────────────

def check_hc_quota():
    """Check HEALTH_CHECKS quota. Returns (ok, usage, limit)."""
    try:
        result = subprocess.run(
            [GCLOUD, 'compute', 'project-info', 'describe',
             '--project=gcp-research-credits-489020', '--format=json(quotas)'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return True, 0, 0  # fail open
        data = json.loads(result.stdout)
        for q in data.get('quotas', []):
            if q.get('metric') == 'HEALTH_CHECKS':
                limit = int(q['limit'])
                usage = int(q['usage'])
                ok = (limit - usage) > 5
                return ok, usage, limit
    except Exception:
        pass
    return True, 0, 0  # fail open


# ── Orphan health check sweep ───────────────────────────────────────────────

def cleanup_orphan_health_checks(live_vm_ids):
    """Delete health checks for VMs no longer in FLEET. Runs periodically."""
    try:
        # Collect all health check names
        result = subprocess.run(
            [GCLOUD, 'compute', 'health-checks', 'list',
             '--project=gcp-research-credits-489020',
             '--format=value(name)'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return
        names = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        import re
        orphan_ids = set()
        for name in names:
            m = re.search(r'tpu-\d+-(\d+)-', name)
            if m and m.group(1) not in live_vm_ids:
                orphan_ids.add(m.group(1))
        if not orphan_ids:
            return
        print(f"[hc_cleanup] {len(orphan_ids)} orphan VM IDs found", flush=True)
        for vm_id in orphan_ids:
            hcs_to_del = [n for n in names if vm_id in n]
            for hc in hcs_to_del:
                subprocess.run(
                    [GCLOUD, 'compute', 'health-checks', 'delete', hc,
                     '--project=gcp-research-credits-489020', '--quiet'],
                    capture_output=True, timeout=30
                )
            print(f"[hc_cleanup] Deleted {len(hcs_to_del)} checks for vm_id={vm_id}", flush=True)
    except Exception as e:
        print(f"[hc_cleanup] Error: {e}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[vm_manager] Starting v3 VM manager", flush=True)
    print(f"[vm_manager] Fleet: {len(FLEET)} VMs", flush=True)
    print(f"[vm_manager] Control plane: {CONTROL_PLANE}", flush=True)

    # Log quota status
    ok, usage, limit = check_hc_quota()
    print(f"[vm_manager] HEALTH_CHECKS quota: {usage}/{limit} ({'OK' if ok else 'NEAR LIMIT'})", flush=True)
    if not ok:
        print(f"[vm_manager] WARNING: HEALTH_CHECKS quota near limit — VM creation may fail", flush=True)

    stop_event = threading.Event()
    slots = [VMSlot(name, zone, accel, bucket) for name, zone, accel, bucket in FLEET]
    threads = []

    # Stagger thread starts slightly to avoid GCS thundering herd at t=0
    for i, slot in enumerate(slots):
        t = threading.Thread(
            target=slot.run,
            args=(stop_event,),
            name=f"vm-{slot.name}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # 2s stagger between thread starts

    print(f"[vm_manager] {len(threads)} VM threads started", flush=True)

    hc_cleanup_interval = 30  # every 30 min (30 * 60s poll)
    cycle = 0

    try:
        while True:
            time.sleep(60)
            cycle += 1

            # Periodic quota check + health check count validation
            if cycle % 5 == 0:
                ok, usage, limit = check_hc_quota()
                active_vms = len([s for s in slots if not s._shutdown])
                expected_max = active_vms * 5  # ~5 health checks per VM empirically
                print(f"[vm_manager] HEALTH_CHECKS: {usage}/{limit} (active_vms={active_vms} expected<={expected_max})", flush=True)
                if usage > expected_max:
                    print(f"[vm_manager] HC count {usage} > expected {expected_max} — running orphan cleanup", flush=True)
                    live_ids = set()
                    for s in slots:
                        if not s._shutdown:
                            # Extract numeric GCP VM ID from telemetry if available
                            raw = gcs_read(f"{CONTROL_PLANE}/telemetry/{s.name}_boot.json")
                            if raw:
                                try:
                                    live_ids.add(json.loads(raw).get('vm_id', ''))
                                except Exception:
                                    pass
                    cleanup_orphan_health_checks(live_ids)
                if not ok:
                    # Quota near limit — scale down VMs whose chips are all idle
                    pending, running, _, _ = get_queue_counts()
                    if pending == 0:
                        for slot in slots:
                            if slot._shutdown:
                                continue
                            if slot._all_chips_idle():
                                now = time.time()
                                if slot._idle_since is None:
                                    slot._idle_since = now
                                elif now - slot._idle_since >= IDLE_SCALE_DOWN_S:
                                    print(f"[vm_manager] Scale-down: {slot.name} idle "
                                          f"{now - slot._idle_since:.0f}s, no pending work, "
                                          f"quota {usage}/{limit}", flush=True)
                                    slot.mark_shutdown()
                            else:
                                slot._idle_since = None  # reset if chip becomes active

            # Check thread health
            dead = [t for t in threads if not t.is_alive()]
            if dead:
                print(f"[vm_manager] WARNING: {len(dead)} dead VM threads: {[t.name for t in dead]}", flush=True)

    except KeyboardInterrupt:
        print("[vm_manager] Caught SIGINT — stopping all VM slots...", flush=True)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)
        print("[vm_manager] All threads stopped", flush=True)


if __name__ == '__main__':
    main()
