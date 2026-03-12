#!/usr/bin/env python3
"""
coordinator.py — Centralized push coordinator for TPU sweeps.

OOP rewrite: SweepState is the single source of truth.
VMState tracks per-VM live state. State persists to disk between cycles.

Usage (blocklab):
    EXP=exp12c python3 coordinator.py --init          # Distribute configs to VMs
    EXP=exp12c python3 coordinator.py --monitor        # Long-running loop until all done
    EXP=exp12c python3 coordinator.py --status         # One-shot status
    EXP=exp12c python3 coordinator.py --dry-run        # Print all configs
    EXP=exp12c python3 coordinator.py --preflight      # Run experiment preflight

Usage (TPU VM — launched by submit.sh):
    EXP=exp12c python3 coordinator.py --sweep --proc-idx 0 --num-procs 16
"""

import argparse
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

# ── Config ──────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = 300   # 5 min between heartbeat writes
STALE_TTL = 1500           # 25 min — worker considered dead if no heartbeat
MAX_RETRIES = 2            # Max times to retry a failed config
MONITOR_POLL_S = 60        # Coordinator poll interval (Patch 4)
REBALANCE_DELAY_S = 1800   # 30 min — wait for real data before first rebalance
REBALANCE_INTERVAL_S = 3600  # 1 hour — rebalance periodically after first

# ── Experiment config loading ───────────────────────────────────────────────

def load_exp_config():
    """Load experiment config from EXP env var → experiments/<exp>.env file."""
    exp = os.environ.get('EXP')
    if not exp:
        print("ERROR: EXP env var not set. Usage: EXP=exp12c python3 coordinator.py ...", file=sys.stderr)
        sys.exit(1)

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', f'{exp}.env'),
        os.path.expanduser(f'~/distributed_tpu_training/experiments/{exp}.env'),
    ]
    env_file = None
    for c in candidates:
        if os.path.isfile(c):
            env_file = c
            break
    if not env_file:
        print(f"ERROR: experiment config not found: experiments/{exp}.env", file=sys.stderr)
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

    required = ['EXP_NAME', 'EXP_MODULE', 'WORK_DIR', 'CODE_DIRS', 'STEPS_PER_CONFIG']
    for r in required:
        if r not in cfg:
            print(f"ERROR: {r} missing in {env_file}", file=sys.stderr)
            sys.exit(1)

    return cfg


def load_exp_module(cfg):
    """Import the experiment module (e.g. exp12c_tpu.run_tpu_v2)."""
    module_name = cfg['EXP_MODULE']
    work_dir = os.path.expanduser(f"~/sf_bema/experiments/{cfg['WORK_DIR']}")
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
    parent = os.path.expanduser("~/sf_bema/experiments")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(module_name)


# ── VM config loading ──────────────────────────────────────────────────────

def parse_env_file(path):
    """Parse a .env file into a dict."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                val = val.strip().strip("'\"")
                cfg[key.strip()] = val
    return cfg


def load_vm_configs():
    """Read all ~/distributed_tpu_training/vm_configs/*.env, return list of dicts.

    Skips VMs with PROCS_PER_HOST=0 (OOM) or unknown accelerator type.
    """
    vm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vm_configs')
    vms = []
    for f in sorted(os.listdir(vm_dir)):
        if not f.endswith('.env'):
            continue
        cfg = parse_env_file(os.path.join(vm_dir, f))
        pph = int(cfg.get('PROCS_PER_HOST', '0'))
        if pph == 0:
            continue  # Skip OOM VMs (e.g. v5e)

        # Derive step time from accelerator type
        accel = cfg.get('ACCELERATOR_TYPE', '')
        if accel.startswith('v6e'):
            cfg['STEP_S'] = 4.9
        elif accel.startswith('v4'):
            cfg['STEP_S'] = 8.4
        elif accel.startswith('v5'):
            cfg['STEP_S'] = 6.0
        else:
            continue  # Unknown accelerator, skip

        # Ensure numeric fields are ints
        cfg['TPU_NUM_WORKERS'] = int(cfg.get('TPU_NUM_WORKERS', '1'))
        cfg['PROCS_PER_HOST'] = pph
        cfg['CHIPS_PER_HOST'] = int(cfg.get('CHIPS_PER_HOST', '8'))
        vms.append(cfg)
    return vms


# ── GCS helpers ─────────────────────────────────────────────────────────────
# Auto-detect: use `gcloud storage` if available (SDK 400+), else fall back to `gsutil`.
# v4 VMs have old SDK 347 that only has gsutil.

_USE_GSUTIL = None

def _use_gsutil():
    """Detect whether to use gsutil (old SDK) or gcloud storage (new SDK)."""
    global _USE_GSUTIL
    if _USE_GSUTIL is None:
        try:
            r = subprocess.run(['gcloud', 'storage', '--help'],
                               capture_output=True, text=True, timeout=5)
            _USE_GSUTIL = r.returncode != 0
        except Exception:
            _USE_GSUTIL = True
    return _USE_GSUTIL


def gcs_write(path, content):
    """Write content to a GCS path via stdin pipe. Returns True on success."""
    try:
        if _use_gsutil():
            # Use stdin piping — NOT repr(content) which produces invalid JSON
            proc = subprocess.run(
                ['gsutil', 'cp', '-', path],
                input=content, capture_output=True, text=True, timeout=30
            )
        else:
            proc = subprocess.run(
                ['gcloud', 'storage', 'cp', '-', path],
                input=content, capture_output=True, text=True, timeout=30
            )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def gcs_read(path):
    """Read content from a GCS path. Returns content string or None."""
    try:
        if _use_gsutil():
            result = subprocess.run(
                ['gsutil', 'cat', path],
                capture_output=True, text=True, timeout=30
            )
        else:
            result = subprocess.run(
                ['gcloud', 'storage', 'cat', path],
                capture_output=True, text=True, timeout=30
            )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def gcs_read_batch(paths, max_workers=20):
    """Read multiple GCS paths in parallel. Returns dict: {path: content_or_None}."""
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(gcs_read, p): p for p in paths}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                results[p] = fut.result()
            except Exception:
                results[p] = None
    return results


def gcs_exists(path):
    """Check if a GCS object exists."""
    try:
        if _use_gsutil():
            result = subprocess.run(
                ['gsutil', 'ls', path],
                capture_output=True, text=True, timeout=30
            )
        else:
            result = subprocess.run(
                ['gcloud', 'storage', 'ls', path],
                capture_output=True, text=True, timeout=30
            )
        return result.returncode == 0 and result.stdout.strip() != ''
    except (subprocess.TimeoutExpired, Exception):
        return False


def gcs_list(prefix):
    """List objects under a GCS prefix. Returns list of full paths."""
    try:
        if _use_gsutil():
            result = subprocess.run(
                ['gsutil', 'ls', f'{prefix}/'],
                capture_output=True, text=True, timeout=30
            )
        else:
            result = subprocess.run(
                ['gcloud', 'storage', 'ls', f'{prefix}/'],
                capture_output=True, text=True, timeout=30
            )
        if result.returncode == 0:
            return [l.strip().rstrip('/') for l in result.stdout.strip().split('\n') if l.strip()]
        return []
    except (subprocess.TimeoutExpired, Exception):
        return []


def gcs_delete(path):
    """Delete a GCS object (silently ignores errors)."""
    try:
        cmd = ['gsutil', 'rm', path] if _use_gsutil() else ['gcloud', 'storage', 'rm', path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, Exception):
        pass


def gcs_delete_prefix(prefix):
    """Delete all objects under a GCS prefix."""
    try:
        if _use_gsutil():
            subprocess.run(['gsutil', '-m', 'rm', '-r', f'{prefix}/'],
                           capture_output=True, text=True, timeout=30)
        else:
            subprocess.run(['gcloud', 'storage', 'rm', '-r', f'{prefix}/'],
                           capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, Exception):
        pass


def gcs_copy(src, dst):
    """Copy a GCS object (or download to local)."""
    try:
        cmd = ['gsutil', 'cp', src, dst] if _use_gsutil() else ['gcloud', 'storage', 'cp', src, dst]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# ── Local path helpers ────────────────────────────────────────────────────

def results_dir(cfg):
    """Local results directory: ~/sf_bema/results/<EXP>/"""
    d = os.path.expanduser(f"~/sf_bema/results/{cfg['EXP_NAME']}")
    os.makedirs(d, exist_ok=True)
    return d


def validated_dir(cfg):
    """Local validated results: ~/sf_bema/results/<EXP>/validated/"""
    d = os.path.join(results_dir(cfg), 'validated')
    os.makedirs(d, exist_ok=True)
    return d


def state_path(cfg):
    """Local state file: ~/sf_bema/results/<EXP>/state.json"""
    return os.path.join(results_dir(cfg), 'state.json')


def scan_validated_dir(cfg):
    """Return set of labels that have validated result JSONs locally."""
    vdir = validated_dir(cfg)
    labels = set()
    for f in os.listdir(vdir):
        if f.endswith('.json'):
            labels.add(f[:-5])  # strip .json
    return labels


# ── Heartbeat helpers ──────────────────────────────────────────────────────

def read_heartbeats(bucket, cfg, tpu_name):
    """Read heartbeat files for a VM from its GCS bucket.

    Only reads files matching this VM's name (filters by tpu_name prefix).
    Returns list of dicts: [{worker_id, vm, timestamp, step, label}, ...]
    """
    prefix = f"{bucket}/coord/{cfg['EXP_NAME']}/heartbeat"
    paths = gcs_list(prefix)
    # Filter to only this VM's heartbeat files (e.g. v6e-ew4a-2_0_0.json)
    vm_paths = [p for p in paths if os.path.basename(p).startswith(tpu_name)]
    heartbeats = []
    for p in vm_paths:
        content = gcs_read(p)
        if not content:
            continue
        try:
            hb = json.loads(content)
            heartbeats.append(hb)
        except json.JSONDecodeError:
            # Legacy format: "worker_id timestamp step label"
            parts = content.split()
            if len(parts) >= 4:
                heartbeats.append({
                    'worker_id': parts[0],
                    'vm': tpu_name,
                    'timestamp': float(parts[1]),
                    'step': int(parts[2]),
                    'label': parts[3],
                })
    return heartbeats


def read_all_heartbeats_from_bucket(bucket, cfg):
    """Read ALL heartbeat files from a bucket in one pass (parallel).

    Returns dict: {tpu_name: [heartbeat_dicts]}
    Uses batch parallel reads to avoid sequential subprocess bottleneck.
    """
    prefix = f"{bucket}/coord/{cfg['EXP_NAME']}/heartbeat"
    paths = gcs_list(prefix)
    by_vm = {}
    if not paths:
        return by_vm
    batch = gcs_read_batch(paths, max_workers=min(len(paths), 20))
    for p, content in batch.items():
        if not content:
            continue
        try:
            hb = json.loads(content)
            vm = hb.get('vm', '')
            by_vm.setdefault(vm, []).append(hb)
        except json.JSONDecodeError:
            pass
    return by_vm


def read_done_receipts(bucket, cfg):
    """Read done/ receipts from a VM's GCS bucket. Returns set of labels."""
    prefix = f"{bucket}/coord/{cfg['EXP_NAME']}/done"
    paths = gcs_list(prefix)
    labels = set()
    for p in paths:
        basename = os.path.basename(p)
        if basename.endswith('.done'):
            labels.add(basename[:-5])
    return labels


def is_stale(hb):
    """Is a heartbeat stale (older than TTL)?"""
    age = time.time() - hb.get('timestamp', 0)
    return age > STALE_TTL


def is_vm_dead(heartbeats):
    """A VM is dead if ALL heartbeats are stale.

    A VM with NO heartbeats is 'starting', not 'dead' — don't delete its assignment.
    Only mark dead once we've seen heartbeats that have all gone stale.
    Grace period: if any heartbeat has status 'starting' and is less than 45 min old,
    treat as still starting up (XLA compilation can take 10-15 min).
    """
    if not heartbeats:
        return False  # No heartbeats = hasn't started yet, not dead
    # Grace period for VMs in startup/XLA compile phase
    STARTUP_GRACE = 2700  # 45 min grace for XLA compilation
    for hb in heartbeats:
        age = time.time() - hb.get('timestamp', 0)
        if hb.get('status') in ('starting', 'xla_compile') and age < STARTUP_GRACE:
            return False  # Still starting up, don't mark as dead
    return all(is_stale(hb) for hb in heartbeats)


# ── Result validation ──────────────────────────────────────────────────────

def validate_result(result_path, cfg, module):
    """Validate a result JSON. Returns (is_valid, reason)."""
    try:
        with open(result_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "invalid JSON"

    summary = data.get('summary', {})
    if summary.get('best_val_loss') is None:
        return False, "no best_val_loss"
    if math.isnan(summary.get('best_val_loss', float('nan'))):
        return False, "NaN loss"
    steps_per_config = int(cfg.get('STEPS_PER_CONFIG', 0))
    if steps_per_config > 0 and summary.get('total_steps', 0) < steps_per_config * 0.5:
        return False, f"only {summary.get('total_steps')} steps (need >= {steps_per_config * 0.5})"

    # Module-specific validation (optional)
    if hasattr(module, 'validate_result'):
        return module.validate_result(data)

    return True, "ok"


def pull_result_summary(bucket, cfg, label):
    """Download a result summary JSON from GCS to local tmp. Returns local path or None."""
    # Try results/<label>/summary.json first, then results/<label>.json
    local_dir = os.path.join(results_dir(cfg), 'tmp_pull')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f'{label}.json')

    prefix = f"{bucket}/coord/{cfg['EXP_NAME']}/results/{label}"
    # Try summary.json inside the results dir
    if gcs_copy(f"{prefix}/summary.json", local_path):
        return local_path
    # Try direct result file
    if gcs_copy(f"{prefix}.json", local_path):
        return local_path
    # Try listing the results dir for any .json
    files = gcs_list(prefix)
    for f in files:
        if f.endswith('.json'):
            if gcs_copy(f, local_path):
                return local_path
    return None


def move_to_validated(local_path, cfg, label):
    """Move a validated result to the validated/ directory."""
    dst = os.path.join(validated_dir(cfg), f'{label}.json')
    os.rename(local_path, dst)


# ── VMState ────────────────────────────────────────────────────────────────

class VMState:
    """Tracks a single TPU VM's live state."""

    def __init__(self, config):
        self.name = config['TPU_NAME']
        self.bucket = config['BUCKET']
        self.config = config
        self.num_procs = config['TPU_NUM_WORKERS'] * config['PROCS_PER_HOST']
        self.step_s = config['STEP_S']

        # Live state (refreshed each cycle)
        self.heartbeats = []       # Latest heartbeat per worker
        self.done_labels = set()   # Labels with done/ receipts on GCS
        self.assigned_labels = []  # Current assignment list (from GCS)
        self.is_alive = True
        self.last_refresh = 0

    @staticmethod
    def label_of(item):
        """Extract label string from assignment item (handles both str and dict)."""
        return item['label'] if isinstance(item, dict) else item

    @property
    def throughput(self):
        """Estimated configs/hour based on active procs and step time."""
        active = self.active_procs
        if active <= 0:
            return 0.0
        steps_per_config = int(self.config.get('STEPS_PER_CONFIG',
                               os.environ.get('STEPS_PER_CONFIG', '1778')))
        cfg_hours = steps_per_config * self.step_s / 3600
        return active / cfg_hours

    @property
    def active_procs(self):
        """Count of non-stale heartbeats (= working processes)."""
        return sum(1 for hb in self.heartbeats if not is_stale(hb))

    @property
    def current_labels(self):
        """Labels being actively worked on (from non-stale heartbeats)."""
        return {hb['label'] for hb in self.heartbeats if not is_stale(hb)}

    def refresh_from_gcs(self, cfg, seen_buckets_hb, seen_buckets_done):
        """Read heartbeats, done receipts, assignment from this VM's bucket.

        Uses seen_buckets_* dicts to avoid duplicate GCS reads for shared buckets.
        """
        bucket = self.bucket

        # Heartbeats: batch per bucket
        if bucket not in seen_buckets_hb:
            seen_buckets_hb[bucket] = read_all_heartbeats_from_bucket(bucket, cfg)
        self.heartbeats = seen_buckets_hb[bucket].get(self.name, [])

        # Done receipts: batch per bucket
        if bucket not in seen_buckets_done:
            seen_buckets_done[bucket] = read_done_receipts(bucket, cfg)
        self.done_labels = seen_buckets_done[bucket]

        # Assignment
        assign_path = f"{bucket}/coord/{cfg['EXP_NAME']}/assignments/{self.name}.json"
        raw = gcs_read(assign_path)
        if raw:
            try:
                self.assigned_labels = json.loads(raw)
            except json.JSONDecodeError:
                self.assigned_labels = []
        else:
            self.assigned_labels = []

        # Liveness
        self.is_alive = not is_vm_dead(self.heartbeats)
        self.last_refresh = time.time()

    def to_dict(self):
        """Serialize for state persistence."""
        return {
            'name': self.name,
            'is_alive': self.is_alive,
            'active_procs': self.active_procs,
            'num_assigned': len(self.assigned_labels),
            'num_done': len(self.done_labels),
        }

    def status_lines(self):
        """Return formatted status lines for this VM."""
        lines = []
        status_str = "ALIVE" if self.is_alive else "DEAD"
        n_assigned = len(self.assigned_labels)
        n_done = len(self.done_labels)
        lines.append(f"  {self.name}: {status_str}  assigned={n_assigned}  "
                      f"done_on_gcs={n_done}  active_procs={self.active_procs}")
        for hb in self.heartbeats:
            age_min = (time.time() - hb.get('timestamp', 0)) / 60
            stale_str = " STALE" if is_stale(hb) else ""
            lines.append(f"    {hb.get('worker_id', '?')}: label={hb.get('label', '?')} "
                         f"step={hb.get('step', '?')} age={age_min:.0f}min{stale_str}")
        return lines


# ── SweepState ─────────────────────────────────────────────────────────────

class SweepState:
    """Coordinator brain. Single source of truth. Persists to disk."""

    def __init__(self, cfg, module, vm_configs):
        self.cfg = cfg
        self.module = module
        self.exp_name = cfg['EXP_NAME']
        self.total_configs = 0

        # Ground truth — built once from module
        self.all_configs = []          # [(label, overrides)]
        self.all_labels = set()
        self.overrides_map = {}        # {label: overrides}

        # VM tracking
        self.vms = {}                  # {vm_name: VMState}
        self.vm_configs = vm_configs   # Raw config dicts (for push_assignments etc.)

        # Result tracking (persisted)
        self.validated = set()         # Labels validated locally
        self.retries = {}              # {label: retry_count}
        self.failed = set()            # Labels that exceeded MAX_RETRIES

        # Timeline tracking (persisted)
        self.init_time = 0
        self.last_poll_time = 0
        self.last_rebalance_time = 0
        self.events = []               # [{time, event, details}]

        # Derived (computed each cycle by refresh())
        self.missing = set()           # all_labels - validated
        self.gcs_done = set()          # All done receipts across all VMs
        self.alive_vms = []            # [VMState]
        self.dead_vms = []             # [VMState]

    # ── Lifecycle ──

    def initialize(self):
        """Build configs from module, create VMState objects, load existing state."""
        self.all_configs = self.module.build_configs()
        self.total_configs = len(self.all_configs)
        self.all_labels = {l for l, _ in self.all_configs}
        self.overrides_map = {l: o for l, o in self.all_configs}

        # Create VMState for each VM config
        for vc in self.vm_configs:
            vm = VMState(vc)
            # Inject steps_per_config into VM config for throughput calc
            vm.config['STEPS_PER_CONFIG'] = self.cfg['STEPS_PER_CONFIG']
            self.vms[vm.name] = vm

        # Load any persisted state
        self.load()

        # Refresh validated from disk (authoritative)
        self.validated = scan_validated_dir(self.cfg)
        self.missing = self.all_labels - self.validated

    def load(self):
        """Load persisted state from disk."""
        sp = state_path(self.cfg)
        if not os.path.isfile(sp):
            return

        try:
            with open(sp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        self.init_time = data.get('init_time', 0)
        self.last_poll_time = data.get('last_poll', 0)
        self.last_rebalance_time = data.get('last_rebalance', 0)
        self.retries = data.get('retries', {})
        self.failed = set(data.get('failed', []))
        self.events = data.get('events', [])
        # validated is loaded from disk dir, not state.json (disk is authoritative)

    def save(self):
        """Persist state to disk. Called after every mutation."""
        data = {
            'exp_name': self.exp_name,
            'total': self.total_configs,
            'init_time': self.init_time,
            'validated': sorted(self.validated),
            'retries': self.retries,
            'failed': sorted(self.failed),
            'assignments': {name: [VMState.label_of(item) for item in vm.assigned_labels]
                            for name, vm in self.vms.items()},
            'events': self.events[-500:],  # Keep last 500 events
            'last_poll': self.last_poll_time,
            'last_rebalance': self.last_rebalance_time,
        }
        sp = state_path(self.cfg)
        with open(sp, 'w') as f:
            json.dump(data, f, indent=2)

    # ── Refresh cycle ──

    def refresh(self):
        """Full refresh: GCS reads -> state updates -> derived computation."""
        # 0. Reload vm_configs from disk so monitor can pick up newly created/recovered VMs.
        # Without this, fleet expansion adds VMs that will never receive assignments.
        try:
            latest_vm_configs = load_vm_configs()
        except Exception:
            latest_vm_configs = self.vm_configs

        latest_by_name = {vc.get('TPU_NAME'): vc for vc in latest_vm_configs if vc.get('TPU_NAME')}

        # Add new VMs and refresh existing VM configs (bucket/zone/runtime may change).
        for name, vc in latest_by_name.items():
            if name not in self.vms:
                vm = VMState(vc)
                vm.config['STEPS_PER_CONFIG'] = self.cfg['STEPS_PER_CONFIG']
                self.vms[name] = vm
                self.log_event("VM_ADDED", name)
            else:
                self.vms[name].config.update(vc)

        # Remove VMs deleted from vm_configs/ (rare, but avoids stale state).
        for name in list(self.vms.keys()):
            if name not in latest_by_name:
                self.vms.pop(name, None)
                self.log_event("VM_REMOVED", name)

        self.vm_configs = latest_vm_configs

        # 1. Refresh each VM from GCS (batch by bucket)
        seen_buckets_hb = {}
        seen_buckets_done = {}
        for vm in self.vms.values():
            vm.refresh_from_gcs(self.cfg, seen_buckets_hb, seen_buckets_done)

        # 2. Aggregate GCS done across all VMs
        self.gcs_done = set()
        for vm in self.vms.values():
            self.gcs_done |= vm.done_labels

        # 3. Refresh validated from disk (authoritative source)
        self.validated = scan_validated_dir(self.cfg)

        # 4. Compute derived state
        self.missing = self.all_labels - self.validated
        self.alive_vms = [vm for vm in self.vms.values() if vm.is_alive]
        self.dead_vms = [vm for vm in self.vms.values() if not vm.is_alive]

        # 5. Ensure assignment files exist for alive VMs.
        #
        # If an assignment file is missing, workers immediately exit:
        #   "[sweep] Assignment file gone. Exiting."
        # That causes the fleet manager to repeatedly "re-sweep" without progress.
        for vm in self.alive_vms:
            assign_path = f"{vm.bucket}/coord/{self.exp_name}/assignments/{vm.name}.json"
            if not gcs_exists(assign_path):
                if gcs_write(assign_path, json.dumps([], indent=2)):
                    vm.assigned_labels = []
                    self.log_event("ASSIGNMENT_CREATED", vm.name)

        # 6. Repair: if any missing labels are not currently assigned anywhere, append them.
        #
        # This prevents "tail stalls" where remaining configs become unassigned due to VM death,
        # assignment deletion, or adding VMs mid-run.
        assigned = set()
        for vm in self.vms.values():
            for item in (vm.assigned_labels or []):
                assigned.add(VMState.label_of(item))
        current = set()
        for vm in self.alive_vms:
            current |= set(vm.current_labels)

        unassigned = (self.missing - self.gcs_done - self.failed) - assigned - current
        if unassigned and self.alive_vms:
            alive_vm_configs = [vm.config for vm in self.alive_vms]
            self._append_to_assignments(unassigned, alive_vm_configs)
            self.log_event("UNASSIGNED_REPAIRED", f"{len(unassigned)} labels appended")

        self.last_poll_time = time.time()

    # ── Actions ──

    def pull_results(self):
        """Pull new done results from GCS, validate, move to validated/.

        Returns count of newly validated results this cycle.
        """
        new_done = self.gcs_done - self.validated - self.failed
        pulled = 0

        for label in new_done:
            # Find which bucket has this result
            for vm in self.vms.values():
                if label in vm.done_labels:
                    local_path = pull_result_summary(vm.bucket, self.cfg, label)
                    if local_path:
                        valid, reason = validate_result(local_path, self.cfg, self.module)
                        if valid:
                            move_to_validated(local_path, self.cfg, label)
                            self.validated.add(label)
                            self.missing.discard(label)
                            pulled += 1
                            self.log_event("VALIDATED", label)
                        else:
                            print(f"  [validate] REJECTED {label}: {reason}")
                            os.remove(local_path)
                            self.retries[label] = self.retries.get(label, 0) + 1
                            if self.retries[label] >= MAX_RETRIES:
                                other_zone_vms = [v.config for v in self.alive_vms
                                                  if v.config.get('ZONE') != vm.config.get('ZONE')]
                                if other_zone_vms:
                                    print(f"  [validate] {label}: failed {self.retries[label]}x, reassigning to different zone")
                                    done_path = f"{vm.bucket}/coord/{self.cfg['EXP_NAME']}/done/{label}.done"
                                    gcs_delete(done_path)
                                    self._append_to_assignments({label}, other_zone_vms)
                                    self.retries[label] = 0
                                    self.log_event("CROSS_ZONE_RETRY", f"{label}: validation failed, retrying in other zone")
                                else:
                                    print(f"  [validate] {label} failed {self.retries[label]}x, marking as FAILED")
                                    self.failed.add(label)
                                    self.log_event("FAILED", f"{label} after {self.retries[label]} retries")
                    else:
                        # No result uploaded — likely worker crashed (failed_rc*)
                        self.retries[label] = self.retries.get(label, 0) + 1
                        done_path = f"{vm.bucket}/coord/{self.cfg['EXP_NAME']}/done/{label}.done"
                        if self.retries[label] >= MAX_RETRIES:
                            # Instead of permanent failure, try reassigning to a different zone
                            other_zone_vms = [v.config for v in self.alive_vms
                                              if v.config.get('ZONE') != vm.config.get('ZONE')]
                            if other_zone_vms:
                                print(f"  [retry] {label}: failed {self.retries[label]}x on {vm.name}, reassigning to different zone")
                                gcs_delete(done_path)
                                self._append_to_assignments({label}, other_zone_vms)
                                self.retries[label] = 0  # Reset retries for new zone
                                self.log_event("CROSS_ZONE_RETRY", f"{label}: from {vm.name} zone to other zone")
                            else:
                                print(f"  [retry] {label}: no result found {self.retries[label]}x, marking as FAILED (no other zones)")
                                self.failed.add(label)
                                self.log_event("FAILED", f"{label}: no result after {self.retries[label]} retries")
                        else:
                            print(f"  [retry] {label}: no result found (attempt {self.retries[label]}/{MAX_RETRIES}), deleting done receipt for retry")
                            gcs_delete(done_path)
                            self.log_event("RETRY", f"{label}: deleted done receipt (attempt {self.retries[label]})")
                    break  # Found the bucket, move on

        return pulled

    def handle_dead_vms(self):
        """Detect dead VMs, reassign their unfinished work to alive VMs."""
        if not self.dead_vms or not self.alive_vms:
            return

        alive_vm_configs = [vm.config for vm in self.alive_vms]

        for vm in self.dead_vms:
            if not vm.assigned_labels:
                continue

            orphaned = [VMState.label_of(item) for item in vm.assigned_labels
                        if VMState.label_of(item) not in self.validated
                        and VMState.label_of(item) not in self.gcs_done]

            if orphaned and alive_vm_configs:
                print(f"  [dead] {vm.name}: {len(orphaned)} orphaned configs -> alive VMs")
                self._append_to_assignments(set(orphaned), alive_vm_configs)
                self.log_event("DEAD_VM_REASSIGN",
                               f"{vm.name}: {len(orphaned)} configs reassigned")

            # Delete assignment for dead VM
            assign_path = f"{vm.bucket}/coord/{self.exp_name}/assignments/{vm.name}.json"
            if gcs_exists(assign_path):
                print(f"  [dead] Deleting assignment for dead VM: {vm.name}")
                gcs_delete(assign_path)

    def should_rebalance(self):
        """Determine if we should rebalance now."""
        if len(self.alive_vms) < 2:
            return False
        if len(self.missing) <= 10:
            return False

        now = time.time()
        elapsed_since_start = now - self.init_time if self.init_time else 0

        if self.last_rebalance_time == 0:
            # First rebalance: wait for enough data
            return elapsed_since_start > REBALANCE_DELAY_S
        else:
            # Subsequent: periodic
            return (now - self.last_rebalance_time) > REBALANCE_INTERVAL_S

    def rebalance(self):
        """Smart rebalance: keep position 0-1 per proc, redistribute the rest.

        For each alive VM:
        1. Read current assignment
        2. Protect first 2*num_procs items (current + next task per proc)
        3. Remove done items from unprotected zone
        4. Pool all unprotected-and-not-done items
        5. Redistribute pool proportional to actual throughput
        6. Push updated assignments (protected + new share)
        """
        # Measure actual throughputs from active procs
        actual_tputs = {}
        for vm in self.alive_vms:
            if vm.active_procs > 0:
                steps_per_config = int(self.cfg['STEPS_PER_CONFIG'])
                cfg_hours = steps_per_config * vm.step_s / 3600
                actual_tputs[vm.name] = vm.active_procs / cfg_hours

        if not actual_tputs:
            print("  [rebalance] No throughput data yet, skipping")
            return False

        done_labels = self.validated | self.gcs_done

        # Phase 1: Read all current assignments, split into protected + movable
        pool = []
        vm_protected = {}

        for vm in self.alive_vms:
            assignment = vm.assigned_labels  # Already loaded by refresh()
            if not assignment:
                vm_protected[vm.name] = []
                continue

            protect_count = min(2 * vm.num_procs, len(assignment))
            protected = assignment[:protect_count]
            movable = assignment[protect_count:]

            # Filter out done items (handle both str and dict formats)
            protected = [item for item in protected if VMState.label_of(item) not in done_labels]
            not_done_movable = [item for item in movable if VMState.label_of(item) not in done_labels]

            vm_protected[vm.name] = protected
            pool.extend(not_done_movable)

        if not pool:
            print("  [rebalance] No movable configs to redistribute")
            return False

        # Phase 2: Redistribute pool proportional to actual throughput
        total_tput = sum(actual_tputs.get(vm.name, 0) for vm in self.alive_vms)
        if total_tput <= 0:
            print("  [rebalance] Zero total throughput, skipping")
            return False

        pool_assignments = {}
        idx = 0
        sorted_vms = sorted(self.alive_vms, key=lambda vm: actual_tputs.get(vm.name, 0))
        for vm in sorted_vms:
            tput = actual_tputs.get(vm.name, 0)
            share = round(len(pool) * tput / total_tput)
            pool_assignments[vm.name] = pool[idx:idx + share]
            idx += share
        # Leftover to fastest
        if idx < len(pool):
            fastest = max(self.alive_vms, key=lambda vm: actual_tputs.get(vm.name, 0))
            pool_assignments.setdefault(fastest.name, [])
            pool_assignments[fastest.name].extend(pool[idx:])

        # Phase 3: Write updated assignments = protected + new share
        for vm in self.alive_vms:
            new_assignment = vm_protected.get(vm.name, []) + pool_assignments.get(vm.name, [])
            path = f"{vm.bucket}/coord/{self.exp_name}/assignments/{vm.name}.json"
            gcs_write(path, json.dumps(new_assignment, indent=2))
            n_protected = len(vm_protected.get(vm.name, []))
            n_new = len(pool_assignments.get(vm.name, []))
            tput = actual_tputs.get(vm.name, 0)
            print(f"  [rebalance] {vm.name}: {n_protected} protected + {n_new} new "
                  f"= {n_protected + n_new} total (tput={tput:.2f} cfg/h)")

        self.last_rebalance_time = time.time()
        elapsed = time.time() - self.init_time if self.init_time else 0
        self.log_event("REBALANCE", f"{len(pool)} configs redistributed at {elapsed/60:.0f}min")
        return True

    def distribute(self):
        """Initial distribution of configs to VMs proportional to throughput."""
        # Skip already-validated
        remaining = [(l, o) for l, o in self.all_configs if l not in self.validated]
        print(f"[init] {len(self.validated)} already validated, {len(remaining)} remaining")

        if not remaining:
            print("[init] All configs already validated. Nothing to do.")
            return

        # Clean stale coord data from all VM buckets
        print(f"[init] Cleaning stale coord data from GCS...")
        for vm in self.vms.values():
            prefix = f"{vm.bucket}/coord/{self.exp_name}"
            gcs_delete_prefix(f"{prefix}/done")
            gcs_delete_prefix(f"{prefix}/heartbeat")
            print(f"  Cleaned {vm.name} ({vm.bucket})")

        # Compute throughputs
        throughputs = {}
        steps = int(self.cfg['STEPS_PER_CONFIG'])
        for vm in self.vms.values():
            cfg_hours = steps * vm.step_s / 3600
            throughputs[vm.name] = vm.num_procs / cfg_hours

        print(f"[init] VM throughputs (configs/hour):")
        for name, tput in sorted(throughputs.items(), key=lambda x: -x[1]):
            print(f"  {name}: {tput:.2f} cfg/h")

        # Distribute proportional to throughput
        assignments = self._distribute_configs(remaining, throughputs)

        print(f"\n[init] Distribution:")
        for name, cfgs in sorted(assignments.items()):
            print(f"  {name}: {len(cfgs)} configs")
        total_assigned = sum(len(v) for v in assignments.values())
        print(f"  TOTAL: {total_assigned} (should be {len(remaining)})")
        assert total_assigned == len(remaining), \
            f"Assignment mismatch: {total_assigned} != {len(remaining)}"

        # Push to each VM's bucket
        self._push_assignments(assignments)

        # Record
        self.init_time = time.time()
        self.log_event("INIT", f"{self.total_configs} configs, {len(remaining)} distributed")
        self.save()
        print(f"\n[init] Done. Run --monitor to start coordination loop.")

    # ── Internal helpers ──

    def _distribute_configs(self, remaining, throughputs):
        """Distribute configs proportional to throughput.

        Returns {vm_name: [(label, overrides), ...]}.
        """
        if not remaining or not throughputs:
            return {}

        total_tput = sum(throughputs.values())
        assignments = {}
        idx = 0

        # Sort VMs by throughput (fastest last, so leftover goes to fastest)
        sorted_vms = sorted(throughputs.items(), key=lambda x: x[1])

        for vm_name, tput in sorted_vms:
            share = round(len(remaining) * tput / total_tput)
            assignments[vm_name] = remaining[idx:idx + share]
            idx += share

        # Leftover to fastest VM
        if idx < len(remaining):
            fastest = max(throughputs, key=throughputs.get)
            assignments.setdefault(fastest, [])
            assignments[fastest].extend(remaining[idx:])

        return assignments

    def _push_assignments(self, assignments):
        """Push each VM's assignment to its own GCS bucket."""
        for vm_name, configs in assignments.items():
            vm = self.vms.get(vm_name)
            if not vm:
                continue
            payload = json.dumps([{'label': l, 'overrides': o} for l, o in configs], indent=2)
            path = f"{vm.bucket}/coord/{self.exp_name}/assignments/{vm_name}.json"
            if gcs_write(path, payload):
                print(f"  [push] {vm_name}: {len(configs)} configs -> {path}")
            else:
                print(f"  [push] FAILED to write assignment for {vm_name}")

    def _append_to_assignments(self, labels, alive_vm_configs):
        """Append unassigned configs to alive VMs proportional to throughput."""
        remaining = [(l, self.overrides_map[l]) for l in sorted(labels)]
        if not remaining:
            return

        throughputs = {}
        steps = int(self.cfg['STEPS_PER_CONFIG'])
        for vc in alive_vm_configs:
            name = vc['TPU_NAME']
            procs = vc['TPU_NUM_WORKERS'] * vc['PROCS_PER_HOST']
            cfg_hours = steps * vc['STEP_S'] / 3600
            throughputs[name] = procs / cfg_hours

        assignments = self._distribute_configs(remaining, throughputs)

        for vm_name, configs in assignments.items():
            vm = self.vms.get(vm_name)
            if not vm:
                continue
            path = f"{vm.bucket}/coord/{self.exp_name}/assignments/{vm_name}.json"
            existing_raw = gcs_read(path)
            existing = json.loads(existing_raw) if existing_raw else []
            existing_labels = {VMState.label_of(item) for item in existing}
            new_items = [{'label': l, 'overrides': o} for l, o in configs
                         if l not in existing_labels]
            if new_items:
                existing.extend(new_items)
                gcs_write(path, json.dumps(existing, indent=2))
                print(f"  [append] {vm_name}: +{len(new_items)} configs (total {len(existing)})")

    # ── Queries ──

    @property
    def is_complete(self):
        return len(self.validated) >= self.total_configs

    @property
    def progress_str(self):
        return f"{len(self.validated)}/{self.total_configs}"

    def log_event(self, event, details=""):
        """Log a timestamped event."""
        self.events.append({
            'time': time.time(),
            'event': event,
            'details': details,
        })

    def status_report(self):
        """Generate a full status report string."""
        lines = []
        lines.append(f"[status] Experiment: {self.exp_name}")
        lines.append(f"[status] Validated: {self.progress_str}")
        lines.append(f"[status] Failed: {len(self.failed)}")
        lines.append("")

        for vm in self.vms.values():
            lines.extend(vm.status_lines())

        if self.failed:
            lines.append(f"\n  Failed configs ({len(self.failed)}):")
            for label in sorted(self.failed)[:20]:
                retries = self.retries.get(label, 0)
                lines.append(f"    {label} (retries={retries})")

        return '\n'.join(lines)

    def print_status(self, pulled=0, cycle_elapsed=0):
        """Print a one-line monitor status."""
        alive_str = ', '.join(sorted(vm.name for vm in self.alive_vms)) or 'none'
        dead_str = ', '.join(sorted(vm.name for vm in self.dead_vms)) or 'none'
        print(f"[monitor] {self.progress_str} validated | "
              f"missing={len(self.missing)} pulled={pulled} | "
              f"alive=[{alive_str}] dead=[{dead_str}] | "
              f"cycle={cycle_elapsed:.1f}s")


# ── Experiment Queue ───────────────────────────────────────────────────────

class ExperimentQueue:
    """Manages a sequential queue of experiments in ~/distributed_tpu_training/queue.json."""
    QUEUE_FILE = os.path.expanduser('~/distributed_tpu_training/queue.json')

    def __init__(self):
        self.experiments = []
        self.load()

    def load(self):
        if os.path.isfile(self.QUEUE_FILE):
            with open(self.QUEUE_FILE) as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []

    def save(self):
        with open(self.QUEUE_FILE, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def enqueue(self, exp_name):
        # Validate experiment config exists
        env_path = os.path.expanduser(f'~/distributed_tpu_training/experiments/{exp_name}.env')
        if not os.path.isfile(env_path):
            print(f"[queue] ERROR: experiment config not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        # Check not already in queue
        for e in self.experiments:
            if e['exp'] == exp_name and e['status'] in ('pending', 'running'):
                print(f"[queue] {exp_name} already in queue (status={e['status']})")
                return
        self.experiments.append({
            'exp': exp_name,
            'status': 'pending',
            'enqueued_at': time.time(),
        })
        self.save()
        print(f"[queue] Enqueued: {exp_name} (position {len([e for e in self.experiments if e['status'] == 'pending'])})")

    def dequeue(self, exp_name):
        before = len(self.experiments)
        self.experiments = [e for e in self.experiments if e['exp'] != exp_name]
        self.save()
        removed = before - len(self.experiments)
        if removed:
            print(f"[queue] Removed: {exp_name}")
        else:
            print(f"[queue] {exp_name} not found in queue")

    def next_pending(self):
        for e in self.experiments:
            if e['status'] == 'pending':
                return e
        return None

    def mark_running(self, exp_name):
        for e in self.experiments:
            if e['exp'] == exp_name:
                e['status'] = 'running'
                e['started_at'] = time.time()
        self.save()

    def mark_done(self, exp_name):
        for e in self.experiments:
            if e['exp'] == exp_name:
                e['status'] = 'done'
                e['finished_at'] = time.time()
        self.save()

    def status_report(self):
        if not self.experiments:
            print("[queue] Queue is empty")
            return
        print(f"[queue] {len(self.experiments)} experiment(s):")
        for i, e in enumerate(self.experiments):
            status = e['status'].upper()
            exp = e['exp']
            extra = ''
            if e.get('total_configs'):
                extra += f" configs={e['total_configs']}"
            if e.get('started_at') and e['status'] == 'running':
                elapsed = (time.time() - e['started_at']) / 3600
                extra += f" running={elapsed:.1f}h"
            if e.get('finished_at') and e['status'] == 'done':
                dur = (e['finished_at'] - e.get('started_at', e['finished_at'])) / 3600
                extra += f" duration={dur:.1f}h"
            print(f"  {i+1}. [{status}] {exp}{extra}")


def deploy_to_fleet(exp_name, cfg, vm_configs, prev_exp=None):
    """Deploy code and launch sweep workers on all VMs."""
    submit_sh = os.path.expanduser('~/distributed_tpu_training/submit.sh')
    for vc in vm_configs:
        tpu_name = vc['TPU_NAME']
        env = {**os.environ}
        for key in ['TPU_NAME', 'ZONE', 'BUCKET', 'TPU_NUM_WORKERS',
                     'PROCS_PER_HOST', 'CHIPS_PER_HOST', 'ACCELERATOR_TYPE',
                     'WANDB_MODE', 'LAUNCH_MODE', 'LIBTPU_INIT_ARGS',
                     'MODEL_PATH']:
            if key in vc:
                env[key] = str(vc[key])
        env['EXP'] = exp_name

        # Cancel previous experiment workers if switching
        if prev_exp and prev_exp != exp_name:
            cancel_env = {**env, 'EXP': prev_exp}
            try:
                subprocess.run(['bash', submit_sh, '--cancel'],
                              env=cancel_env, timeout=120, capture_output=True)
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"[deploy] Warning: cancel on {tpu_name} failed: {e}")

        # Deploy + launch
        print(f"[deploy] {exp_name} -> {tpu_name}")
        try:
            subprocess.run(['bash', submit_sh, '--sweep'], env=env, timeout=600)
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"[deploy] Warning: sweep launch on {tpu_name} failed: {e}")


def copy_validated_results(cfg):
    """Copy validated results to experiment folder."""
    src = os.path.expanduser(f"~/sf_bema/results/{cfg['EXP_NAME']}/validated/")
    if not os.path.isdir(src):
        print(f"[queue] No validated results dir: {src}")
        return
    module_dir = cfg['EXP_MODULE'].split('.')[0]
    dst = os.path.expanduser(f"~/sf_bema/experiments/{cfg['WORK_DIR']}/{module_dir}/results/")
    os.makedirs(dst, exist_ok=True)
    count = 0
    for f in os.listdir(src):
        if f.endswith('.json'):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            count += 1
    print(f"[queue] Copied {count} results to {dst}")


def queue_monitor():
    """Process experiment queue sequentially. Long-running."""
    queue = ExperimentQueue()
    prev_exp = None

    while True:
        queue.load()
        entry = queue.next_pending()
        if not entry:
            # Check for crash recovery — resume running experiment
            running = [e for e in queue.experiments if e['status'] == 'running']
            if running:
                entry = running[0]
                print(f"[queue] Resuming running experiment: {entry['exp']}")
            else:
                print("[queue] No pending experiments. Sleeping 60s...")
                time.sleep(60)
                continue

        exp_name = entry['exp']
        queue.mark_running(exp_name)
        print(f"\n[queue] === Starting: {exp_name} ===")

        os.environ['EXP'] = exp_name
        cfg = load_exp_config()
        module = load_exp_module(cfg)
        vm_configs = load_vm_configs()

        # Update queue entry with config count
        for e in queue.experiments:
            if e['exp'] == exp_name:
                try:
                    configs = module.build_configs()
                    e['total_configs'] = len(configs)
                except Exception:
                    pass
        queue.save()

        # Init + distribute
        init_experiment(cfg, module, vm_configs)

        # Deploy code + launch workers on ALL VMs
        deploy_to_fleet(exp_name, cfg, vm_configs, prev_exp)

        # Monitor until complete (blocks)
        monitor_experiment(cfg, module, vm_configs)

        # Copy results
        copy_validated_results(cfg)

        queue.mark_done(exp_name)
        prev_exp = exp_name
        print(f"[queue] === {exp_name} COMPLETE ===\n")


# ── Mode: --init ───────────────────────────────────────────────────────────

def init_experiment(cfg, module, vm_configs):
    """Build configs, distribute to VMs proportional to throughput, push to GCS."""
    state = SweepState(cfg, module, vm_configs)
    state.initialize()

    print(f"[init] {state.total_configs} configs from {cfg['EXP_MODULE']}")
    state.distribute()


# ── Mode: --monitor ────────────────────────────────────────────────────────

def monitor_experiment(cfg, module, vm_configs):
    """The brain. Each cycle: refresh -> pull -> handle dead -> rebalance -> save."""
    state = SweepState(cfg, module, vm_configs)
    state.initialize()

    print(f"[monitor] Watching {state.total_configs} configs. "
          f"Poll every {MONITOR_POLL_S}s. Exit when all validated.")
    print(f"[monitor] First rebalance after {REBALANCE_DELAY_S}s, "
          f"then every {REBALANCE_INTERVAL_S}s.")

    while True:
        cycle_start = time.time()

        # Full refresh from GCS + disk
        state.refresh()

        # Check completion (before pulling, in case we finished last cycle)
        if state.is_complete:
            print(f"\n[monitor] All {state.total_configs} configs validated. DONE!")
            state.log_event("COMPLETE", f"All {state.total_configs} validated")
            state.save()
            break

        # Pull new results from GCS, validate
        pulled = state.pull_results()

        # Handle dead VMs
        state.handle_dead_vms()

        # Smart rebalance
        if state.should_rebalance():
            elapsed = time.time() - state.init_time if state.init_time else 0
            print(f"  [rebalance] Triggering smart rebalance "
                  f"({elapsed/60:.0f}min since start)")
            state.rebalance()

        # Save and print status
        state.save()
        cycle_elapsed = time.time() - cycle_start
        state.print_status(pulled=pulled, cycle_elapsed=cycle_elapsed)

        # Check completion again (after pulling)
        if state.is_complete:
            print(f"\n[monitor] All {state.total_configs} configs validated. DONE!")
            state.log_event("COMPLETE", f"All {state.total_configs} validated")
            state.save()
            break

        time.sleep(MONITOR_POLL_S)


# ── Mode: --status ─────────────────────────────────────────────────────────

def show_status(cfg, vm_configs):
    """One-shot status: read state + GCS, print VM-aware status."""
    module = None
    try:
        module = load_exp_module(cfg)
    except Exception:
        pass

    if module:
        state = SweepState(cfg, module, vm_configs)
        state.initialize()
        state.refresh()
        print(state.status_report())
    else:
        # Fallback: no module available, use raw state file
        sp = state_path(cfg)
        raw_state = {}
        if os.path.isfile(sp):
            with open(sp) as f:
                raw_state = json.load(f)

        validated = scan_validated_dir(cfg)
        total = raw_state.get('total', '?')

        print(f"[status] Experiment: {cfg['EXP_NAME']}")
        print(f"[status] Validated: {len(validated)}/{total}")
        print(f"[status] Failed: {len(raw_state.get('failed', []))}")
        print()

        for vc in vm_configs:
            name = vc['TPU_NAME']
            bucket = vc['BUCKET']
            hbs = read_heartbeats(bucket, cfg, name)
            done = read_done_receipts(bucket, cfg)

            assign_raw = gcs_read(f"{bucket}/coord/{cfg['EXP_NAME']}/assignments/{name}.json")
            n_assigned = 0
            if assign_raw:
                try:
                    n_assigned = len(json.loads(assign_raw))
                except json.JSONDecodeError:
                    pass

            dead = is_vm_dead(hbs)
            status_str = "DEAD" if dead else "ALIVE"
            n_done = len(done)

            print(f"  {name}: {status_str}  assigned={n_assigned}  done_on_gcs={n_done}")
            for hb in hbs:
                age_min = (time.time() - hb.get('timestamp', 0)) / 60
                stale_mark = " STALE" if is_stale(hb) else ""
                print(f"    {hb.get('worker_id', '?')}: label={hb.get('label', '?')} "
                      f"step={hb.get('step', '?')} age={age_min:.0f}min{stale_mark}")


# ── Mode: --sweep (worker) ─────────────────────────────────────────────────

def write_heartbeat(prefix, worker_id, step, label, tpu_name, status="running"):
    """Write a JSON heartbeat to GCS."""
    hb = {
        'worker_id': worker_id,
        'vm': tpu_name,
        'timestamp': time.time(),
        'step': step,
        'label': label,
        'status': status,
    }
    path = f"{prefix}/heartbeat/{worker_id}.json"
    gcs_write(path, json.dumps(hb))


def run_with_heartbeat(cmd, prefix, worker_id, label, tpu_name, log_path):
    """Run a subprocess with step-coupled heartbeat.

    Heartbeat only fires when training produces stdout (step-coupled).
    If training deadlocks -> no stdout -> no heartbeat -> coordinator detects.
    """
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
        preexec_fn=os.setpgrp  # new process group — killpg catches all XLA children
    )
    # Capture pgid immediately — proc.pid may be gone by cleanup time
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None

    step_ref = [0]   # mutable ref shared with heartbeat thread
    captured_run_name = None
    step_re = re.compile(r'step (\d+)/')
    done_re = re.compile(r'^DONE: (.+)$')

    # Background heartbeat thread — decoupled from stdout so silent training doesn't stale out
    hb_stop = threading.Event()
    def _hb_loop():
        write_heartbeat(prefix, worker_id, 0, label, tpu_name, status="xla_compile")
        while not hb_stop.is_set():
            hb_stop.wait(HEARTBEAT_INTERVAL)
            if hb_stop.is_set():
                break
            status = "training" if step_ref[0] > 0 else "xla_compile"
            try:
                write_heartbeat(prefix, worker_id, step_ref[0], label, tpu_name, status=status)
            except Exception:
                pass
    hb_thread = threading.Thread(target=_hb_loop, daemon=True)
    hb_thread.start()

    with open(log_path, 'w') as log:
        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)
            log.write(line)

            m = step_re.search(line)
            if m:
                new_step = int(m.group(1))
                if new_step > 0 and step_ref[0] == 0:
                    # First real step — XLA compile done
                    write_heartbeat(prefix, worker_id, new_step, label, tpu_name, status="training")
                step_ref[0] = new_step

            dm = done_re.search(line.strip())
            if dm:
                captured_run_name = dm.group(1).strip()

    hb_stop.set()
    hb_thread.join(timeout=5)
    proc.wait()

    # Kill entire process group — catches orphaned torch_xla children
    import signal as _signal
    if pgid is not None:
        try:
            os.killpg(pgid, _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # Final heartbeat
    write_heartbeat(prefix, worker_id, step_ref[0], label, tpu_name)
    return proc.returncode, captured_run_name


def upload_result(label, prefix, work_dir, run_name=None):
    """Upload result JSON to GCS (two-phase: tmp_ -> final).

    Searches outputs/ for matching result files and uploads the summary JSON.
    Uses run_name (captured from training stdout) for reliable file matching.
    Falls back to label-based glob if run_name not available.
    """
    import glob as globmod
    # Find result file — prefer run_name (exact match), fall back to label (substring)
    result_files = []
    if run_name:
        result_files = globmod.glob(os.path.join(work_dir, f'outputs/**/{run_name}.json'), recursive=True)
    if not result_files:
        result_files = globmod.glob(os.path.join(work_dir, f'outputs/**/*{label}*.json'), recursive=True)
    if not result_files:
        # Last resort: find most recently modified .json in outputs/
        all_json = globmod.glob(os.path.join(work_dir, 'outputs/**/*.json'), recursive=True)
        if all_json:
            all_json.sort(key=os.path.getmtime, reverse=True)
            result_files = [all_json[0]]
    if not result_files:
        return False

    result_file = result_files[0]

    # Stage to tmp_<label>/
    tmp_path = f"{prefix}/results/tmp_{label}/summary.json"
    if not gcs_copy(result_file, tmp_path):
        return False

    # Commit: copy to final location
    final_path = f"{prefix}/results/{label}/summary.json"
    if not gcs_copy(tmp_path, final_path):
        return False

    # Clean up staging
    gcs_delete_prefix(f"{prefix}/results/tmp_{label}")
    return True


def cleanup_checkpoint(cfg, label, proc_idx):
    """Delete rolling checkpoint after a config run completes.

    Checkpoints live in /tmp/ or CHECKPOINT_DIR. Rolling = only 1 per process.
    Delete to free disk space (~2.6GB per checkpoint).
    """
    import glob as globmod
    import shutil

    ckpt_dir = os.environ.get('CHECKPOINT_DIR', '/tmp')
    # Common patterns: /tmp/checkpoint_*, /tmp/<exp>_checkpoint_*
    patterns = [
        os.path.join(ckpt_dir, f'checkpoint*'),
        os.path.join(ckpt_dir, f'{cfg["EXP_NAME"]}*checkpoint*'),
        os.path.join(ckpt_dir, f'*{label}*checkpoint*'),
    ]
    cleaned = 0
    for pattern in patterns:
        for path in globmod.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                cleaned += 1
            except OSError:
                pass
    if cleaned:
        print(f"[sweep] Cleaned {cleaned} checkpoint files for {label}")


def worker_sweep(cfg, module, worker_id, tpu_name, proc_idx, num_procs):
    """Worker: read assignment, execute configs. Loops until assignment deleted (Patch 1)."""
    bucket = os.environ.get('BUCKET', '')
    if not bucket:
        print("[sweep] ERROR: BUCKET env var not set", file=sys.stderr)
        sys.exit(1)

    prefix = f"{bucket}/coord/{cfg['EXP_NAME']}"
    my_done_labels = set()  # Only labels THIS worker completed (not from GCS receipts)
    work_dir = os.getcwd()

    print(f"[sweep] Worker {worker_id} (proc {proc_idx}/{num_procs}) on {tpu_name}", flush=True)
    print(f"[sweep] Bucket: {bucket}", flush=True)
    print(f"[sweep] Work dir: {work_dir}", flush=True)

    # Helper: upload last N lines of this worker's log to GCS for remote debugging
    def upload_log_tail(msg=""):
        """Upload last 100 lines of worker log to GCS so blocklab can debug without SSH."""
        log_file = f"/tmp/{cfg['EXP_NAME']}_{proc_idx}.log"
        try:
            lines = []
            if os.path.exists(log_file):
                with open(log_file) as f:
                    lines = f.readlines()[-100:]
            content = f"=== {worker_id} log tail ({msg}) @ {time.strftime('%H:%M:%S')} ===\n" + "".join(lines)
            gcs_write(f"{prefix}/logs/{worker_id}.log", content)
        except Exception as e:
            print(f"[sweep] Warning: log upload failed: {e}", flush=True)

    # Write initial heartbeat so dashboard knows we're alive
    write_heartbeat(prefix, worker_id, 0, "starting", tpu_name, status="starting")

    while True:
        # Read this VM's assignment (coordinator may update it with rebalanced configs)
        print(f"[sweep] Reading assignment from {prefix}/assignments/{tpu_name}.json ...", flush=True)
        raw = gcs_read(f"{prefix}/assignments/{tpu_name}.json")
        if raw is None:
            # Assignment file deleted -> coordinator says we're done (or we're dead)
            print(f"[sweep] Assignment file gone. Exiting.", flush=True)
            upload_log_tail("assignment_gone")
            break

        try:
            assignment = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[sweep] Failed to parse assignment JSON. Retrying in 60s...", flush=True)
            upload_log_tail("bad_json")
            time.sleep(60)
            continue

        # Static partition: this proc takes every num_procs-th config
        my_configs = assignment[proc_idx::num_procs]
        all_labels = [item['label'] for item in my_configs]
        print(f"[sweep] Assignment has {len(assignment)} total configs, my partition ({proc_idx}/{num_procs}): {len(my_configs)} configs = {all_labels}", flush=True)

        did_work = False
        for item in my_configs:
            label = item['label']
            overrides = item['overrides']

            # Skip if this worker already completed it
            if label in my_done_labels:
                print(f"[sweep] Skipping {label} (already done locally)", flush=True)
                continue
            done_path = f"{prefix}/done/{label}.done"
            if gcs_exists(done_path):
                print(f"[sweep] Skipping {label} (done receipt exists on GCS)", flush=True)
                my_done_labels.add(label)
                continue

            did_work = True
            print(f"\n[sweep] === Starting: {label} (proc {proc_idx}) ===", flush=True)
            upload_log_tail(f"starting_{label}")

            # v5e memory fix: halve batch_size, double gradient_accumulation_steps
            # Preserves effective BS: (bs/2) * (ga*2) == bs * ga
            accel_type = os.environ.get('ACCELERATOR_TYPE', '')
            if 'v5' in accel_type or 'v5' in tpu_name:
                overrides = list(overrides)  # copy to avoid mutating assignment
                cur_bs = 8   # config.yaml default
                cur_ga = 16  # config.yaml default
                for o in overrides:
                    if o.startswith('training.batch_size='):
                        cur_bs = int(o.split('=')[1])
                    elif o.startswith('training.gradient_accumulation_steps='):
                        cur_ga = int(o.split('=')[1])
                new_bs = max(1, cur_bs // 2)
                new_ga = cur_ga * 2
                overrides.append(f'training.batch_size={new_bs}')
                overrides.append(f'training.gradient_accumulation_steps={new_ga}')
                print(f"  [v5e] Memory fix: bs {cur_bs}->{new_bs}, ga {cur_ga}->{new_ga}, "
                      f"eff_bs {cur_bs*cur_ga}->{new_bs*new_ga} (unchanged)", flush=True)

            # Clean slate: delete orphaned staging
            gcs_delete_prefix(f"{prefix}/results/tmp_{label}")

            # Build command
            if hasattr(module, 'build_command'):
                cmd = module.build_command(overrides)
            else:
                # Fallback: use run_single pattern
                model_path = os.environ.get('MODEL_PATH')
                if model_path and os.path.isdir(model_path):
                    overrides = overrides + [f'model.name={model_path}']
                script_dir = os.path.dirname(os.path.abspath(module.__file__))
                script = os.path.join(script_dir, 'train_tpu_v2.py')
                cmd = [sys.executable, script] + overrides

            log_path = f"/tmp/{cfg['EXP_NAME']}_{proc_idx}_{label}.log"

            # Run with step-coupled heartbeat
            rc, run_name = run_with_heartbeat(cmd, prefix, worker_id, label, tpu_name, log_path)

            if rc == 0:
                # Two-phase commit: upload result, then write done receipt
                print(f"[sweep] Training done for {label}, uploading result...", flush=True)
                write_heartbeat(prefix, worker_id, 0, label, tpu_name, status="uploading")
                uploaded = upload_result(label, prefix, work_dir, run_name=run_name)
                gcs_write(f"{prefix}/done/{label}.done", f"{worker_id} {time.time()}")
                my_done_labels.add(label)

                # Clean up rolling checkpoint (free disk space)
                cleanup_checkpoint(cfg, label, proc_idx)

                upload_status = "uploaded" if uploaded else "done (no result file found)"
                print(f"[sweep] Completed: {label} ({upload_status})", flush=True)
                upload_log_tail(f"completed_{label}")
            else:
                print(f"[sweep] FAILED: {label} (exit code {rc})", flush=True)
                write_heartbeat(prefix, worker_id, 0, label, tpu_name, status=f"failed_rc{rc}")
                # Clean up checkpoint even on failure
                cleanup_checkpoint(cfg, label, proc_idx)
                # Still mark done (with failure) so coordinator can detect and retry
                gcs_write(f"{prefix}/done/{label}.done", f"{worker_id} {time.time()} failed_rc{rc}")
                my_done_labels.add(label)
                upload_log_tail(f"failed_{label}")

        if not did_work:
            # All assigned configs done, wait for potential rebalanced work (Patch 1)
            write_heartbeat(prefix, worker_id, 0, "idle", tpu_name, status="idle")
            print(f"[sweep] All {len(my_configs)} assigned configs done (done_labels={my_done_labels}). Waiting for rebalanced work...", flush=True)
            upload_log_tail("idle")
            time.sleep(60)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Centralized push coordinator for TPU sweeps')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--init', action='store_true',
                      help='(blocklab) Build configs, distribute to VMs, push assignments')
    mode.add_argument('--monitor', action='store_true',
                      help='(blocklab) Long-running loop: pull results, validate, rebalance')
    mode.add_argument('--sweep', action='store_true',
                      help='(TPU VM) Worker mode: read assignment, execute configs')
    mode.add_argument('--status', action='store_true',
                      help='(blocklab) One-shot status report')
    mode.add_argument('--dry-run', action='store_true',
                      help='Print all configs without running')
    mode.add_argument('--preflight', action='store_true',
                      help='Run experiment preflight')
    mode.add_argument('--enqueue', type=str, metavar='EXP_NAME',
                      help='Add experiment to queue')
    mode.add_argument('--dequeue', type=str, metavar='EXP_NAME',
                      help='Remove experiment from queue')
    mode.add_argument('--queue-status', action='store_true',
                      help='Show experiment queue')
    mode.add_argument('--queue-monitor', action='store_true',
                      help='Process experiment queue (long-running)')

    # Worker args
    parser.add_argument('--worker-id', type=str, default=None,
                        help='Worker ID (auto-generated if not set)')
    parser.add_argument('--proc-idx', type=int, default=0,
                        help='(sweep) This process index (0-based)')
    parser.add_argument('--num-procs', type=int, default=1,
                        help='(sweep) Total processes on this VM')

    args = parser.parse_args()

    # Queue commands don't require EXP env var
    if args.enqueue:
        queue = ExperimentQueue()
        queue.enqueue(args.enqueue)
        return
    elif args.dequeue:
        queue = ExperimentQueue()
        queue.dequeue(args.dequeue)
        return
    elif args.queue_status:
        queue = ExperimentQueue()
        queue.status_report()
        return
    elif args.queue_monitor:
        queue_monitor()
        return

    cfg = load_exp_config()

    if args.init:
        module = load_exp_module(cfg)
        vm_configs = load_vm_configs()
        init_experiment(cfg, module, vm_configs)

    elif args.monitor:
        module = load_exp_module(cfg)
        vm_configs = load_vm_configs()
        monitor_experiment(cfg, module, vm_configs)

    elif args.sweep:
        module = load_exp_module(cfg)
        tpu_name = os.environ.get('TPU_NAME', 'local')
        worker_id = args.worker_id
        if not worker_id:
            chip = os.environ.get('TPU_VISIBLE_CHIPS', '0')
            worker_idx = os.environ.get('WORKER_IDX', '0')
            worker_id = f"{tpu_name}_{worker_idx}_{chip}"
        worker_sweep(cfg, module, worker_id, tpu_name, args.proc_idx, args.num_procs)

    elif args.status:
        vm_configs = load_vm_configs()
        show_status(cfg, vm_configs)

    elif args.preflight:
        module = load_exp_module(cfg)
        module.preflight()

    elif args.dry_run:
        module = load_exp_module(cfg)
        configs = module.build_configs()
        print(f"[dry-run] {len(configs)} configs:")
        for i, (label, overrides) in enumerate(configs):
            print(f"  {i + 1:3d}. {label}")
            for o in overrides:
                print(f"       {o}")


if __name__ == '__main__':
    main()
