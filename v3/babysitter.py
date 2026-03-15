#!/usr/bin/env python3
"""
babysitter.py — Pull-based TPU worker.

Runs on each TPU VM. Detects chips, claims tasks from GCS pending/,
runs training, writes completion. Fully autonomous — no central push needed.

Usage:
    flock -n /tmp/tpu_babysitter.lock python3 -u babysitter.py

Env vars (set by deploy.sh):
    CONTROL_PLANE  — GCS path for task state (default: gs://..../coord_v2)
    BUCKET         — Regional GCS bucket for results/XLA cache
    MODEL_PATH     — Local path to model weights
    TPU_NAME       — VM name (for worker ID)
    WANDB_MODE     — online/disabled
"""

import glob as globmod
import json
import os
import argparse
import random
import re
import shutil
import subprocess
import sys
import threading
import time

# Add pull/ to path for gcs module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import (claim_task, complete_task, fail_task, write_heartbeat,
                 gcs_write, gcs_copy, gcs_delete_prefix, gcs_exists, CONTROL_PLANE)

DRAIN_FLAG_PATH = f"{CONTROL_PLANE}/flags/drain.json"

# ── Config ───────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = 120  # 2 min (was 5 min — reduces false stale reclaims)
IDLE_POLL_INTERVAL = 30   # seconds between checking for new tasks when idle
MAX_IDLE_CYCLES = 0       # 0 = never exit (run forever)

# Serialize PJRT initialization across chip workers.
# v6e-8: all chips on the same host share a PJRT session setup.
# If multiple chips call initialize_singleprocess() simultaneously, they hit
# PRE_START_SESSION_BARRIER deadlock. Fix: only one chip at a time may start training.
# Hold lock for 90s — enough for PJRT to initialize before next chip starts.
_TPU_INIT_LOCK = threading.Lock()
_TPU_INIT_STAGGER_S = 90

TPU_NAME = ""  # resolved at runtime in main() after parsing CLI/env

# ── Preemption detection (SkyPilot pattern) ───────────────────────────────────

_preemption_detected = threading.Event()


def _check_preemption():
    """GCP spot VM preemption signal via metadata server.

    Checks two endpoints (SkyPilot pattern):
    1. abort_signal — fires ~30s before termination (earliest warning)
    2. instance/preempted — fires at termination

    Runs as background thread.
    """
    import urllib.request
    headers = {"Metadata-Flavor": "Google"}
    base = "http://metadata.google.internal/computeMetadata/v1/instance"
    abort_url = f"{base}/guest-attributes/google.compute/abort_signal"
    preempted_url = f"{base}/preempted"

    while not _preemption_detected.is_set():
        triggered = False

        # Check abort_signal first (earlier warning, ~30s before termination)
        try:
            req = urllib.request.Request(abort_url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                val = resp.read().decode().strip()
                if val:  # attribute present = preemption imminent
                    print(f"[babysitter] ⚠️  PREEMPTION abort_signal detected", flush=True)
                    triggered = True
        except Exception:
            pass

        # Fallback: check instance/preempted flag
        if not triggered:
            try:
                req = urllib.request.Request(preempted_url, headers=headers)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    val = resp.read().decode().strip()
                    if val.upper() == "TRUE":
                        print(f"[babysitter] ⚠️  PREEMPTION instance/preempted=TRUE", flush=True)
                        triggered = True
            except Exception:
                pass

        if triggered:
            _preemption_detected.set()
            try:
                gcs_write(f"{CONTROL_PLANE}/telemetry/{TPU_NAME}_preempted.json",
                          json.dumps({"tpu_name": TPU_NAME, "timestamp": time.time(),
                                      "status": "PREEMPTED"}))
            except Exception:
                pass
            return

        time.sleep(10)


def start_preemption_monitor():
    """Start background preemption detection thread."""
    t = threading.Thread(target=_check_preemption, name="preemption_monitor", daemon=True)
    t.start()
    return t


def _apply_cli_env_overrides(argv=None):
    """Apply CLI overrides for deploy_ue1d.sh / deploy_uc2b.sh.

    Those scripts historically passed --tpu-name/--zone/--bucket/--exp but
    babysitter only read env vars. This hook bridges that gap so ue1d can
    upload results and write heartbeats under the expected VM name.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-name", dest="tpu_name", default=None)
    parser.add_argument("--zone", dest="zone", default=None)
    parser.add_argument("--bucket", dest="bucket", default=None)
    parser.add_argument("--exp", dest="exp", default=None)
    parser.add_argument("--control-plane", dest="control_plane", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    args, _unknown = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.tpu_name:
        os.environ["TPU_NAME"] = args.tpu_name
    if args.zone:
        os.environ["ZONE"] = args.zone
    if args.bucket:
        os.environ["BUCKET"] = args.bucket
    if args.exp:
        # Used as a fallback when tasks omit "experiment".
        os.environ["EXP"] = args.exp
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    if args.control_plane:
        # gcs.py snapshots CONTROL_PLANE at import time. Update both env + module globals.
        os.environ["CONTROL_PLANE"] = args.control_plane
        try:
            import gcs as _gcs
            _gcs.CONTROL_PLANE = args.control_plane
        except Exception:
            pass
        global CONTROL_PLANE, DRAIN_FLAG_PATH
        CONTROL_PLANE = args.control_plane
        DRAIN_FLAG_PATH = f"{CONTROL_PLANE}/flags/drain.json"


def _resolve_tpu_name():
    """Resolve TPU_NAME from env, else derive from hostname."""
    global TPU_NAME
    tpu_name = os.environ.get("TPU_NAME", "")
    if tpu_name and tpu_name not in ("unknown", "none", ""):
        TPU_NAME = tpu_name
        return

    import socket as _socket
    derived = _socket.gethostname().split(".")[0]
    if derived and derived not in ("localhost", ""):
        print(f"[babysitter] WARNING: TPU_NAME not set — derived from hostname: {derived}", flush=True)
        os.environ["TPU_NAME"] = derived
        TPU_NAME = derived
        return

    print("[babysitter] FATAL: TPU_NAME is unset and hostname is unusable. "
          "Set TPU_NAME env var (or pass --tpu-name). Exiting.", flush=True)
    sys.exit(1)


# ── Sanity check ─────────────────────────────────────────────────────────

def _bootstrap_libtpu():
    """Detect libtpu.so and set TPU_LIBRARY_PATH if not already set.

    _XLAC (C extension) reads TPU_LIBRARY_PATH. Without it, torch_xla device
    init fails even when 'import torch_xla' succeeds.

    Tries in order:
      1. Existing TPU_LIBRARY_PATH env var
      2. libtpu Python package (get_library_path())

    Sets os.environ['TPU_LIBRARY_PATH'] so all subprocess envs inherit it.
    Returns the path, or '' if not found.
    """
    existing = os.environ.get('TPU_LIBRARY_PATH', '')
    if existing and os.path.isfile(existing):
        print(f"[babysitter] TPU_LIBRARY_PATH already set: {existing}", flush=True)
        return existing

    try:
        import libtpu
        p = libtpu.get_library_path()
        if p and os.path.isfile(p):
            os.environ['TPU_LIBRARY_PATH'] = p
            print(f"[babysitter] TPU_LIBRARY_PATH detected: {p}", flush=True)
            return p
    except Exception as e:
        print(f"[babysitter] libtpu import failed: {e}", flush=True)

    print("[babysitter] WARNING: libtpu.so not found — TPU init will likely fail", flush=True)
    return ''


def verify_environment():
    """Check critical dependencies before starting. Exit if broken."""
    errors = []
    for mod in ['torch', 'hydra', 'omegaconf', 'transformers']:
        try:
            __import__(mod)
        except ImportError:
            errors.append(mod)

    if not os.path.isdir(os.environ.get('MODEL_PATH', '/tmp/SmolLM2-135M')):
        errors.append('MODEL_PATH missing')

    if not (globmod.glob('/dev/accel*') or globmod.glob('/dev/vfio/devices/vfio*')):
        # Only warn, don't fail — some setups use different device paths
        print("[babysitter] WARNING: No /dev/accel* or /dev/vfio* found, using ACCELERATOR_TYPE", flush=True)

    # Hard TPU-init gate: libtpu must be present and xla_device() must succeed.
    # IMPORTANT: run in a subprocess that exits — babysitter's main process must NOT hold
    # the TPU device open (it would conflict when the training subprocess tries to open it).
    _bootstrap_libtpu()
    print("[babysitter] Testing TPU init via subprocess...", flush=True)
    try:
        _tpu_test = subprocess.run(
            [sys.executable, '-c',
             'import torch_xla.core.xla_model as xm; d=xm.xla_device(); print(f"TPU_INIT_OK:{d}")'],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, 'TPU_VISIBLE_CHIPS': '0', 'PJRT_DEVICE': 'TPU',
                 'TPU_LIBRARY_PATH': os.environ.get('TPU_LIBRARY_PATH', '')},
        )
        if 'TPU_INIT_OK' in _tpu_test.stdout:
            dev = _tpu_test.stdout.strip().split('TPU_INIT_OK:')[-1]
            print(f"[babysitter] TPU init OK: {dev}", flush=True)
        else:
            errors.append(f'TPU_INIT_FAILED: {_tpu_test.stderr[:300]}')
    except subprocess.TimeoutExpired:
        errors.append('TPU_INIT_FAILED: timeout (120s)')

    if errors:
        msg = f"FATAL: Environment check failed: {errors}"
        print(msg, flush=True)
        # Upload error to GCS
        gcs_write(f"{CONTROL_PLANE}/logs/env_fail_{TPU_NAME}.log",
                  f"{time.strftime('%H:%M:%S')} {msg}")
        sys.exit(1)

    print(f"[babysitter] Environment OK", flush=True)


# ── Chip detection ───────────────────────────────────────────────────────

def detect_chips():
    """Count TPU chips. v4 uses /dev/accel*, v6e uses /dev/vfio/devices/."""
    # v4: /dev/accel0, /dev/accel1, ...
    accel_devs = globmod.glob('/dev/accel[0-9]*')
    if accel_devs:
        return len(accel_devs)
    # v6e: /dev/vfio/devices/vfio0, vfio1, ...
    vfio_devs = globmod.glob('/dev/vfio/devices/vfio[0-9]*')
    if vfio_devs:
        return len(vfio_devs)
    # Derive from accelerator type
    accel = os.environ.get('ACCELERATOR_TYPE', '')
    if 'v6e-8' in accel:
        return 8
    elif 'v4-8' in accel:
        return 4  # 4 chips (not 8 — 8 cores, 2 per chip)
    elif 'v5litepod-4' in accel or 'v5e' in accel:
        return 4
    # Last fallback: env var
    return int(os.environ.get('CHIPS_PER_HOST', '8'))


# ── v5e OOM fix ──────────────────────────────────────────────────────────

def apply_v5e_fix(overrides):
    """Halve batch_size, double ga for v5e (16GB/chip). Preserves effective BS."""
    accel = os.environ.get('ACCELERATOR_TYPE', '')
    if 'v5' not in accel and 'v5' not in TPU_NAME:
        return overrides

    overrides = list(overrides)
    cur_bs, cur_ga = 8, 16  # defaults
    for o in overrides:
        if o.startswith('training.batch_size='):
            cur_bs = int(o.split('=')[1])
        elif o.startswith('training.gradient_accumulation_steps='):
            cur_ga = int(o.split('=')[1])

    new_bs = max(1, cur_bs // 2)
    new_ga = cur_ga * 2
    overrides.append(f'training.batch_size={new_bs}')
    overrides.append(f'training.gradient_accumulation_steps={new_ga}')
    print(f"  [v5e] OOM fix: bs {cur_bs}->{new_bs}, ga {cur_ga}->{new_ga}", flush=True)
    return overrides


# ── Training execution ───────────────────────────────────────────────────

def _heartbeat_loop(worker_id, task_id, label, step_ref, stop_event):
    """Background heartbeat thread — independent of stdout blocking."""
    while not stop_event.is_set():
        try:
            status = "training" if step_ref[0] > 0 else "xla_compile"
            write_heartbeat(worker_id, task_id, step_ref[0], label, status=status)
        except Exception:
            pass
        stop_event.wait(HEARTBEAT_INTERVAL)


def run_training(task, chip_idx, worker_id):
    """Run one training config. Returns (exit_code, run_name)."""
    label = task['label']
    overrides = apply_v5e_fix(list(task['overrides']))
    work_dir = os.path.expanduser(f"~/sf_bema/experiments/{task['work_dir']}")
    train_script = os.path.join(work_dir, task['train_script'])
    bucket = os.environ.get('BUCKET', '')

    # Add model path if available
    model_path = os.environ.get('MODEL_PATH', '/tmp/SmolLM2-135M')
    if os.path.isdir(model_path):
        overrides.append(f'model.name={model_path}')

    # Per-experiment checkpoint dirs: prevents cross-experiment checkpoint sharing
    exp = task.get('experiment', 'default')
    task_ckpt_dir = f'/tmp/ckpts/{exp}'
    os.makedirs(task_ckpt_dir, exist_ok=True)
    # Per-experiment GCS checkpoint dir (add experiment subfolder to base path)
    gcs_base = os.environ.get('GCS_CHECKPOINT_DIR', '')
    task_gcs_ckpt_dir = f'{gcs_base}/{exp}' if gcs_base else ''

    # TPU_LIBRARY_PATH was set by _bootstrap_libtpu() at startup — just propagate via os.environ
    _libtpu_path = os.environ.get('TPU_LIBRARY_PATH', '')
    if _libtpu_path:
        print(f"[chip{chip_idx}] TPU_LIBRARY_PATH={_libtpu_path}", flush=True)
    else:
        print(f"[chip{chip_idx}] WARNING: TPU_LIBRARY_PATH not in env", flush=True)

    cmd = [sys.executable, '-u', train_script] + overrides
    env = {
        **os.environ,
        'TPU_LIBRARY_PATH': _libtpu_path,  # always explicit — never rely on inheritance alone
        'TPU_VISIBLE_CHIPS': str(chip_idx),
        'TPU_PROCESS_BOUNDS': '1,1,1',
        'TPU_NUM_WORKERS': '1',
        'CHIPS_PER_HOST': '1',
        'LAUNCH_MODE': 'single',  # debug_single_process=True — only mode confirmed working (no child proc libtpu issue)
        'PJRT_DEVICE': 'TPU',
        'PYTHONUNBUFFERED': '1',
        'CHECKPOINT_DIR': task_ckpt_dir,
        'GCS_CHECKPOINT_DIR': task_gcs_ckpt_dir,
        'XLA_PERSISTENT_CACHE_PATH': '/tmp/xla_cache',
        'XLA_COMPILATION_CACHE_PATH': '/tmp/xla_cache',
        'LLVM_NUM_THREADS': '32',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HYDRA_FULL_ERROR': '1',
        # Force WANDB_MODE=disabled — no W&B API key on VMs. Override parent env.
        'WANDB_MODE': 'disabled',
        'TRAIN_LOSS_JSONL': f'/tmp/{task["task_id"]}_train_loss.jsonl',
    }

    print(f"[chip{chip_idx}] Running: {label}", flush=True)
    print(f"[chip{chip_idx}] Dir: {work_dir}", flush=True)
    print(f"[chip{chip_idx}] Cmd: {' '.join(cmd[:3])} + {len(overrides)} overrides", flush=True)

    # Background heartbeat thread — survives stdout blocking between eval intervals
    step_ref = [0]  # mutable ref for thread to read current step
    hb_stop = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(worker_id, task['task_id'], label, step_ref, hb_stop),
        daemon=True
    )
    hb_thread.start()

    log_path = f"/tmp/babysitter_chip{chip_idx}_{label}.log"
    # Serialize PJRT initialization: only one chip starts at a time.
    # v6e PRE_START_SESSION_BARRIER requires chips not to clash at PJRT init.
    print(f"[chip{chip_idx}] Waiting for TPU init lock...", flush=True)
    with _TPU_INIT_LOCK:
        print(f"[chip{chip_idx}] Launching training subprocess...", flush=True)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=work_dir,
            preexec_fn=os.setpgrp  # new process group (SkyPilot pattern): killpg catches all children
        )
        # Hold lock for _TPU_INIT_STAGGER_S to let PJRT initialize before next chip
        time.sleep(_TPU_INIT_STAGGER_S)
    # Capture pgid immediately after fork — proc may be gone by cleanup time
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None

    step_re = re.compile(r'step (\d+)/')
    done_re = re.compile(r'^DONE: (.+)$')
    captured_run_name = None

    with open(log_path, 'w') as log:
        for line in iter(proc.stdout.readline, ''):
            print(f"[chip{chip_idx}] {line}", end='', flush=True)
            log.write(line)

            m = step_re.search(line)
            if m:
                step_ref[0] = int(m.group(1))

            dm = done_re.search(line.strip())
            if dm:
                captured_run_name = dm.group(1).strip()

    # Stop heartbeat thread
    hb_stop.set()
    hb_thread.join(timeout=5)

    proc.wait()

    # Kill entire process group to catch orphan children from torch_xla.launch
    # (SkyPilot pattern: killpg is the only reliable way to kill the full tree)
    import signal
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    return proc.returncode, captured_run_name, task_ckpt_dir


def _resolve_exp(task):
    """Resolve experiment name from task dict. Falls back to task_id prefix if field is None."""
    exp = task.get('experiment')
    if not exp:
        exp = task.get('task_id', 'unknown').split('__')[0]
    return exp


def upload_result(task, run_name, work_dir):
    """Upload result JSON to GCS. Returns result summary dict or None."""
    label = task['label']
    bucket = os.environ.get('BUCKET', '')
    exp = _resolve_exp(task)

    # Find result file
    result_files = []
    if run_name:
        result_files = globmod.glob(os.path.join(work_dir, f'outputs/**/{run_name}.json'), recursive=True)
    if not result_files:
        result_files = globmod.glob(os.path.join(work_dir, f'outputs/**/*{label}*.json'), recursive=True)
    if not result_files:
        all_json = globmod.glob(os.path.join(work_dir, 'outputs/**/*.json'), recursive=True)
        if all_json:
            all_json.sort(key=os.path.getmtime, reverse=True)
            result_files = [all_json[0]]

    if not result_files:
        print(f"  WARNING: No result file found for {label}", flush=True)
        return None

    result_file = result_files[0]
    print(f"  Uploading: {result_file}", flush=True)

    # Upload to regional bucket (heavy data stays regional)
    gcs_path = f"{bucket}/coord/{exp}/results/{label}/summary.json"
    if gcs_copy(result_file, gcs_path):
        # Read result for summary
        try:
            with open(result_file) as f:
                data = json.load(f)
            return data.get('summary', {})
        except Exception:
            return {'uploaded': True}

    return None


def cleanup_checkpoint(run_name, ckpt_dir='/tmp'):
    """Delete rolling checkpoint by exact run_name."""
    if not run_name:
        return
    path = os.path.join(ckpt_dir, f'ckpt_{run_name}.pt')
    try:
        os.remove(path)
    except OSError:
        pass


# ── Chip worker loop ─────────────────────────────────────────────────────

def chip_worker(chip_idx, num_chips):
    """One chip's main loop: claim -> train -> complete -> repeat."""
    worker_id = f"{TPU_NAME}_chip{chip_idx}"
    idle_count = 0

    print(f"[chip{chip_idx}] Worker {worker_id} started (chip {chip_idx}/{num_chips})", flush=True)

    while True:
        # Stop if preemption detected — don't claim new tasks
        if _preemption_detected.is_set():
            print(f"[chip{chip_idx}] Preemption detected — stopping task claiming", flush=True)
            break

        # Drain mode: populate.py --drain sets this flag before --clear.
        # Don't claim new tasks while drain is active — finish current task then pause.
        if gcs_exists(DRAIN_FLAG_PATH):
            write_heartbeat(worker_id, None, 0, "drain", status="idle")
            if idle_count % 10 == 0:
                print(f"[chip{chip_idx}] Drain flag set — not claiming new tasks", flush=True)
            idle_count += 1
            time.sleep(IDLE_POLL_INTERVAL)
            continue

        # Claim a task
        task = claim_task(worker_id)

        if task is None:
            idle_count += 1
            write_heartbeat(worker_id, None, 0, "idle", status="idle")
            if idle_count % 10 == 1:
                print(f"[chip{chip_idx}] No pending tasks. Idle ({idle_count}).", flush=True)
            if MAX_IDLE_CYCLES > 0 and idle_count >= MAX_IDLE_CYCLES:
                print(f"[chip{chip_idx}] Max idle cycles reached. Exiting.", flush=True)
                break
            time.sleep(IDLE_POLL_INTERVAL)
            continue

        idle_count = 0
        task_id = task['task_id']
        label = task['label']
        work_dir = os.path.expanduser(f"~/sf_bema/experiments/{task['work_dir']}")

        print(f"\n[chip{chip_idx}] ═══ CLAIMED: {label} (task_id={task_id}) ═══", flush=True)

        # No pre-training cleanup: per-experiment CHECKPOINT_DIR prevents cross-exp sharing.
        # train_v2_tpu.py load_checkpoint() handles: (1) local resume, (2) GCS download if local missing.

        # Clean staging
        bucket = os.environ.get('BUCKET', '')
        if bucket:
            gcs_delete_prefix(f"{bucket}/coord/{_resolve_exp(task)}/results/tmp_{label}")

        # Run training — returns (rc, run_name, task_ckpt_dir)
        rc, run_name, task_ckpt_dir = run_training(task, chip_idx, worker_id)

        if rc == 0:
            print(f"[chip{chip_idx}] Training DONE for {label}, uploading result...", flush=True)
            write_heartbeat(worker_id, task_id, 0, label, status="uploading")
            result_summary = upload_result(task, run_name, work_dir)
            complete_task(task_id, worker_id, result_summary)
            # Upload per-step train loss JSONL alongside result (survives cleanup_gcs.sh)
            jsonl_local = f'/tmp/{task_id}_train_loss.jsonl'
            if os.path.exists(jsonl_local) and bucket:
                gcs_copy(jsonl_local, f"{bucket}/coord/{_resolve_exp(task)}/results/{label}/train_loss.jsonl")
            cleanup_checkpoint(run_name, task_ckpt_dir)
            print(f"[chip{chip_idx}] ✓ COMPLETED: {label}", flush=True)
        else:
            print(f"[chip{chip_idx}] ✗ FAILED: {label} (rc={rc})", flush=True)
            cleanup_checkpoint(run_name, task_ckpt_dir)

            # Read log for error classification and GCS upload
            log_file_check = f"/tmp/babysitter_chip{chip_idx}_{label}.log"
            last_lines = ''
            wait_s = 10 + random.randint(0, 30)
            try:
                with open(log_file_check) as f:
                    last_lines = f.read()[-4000:]
            except Exception:
                pass

            # Retry classification: infra crash vs code error
            current_retries = task.get('retries', 0)
            infra_fail = (rc == -9) or any(s in last_lines for s in [
                'Killed', 'preempted', 'SIGKILL', 'preemption',
                'libtpu not found', 'libtpu.so',  # missing TPU library — deploy issue, not code bug
                'OSError: [Errno', 'No module named',  # env setup failures
                'TPU initialization failed',  # TPU barrier timeout — transient infra issue
                'Deadline exceeded', 'deadline exceeded',  # gRPC/XLA timeouts
                'PRE_START_SESSION_BARRIER',  # multi-host barrier timeout on single VM
            ])
            code_fail = (rc == 1) and any(s in last_lines for s in ['Traceback', 'Error:', 'RuntimeError']) and not infra_fail
            if infra_fail:
                should_retry = True
                error_label = f"infra_crash rc={rc}"
            elif code_fail and current_retries >= 3:
                should_retry = False
                error_label = f"code_error_permanent rc={rc} retries={current_retries}"
            else:
                should_retry = True
                error_label = f"exit_code={rc}"
            fail_task(task_id, worker_id, error_label, retry=should_retry)

            # Upload failure log for remote diagnosis
            try:
                gcs_write(f"{CONTROL_PLANE}/logs/task_fail_{task_id}.log",
                          f"=== FAILURE: {worker_id} @ {time.strftime('%H:%M:%S')} rc={rc} ===\n" + last_lines)
            except Exception:
                pass

            # Adjust cooldown based on error type
            if 'Device or resource busy' in last_lines or 'vfio' in last_lines.lower():
                print(f"[chip{chip_idx}] Device lock detected. Waiting 90s.", flush=True)
                wait_s = 90
            elif rc == -9:
                print(f"[chip{chip_idx}] SIGKILL detected (OOM?). Waiting 30s.", flush=True)
                wait_s = 30 + random.randint(0, 30)
            time.sleep(wait_s)

        # Upload log tail for remote debugging
        log_file = f"/tmp/babysitter_chip{chip_idx}_{label}.log"
        try:
            with open(log_file) as f:
                tail = f.readlines()[-50:]
            gcs_write(f"{CONTROL_PLANE}/logs/{worker_id}_last.log",
                      f"=== {worker_id} @ {time.strftime('%H:%M:%S')} ===\n" + "".join(tail))
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────────────────

def kill_orphan_training():
    """Kill stale training processes from previous babysitter runs.

    Uses pkill -KILL (sends SIGKILL to entire process group match).
    Covers: torch_xla.launch child processes, orphaned training loops.
    """
    import signal
    patterns = ['train_v2_tpu.py', 'train_tpu.py', 'train_tpu_v2.py']
    for pattern in patterns:
        try:
            # First find what's running (for logging)
            result = subprocess.run(
                ['pgrep', '-f', pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"[babysitter] Found {len(pids)} orphan '{pattern}' — killing process groups", flush=True)
                # Kill each by process group (catches all torch_xla children)
                for pid_str in pids:
                    try:
                        pid = int(pid_str.strip())
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, ValueError):
                        # Process already gone or permission denied — try direct kill
                        try:
                            os.kill(int(pid_str.strip()), signal.SIGKILL)
                        except Exception:
                            pass
        except Exception:
            pass
    # Also nuke any lingering python3 with TPU device args
    try:
        subprocess.run(['pkill', '-KILL', '-f', 'TPU_VISIBLE_CHIPS'],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(3)


def main():
    _apply_cli_env_overrides()
    _resolve_tpu_name()
    kill_orphan_training()
    verify_environment()
    start_preemption_monitor()  # Background GCP preemption detection

    num_chips = detect_chips()
    print(f"[babysitter] TPU={TPU_NAME}, chips={num_chips}", flush=True)
    print(f"[babysitter] Control plane: {CONTROL_PLANE}", flush=True)
    print(f"[babysitter] BUCKET: {os.environ.get('BUCKET', 'NOT SET')}", flush=True)

    if num_chips == 1:
        # Single chip — run directly (no threading)
        chip_worker(0, 1)
    else:
        # Multi-chip — one thread per chip
        threads = []
        for i in range(num_chips):
            t = threading.Thread(target=chip_worker, args=(i, num_chips),
                                 name=f"chip_{i}", daemon=True)
            t.start()
            threads.append(t)
            time.sleep(45)  # stagger starts: TPU init conflicts if multiple xmp.spawn at once

        # Wait for all threads
        for t in threads:
            t.join()

    print("[babysitter] All chip workers exited.", flush=True)


if __name__ == '__main__':
    main()
