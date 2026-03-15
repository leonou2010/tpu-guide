"""
gcs.py — GCS operations layer for pull-based coordinator.

Atomic task claiming, heartbeats, result management.
Uses gsutil/gcloud CLI (no Python storage library needed).
"""

import json
import os
import random
import subprocess
import time

CONTROL_PLANE = os.environ.get(
    'CONTROL_PLANE',
    'gs://gcp-researchcredits-blocklab-europe-west4/coord_v2'
)

# ── CLI detection ────────────────────────────────────────────────────────

_USE_GSUTIL = None

def _use_gsutil():
    global _USE_GSUTIL
    if _USE_GSUTIL is None:
        try:
            r = subprocess.run(['gcloud', 'storage', '--help'],
                               capture_output=True, text=True, timeout=5)
            _USE_GSUTIL = r.returncode != 0
        except Exception:
            _USE_GSUTIL = True
    return _USE_GSUTIL


# ── Primitives ───────────────────────────────────────────────────────────

def gcs_write(path, content):
    try:
        # Both paths use input=content — avoids heredoc shell escaping issues that
        # produce silent 0-byte uploads when content contains special characters.
        if _use_gsutil():
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
    except Exception:
        return False


def gcs_read(path):
    try:
        cmd = ['gsutil', 'cat', path] if _use_gsutil() else ['gcloud', 'storage', 'cat', path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def gcs_read_batch(paths, max_workers=20):
    """Read multiple GCS paths in parallel. Returns dict: {path: content_or_None}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(gcs_read, p): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                results[p] = fut.result()
            except Exception:
                results[p] = None
    return results


def gcs_list(prefix):
    try:
        cmd = ['gsutil', 'ls', f'{prefix}/'] if _use_gsutil() else ['gcloud', 'storage', 'ls', f'{prefix}/']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return [l.strip().rstrip('/') for l in result.stdout.strip().split('\n') if l.strip()]
        return []
    except Exception:
        return []


def gcs_exists(path):
    try:
        cmd = ['gsutil', 'ls', path] if _use_gsutil() else ['gcloud', 'storage', 'ls', path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and result.stdout.strip() != ''
    except Exception:
        return False


def gcs_write_atomic(final_path, content, tmp_suffix="_tmp"):
    """Metaflow-style atomic write: write to tmp path then rename.
    Prevents partial reads during concurrent access.
    Returns True if successful, False otherwise.
    """
    tmp_path = final_path + tmp_suffix
    if not gcs_write(tmp_path, content):
        return False
    return gcs_move(tmp_path, final_path)


def gcs_write_if_new(path, content):
    """Atomic create: write to GCS only if the object does not already exist.

    Uses the GCS x-goog-if-generation-match:0 precondition so the
    existence-check and write are a single server-side RPC — no TOCTOU gap.
    Returns True if this call created the object (this caller won).
    Returns False if the object already existed (another caller won).
    """
    try:
        if _use_gsutil():
            proc = subprocess.run(
                ['gsutil', '-h', 'x-goog-if-generation-match:0', 'cp', '-', path],
                input=content, capture_output=True, text=True, timeout=30
            )
        else:
            proc = subprocess.run(
                ['gcloud', 'storage', 'cp', '--if-generation-match=0', '-', path],
                input=content, capture_output=True, text=True, timeout=30
            )
        return proc.returncode == 0
    except Exception:
        return False


def gcs_move(src, dst):
    """Move/rename a GCS object (copy then delete)."""
    try:
        if _use_gsutil():
            r = subprocess.run(['gsutil', 'mv', src, dst],
                               capture_output=True, text=True, timeout=30)
        else:
            r = subprocess.run(['gcloud', 'storage', 'mv', src, dst],
                               capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def gcs_delete(path):
    try:
        cmd = ['gsutil', 'rm', path] if _use_gsutil() else ['gcloud', 'storage', 'rm', path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        pass


def gcs_copy(src, dst):
    try:
        cmd = ['gsutil', 'cp', src, dst] if _use_gsutil() else ['gcloud', 'storage', 'cp', src, dst]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def gcs_delete_prefix(prefix):
    try:
        if _use_gsutil():
            subprocess.run(['gsutil', '-m', 'rm', '-r', f'{prefix}/'],
                           capture_output=True, text=True, timeout=30)
        else:
            subprocess.run(['gcloud', 'storage', 'rm', '-r', f'{prefix}/'],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        pass


# ── Task operations ──────────────────────────────────────────────────────

def claim_task(worker_id):
    """Atomically claim one task from pending/. Returns task dict or None.

    Uses gcs_write_if_new() (generation-match=0) so the write to running/
    is a single atomic RPC — at most one concurrent worker can win per task.
    No post-write verify read is needed.
    """
    pending = gcs_list(f"{CONTROL_PLANE}/pending")
    if not pending:
        return None

    random.shuffle(pending)  # spread load, reduce contention

    for task_path in pending:
        task_id = os.path.basename(task_path).replace('.json', '')

        # Read task spec
        raw = gcs_read(task_path)
        if raw is None:
            continue

        try:
            task = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Atomic claim: create running/<task_id>.json only if it doesn't exist.
        # If another worker already claimed this task, gcs_write_if_new returns False
        # and we move on to the next candidate — no duplicate work possible.
        claim = {**task, 'worker_id': worker_id, 'claimed_at': time.time()}
        if not gcs_write_if_new(f"{CONTROL_PLANE}/running/{task_id}.json", json.dumps(claim)):
            continue  # lost the race — another worker claimed first

        # Won the claim — remove from pending (best-effort, non-critical)
        gcs_delete(task_path)
        return task

    return None


def _gcs_write_with_retry(path, content, retries=3, backoff=2.0):
    """Write to GCS with retry/backoff. Returns True only if confirmed written."""
    for attempt in range(retries):
        if gcs_write(path, content):
            return True
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    return False


def complete_task(task_id, worker_id, result_summary=None):
    """Mark task as completed. Lossless: write-confirmed → then delete running/."""
    completion = {
        'task_id': task_id,
        'worker_id': worker_id,
        'completed_at': time.time(),
        'result': result_summary or {},
    }
    dest = f"{CONTROL_PLANE}/completed/{task_id}.json"
    if not _gcs_write_with_retry(dest, json.dumps(completion)):
        # Write failed after retries — leave running/ intact so monitor can reclaim
        print(f"[complete_task] ERROR: failed to write {dest} after retries — NOT deleting running/")
        return
    gcs_delete(f"{CONTROL_PLANE}/running/{task_id}.json")
    # Also clean up from pending/ in case task was reclaimed while we were training
    gcs_delete(f"{CONTROL_PLANE}/pending/{task_id}.json")


def fail_task(task_id, worker_id, error_msg, retry=True, max_retries=10):
    """Handle failed task. Lossless: write-confirmed → then delete running/."""
    # Read current task from running/
    raw = gcs_read(f"{CONTROL_PLANE}/running/{task_id}.json")
    if raw is None:
        return

    try:
        task = json.loads(raw)
    except json.JSONDecodeError:
        gcs_delete(f"{CONTROL_PLANE}/running/{task_id}.json")
        return

    retries = task.get('retries', 0) + 1

    if retry and retries <= max_retries:
        # Back to pending with incremented retry count
        task.pop('worker_id', None)
        task.pop('claimed_at', None)
        task['retries'] = retries
        task['last_error'] = error_msg
        dest = f"{CONTROL_PLANE}/pending/{task_id}.json"
    else:
        # Permanently failed
        task['retries'] = retries
        task['last_error'] = error_msg
        task['failed_at'] = time.time()
        dest = f"{CONTROL_PLANE}/failed/{task_id}.json"

    if not _gcs_write_with_retry(dest, json.dumps(task)):
        # Write failed after retries — leave running/ intact so monitor can reclaim
        print(f"[fail_task] ERROR: failed to write {dest} after retries — NOT deleting running/")
        return
    gcs_delete(f"{CONTROL_PLANE}/running/{task_id}.json")


def write_heartbeat(worker_id, task_id, step, label, status="training"):
    """Write heartbeat to GCS."""
    import os as _os
    hb = {
        'worker_id': worker_id,
        'task_id': task_id,
        'timestamp': time.time(),
        'step': step,
        'label': label,
        'status': status,
        'zone': _os.environ.get('ZONE', ''),
        'tpu_name': _os.environ.get('TPU_NAME', ''),
    }
    gcs_write(f"{CONTROL_PLANE}/heartbeats/{worker_id}.json", json.dumps(hb))


def read_heartbeat(worker_id):
    """Read a worker's heartbeat. Returns dict or None."""
    raw = gcs_read(f"{CONTROL_PLANE}/heartbeats/{worker_id}.json")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


# ── Monitor operations ───────────────────────────────────────────────────

def reclaim_stale(stale_ttl_s=300, startup_grace_s=2700, max_task_age_s=14400):
    """Find running tasks with stale heartbeats, move back to pending.

    stale_ttl_s: seconds without heartbeat before declaring dead (5 min)
    startup_grace_s: extra grace for XLA compile phase (45 min)
    max_task_age_s: hard upper bound on claimed_at age (4h) — reclaim regardless of heartbeat
    """
    running_paths = gcs_list(f"{CONTROL_PLANE}/running")
    reclaimed = 0

    for path in running_paths:
        raw = gcs_read(path)
        if not raw:
            continue
        try:
            task = json.loads(raw)
        except json.JSONDecodeError:
            continue

        worker_id = task.get('worker_id')
        task_id = task.get('task_id')
        if not worker_id or not task_id:
            continue

        now = time.time()

        # Hard limit: claimed_at max age — catches zombies even with live heartbeats.
        # A task in running/ for >4h is stuck regardless of what the heartbeat says.
        claim_age = now - task.get('claimed_at', 0)
        if claim_age > max_task_age_s:
            print(f"[reclaim] {task_id} (worker {worker_id}) — claimed {claim_age:.0f}s ago (>{max_task_age_s}s), force-reclaiming zombie")
            fail_task(task_id, worker_id, "max_task_age_exceeded", retry=True)
            gcs_delete(f"{CONTROL_PLANE}/heartbeats/{worker_id}.json")
            reclaimed += 1
            continue

        hb = read_heartbeat(worker_id)

        if hb:
            age = now - hb.get('timestamp', 0)
            status = hb.get('status', '')
            # Worker moved on to a different task — this one is orphaned
            hb_task = hb.get('task_id', '')
            if hb_task and hb_task != task_id and age <= stale_ttl_s:
                print(f"[reclaim] {task_id} (worker {worker_id}) — worker moved to {hb_task}, reclaiming")
                fail_task(task_id, worker_id, "worker_moved_on", retry=True)
                reclaimed += 1
                continue
            # Extra grace for XLA compilation
            ttl = startup_grace_s if status in ('starting', 'xla_compile') else stale_ttl_s
            if age <= ttl:
                continue  # still alive
        else:
            # No heartbeat — check claim age
            if claim_age <= startup_grace_s:
                continue  # just started, give it time

        # Worker is dead — reclaim
        print(f"[reclaim] {task_id} (worker {worker_id}) — heartbeat stale, reclaiming")
        fail_task(task_id, worker_id, "heartbeat_stale", retry=True)
        gcs_delete(f"{CONTROL_PLANE}/heartbeats/{worker_id}.json")
        reclaimed += 1

    return reclaimed


def get_queue_counts():
    """Return (pending, running, completed, failed) counts."""
    return (
        len(gcs_list(f"{CONTROL_PLANE}/pending")),
        len(gcs_list(f"{CONTROL_PLANE}/running")),
        len(gcs_list(f"{CONTROL_PLANE}/completed")),
        len(gcs_list(f"{CONTROL_PLANE}/failed")),
    )
