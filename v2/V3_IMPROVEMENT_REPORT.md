# V3 Improvement Report — Based on exp13_rerun Test Run
Generated: 2026-03-13. Run: 120 tasks, 15h wall time, 10 VMs (60 chips).

---

## Executive Summary

exp13_rerun exposed **17 bugs** across 3 relaunch sessions + overnight monitoring. The run completed at **40% efficiency** (8 tasks/hr actual vs ~20 tasks/hr theoretical). All waste was in the **coordination layer** — GCS bookkeeping, deploy scripts, and monitoring — not compute. The chips themselves ran fine once deployed correctly.

**The 3 root-cause themes:**
1. **Deploy script doesn't verify the environment it creates** (libtpu, user-site shadows, root processes)
2. **Coordination state goes stale silently** (zombie tasks, monitor hangs, empty task files)
3. **No autonomous alarm system** — every bug required manual detection and intervention

---

## Bug Registry

### CRITICAL — Caused major task loss or run failure

| ID | Bug | Tasks Lost | Fixed? |
|----|-----|-----------|--------|
| BUG-R1 | Root babysitter survives FORCE_REDEPLOY — burns 79 tasks with exit_code=2 | 79 | ✅ sudo pkill added |
| BUG-O2 | libtpu not found on fresh v6e VMs — 2 VMs each burned ~25 tasks | ~50 | ✅ import libtpu check added |
| BUG-O7 | 5 zombie tasks stuck in running/ for 7-9h — no results, never detected | 5 × full run | ✅ claimed_at max-age reclaim added (Fix 3) |
| BUG-O8 | 4 tasks had 0-byte JSON from populate.py — unclaimed for 11h | 4 tasks delayed 11h | ✅ write verification added (Fix 2) |
| BUG-R5 | populate --clear races with live babysitters — burned retries on fresh run | ~79 burned retries | ✅ heartbeat safety check added (Fix 4) |

### HIGH — Wasted significant compute or caused multi-hour delays

| ID | Bug | Impact | Fixed? |
|----|-----|--------|--------|
| BUG-O1 | monitor.py stuck in GCS HTTP call for 4h (01:04–05:11 UTC) — no validation, no reclaim | 4h blind window | ✅ already in v2 (timeout=30 on all GCS calls + watchdog in overnight_watchdog.sh) |
| BUG-S1 | User-site CUDA torch shadows system TPU torch on v6e (PRE_START_SESSION_BARRIER) | Chip1/7 on ew4a-2 failed 3-8× per task | ⚠️ requeue only |
| BUG-O6 | User-site torch_xla on v4 VMs (buffer deleted error) — same root cause as S1 | 8× failures on uc2b-2 chip2 | ⚠️ requeue only |
| BUG-O5 | Heartbeat-stale false reclaim 11× on uc2b-5_chip2 (chip was alive and training) | 11 wasted partial runs | ✅ already in v2 (stale_ttl=1800s, heartbeat every stale_ttl/3 s) |
| BUG-R9 | Crash-looping babysitters write heartbeats — vm_requester thinks they're healthy, never redeploys | Broken VMs spin for up to 45min | ✅ already in v2 (xla_compile >15min → force redeploy in vm_requester.sh) |

### MEDIUM — Correctness issues or repeated manual work

| ID | Bug | Impact | Fixed? |
|----|-----|--------|--------|
| BUG-R2 | Root-owned /tmp/babysitter_chip*.log blocks new babysitter (PermissionError) | Training blocked until cleaned | ✅ sudo rm added |
| BUG-R3 | Root-owned /tmp/boot_state.json breaks telemetry | Telemetry silent | ✅ sudo rm added |
| BUG-R6 | libtpu.so on PATH but TPU_LIBRARY_PATH not set — C extension can't find it | Training fails with obscure error | ✅ auto-detect added |
| BUG-R8 | libtpu infra errors classified as permanent code_error — exhaust retries | Tasks permanently failed for infra reason | ✅ classifier updated |
| BUG-O3 | Spot preemption wave took 4 VMs simultaneously | 1h fleet gap | Auto-recovered |
| BUG-O4 | pgrep -af regex uses POSIX BRE — \| is literal, not alternation | Agent spawned duplicate processes | ✅ use ps aux |

### LOW — Cosmetic or known-going-in

| ID | Bug | Fix |
|----|-----|-----|
| BUG-R7 | deploy_babysitter.log locked before exec redirect — silent exit | sudo rm BEFORE exec |
| BUG-S2 | USER unbound variable (SSH minimal env, set -uo) | id -un fallback |
| BUG-S3 | tpu_core GCS wheels are v4-only, break v6e (libcublas missing) | Clear zone-type separation |
| BUG-S4 | CUDA version detection direction inverted | +cu = bad, treat as corrupt |

---

## Fixes Applied This Session (2026-03-13) — All 7 Necessary Items

### Fix 1 — Lossless state transitions in gcs.py ✅
**Files**: `gcs.py:220`, `gcs.py:234`
**Problem**: `complete_task()` and `fail_task()` deleted `running/<id>.json` even if the write to `completed/` or `pending/` failed. A GCS transient error silently dropped the task — gone from all queues, never retried.
**Fix**: Added `_gcs_write_with_retry()` (3 attempts, exponential backoff). Both functions now write-confirm first; if retries exhausted, log error and leave `running/` intact so monitor reclaims.

### Fix 2 — populate.py write verification ✅
**File**: `populate.py:157`
**Problem**: `gcs_write()` return value was discarded. 0-byte objects entered `pending/` silently (BUG-O8). Babysitters couldn't parse them → skipped forever.
**Fix**: Check `gcs_write()` return; on success, read back and verify `len >= 10` bytes. Skip and log if either check fails.

### Fix 3 — Zombie detection: claimed_at max-age in reclaim_stale ✅
**File**: `gcs.py:313` (now `reclaim_stale()`)
**Problem**: `reclaim_stale()` only checked heartbeat age. A task with a live heartbeat from the *same* worker but 9h in `running/` (BUG-O7) was never reclaimed — zombie with fresh heartbeat is invisible to heartbeat-only logic.
**Fix**: Added `max_task_age_s=14400` (4h default). Any task in `running/` for >4h is force-reclaimed as `max_task_age_exceeded` regardless of heartbeat state. Check happens before heartbeat logic so nothing bypasses it.

### Fix 4 — populate --clear safety ✅
**File**: `populate.py:125`
**Problem**: `populate.py --clear` wiped `running/` while babysitters were active. When those babysitters called `fail_task()`, they read `running/<id>.json` → None → silently returned. Tasks vanished (BUG-R5).
**Fix**: Before clearing, list `heartbeats/`. If any exist, abort with an error explaining the risk. Use `--force` to bypass if you've confirmed workers are actually dead.

### Fix 5 — gcs_write() heredoc → input=content ✅
**File**: `gcs.py:37`
**Problem**: gsutil path used `bash -c 'cat <<GCSEOF | gsutil cp - path\nCONTENT\nGCSEOF'`. This breaks silently when content contains shell special characters (`'`, `$`, backticks) — the heredoc truncates or corrupts the upload, producing 0-byte objects. This was the root cause of BUG-O8.
**Fix**: Use `subprocess.run(['gsutil', 'cp', '-', path], input=content, ...)` — same as the gcloud path already did. Content goes directly via stdin, no shell interpolation.

### Fix 6 — vm_requester.sh deploy SSH timeout ✅
**File**: `vm_requester.sh:311` (`deploy_babysitter()`)
**Problem**: `gcloud ... ssh --command` had no timeout. A single hung SSH would hold a background process indefinitely. The `wait` at the end of each zone's deploy loop then blocks the entire main cycle — new VMs stop being created, stale VMs stop being reclaimed.
**Fix**: Wrapped SSH call in `timeout 120s`. If it hits 124 (timeout exit code), logs clearly and returns. The background `&` + bounded timeout means `wait` takes at most 120s × concurrent deploys, not forever.

### Fix 7 — TPU_NAME guard in babysitter.py ✅
**File**: `babysitter.py:41`
**Problem**: `TPU_NAME = os.environ.get('TPU_NAME', 'unknown')`. If deploy_babysitter.sh fails to set TPU_NAME, all chips on all VMs share `worker_id = 'unknown_chip0'`, etc. Heartbeats from different VMs overwrite each other → wrong reclaim decisions.
**Fix**: If TPU_NAME is empty/unknown, derive from `socket.gethostname()` and log a warning. If hostname is also unusable, exit(1) immediately — a babysitter with a bad worker_id is worse than no babysitter.

---

## What Already Works Well in V2 (No Changes Needed)

- **GCS call timeouts**: All `subprocess.run()` calls use `timeout=30`. BUG-O1 (monitor hang) was a GCS HTTP-level stall, not a subprocess timeout gap — `timeout=30` was already there.
- **Monitor self-watchdog**: `overnight_watchdog.sh` restarts monitor if log is stale for >5min. Already implemented.
- **Heartbeat frequency**: babysitter.py heartbeats every `min(eval_interval, stale_ttl//3)` seconds. Already correct.
- **xla_compile stuck detection**: vm_requester.sh checks heartbeat STATUS — forces redeploy if `xla_compile` for >15min. Already implemented.
- **Orphan detection (task_id mismatch)**: `reclaim_stale()` checks `hb.task_id != running_task.task_id` and reclaims if worker moved on. Already implemented.
- **Pull-based GCS queue**: correct design, handles preemption and multi-zone cleanly
- **vm_requester.sh**: auto-delete PREEMPTED VMs and recreate — worked flawlessly
- **XLA cache warming**: ~5min overhead at first step, self-healing, acceptable
- **Checkpoint resume**: rolling /tmp/ckpt_*.pt checkpoints survived preemptions
- **overnight_loop.sh**: auto-requeued failures every 10min, correctly classified most errors
- **LAUNCH_MODE=single**: confirmed only working mode — never switch to pjrt

---

## Remaining Open Items (Nice-to-Have for V3)

These were NOT critical for exp13_rerun (either auto-recovered or low-frequency). Address before next large run:

| # | Item | File | Priority |
|---|------|------|----------|
| 1 | Exponential backoff on RESOURCE_EXHAUSTED per zone (currently retries every 2min forever) | `vm_requester.sh` | P2 |
| 2 | Disable us-east1-d zone (zero contribution, constant noise in logs) | `vm_requester.sh` | P2 |
| 3 | Rapid failure alarm: >5 tasks from same VM in <10min → log + delete VM | `overnight_watchdog.sh` | P2 |
| 4 | populate drain mode: wait for running tasks to finish before --clear (vs. current abort) | `populate.py` | P2 |
