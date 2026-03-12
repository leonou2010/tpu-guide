# Changes & Lessons — Session 6 (2026-03-11 ~09:00 UTC)

## Problem 1: XLA Cache Poisoning (CRITICAL)
**Symptom**: Every training step shows `UNIMPLEMENTED: Deserializing serialized executable not supported`. Training estimates ~85h/config instead of ~2h.
**Root Cause**: XLA persistent cache files on GCS were created with an older torch_xla version. When a new VM downloads them, torch_xla can't deserialize the old format. It falls back to recompile-from-scratch for each cached graph.
**Impact**: Self-healing (torch_xla recompiles and caches new entries), but first step per config wastes ~5 min. Warning spam in logs.
**Fix**:
1. Deleted old incompatible cache from all GCS buckets: `gcloud storage rm -r gs://.../xla_cache/`
2. Uploaded fresh cache from working v6e VM (torch_xla 2.9.0, 67 files, 19GB) to `gs://.../xla_cache_v6e_fresh/`
3. Copied fresh cache to standard `gs://.../xla_cache/` path for deploy_babysitter.sh
**Lesson**: XLA cache is tightly coupled to torch_xla + libtpu.so version. NEVER share cache between versions. Tag cache by version: `xla_cache_v6e_txla2.9.0/`. When upgrading torch_xla, ALWAYS delete and regenerate cache.

## Problem 2: Zero Completed Tasks After 6 Hours
**Symptom**: `completed=0` after 6 hours of training across 65+ workers.
**Root Cause**: NOT what it seems. Training WAS progressing (configs at step 700/889). The 0 completed count was because:
  - Training takes ~2h per config on v6e
  - Deploy was at ~03:00 UTC, first configs needed XLA compile (~10 min) + 889 steps
  - Many configs were reclaimed and restarted due to stale TTL issues (earlier sessions)
  - Some VMs cycled through 4-5 configs before any could finish
**Impact**: Delayed completions by hours.
**Fix**: Confirmed training IS working. First completions arriving at ~09:30 UTC.
**Lesson**: `completed=0` doesn't mean broken. Check heartbeat step progress on VMs: `grep 'step ' /tmp/babysitter_chip*.log`. Step 700/889 with 0.31h left = everything is fine.

## Problem 3: Dashboard Shows 0% Utilization, Wrong VM Types
**Symptom**: Dashboard shows "0% utilization" and can't identify v6e/v4/v5e VMs.
**Root Cause**: TPU VMs report internal names (`t1v-n-XXXXXXXX-w-0`) from metadata API, not friendly names (`v6e-ew4a-2`). Dashboard zone detection patterns (`ew4a`, `ue1d`) don't match.
**Fix**: Dashboard infers VM type from chip count: 8 chips = v6e, 4 chips = v4. Zone inferred heuristically.
**Lesson**: Dashboard must handle internal VM names. Best approach: have worker write both internal name AND friendly name in heartbeat.

## Problem 4: No v5e VMs Running
**Symptom**: User asks "why 0 v5e?" — no v5e VMs exist.
**Root Cause**: v5e VMs were never created for the pull-based architecture session. Previous v5e VMs were from push-based coordinator (different session).
**Fix**: Creating v5e-ew4b-1 (europe-west4-b). v5e is very slow (~100s/step, ~25h/config) but contributes throughput.
**Lesson**: Always create all VM types at session start. v5e contributes ~4 configs/day per VM.

## Problem 5: Stale TTL vs Heartbeat Interval Race
**Symptom**: Tasks reclaimed even though workers are alive and training.
**Root Cause**: Earlier monitor was run with `--stale-ttl 90` (90s), but heartbeat interval is 300s. Tasks reclaimed before next heartbeat.
**Fix**: Set `--stale-ttl 900` (15 min). Heartbeat interval (300s) is well within this.
**Lesson**: stale_ttl MUST be > 2× HEARTBEAT_INTERVAL. With 300s heartbeat, use stale_ttl >= 900s.

## Problem 6: Orphaned Tasks in running/
**Symptom**: Tasks stuck in `running/` forever, never complete or fail.
**Root Cause**: When babysitter is redeployed, workers start new tasks but old tasks remain in running/. Worker's heartbeat shows new task_id, but old task's running/ entry is never cleaned up.
**Fix**: Added orphan detection to `reclaim_stale()` in gcs.py: if worker's heartbeat task_id ≠ running/ task_id, reclaim immediately.
**Lesson**: Pull-based systems need orphan detection. Workers can only work on one task at a time — if heartbeat shows different task, old task is orphaned.

## Problem 7: Heartbeat Tied to stdout (ROOT CAUSE of false reclaims)
**Symptom**: v4 workers reclaimed every ~15 min even though training is healthy.
**Root Cause**: Heartbeat updates are INSIDE the `readline()` loop. When training produces no stdout between eval intervals (100 steps × 8.4s on v4 = 840s), `readline()` blocks, heartbeat goes 14 min stale. With stale_ttl=900, any timing jitter causes reclaim.
**Fix**:
1. Added `_heartbeat_loop()` background thread in babysitter.py that writes heartbeats every 300s regardless of stdout
2. Increased monitor stale_ttl to 3600 (1 hour) as immediate mitigation for running VMs
3. Fixed checkpoint cleanup to only delete label-specific files (was deleting ALL ckpt_*.pt)
**Lesson**: Never tie heartbeats to training output. Use a separate daemon thread/process. readline() blocks = heartbeat dies.

## Problem 8: ue1d VMs Missing Python Packages (No Internet)
**Symptom**: `FATAL: Environment check failed: ['transformers']` on ue1d-1.
**Root Cause**: us-east1-d VMs have no internet. deploy_babysitter.sh tries `pip install` which fails silently.
**Fix**: Install from GCS wheel bundles at `gs://gcp-researchcredits-blocklab-us-east1/wheels/all_wheels.tar.gz` before deploying.
**Lesson**: Non-internet VMs need wheel bundles on GCS. deploy_babysitter.sh should detect no-internet and install from GCS wheels.

## Problem 9: v5e VM Creation Failed
**Symptom**: No v5e VMs created.
**Root Cause**: europe-west4-b has no capacity (error code 8). us-central1-a has no quota (error code 7).
**Fix**: Retry periodically. Try v5litepod-8 (different capacity pool).
**Lesson**: v5e capacity is scarce. Have fallback zones. v5e is impractically slow anyway (~100s/step).

## Problem 10: Checkpoint Cleanup Deletes Other Chips' Checkpoints
**Symptom**: Configs restart from step 0 even when checkpoint exists on same VM.
**Root Cause**: `cleanup_checkpoint()` had pattern `ckpt_*.pt` matching ALL checkpoints in /tmp, not just the specific label. When chip0 starts a new config, chip1-7's checkpoints get deleted.
**Fix**: Changed to only match `ckpt_*{label}*` pattern.
**Lesson**: Checkpoint cleanup must be label-specific. Multiple chips share /tmp.

## Changes Made
| File | Change |
|------|--------|
| `gcs.py` | Added orphan detection to `reclaim_stale()` |
| `dashboard.py` | Fixed zone/type detection for internal VM names |
| `babysitter.py` | Background heartbeat thread, fixed checkpoint cleanup |
| `monitor.py` | Restarted with stale_ttl=3600 (was 900) |
| GCS `xla_cache/` | Purged old incompatible, uploaded fresh torch_xla 2.9.0 cache |
| GCS `pull_code/` | Updated babysitter.py and gcs.py on all 3 buckets |
| MEMORY.md | Updated XLA cache docs, session 5/6 status, lessons |

## Problem 11: v4 VMs Running OLD Babysitter Without LAUNCH_MODE=single (CRITICAL)
**Symptom**: ALL v4 configs fail with rc=1 (`BrokenProcessPool`) or rc=-9 (SIGKILL). Zero configs complete on v4.
**Root Cause**: v4 VMs were deployed in a previous session before the LAUNCH_MODE=single fix. The old babysitter.py didn't set this env var, so `torch_xla.launch()` uses ProcessPoolExecutor which conflicts with per-chip isolation.
**Evidence**: Traceback shows `torch_xla.launch(_train_fn, args=(resolved,))` (line 111) instead of `torch_xla.launch(..., debug_single_process=True)` (line 109).
**Impact**: 28 chips on 7 v4 VMs completely wasted — every config fails after step 400.
**Fix**: Redeployed all 7 v4 VMs with latest babysitter.py (has LAUNCH_MODE=single).
**Lesson**: When fixing a bug, ALWAYS redeploy to ALL VMs, not just the ones you're testing on. Old code running on other VMs will silently fail.

## Fleet Status (09:45 UTC)
- 5× v6e-8 (europe-west4-a): 40 chips, training at step 200+ (first completions ~10:30 UTC)
- 3× v6e-8 (us-east1-d): deploying (wheel bundle installed, babysitter deploying)
- 7× v4-8 (us-central2-b): 28 chips, REDEPLOYING with LAUNCH_MODE fix
- 0× v5e: no capacity/quota
- Total: ~68 chips, only 40 (v6e-ew4a) actively training
- Monitor: PID 967156, stale_ttl=3600
