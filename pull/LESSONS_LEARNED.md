# Persistent Errors & Lessons — Pull-Based TPU Training

## Error 1: XLA Cache Version Mismatch
**Error**: `UNIMPLEMENTED: Deserializing serialized executable not supported`
**When**: Every time a VM downloads XLA cache created by a different torch_xla version
**Impact**: Self-healing (recompiles + caches new), but 5 min wasted per first step
**Root Fix**: Cache per torch_xla version. Tag paths: `xla_cache_v6e_txla2.9.0/`
**Quick Fix**: Delete old cache from GCS, upload fresh from working VM

## Error 2: Heartbeat Tied to stdout (Causes False Reclaims)
**Error**: Tasks reclaimed as "stale" even though training is healthy
**When**: Training produces no stdout between eval intervals (100 steps × 7-8s = 12-14 min)
**Impact**: Configs restart from scratch, wasting hours
**Root Fix**: Background heartbeat thread in babysitter.py (`_heartbeat_loop()`)
**Quick Fix**: Increase monitor stale_ttl to 3600 (1 hour)

## Error 3: LAUNCH_MODE Must Be 'single'
**Error**: `BrokenProcessPool` or TPU device conflicts
**When**: torch_xla.launch() spawns child processes that conflict on TPU devices
**Impact**: Training crashes immediately
**Root Fix**: Set `LAUNCH_MODE=single`, `CHIPS_PER_HOST=1`, `TPU_VISIBLE_CHIPS=N` in subprocess env

## Error 4: Non-Internet VMs Can't pip install
**Error**: `ModuleNotFoundError: No module named 'regex'` (or transformers, hydra, etc.)
**When**: Deploying to VMs in us-east1-d, us-central2-b (no internet)
**Impact**: Babysitter fails environment check
**Root Fix**: Download wheels locally, upload to GCS, install from GCS on VM
**Wheel Locations**: `gs://gcp-researchcredits-blocklab-us-east1/wheels/`
**Known Missing**: `regex` (transformers dep), `antlr4-python3-runtime` (hydra dep), `sympy`, `datasets`

## Error 5: Checkpoint Cleanup Deletes Other Chips' Checkpoints
**Error**: `ckpt_*.pt` pattern matches ALL checkpoints, not just this label's
**When**: One chip finishes and claims new config → cleanup deletes other chips' checkpoints
**Impact**: Other chips can't resume from checkpoint, restart from scratch
**Root Fix**: Changed to `ckpt_*{label}*` pattern only

## Error 6: pgrep Matches Command String, Not Process Name
**Error**: `pgrep -f babysitter.py` matches bash processes that CONTAIN "babysitter.py" in args
**When**: Deploy script checks if babysitter is already running
**Impact**: Thinks babysitter is running when it's just stale SSH sessions
**Root Fix**: Use `pgrep -f 'python.*babysitter'` to match actual Python processes

## Error 7: Internal VM Names Don't Match Zone Patterns
**Error**: Dashboard shows 0% utilization, can't identify v6e/v4/v5e
**When**: TPU VMs report `t1v-n-XXXXXXXX-w-0` instead of `v6e-ew4a-2`
**Impact**: Dashboard is useless for monitoring
**Root Fix**: Infer type from chip count (8=v6e, 4=v4). Future: write friendly name in heartbeat.

## Error 8: v5e No Capacity/Quota
**Error**: Code 8 "no capacity" or Code 7 "no permission"
**When**: Trying to create v5e VMs
**Impact**: Can't use v5e fleet
**Quick Fix**: Retry periodically, try different sizes (v5litepod-4 vs -8)

## Error 9: Orphaned Tasks in running/
**Error**: Tasks stuck in running/ forever, never complete
**When**: Babysitter redeployed, old worker moves to new task
**Impact**: Tasks never complete, never get retried
**Root Fix**: Orphan detection in reclaim_stale() — if heartbeat shows different task_id, reclaim

## Error 10: v4 VMs Running Old Code Without LAUNCH_MODE Fix
**Error**: `BrokenProcessPool: A process in the process pool was terminated abruptly`
**When**: v4 VMs deployed before LAUNCH_MODE=single fix, running old babysitter.py
**Impact**: ALL v4 configs fail (28 chips wasted for entire session)
**Root Fix**: Redeploy ALL VMs after any babysitter code change
**Lesson**: ALWAYS redeploy to ALL VMs when fixing bugs. Old code on other VMs fails silently.

## Error 11: Orphan Training Processes Blocking TPU Chips (CRITICAL)
**Error**: Training killed with SIGKILL (exit_code=-9) after step 200-400
**When**: deploy_babysitter.sh run without killing child processes, OR babysitter restarts
**Symptom**: `ps aux` shows 16-20 training processes instead of 8. Old orphans hold TPU chips.
**Impact**: Training NEVER completes. Every config crashes after 5-15 min.
**Root Cause**: `torch_xla.launch(debug_single_process=True)` spawns a child process. When
  babysitter dies/restarts, parent training process dies but child survives (reparented to init).
  New babysitter spawns NEW training processes. They conflict on TPU devices with orphan children.
  Also: `deploy_babysitter.sh` was called multiple times without flock, starting duplicate babysitters.
**Root Fix**:
1. `start_new_session=True` in subprocess.Popen → training in own process group
2. `os.killpg(proc.pid, SIGKILL)` after proc.wait() → kill entire process tree
3. `kill_orphan_training()` on babysitter startup → clear stale training processes
4. `flock -n` in deploy script → prevent duplicate babysitters
5. Deploy script checks if babysitter already running → early exit if healthy
**Evidence**: v6e-ew4a-2 had 1.4TiB RAM (not OOM), 19 training processes, no dmesg OOM.

## Error 12: VM Missing Experiment Data Directory
**Error**: `FileNotFoundError: Directory .../exp13_smollm2_smoltalk/data/train not found`
**When**: VM was set up for exp12_1 only, then pull-based babysitter claims exp13 tasks
**Impact**: All exp13 tasks on this VM fail with rc=1, exhaust retries → permanently failed
**Root Fix**: deploy_babysitter.sh downloads both exp12_1 and exp13 code tarballs.
  Re-queue permanently failed tasks: reset retries=0, move from failed/ to pending/.

## Key Principles
1. **Heartbeats MUST be independent of training output** — use daemon threads
2. **XLA cache is version-locked** — never share between different torch_xla versions
3. **Non-internet VMs need GCS wheel bundles** — can't pip install
4. **stale_ttl > 2× max_eval_interval** — on v4 (8.4s/step), 100 steps = 840s, need stale_ttl > 1700s
5. **Never redeploy WORKING VMs** — kills training, wastes hours of progress
6. **ALWAYS redeploy BROKEN VMs** — old code fails silently, wasting capacity
7. **Each chip gets its own process** — LAUNCH_MODE=single, no multi-process spawn
8. **Checkpoint cleanup must be label-specific** — multiple chips share /tmp
9. **After ANY babysitter.py fix**: Upload to ALL 3 GCS buckets, then redeploy ALL VMs that are failing
10. **Deploy must be idempotent** — check if babysitter running, don't kill healthy training
11. **Kill entire process TREE** — start_new_session=True + os.killpg on exit
12. **Use flock for babysitter** — prevent duplicate instances
