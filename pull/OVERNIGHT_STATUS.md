# Pull-Based Sweep — Overnight Status

## Session Start: 2026-03-11 00:50 UTC

### Architecture
- Pull-based GCS task queue at `gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/`
- Workers (babysitter.py) auto-claim tasks from `pending/`, train, write to `completed/`
- Monitor (monitor.py) validates results, reclaims stale tasks, copies to local folders
- Dashboard (dashboard.py) shows real-time fleet health

### Experiments
| Exp | Target | Validated | Work Dir | Train Script | Result Dest |
|-----|--------|-----------|----------|--------------|-------------|
| exp13 | 120 | 0 | exp13_smollm2_smoltalk | exp13_tpu/train_tpu.py | ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/ |
| exp12_1 | 185 | 176 | exp10_smollm2_smoltalk | exp12c_tpu/train_tpu_v2.py | ~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu/results/ |

### Fleet at Start
| VM | Zone | Type | Chips | Status |
|----|------|------|-------|--------|
| v6e-ew4a-2 | europe-west4-a | v6e-8 | 8 | DEGRADED (stale device locks) |
| v6e-ew4a-4 | europe-west4-a | v6e-8 | 8 | WARMING (just restarted) |
| v6e-ew4a-5 | europe-west4-a | v6e-8 | 8 | HEALTHY |
| v6e-ew4a-6 | europe-west4-a | v6e-8 | 8 | HEALTHY |
| v6e-ew4a-8d | europe-west4-a | v6e-8 | 8 | DEGRADED (no heartbeat) |
| v6e-ew4a-9 | europe-west4-a | v6e-8 | 8 | CREATING |
| v6e-ew4a-10 | europe-west4-a | v6e-8 | 8 | CREATING |
| v6e-ue1d-3 | us-east1-d | v6e-8 | 8 | READY (deploy pending) |
| v6e-ue1d-4 | us-east1-d | v6e-8 | 8 | CREATING |
| v6e-ue1d-5 | us-east1-d | v6e-8 | 8 | CREATING |
| v4-uc2b-1..7,spot3 | us-central2-b | v4-8 | 4 each | HEALTHY (32 chips) |
| v5e-ew4b-1 | europe-west4-b | v5e | 4 | WARMING |

### XLA Cache Locations
- v6e: `gs://gcp-researchcredits-blocklab-europe-west4/xla_cache/` → `/tmp/xla_cache` (WORKS)
- v5e: `gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e/` → `/tmp/xla_cache` (WORKS)
- v4: `gs://gcp-researchcredits-blocklab-1-us-central2/xla_cache/` → broken, recompiles each time (~5 min)

### Known Issues
1. v6e-ew4a-2, v6e-ew4a-8d: stale `/dev/vfio/` locks from old exp12_1 sessions
2. v5e-ew4b-1: SSH 255 errors (IAP tunnel flaky), impractical speed (~100s/step)
3. v4 XLA cache: broken (UNIMPLEMENTED deserialization error), each config recompiles
4. IAP SSH rate limiting: can't run many parallel SSH sessions

### Changes Log
- 00:20 UTC: Created pull-based system (gcs.py, babysitter.py, monitor.py, populate.py, dashboard.py, deploy.sh)
- 00:24 UTC: Populated 129 tasks (120 exp13 + 9 exp12_1)
- 00:27 UTC: Fixed babysitter.py device detection for v6e (vfio)
- 00:33 UTC: Deployed to all v6e VMs with fixed code
- 00:35 UTC: All 14 VMs active, 64 running
- 00:40 UTC: 7 failed tasks (v5e device busy) re-queued
- 00:44 UTC: Added Fleet Health panel to dashboard
- 00:50 UTC: Creating 3 new VMs, fixing degraded VMs

### 01:00:47 UTC Update
- exp13: 0/120
- exp12_1: 176/185
- Queue: pending=101 running=29 completed=0 failed=0
- Fleet: 10/13 VMs healthy

### 01:01:05 UTC Update
- exp13: 0/120
- exp12_1: 176/185
- Queue: pending=101 running=29 completed=0 failed=0
- Fleet: 10/13 VMs healthy

### 01:07 UTC — CRITICAL BUG FIX: Monitor Stale TTL
**Root cause of 0 completions after 4 hours:**
- Monitor `reclaim_stale` TTL was 300s = heartbeat interval (300s)
- Race condition: tasks reclaimed 1s after heartbeat expiry, before next heartbeat
- Workers restarted from scratch endlessly, never completing any task
- **Fix**: Increased stale_ttl to 900s (3x heartbeat interval)
  - `monitor.py` restarted with `--stale-ttl 900`
  - `auto_maintain.py` updated to use `stale_ttl_s=900`
- **Fix 2**: `complete_task()` in gcs.py now also deletes from `pending/` to prevent duplicate work
- **Fix 3**: Monitor now deduplicates pending tasks that are already in completed/

### 01:15 UTC — Dashboard Enhanced
- Added regional VM grouping (zone subtotals)
- Added aggregate stats (total VMs, training/compiling/stuck/idle)
- Fleet Health now driven by heartbeats (not running tasks)
- Rich Live mode for auto-refresh: `python3 dashboard.py --interval 30`
- Scrollable with `--pager`

### 01:15 UTC — Code Deployment Fix
- GCS `pull_code/` was missing `babysitter.py` — only had `gcs.py`
- Uploaded babysitter.py + gcs.py to all 3 GCS buckets (europe-west4, us-east1, us-central2)
- Updated `code/pull/` directory with all latest files

### 01:15 UTC — New VM Creation
- Attempting: v6e-ew4a-9, v6e-ew4a-10, v6e-ue1d-3, v6e-ue1d-4, v6e-ue1d-6 (all with --internal-ips)
- v6e-ue1d-5: READY, being set up
- v6e-ew4a-8d: DEGRADED (no heartbeat), being redeployed

### 01:19 UTC Update
- exp13: 0/120 validated, 35 running, 86 pending
- exp12_1: 176/185 validated, 9 pending
- Fleet: 13 VMs, 55 training, 3 compiling, 8 stuck, 66 total heartbeats
- Leading worker: v6e-ew4a-5_chip0 at step 500/889 — ETA first completion ~01:50 UTC
- v6e-ew4a-2/4: step 1/889 (just restarted after nuclear kill)
- v6e-ew4a-6: step 300/889
- v4: mostly step 100/889 (8.4s/step → ~110 min to complete)
- v5e: still XLA compiling (~35 min in)

### 01:25:29 UTC Update
- exp13: 0/120
- exp12_1: 176/185
- Queue: pending=94 running=36 completed=0 failed=0
- Fleet: 10/13 VMs healthy

### 01:54:10 UTC Update
- exp13: 0/120
- exp12_1: 176/185
- Queue: pending=98 running=31 completed=0 failed=0
- Fleet: 2/14 VMs healthy

### 14:10 UTC — ROOT CAUSE FOUND + CRITICAL FIX: Orphan Processes

**Root cause of 0 completions after ~14 hours:**
Training kept crashing with "Device or resource busy" (SIGKILL, exit_code=-9). Root cause: orphan training processes from babysitter restarts.

**Mechanism:**
1. `torch_xla.launch()` spawns child process via ProcessPoolExecutor (when LAUNCH_MODE != 'single')
2. When babysitter is killed/restarted, parent training dies but child survives (reparented to init)
3. New babysitter spawns NEW training processes → conflict on TPU devices with orphan children
4. Training gets SIGKILL after step 200-400 due to device contention
5. Old deploy_babysitter.sh had idempotency check → didn't restart babysitter → old code ran forever

**Evidence:**
- v6e-ew4a-2 had 19 training processes (should be 8)
- All worker logs show: `RuntimeError: TPU initialization failed: open(/dev/vfio/0): Device or resource busy`
- Workers cycled through 10+ tasks each without any completing

**Fixes applied:**
1. `babysitter.py`: Added `kill_orphan_training()` on startup
2. `babysitter.py`: Added `start_new_session=True` to subprocess.Popen
3. `babysitter.py`: Added `os.killpg(proc.pid, SIGKILL)` after proc.wait()
4. `babysitter.py`: Sets `LAUNCH_MODE=single` in env → `debug_single_process=True` in train script → NO child processes spawned
5. `deploy_babysitter.sh`: Removed idempotency check → ALWAYS force restart
6. `deploy_babysitter.sh`: Uses `flock -n` to prevent duplicate babysitters
7. Re-queued 5 permanently failed tasks (retries exhausted during orphan era)

**Deployed to all 15 VMs:**
- 5 v6e-ew4a (europe-west4-a): SUCCESS
- 7 v4-uc2b (us-central2-b): SUCCESS
- 3 v6e-ue1d (us-east1-d): PENDING (slow SSH)

**Fleet status at 14:18 UTC:**
- 71 chips active on 12 VMs (+ 3 ue1d pending)
- 11 chips training (step 1), 60 compiling XLA
- Queue: pending=16, running=111, completed=0, failed=0
- Monitor: PID 1513336, stale_ttl=3600

**VM Name Mapping:**
| Friendly Name | Internal Name | Type | Chips |
|---|---|---|---|
| v6e-ew4a-2 | t1v-n-dcdd55ee-w-0 | v6e-8 | 8 |
| v6e-ew4a-4 | t1v-n-ad53d7ef-w-0 | v6e-8 | 8 |
| v6e-ew4a-6 | t1v-n-513c8423-w-0 | v6e-8 | 8 |
| v6e-ew4a-8d | t1v-n-59f1a300-w-0 | v6e-8 | 8 |
| v6e-ew4a-10 | t1v-n-13f6f8c7-w-0 | v6e-8 | 8 |
| v4-uc2b-1 | t1v-n-35b9ee83-w-0 | v4-8 | 4 |
| v4-uc2b-2 | t1v-n-2fbbc397-w-0 | v4-8 | 4 |
| v4-uc2b-3 | t1v-n-0dbf6801-w-0 | v4-8 | 4 |
| v4-uc2b-4 | t1v-n-6532ac3a-w-0 | v4-8 | 4 |
| v4-uc2b-5 | t1v-n-a1537dde-w-0 | v4-8 | 4 |
| v4-uc2b-6 | t1v-n-4403cdf6-w-0 | v4-8 | 4 |
| v4-uc2b-spot3 | t1v-n-81f218b9-w-0 | v4-8 | 4 |

**Expected timeline (if fix works):**
- ~14:22 UTC: First v6e chips reach step 100 (heartbeat shows progress)
- ~14:30 UTC: Training past step 200 (previous crash point)
- ~15:30 UTC: First v6e config completes (step 889)
- ~16:20 UTC: First v4 config completes
- ETA all done: ~20:00 UTC (68 chips × ~1.3h/config on v6e, 2.1h on v4)

### 10:43 UTC — STEP 200 REACHED (Past Crash Point!)
- **CRITICAL SUCCESS**: max_step=200. Previous crash window was step 200-400.
- 68 chips training across 13 VMs, 4 compiling
- v6e-ew4a-2: all 8 chips at step 200
- v6e-ew4a-4, v6e-ew4a-6, v6e-ew4a-10: some chips reaching step 200
- Queue: pending=61, running=68, completed=0, failed=0
- exp13: 0/120 validated, exp12_1: 176/185 validated
- **ETA first v6e completion**: ~11:40 UTC (step 889 at 4.9s/step)
- **ETA first v4 completion**: ~12:30 UTC (step 889 at 8.4s/step)
- Orphan fix (LAUNCH_MODE=single + force-restart) confirmed working

### 10:45 UTC — STEP 300 CONFIRMED (Past Full Crash Window!)
- **DEFINITIVE SUCCESS**: max_step=300. Previous crash window was 200-400.
- v6e-ew4a-2: steps 200-300, v6e-ew4a-4/10: step 200
- v4 VMs: all at step 100 (slower but healthy)
- 68 chips training, 4 stale/compiling
- No failed tasks. Queue: pending=61, running=68
- Creating v6e-ew4a-11, deploying to v6e-ue1d-3
- ETA first completion: ~11:30-11:40 UTC (v6e chips at step 300/889)

### 10:49 UTC — STEP 400 CONFIRMED (Full Crash Window Cleared!)
- v6e-ew4a-2 chip6: step 400/889, train=1.4856, val=1.4086, ~0.72h left
- v6e-ew4a-2 chip3: checkpoint saved at step 300
- 4 v6e VMs at step 200-300, v4 VMs at step 100
- **ETA first completion: ~11:32 UTC** (step 889 from step 400 = 489 steps × 4.9s = 40 min)
- VM creation failed: ew4a-11 (internal error), ue1d-1 (no capacity). Will retry later.
- ue1d-3 deploy still in progress

### 11:25 UTC — v6e-ue1d-3 Deployed
- Babysitter running on t1v-n-3d512f92-w-0 (8 chips, PID=27843)
- Fleet now: 5 v6e-ew4a + 2 v6e-ue1d + 7 v4-uc2b = 14 VMs, 76 chips
- Re-queued 2 failed tasks (exp13__ema_lr0.003_k0.7, exp13__v4_c1.0_lr0.003_k0.5)
- First completions imminent (~11:30-11:35 UTC)

### 11:35 UTC — FIRST COMPLETIONS! 🎉
- **completed=2** in GCS queue — first configs EVER completed in pull system!
- 24 chips now compiling XLA for their next config (finished first task)
- Queue: pending=60, running=67, completed=2, failed=1 (re-queued)
- exp13: 0/120 validated (monitor hasn't pulled results yet), exp12_1: 176/185
- Orphan fix + LAUNCH_MODE=single confirmed working at scale

### 12:31 UTC — Round 2 Starting
- exp13: 2/120 validated, exp12_1: 176/185 validated
- Queue: pending=50, running=75, completed=2, failed=0 (2 re-queued)
- Fleet: 80 chips on 14 VMs, most at step 1 (starting round 2 configs)
- 29 chips XLA compiling, 51 training
- v6e-ue1d-4 deploy in progress (SSH active)
- v6e-ew4a-1 creation failed (not in VM list)
- Attempting to create v6e-ew4a-1 and v6e-ue1d-1 again
- ETA next v6e completions: ~13:45 UTC, v4 completions: ~14:30 UTC
