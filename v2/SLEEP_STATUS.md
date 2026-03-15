# Overnight Run Status — UPDATED 2026-03-13 13:51 UTC

## TL;DR (13:51 UTC Update — 99% DONE)
- **119/120 validated (99%)** — 1 task remaining
- **Last task**: v4_c0.05_lr0.003_k0.3 on v4-uc2b-1_chip3 | ~123 min remaining | ETA ~15:56 UTC
- **History of this task**: originally failed by ew4a-2_chip1 (PRE_START_SESSION_BARRIER at 11:25 UTC), auto-requeued by overnight_loop, now running stably on v4 chip
- **All bugs fixed**: O1-O8 logged in TESTRUN_FINDINGS.md

## TL;DR (11:25 UTC Update — CRITICAL BUGS FOUND & FIXED)
- **111/120 validated (93%)** — found and fixed 2 critical bugs blocking completion
- **BUG-O7 FIXED**: 5 tasks zombie-stuck in running/ for 7-9 hours (force-redeploy killed babysitter mid-upload) — requeued all 5
- **BUG-O8 FIXED**: 4 tasks had 0-byte empty JSON in pending/ entire run (populate.py write failure) — rebuilt from module, immediately claimed
- **Now 10 tasks running**: all on v6e chips (2-6 min old) except 1 v4. ETA completion ~14:00 UTC
- **Watch**: ew4a-2_chip1 claimed v4_c0.05_lr0.003_k0.3 — may fail with PRE_START_SESSION_BARRIER

## TL;DR (10:00 UTC Update — NEAR COMPLETION)
- **101/120 validated (84%)** — 37 tasks completed in last 70 min burst
- **Only 19 tasks remain**: 15 running + 4 pending (all will be claimed by idle chips)
- **BUG-O6 found**: uc2b-2_chip2 failing with "buffer deleted" (user-site torch_xla on v4) — requeued
- **Fleet**: 10 VMs, 58 chips. Most v4 chips now idle (finished their tasks). v6e ew4a-1 still churning.
- **ETA**: ~12:00-12:30 UTC for final 19 tasks (slowest = v4 170 min/task)

## TL;DR (08:08 UTC Update)
- **57/120 validated (48%)** — rate jumped to ~42/hr as ew4a-3/4/5 completing first tasks
- **Only 7 pending/failed tasks remain** (5 pending + 2 just requeued) — 53 running
- **All 10 VMs healthy**: 60 chips, 49 training, 10 xla_compile, 1 idle
- **ew4a-2 still failing**: chip7 got PRE_START_SESSION_BARRIER again (v4_c0.0002_lr0.001_k0.7 requeued)
- **BUG-O5 found**: v4-uc2b-5_chip2 false-reclaimed 11× (heartbeat_stale) — logged to TESTRUN_FINDINGS.md
- **ETA**: ~10:00-10:30 UTC for all 120 validated

## TL;DR (07:45 UTC Update)
- **exp13_rerun RUNNING WELL** — 5 v6e VMs + 5 v4 VMs = 60 chips training
- **40/120 validated (33%)** — rate ~7-15/hr, increasing as ew4a-3/4/5 near first completion
- **All systems healthy**: monitor PID 2890022, overnight_loop PID 1937843, vm_requester PID 109604
- **ew4a-2 PRE_START_SESSION_BARRIER**: chip1/chip7 occasional failures — 1 more task requeued at 07:42 UTC
- **vm_requester**: trying ew4a-6/7/8 but getting GCP "internal error" — will auto-retry every 2 min
- **First ew4a-3/4/5 completions**: expected ~08:45 UTC (step 600-700/1778, ~80 min remain)
- **ETA**: ~11:30 UTC for all 120 validated (conservative)

## TL;DR (07:06 UTC Update — older)
- **exp13_rerun RUNNING WELL** — 5 v6e VMs + 5 v4 VMs = 60 chips training
- **36/120 validated (30%)** — progressing at ~15 tasks/30min from v6e
- **libtpu bug FIXED AND CONFIRMED**: ew4a-3/4/5 deployed with fix, all training
- **Monitor running**: PID 2890022, validates ~30 tasks/hr
- **Remaining ew4a-2 issue**: chip7 gets PRE_START_SESSION_BARRIER occasionally — auto-requeued
- **ETA**: ~10:00 UTC for all 120 validated (if preemption stays low)

## TL;DR (05:50 UTC Update — older)
- **exp13_rerun still RUNNING** — 2 v6e VMs + 5 v4 VMs = 36 chips active
- **Preemption wave**: ew4a-3/6/7/8 all PREEMPTED ~05:30 UTC. Only ew4a-1/2 stable.
- **20/120 validated** (17%) — monitor was stuck 4h, fixed at 05:11 UTC
- **First completions imminent**: ew4a-1 chip7 at step 1600/1778 → ~06:00 UTC
- **libtpu bug FIXED**: deploy_babysitter.sh now installs libtpu on v6e — uploaded to all 3 buckets
- **Monitor running**: PID 2890022, 60s interval
- **v3 relaunch plan**: consider doing clean relaunch when you wake up to recover fully

## Critical Sessions Overnight Summary
1. **01:04 UTC**: monitor.py went silent (stuck), no validation for 4h
2. **03:30–04:37 UTC**: 50+ failures from code bundle bug + root babysitter + user-site torch
3. **05:11 UTC**: Monitor restarted, discovered all processes were running but monitor stuck
4. **05:13–05:30 UTC**: ew4a-7 burned 25 tasks, ew4a-4 burned 25 tasks (libtpu not found)
5. **05:25 UTC**: Fixed deploy_babysitter.sh (add libtpu install), deleted ew4a-4 and ew4a-7
6. **05:30 UTC**: Preemption wave — ew4a-3/6/7/8 all preempted simultaneously

## Original TL;DR (03:32 UTC)
- **exp13_rerun is RUNNING** — 10 VMs active, 0 failed tasks, steps monotonically increasing
- **First completions expected ~03:50–04:00 UTC** (ew4a-1 at step 700/889)
- **All tasks should complete by ~09:00 UTC** (conservative estimate)
- **Deploy script fix deployed** — new VMs will no longer install CUDA torch to user-site

---

## Queue State (03:32 UTC)
| State    | Count |
|----------|-------|
| pending  | 76    |
| running  | 45    |
| completed| 1 (exp13, not exp13_rerun) |
| failed   | 0     |
| **total exp13_rerun** | **122** |

---

## Fleet State (03:30 UTC)

### Active VMs (all READY)
| VM          | Chips | Max Step | Status  | Notes |
|-------------|-------|----------|---------|-------|
| v6e-ew4a-1  | 7/8   | 700      | training | furthest along |
| v6e-ew4a-2  | 8/8   | 600      | training+compiling |  |
| v6e-ew4a-3  | 7/8   | 600      | training+compiling |  |
| v6e-ew4a-4  | 8/8   | 0        | xla_compile | just deployed, fix applied |
| v6e-ew4a-5  | 8/8   | 0        | xla_compile | just deployed, fix applied |
| v4-uc2b-1   | 3/4   | 300      | training |  |
| v4-uc2b-2   | 4/4   | 500-600  | training |  |
| v4-uc2b-3   | 4/4   | 500      | training |  |
| v4-uc2b-4   | 4/4   | 600      | training |  |
| v4-uc2b-5   | 4/4   | 600      | training |  |

**Total: 79 chips active, 35 training + 44 xla_compile**

### Zones
- **europe-west4-a**: 5/8 v6e VMs in use. vm_requester trying to create ew4a-6/7/8 (pending HEALTH_CHECKS quota)
- **us-east1-d**: RESOURCE_EXHAUSTED — all creation attempts fail, no v6e capacity
- **us-central2-b**: 5/8 v4 VMs, healthy

---

## ETA for Completions
- **v6e at step 700**: ~189 steps × 5.5s/step ≈ 17 min → **~03:50 UTC first v6e task done**
- **v4 at step 300-600**: 289-589 steps × 8.5s/step = 41-83 min → **~04:15–05:00 UTC first v4 tasks done**
- **All 122 tasks**: v6e does ~1 task/96 min × 5 VMs × 8 chips, v4 ~1 task/170 min × 5 VMs × 4 chips
  - ~40 completions/hour at peak → **~09:00 UTC all done** (conservative, assuming no preemptions)

---

## Background Processes (all running)
| Process | PID | Status |
|---------|-----|--------|
| vm_requester.sh | 109604 | Running — 2-min cycles |
| monitor.py | 352404 | Running — 60s interval, stale-ttl=1800s |
| step_monitor.py | 1268060 | Running — 5-min interval |
| auto_requeue_daemon | 1086057 | Running — 5-min interval |

---

## Known Issues During This Run

### FIXED THIS SESSION
1. **CUDA torch user-site shadow** — deploy script now detects `+cu` or `~/.local/` torch → purges user-site → uses system torch. Uploaded to all 3 GCS buckets.
2. **USER unbound variable** — `sudo chown` used `${USER}` which is unset in SSH context. Fixed with `_cur_user=$(id -un)`.
3. **ew4a-5 BOOTING stuck** — old `/tmp/deploy_startup.sh` had USER bug. Fixed by deploying new script from GCS.
4. **1 failed task `ema_lr0.005_k0.3`** — code_error_permanent (likely CUDA torch issue before fix). Manually requeued at 03:31 UTC.

### STILL OUTSTANDING (vm_requester.sh)
- **us-east1-d RESOURCE_EXHAUSTED**: vm_requester cycles every 2 min trying to create ue1d VMs, fails 100% of the time. No backoff on repeated RESOURCE_EXHAUSTED. Wastes ~230 API calls logged so far. **v3 fix needed**: exponential backoff per zone on RESOURCE_EXHAUSTED.
- **ew4a-7 kept FAILED_ENV** before fix — vm_requester still trying to create ew4a-6/7/8 (HEALTH_CHECKS quota 75/75 = no more v6e in ew4a without quota increase)
- **HEALTH_CHECKS quota**: 75/75 used. Max ~9-10 VMs in europe-west4-a. Requested 5000 but not yet granted.

---

## What to Do When You Wake Up (Updated 05:50 UTC)

### Step 1: Check current status
```bash
python3 ~/distributed_tpu_training/v2/check_progress.py
```

### Step 2: If run still in progress (likely) — do NOT interrupt working VMs
```bash
# Check what's validated
ls ~/sf_bema/results/exp13_rerun/validated/ | wc -l

# Check monitor is running
pgrep -af 'monitor.py'  # Should show PID 2890022

# Check for failed tasks (requeue any found)
gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/failed/ | grep exp13_rerun
```

### Step 3: If you want to do the CLEAN RELAUNCH now (recommended)
Since exp13_rerun has had so many issues and may take until ~11:00 UTC to fully complete:
**Option A**: Kill current run and do clean relaunch with all fixes
**Option B**: Wait for current run to complete, then relaunch clean

See `TESTRUN_FINDINGS.md` for all bugs found. Key fixes already applied:
1. Code bundle includes both `exp13_tpu/` and `exp13_tpu_rerun/` ✅
2. deploy_babysitter.sh: libtpu install on v6e ✅
3. deploy_babysitter.sh: user-site torch purge ✅
4. Uploaded to all 3 GCS buckets ✅

### Background Processes (07:06 UTC)
| Process | PID | Status |
|---------|-----|--------|
| vm_requester.sh | 109604 | Running — creating/recreating preempted VMs |
| monitor.py | 2890022 | Running — 60s interval, validates completions |
| overnight_loop.sh | 1937843 | Running — 10 min interval, requeues failures |

### For the CLEAN RELAUNCH:
See `TESTRUN_FINDINGS.md` for full list. Key items:
1. Use `populate.py --clear` with exp13_rerun config (same grid)
2. Deploy script in GCS already has libtpu fix
3. Consider disabling us-east1-d in vm_requester (no capacity)

---

## Deploy Script State (GCS)
- **Version**: 2026-03-13 03:27 UTC, 23.1 KiB
- **Key fix**: v6e torch — detects `+cu`/user-site shadow, purges `~/.local/site-packages/torch*`, then reinstalls system-wide with `sudo pip` if still missing
- **Buckets**: europe-west4 ✅ | us-east1 ✅ | us-central2 ✅

---

## Stale Heartbeats (not real VMs)
These appear in check_progress but are old dead VMs — ignore:
- `t1v-n-7c7d31e9-w-0` — age ~14400s (4h stale)
- `t1v-n-852c56ff-w-0` — age ~14300s (4h stale)
- `t1v-n-df66c15b-w-0` — age ~14300s (4h stale)
- `v6e-ew4a-6` — age ~17000s (4.7h stale)
All will expire naturally. Monitor won't reclaim their tasks (they were already reclaimed hours ago).
