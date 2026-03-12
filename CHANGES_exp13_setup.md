# exp13 Setup Changes — 2026-03-10

## Overview
Setting up exp13 (120 configs, 882 steps/config) to run on full fleet: v6e, v4, v5e.

## Files Modified

### 1. coordinator.py
- **import shutil** added (for copy_validated_results)
- **load_vm_configs()**: Added `elif accel.startswith('v5'): cfg['STEP_S'] = 6.0` — v5e VMs now included
- **worker_sweep()**: v5e OOM auto-fix — detects `ACCELERATOR_TYPE` containing 'v5' or TPU name containing 'v5', halves batch_size and doubles gradient_accumulation_steps to preserve effective BS=128
- **ExperimentQueue class**: Queue management (enqueue/dequeue/mark_running/mark_done/status_report). File: ~/distributed_tpu_training/queue.json
- **deploy_to_fleet()**: Iterates VM configs, cancels prev exp workers, runs submit.sh --sweep
- **copy_validated_results()**: Copies validated JSON results to experiment folder
- **queue_monitor()**: Main loop — dequeues pending exp, inits, deploys, monitors, copies results, marks done. Crash recovery via 'running' status
- **main()**: New CLI args: --enqueue, --dequeue, --queue-status, --queue-monitor (no EXP needed)

### 2. submit.sh
- **ACCELERATOR_TYPE exported**: Added to both launch_all() and launch_parallel() env exports
- **tmux cleanup**: launch_parallel() now kills ALL tmux sessions (not just matching exp name)
- **push_code() data symlink**: Creates `$WORK_DIR/data -> exp10_smollm2_smoltalk/data` symlink on VMs when WORK_DIR differs
- **pull_xla_cache()**: v5e uses `$BUCKET/xla_cache_v5e/` (v6e/v4/v5e caches incompatible)
- **push_xla_cache()**: Same v5e cache path fix

### 3. experiments/exp13.env
- Fixed config count comment: 102 → 120

### 4. New VM configs
- v6e-ue1d-2.env, v6e-ue1d-3.env (us-east1-d, no internet)
- v4-uc2b-5.env, v4-uc2b-6.env, v4-uc2b-7.env (us-central2-b, no internet)

### 5. Data symlink (local)
- Fixed broken symlink: `~/sf_bema/experiments/exp13_smollm2_smoltalk/data` → `~/sf_bema/experiments/exp10_smollm2_smoltalk/data`

## Per-VM-Type Environment (from probe)

| Property | v6e-ew4a | v4-uc2b | v5e-ew4b | v6e-ue1d |
|----------|----------|---------|----------|----------|
| Python | 3.10.12 | 3.10.6 | 3.10.12 | 3.10.12 |
| torch/xla | 2.9.0 | 2.9.0 | 2.9.0 | NEEDS SETUP |
| Chips | 8 | 8 | 4 | 8 |
| Internet | YES | NO | YES | NO |
| Bucket | europe-west4 | us-central2 | europe-west4 | us-east1 |
| W&B | online | disabled | online | disabled |
| XLA cache | 4.8GB | 2.1GB (broken) | 576MB | none |
| gsutil ver | 5.36 | 5.23 | gsutil avail | 5.30 |
| gcloud storage | YES | YES | YES | YES |

## v5e OOM Fix Logic
- Default: bs=8, ga=16 → effective BS=128
- v5e fix: bs=4, ga=32 → effective BS=128 (same)
- Detection: `ACCELERATOR_TYPE` env var or TPU name contains 'v5'
- Applied in worker_sweep() BEFORE building the training command

## Session 2 Changes (2026-03-10 ~21:30)

### 6. dashboard.py — Multi-exp refactor
- Fleet/quota shown ONCE at top (not duplicated per experiment)
- `build_automation_panel()`: Shows babysitter/monitor PIDs, process counts, cycle info
- `render_all()`: Single function for multi-exp rendering (fleet → automation → per-exp panels)
- Per-VM Workers table filtered to only show VMs assigned to that experiment
- Assignments list hides VMs with 0 configs
- Header shows experiment names (e.g. "exp12_1, exp13")
- `--exp` accepts multiple experiments: `--exp exp12_1 exp13`

### 7. vm_scanner.sh — Fleet expansion
- Added us-central1-a zone (v5e, v5litepod-4)
- Added ZONE_ACCEL, ZONE_MAX_CHIPS maps for VM creation
- Phase 2: Auto-creates VMs in zones with spare quota (max 3 per zone per cycle)
- Generates names: v6e-ew4a-N, v6e-ue1d-N, v4-uc2b-N, v5e-ew4b-N, v5e-uc1a-N
- Creates VM configs automatically for discovered VMs

### 8. vm_scan.sh — Updated zones
- Added us-central1-a and europe-west4-b to GRANTED_ZONES
- v5e OOM messages updated ("OOM fixed — bs halved automatically")

## Running Processes (2026-03-10 ~21:55)
| Process | PID | Log |
|---------|-----|-----|
| babysit_exp13 | 1590436 | /tmp/babysit_exp13.log |
| babysit_exp12_1 | 3835484 | /tmp/babysit_exp12_1.log |
| coordinator --monitor (exp13) | 1588112 | /tmp/monitor_exp13.log |
| vm_scanner (exp13) | 1914688 | /tmp/vm_scanner_exp13.log |

## Recovery Commands
```bash
# Dashboard (multi-exp)
watch -c -n30 'python3 ~/distributed_tpu_training/dashboard.py --exp exp12_1 exp13 --once'

# If orchestrator dies:
# 1. Check status
ls ~/sf_bema/results/exp12_1/validated/ | wc -l  # should be 185
ls ~/sf_bema/results/exp13/validated/ | wc -l    # target: 120

# 2. Restart monitor
cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
nohup bash -c 'EXP=exp13 python3 -u ~/distributed_tpu_training/coordinator.py --monitor' >> /tmp/monitor_exp13.log 2>&1 &

# 3. Restart babysitter
nohup bash ~/distributed_tpu_training/babysit_exp13.sh >> /tmp/babysit_exp13.log 2>&1 &

# 4. Restart VM scanner
nohup bash -c 'EXP=exp13 SCAN_INTERVAL=600 bash ~/distributed_tpu_training/vm_scanner.sh' >> /tmp/vm_scanner_exp13.log 2>&1 &

# 5. Deploy to specific VM
EXP=exp13 TPU_NAME=<vm_name> bash ~/distributed_tpu_training/submit.sh --sweep

# 6. Copy results when done
mkdir -p ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
cp ~/sf_bema/results/exp13/validated/*.json ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
```

## Session 3 Changes (2026-03-11 ~02:30)

### 9. coordinator.py — VMState.label_of() helper
- Added `VMState.label_of(item)` static method: handles both str and dict assignment items
- Fixed `save()` line 650: `VMState.label_of(item)` instead of `item['label']` (crashed exp12_1 monitor)
- Fixed `rebalance()`: uses `VMState.label_of()` for protected/movable filtering
- Fixed `orphan handling`: uses `VMState.label_of()` for orphan detection
- Fixed `existing_labels`: uses `VMState.label_of()` for set comprehension

### 10. v4 VMs — antlr4 missing (killed all v4 training)
- **Root cause**: v4 VMs missing `antlr4-python3-runtime` → hydra import fails → all training crashes with rc=1
- Workers wrote `failed_rc1` done receipts for every config → went idle → appeared "stuck"
- **Fix**: Installed antlr4 on all 8 v4 VMs, cleared 45 stale done receipts, redeployed
- **setup.sh fix**: Added `gcloud storage cp` fallback for antlr4 tarball (was only trying .whl)

### 11. vm_scanner.sh — Fixed VM creation flags
- `--runtime-version` → `--version` (correct gcloud flag)
- Added `--internal-ips` (required by org policy)

### 12. v6e-ue1d-1, v6e-ue1d-2 preempted
- Both VMs preempted, 16 orphaned exp13 configs
- Monitor auto-reassigned via cross-zone retry to v6e-ew4a-2
- Recreation attempts failed (external IP policy)

### 13. babysitting skill created
- `~/.claude/skills/babysitting/SKILL.md` — docs for autonomous overnight operation

## Running Processes (2026-03-11 ~02:45)
| Process | PID | Log |
|---------|-----|-----|
| babysit_exp13 | 1590436 | /tmp/babysit_exp13.log |
| babysit_exp12_1 | 3835484 | /tmp/babysit_exp12_1.log |
| coordinator --monitor (exp13) | 2159055 | /tmp/monitor_exp13.log |
| coordinator --monitor (exp12_1) | 2159509 | /tmp/monitor_exp12_1.log |
| vm_scanner (exp13) | 2592253 | /tmp/vm_scanner_exp13.log |

## Active Fleet (2026-03-11 ~02:45)
| VM | Zone | Type | Experiment | Sessions |
|----|------|------|------------|----------|
| v6e-ew4a-2 | europe-west4-a | v6e-8 | exp13 | 8 |
| v6e-ew4a-4 | europe-west4-a | v6e-8 | exp12_1 | 8 |
| v6e-ew4a-5 | europe-west4-a | v6e-8 | exp13 | 8 |
| v6e-ew4a-6 | europe-west4-a | v6e-8 | exp13 | 8 |
| v6e-ew4a-8d | europe-west4-a | v6e-8 | exp13 | 7 |
| v6e-ue1d-3 | us-east1-d | v6e-8 | exp13 | 8 |
| v4-uc2b-1..7 | us-central2-b | v4-8 | exp13 | 8 each |
| v4-uc2b-spot3 | us-central2-b | v4-8 | exp13 | 8 |
| v5e-ew4b-1 | europe-west4-b | v5litepod-4 | exp13 | 4 |

## Session 4 (2026-03-11 ~03:20 UTC)

### 14. exp13 deployment status
- All VMs deployed with exp13 workers ~03:00 UTC
- v6e VMs: XLA compiling, step 1 reached on v6e-ew4a-2
- v4 VMs: stuck at step 1/889 for 15+ min (investigating — likely slow first XLA compile with broken cache)
- v5e-ew4b-1: running but impractical (~100s/step)
- v6e-ew4a-4: dual sessions (exp12_1 on chips 2-7 + exp13 idle with 0 configs)

### 15. Stale done receipts → retry storms
- 8 stale done receipts found at `gs://gcp-researchcredits-blocklab-1-us-central2/coord/exp13/done/`
- Monitor sees "done but no result" → retries → cross-zone retry to v6e-ew4a-2
- v6e-ew4a-2 now has 19 configs (overloaded), while v6e-ew4a-4/5 have 0
- 48 retry entries in state.json

### 16. exp13 config distribution (uneven)
| VM | Configs |
|----|---------|
| v6e-ew4a-2 | 19 (overloaded from retries) |
| v4-uc2b-1/2/3 | 11 each |
| v4-uc2b-4/5/6/7, spot3 | 9 each |
| v5e-ew4b-1 | 8 |
| v6e-ew4a-6, 8d, ue1d-3 | 9 each |
| v6e-ew4a-4, 5, ue1d-1, ue1d-2 | 0 each |

### 17. v4-8 has 4 chips, not 8 — CRITICAL FIX
- **Root cause**: v4-8 = 8 cores = 4 chips (2 cores/chip). VM configs had CHIPS_PER_HOST=8, PROCS_PER_HOST=8
- Workers 4-7 crash: `RuntimeError: Failed to get global TPU topology` (/dev/accel4-7 don't exist)
- Workers 0-3 train normally (~11s/step, ~3.1h/config)
- **Fix**: Changed all 8 v4 VM configs to CHIPS_PER_HOST=4, PROCS_PER_HOST=4
- **Impact**: Half of v4 worker slots were dead. Configs assigned to workers 4-7 were silently lost.
- **Redeployment**: All 8 v4 VMs redeployed with `--sweep`

### 18. exp12_1 remaining 9 configs — actively training
- v6e-ew4a-4 training all 9 at step 100/1778. ETA ~4h (07:15 UTC) for 8/9, ~6.5h for all 9.
- Redundant training also on v6e-ew4a-5/6/8d (chips 0-1). First to finish wins.
- No action needed.

## Running Processes (2026-03-11 ~03:20)
| Process | PID | Log |
|---------|-----|-----|
| babysit_exp13 | 2816752 (+dup 3013477) | /tmp/babysit_exp13.log |
| babysit_exp12_1 | 3835484 | /tmp/babysit_exp12_1.log |
| coordinator --monitor (exp12_1) | 2845832 | /tmp/monitor_exp12_1.log |
| coordinator --monitor (exp13) | 2851048 | /tmp/monitor_exp13.log |
| vm_scanner (exp13) | 2592253 | /tmp/vm_scanner_exp13.log |

## GCS Heartbeat Paths
Workers write heartbeats to: `$BUCKET/coord/$EXP/heartbeat/$WORKER_ID.json`
- v6e: `gs://gcp-researchcredits-blocklab-europe-west4/coord/exp13/heartbeat/`
- v4: `gs://gcp-researchcredits-blocklab-1-us-central2/coord/exp13/heartbeat/`
- v5e: `gs://gcp-researchcredits-blocklab-europe-west4/coord/exp13/heartbeat/`
- ue1d: `gs://gcp-researchcredits-blocklab-us-east1/coord/exp13/heartbeat/`

---

## Session 2026-03-11 Afternoon (pull-based architecture - overnight fixes)

### Fix 1: All exp13 v4-type tasks failing (24 tasks)
**Root cause**: VM `t1v-n-7fd447a3-w-0` had env_fail (torch/hydra/omegaconf/transformers missing).
All tasks failing with exit_code=1. NOT a hardware or training script issue.
**Fix**: Re-queued all 24 failed exp13 tasks back to pending (retries=0).
**Status**: Tasks back in pending queue, will be picked up by working VMs.

### Fix 2: deploy_babysitter.sh missing PATH (v4 VM Python issue)
**Root cause**: deploy_babysitter.sh starts without setting PATH. On v4 VMs,
system `python3` may not have torch/hydra. Also installs torch* skipped for v6e
but v4 VMs need torch from tpu_core wheels.
**Fix**: 
- Added `export PATH=$HOME/miniconda3/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH` at top
- Added v4-specific torch install from `gs://.../wheels/tpu_core/` if /dev/accel exists and torch missing
**File**: `/tmp/deploy_babysitter.sh` → uploaded to all 3 GCS buckets

### Fix 3: deploy_babysitter.sh overrides TPU_NAME with GCP internal name
**Root cause**: Line 140: `TPU_NAME=$(curl ... instance/name)` overrides env var with GCP
internal name (t1v-n-XXXXXXXX-w-0). This breaks zone mapping in dashboard.
**Fix**: Changed to only use metadata if TPU_NAME env var is empty.
Also added ZONE variable preservation.
**File**: `/tmp/deploy_babysitter.sh` → uploaded to all 3 GCS buckets

### Fix 4: vm_requester.sh not passing TPU_NAME/ZONE to deploy_babysitter.sh
**Root cause**: deploy_babysitter() function calls `WANDB_MODE=xxx bash deploy_babysitter.sh`
without setting TPU_NAME or ZONE. GCP VM env provides internal name instead.
**Fix**: Changed to `TPU_NAME=${name} ZONE=${zone} WANDB_MODE=${wandb_mode} bash deploy_babysitter.sh`
**File**: `~/distributed_tpu_training/pull/vm_requester.sh`

### Fix 5: vm_requester.sh too slow (only 1 VM/zone/cycle, blocks on deploys)
**Root cause**: `wait` after deploy loop blocked VM creation. `break` limited to 1 new VM/zone/cycle.
**Fix**: 
- Removed `wait` from deploy loop (deploys run in background)
- Changed to create up to 3 VMs/zone/cycle
- Wait only happens after Phase 3 VM creation

### Fix 6: gcs.py write_heartbeat adds zone/tpu_name fields
Added `zone` and `tpu_name` from env vars to heartbeat JSON.
This allows dashboard.py to correctly map internal VM names to zones.
**File**: `~/distributed_tpu_training/pull/gcs.py`

### Fix 7: dashboard.py _zone_key improved zone detection
Now reads zone/tpu_name from heartbeat JSON when available.
Falls back to chip count inference only if zone not in heartbeat.
**File**: `~/distributed_tpu_training/pull/dashboard.py`

### Processes Running at Session End
| Process | PID | Log |
|---------|-----|-----|
| monitor.py | 1513336 | /tmp/monitor_pull.log |
| vm_requester.sh | 3985042 | /tmp/vm_requester.log |
| overnight_watchdog.sh | 3338877 | /tmp/overnight_watchdog.log |

### Queue Status at Session End (13:27 UTC)
- pending: 43, running: 79, completed: 2, failed: 5, total: 129
- 88 chips on 15 VMs training/compiling
- v6e VMs at step 300-400 — expect first completions in ~1h
- v4 VMs at step 100-200 — expect completions in ~3h

## Session 8 Changes (2026-03-11 ~14:00 UTC)

### Root Cause Found: torch install failure on v6e VMs
- v6e VMs do NOT have torch pre-installed in system python
- torch lives in `~/.local/lib/python3.10/site-packages/` (installed by deploy_babysitter.sh)
- Old script did `pip install torch torchvision` — installs CUDA torch, fails with `libcudart.so.12`
- Fix: `pip install torch==2.9.0 torch_xla==2.9.0 -f https://storage.googleapis.com/libtpu-releases/index.html`

### deploy_babysitter.sh: 4 Golden Rules applied
- Rule 1: `set -uo pipefail`
- Rule 2: `report_phase()` writes boot state JSON to `CTRL/telemetry/<TPU_NAME>_boot.json`
- Rule 3: Idempotent installs — check before installing, skip if already present
- Rule 4: PRE_FLIGHT_CHECK verifies torch+torch_xla+transformers+hydra before launching babysitter
- Separate v4 vs v6e/v5e torch install paths (v4 from GCS wheels, v6e from libtpu-releases)

### train_v2_tpu.py: GCS checkpoint backup
- `save_checkpoint()`: uploads to `$GCS_CHECKPOINT_DIR/{run_name}.pt` after local save
- `load_checkpoint()`: downloads from GCS if local missing
- Uses gcloud storage cp with gsutil fallback for v4 VMs (old SDK)
- `deploy_babysitter.sh` sets `GCS_CHECKPOINT_DIR=${BUCKET}/checkpoints/exp13`

### New tools
- `~/distributed_tpu_training/pull/vm_status.py`: Rich per-VM health dashboard
  - Shows: VM type, zone, chips, steps, status, task, age, env_fail, boot phase
  - Color coded: green=training, yellow=compiling, red=stale/failed
  - `--watch N` for live refresh
  - Upload: `gsutil cp ~/distributed_tpu_training/pull/vm_status.py gs://.../pull_code/vm_status.py`

### Code bundle rebuilt
- Rebuilt `/tmp/sf_bema_code_exp13.tar.gz` with updated train_v2_tpu.py
- Uploaded to all 3 GCS buckets

### Failed tasks re-queued
- 5 exp12_1 tasks (all from v6e-ue1d-2 missing data) reset retries=0 and moved to pending

### SSH pattern established
- gcloud --command with multi-command doesn't work without bash -c wrapper
- Pattern: write script to GCS, then `--command="gsutil cp gs://...script.sh /tmp/ && bash /tmp/script.sh"`
- Audit scripts: `/tmp/vm_audit.sh`, `/tmp/vm_deep_audit.sh` on GCS pull_code/

## 2026-03-11 Session: Overnight Babysitting Agent

### Issues Found and Fixed

1. **v6e-ew4a-1**: torch/torch_xla missing (fresh VM, not pre-installed).
   - Root cause: pip install torch installed CUDA torch 2.10.0 which shadowed system TPU torch
   - Fix: Removed CUDA torch, installed torch_xla 2.9.0 via libtpu-releases
   - Status: torch_xla working, babysitter deploy in progress

2. **v6e-ue1d-1**: PREEMPTED at 17:57 UTC.
   - Fix: Deleted, recreating as v6e-ue1d-1

3. **v4-uc2b-7**: Dead VM (heartbeats stale 45+ min, 6+ retries)
   - Fix: Reclaimed all 5+ tasks to pending, cleaned stale heartbeats
   - VM recreated as v4-uc2b-8 (immediately PREEMPTED again)

4. **v4-uc2b-8**: Immediately PREEMPTED after creation
   - Deleted

5. **v6e-ue1d-2**: Stuck at step 1 for 35+ min (XLA cache deserialization)
   - 2458 lines of UNIMPLEMENTED warnings filling log
   - Root cause: Stale XLA cache (v6e fresh cache is 67 files)
   - Fix: Cleared /tmp/xla_cache, redeployed babysitter (SSH intermittent)

6. **deploy_babysitter.sh**: Critical bug - `pip install torch` installs CUDA torch
   - LESSON: NEVER pip install torch on v6e VMs
   - Fix: Removed torch from pip install list, added proper detection logic
   - Uploaded to all 3 GCS buckets

### GCS Code Updates
- train_v2_tpu.py: Added GCS checkpoint backup (GCS_CHECKPOINT_DIR env var)
- deploy_babysitter.sh: v3 with pre-flight assertions, proper torch handling
- sf_bema_code_exp13.tar.gz: Updated in all 3 buckets

### Task Management
- 11 failed tasks re-queued with retries=0
- Stale heartbeats from 4 unknown chips + 8 ad53d7ef chips cleaned
- 2 tasks reclaimed from dead VMs

### Fleet Status at 14:23 UTC
- Active: 11 VMs, 60 chips, all at step 100-200
- CREATING: v6e-ew4a-3, v6e-ue1d-5/6/7 (4 new VMs)
- Expected first completions: 15:00-16:00 UTC

### [2026-03-11 20:30-21:03 UTC] — WANDB_MODE=online fix + v5e device detection fix

#### Session: Overnight babysitter initial run

**WANDB_MODE=online causing mass task failures:**
- Root cause: v6e-ew4a VMs (europe-west4-a, internet zone) had WANDB_MODE=online in their babysitter process environment
- Effect: Training jobs called wandb.init() which needed API key → crash → exit_code=1 → retries++
- After 10 retries: 22+ exp13 tasks moved to failed/ and 2 exp12_1 tasks
- Fix 1: Added 'WANDB_MODE': 'disabled' to babysitter.py subprocess env dict (overrides parent env)
- Fix 2: Force-redeployed all 3 v6e-ew4a VMs with WANDB_MODE=disabled
- Fix 3: Re-queued all failed tasks (reset retries=0)
- Files: ~/distributed_tpu_training/pull/babysitter.py (line 229: WANDB_MODE disabled in env dict)

**v5e device path detection bug:**
- Root cause: deploy_babysitter.sh checked /dev/vfio/devices/vfio* for chip count
- v5e uses /dev/vfio/0,1,2,3 (NOT /dev/vfio/devices/)
- Effect: FAILED_NO_DEVICES error on v5e-ew4b-1
- Fix: Added /dev/vfio/[0-9]* path check before the error fallback, both in VM type detection and chip count
- Files: ~/distributed_tpu_training/pull/deploy_babysitter.sh (lines ~122-131, 340-348)

**v6e-ue1d-2 torch missing:**
- torch_tpu_wheels.tar.gz (930MB) download failed silently during deploy (directory empty)
- Manually triggered download/install via /tmp/install_torch.sh script (PID 29319)
- Will complete in ~5-10 min, then need FORCE_REDEPLOY to pick it up

**v5e-ew4b-1 redeployed** with fixed deploy_babysitter.sh + WANDB_MODE=disabled

### [2026-03-11 21:09 UTC] — Fix v4_pullback optimizer buffer deletion crash + relaunch disruption analysis

#### Changes
1. **`shared/optimizers/ema_regularized_adamw_v4_tpu.py`**: Changed `foreach` default from `hasattr(torch, "_foreach_mul_")` (True on v6e torch 2.9) to `False`.
   - Root cause: `_step_foreach` did mixed bf16/fp32 in-place ops (`torch._foreach_add_(p_data, adam_step)` where p_data=bf16, adam_step=fp32). On XLA, each in-place mixed-precision op replaces the bf16 storage buffer. Accumulated lazy graph state at step ~700 caused old buffer pointer to be referenced after replacement → `RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576]`.
   - Fix: `_step_loop` path processes params individually with explicit fp32 casts — safer on XLA.
   - Code bundle rebuilt and uploaded to all 3 GCS buckets.

2. **Relaunch disruption (root cause identified)**: Mass xla_compile wave after relaunch was caused by:
   - Killing old pull processes created a ~5min gap with no heartbeat updates
   - New vm_requester started and saw all VMs with heartbeat age > 2700s
   - Deployed babysitters to ALL VMs simultaneously, killing healthy training
   - All chips restarted → mass xla_compile
   - **Prevention**: Before any future relaunch, either wait for natural stale heartbeats or add startup delay to vm_requester (skip first N cycles).

3. **us-central1-a v5e**: "User does not have permission" — zone quota not usable. Already in ZONES list but all creates fail. Will remove from active zones.

#### Files Modified
- `~/sf_bema/experiments/shared/optimizers/ema_regularized_adamw_v4_tpu.py` (foreach=False)
- Code bundle: `/tmp/sf_bema_code_exp13.tar.gz` → all 3 GCS buckets `/code/`

---
### Session 2026-03-12 ~01:20-01:35 UTC (Context resumption)

#### vm_requester.sh — us-central1-a removed from ZONES
- **Removed** `"us-central1-a v5litepod-4 v2-alpha-tpuv5-lite 16 v5e-uc1a"` from ZONES array
- Was generating 16 `RESOURCE_EXHAUSTED` errors per cycle (wasting API quota)
- Uploaded to all 3 GCS buckets; vm_requester restarted at PID 249637

#### vm_requester.sh — FORCE_REDEPLOY=1 in startup script
- **Bug found**: newly created VMs boot, run startup script → deploy_babysitter.sh sees stale heartbeat from preempted old VM (<2700s old) → guard triggers → exits without deploying
- **Fix**: `STARTUP_META` now includes `FORCE_REDEPLOY=1` so fresh boots always deploy
- Root cause: preempted VM leaves stale heartbeat; new VM with same name treats it as "healthy"
- Uploaded to all 3 GCS buckets; vm_requester restarted

#### Files Modified
- `~/distributed_tpu_training/pull/vm_requester.sh` (removed us-central1-a, added FORCE_REDEPLOY=1 in startup)
- All 3 GCS buckets `/pull_code/vm_requester.sh` updated

---
### Session 2026-03-12 ~02:00-02:15 UTC (Standalone system hardening)

#### dashboard.py — fixed hardcoded /889 step denominator
- Line 388: `f"{step}/889"` → `str(step)` (exp12_1 runs 1778 steps, not 889)

#### babysitter.py — per-experiment checkpoint isolation
- `run_training` now sets `CHECKPOINT_DIR=/tmp/ckpts/{experiment}/` per-task
- `GCS_CHECKPOINT_DIR` computed per-task: `{base}/{experiment}/`
- Prevents cross-experiment checkpoint sharing (exp12_1 ≠ exp13 checkpoints)
- `cleanup_checkpoint(run_name, ckpt_dir)` now uses exact run_name (not label glob)
- Pre-training cleanup removed: per-exp dirs prevent contamination, train_v2_tpu.py handles resume from GCS

#### vm_requester.sh — dead VM auto-delete + HEALTH_CHECKS quota guard
- **Deploy attempt counter**: track consecutive deploy attempts per VM
  - After MAX_DEPLOY_ATTEMPTS=5 (2.5h at 30min cooldown), delete dead READY VM
  - Frees health-check quota consumed by env_fail/dead VMs
  - Counter resets when VM gets a fresh heartbeat
- **HEALTH_CHECKS quota guard**: check quota once per cycle
  - If within 5 of limit: log "QUOTA GUARD: blocked" and skip all VM creates
  - Prevents "You have reached HEALTH_CHECKS limit" errors
- FORCE_REDEPLOY=1 in startup metadata (fix: stale heartbeat blocked fresh boot deploy)
- us-central1-a removed from ZONES (16 API failures/cycle saved)

#### Files Modified
- `~/distributed_tpu_training/pull/dashboard.py`
- `~/distributed_tpu_training/pull/babysitter.py` (checkpoint isolation)
- `~/distributed_tpu_training/pull/vm_requester.sh` (dead VM + quota guard + FORCE_REDEPLOY + no us-central1-a)
- All 3 GCS buckets updated

---

## Session 10 — 2026-03-12 (~02:00–04:00 UTC)

### Bugs Found and Fixed

#### Bug 1: `overnight_watchdog.sh` flock guard — FD inheritance
**Root cause**: Used `exec 9>lock; flock -n 9` (same as old vm_requester bug). Subshells from `$()` and `&` jobs inherit FD 9. When parent watchdog died (unknown cause), children held FD 9 → new watchdog couldn't start → watchdog stayed dead for ~90 min (03:12–03:45 UTC), leaving no process watchdog or failed-task requeuer.

**Fix**: Same PID file + kill -0 pattern as vm_requester.sh:
```bash
_WD_PID_FILE="$HOME/.locks/overnight_watchdog.pid"
if [ -f "$_WD_PID_FILE" ]; then
  _OLD_PID=$(cat "$_WD_PID_FILE" 2>/dev/null)
  if [ -n "$_OLD_PID" ] && kill -0 "$_OLD_PID" 2>/dev/null; then
    echo "overnight_watchdog already running (PID $_OLD_PID) — exiting"; exit 0
  fi
fi
echo $$ > "$_WD_PID_FILE"
trap 'rm -f "$_WD_PID_FILE"' EXIT
```

**Files**: `~/distributed_tpu_training/pull/overnight_watchdog.sh` + uploaded to all 3 GCS buckets.

#### Bug 2: Multiple `vm_requester.sh` instances (4 running simultaneously)
**Root cause**: Watchdog's `ensure_vm_requester()` spawns new instance. But PID file check sometimes races/fails — especially after watchdog restarts. Result: 4 concurrent vm_requester instances (PIDs 2686413, 2745115, 2911218, 3113535) consuming 4× HEALTH_CHECKS quota.

**Fix**: Manually killed extra instances. Root cause is the watchdog itself being unstable (Bug 1 above). Fixed watchdog guard → fewer spurious restarts.

#### Bug 3: 9 tasks stuck in `failed/` (never auto-requeued)
**Root cause**: `requeue_failed()` in watchdog only requeues tasks with `retries < MAX_RETRIES_REQUEUE (8)`. Some tasks had retries=11+ from prior sessions and were permanently skipped. After watchdog died (Bug 1), no requeuing happened for ~90 min.

**Fix**: Manually requeued all 9 failed tasks with retries reset to 0. (3 exp12_1 + 6 exp13)

**Note for v2**: Need to either (a) separate "soft fail" vs "hard fail" retries, or (b) raise MAX_RETRIES or (c) allow operator override. In current setup, tasks with retries>8 stay failed forever even if the failure was transient.

### Progress at Session End
- exp13: 93/120 validated
- exp12_1: 181/185 validated
- Fleet: 4 v6e-ew4a (READY) + 1 CREATING + 7 v4-uc2b + 2 v6e-ue1d = 13 VMs
- Failed tasks: 0 (all requeued)
- Control plane: 1× monitor.py, 1× vm_requester.sh, 1× overnight_watchdog.sh

### Files Modified
- `~/distributed_tpu_training/pull/overnight_watchdog.sh` — PID-based single-instance guard (same fix as vm_requester.sh)
- All 3 GCS buckets updated

### V2 Design Notes Added
- Flock guard is fundamentally broken for long-running scripts with background children. Always use PID file + kill -0.
- Permanent task failure (retries > threshold) needs operator override mechanism.
