# Next Iteration TODO (Pull System V2)

Items discovered during exp13 that must be fixed before the next experiment.
**Priority order: P0 = blocks correctness, P1 = blocks scale, P2 = nice-to-have.**

---

## P0 — Correctness / Data Quality

### 1. Per-step training loss (W&B or local JSONL)
- **Problem**: W&B is disabled on all VMs (`WANDB_MODE=disabled`). Validated JSON has only 8 eval points (every 100 steps). No per-step training loss available.
- **Fix option A** (preferred): Enable `WANDB_MODE=offline` on internet VMs (europe-west4). Sync W&B runs to cloud after task completes or per-checkpoint. Requires deploying WANDB_API_KEY securely (GCS secret or env in deploy_babysitter.sh).
- **Fix option B** (simpler): Add `train_loss.jsonl` — append `{"step": N, "loss": X}` every step in `train_v2_tpu.py`. Upload to `gs://.../results/<task_id>_train_loss.jsonl` at end of task (or every 100 steps).
- **Impact**: Without this, sweeps only give 8 eval points. Can't see per-step convergence or divergence.

### 2. LAUNCH_MODE: single → pjrt
- **Problem**: `babysitter.py` line 219 sets `LAUNCH_MODE='single'` which maps to `torch_xla.launch(debug_single_process=True)` — explicitly a DEBUG path per `train_tpu.py` docstring.
- **Fix**: Change to `LAUNCH_MODE='pjrt'` in babysitter subprocess env. `pjrt` = `torch_xla.launch()` without debug flag — the RECOMMENDED path.
- **Note**: The session 5 history says `single` was chosen to avoid process conflicts. But `pjrt` with `CHIPS_PER_HOST=1` + `TPU_VISIBLE_CHIPS=N` should also be 1 proc/chip with no conflicts. Worth retesting.
- **File**: `~/distributed_tpu_training/pull/babysitter.py` line 219.

---

## P1 — Scale / Throughput

### 3. HEALTH_CHECKS quota (hard VM cap)
- **Problem**: Project quota `HEALTH_CHECKS` limit=75, usage=70. ~5 health checks per TPU VM → max ~15 VMs total. Already hitting this: `RESOURCE_EXHAUSTED` errors in vm_requester.log.
- **Fix**: Request quota increase to **300–500** at `console.cloud.google.com → IAM & Admin → Quotas → HEALTH_CHECKS`.
- **Cost**: Free — just a limit increase. Approval: minutes to 1–3 business days for research accounts.
- **Impact**: With 300 quota → ~60 VMs. With 500 → ~100 VMs.

### 4. Remove us-central1-a from ZONES (permission denied)
- **Problem**: Every vm_requester cycle attempts to create v5e VMs in us-central1-a and gets `User does not have permission to submit requests for accelerator type "v5litepod-4" in location us-central1-a`. Creates 16 failed API calls per cycle (max_vms=16).
- **Fix**: Remove `"us-central1-a v5litepod-4 v2-alpha-tpuv5-lite 16 v5e-uc1a"` from ZONES in `vm_requester.sh`.
- **File**: `~/distributed_tpu_training/pull/vm_requester.sh` ZONES array.

### 5. buffer->IsDeleted crash rate tracking
- **Problem**: `RuntimeError: !buffer->IsDeleted(): bf16[49152,576]` crashes are a major utilization killer — task retries → xla_compile wave → ~10 min dead time per chip per crash. The `foreach=False` fix is deployed but not yet on running VMs.
- **Fix**: Add crash classification in babysitter.py — detect `IsDeleted` in exit log → tag task as `infra_crash` → longer cooldown before requeue (e.g., 10 min) vs immediate requeue.
- **Also**: Add alert if crash rate > N per hour in monitor.py/watchdog.

### 6. Dashboard "claimed" metric fix
- **Problem**: Dashboard counts `VMs × chips_per_vm` as "claimed", even if only some chips have fresh heartbeats. Overstates utilization.
- **Fix**: Change "Claimed" to count chips with `hb_age < 2700s` (fresh heartbeats). Show both `fresh_chips` and `nominal_chips` if needed.
- **File**: `~/distributed_tpu_training/pull/dashboard.py`

---

## P2 — Reliability / Ops

### 7. Relaunch grace period in vm_requester
- **Problem**: On relaunch of pull trio, new vm_requester's first cycle sees all VMs with stale heartbeats (gap during kill+restart) and mass-deploys babysitters, killing healthy training.
- **Fix**: Add `STARTUP_GRACE_CYCLES=2` — skip deploy decisions for the first 2 cycles after vm_requester start. Only create new VMs, don't deploy to existing ones.

### 8. Per-task retry classification
- **Problem**: All failures are treated equally (increment retries → requeue). infra failures (SIGKILL, preemption, lost heartbeat) should get high retry limit; deterministic code errors (exit_code=1, Python exception) should get low limit and alert.
- **Fix**: Parse `last_log` for known infra signatures (rc=-9, preempted, heartbeat lost) vs code errors (Traceback, RuntimeError). Set different TTLs in watchdog requeue logic.

### 9. Hostname cache persistence across vm_requester restarts
- **Problem**: `vm_heartbeat_age()` hostname cache lives in `/tmp/vm_hostname_cache/`. Lost on relaunch. First cycle after relaunch won't have hostname fallback.
- **Fix**: Store hostname cache in `$HOME/.cache/vm_hostname/` (survives restarts). Or write to GCS `coord_v2/vm_hostnames/<name>.txt`.

---

## Done this session (reference)
- [x] vm_requester: all-chip heartbeat scan (min age across chips)
- [x] vm_requester: deploy threshold 1800s → 2700s
- [x] vm_requester: per-VM 30-min deploy cooldown
- [x] vm_requester: hostname cache fallback for internal-name heartbeats
- [x] deploy_babysitter.sh: GCS heartbeat guard (TTL 2700s, training/xla_compile/uploading)
- [x] deploy_babysitter.sh: hostname-based heartbeat check (catches t1v-n-* internal names)
- [x] Lock files moved from /tmp to $HOME/.locks/
- [x] ema_regularized_adamw_v4_tpu.py: foreach=False (fixes buffer->IsDeleted crash)
- [x] vm_requester: removed us-central1-a from ZONES (was wasting 16 API calls/cycle)
- [x] vm_requester: startup script now uses FORCE_REDEPLOY=1 (fix: stale heartbeat from preempted VM blocked fresh boot deploy)
- [x] Stopped push system (fleet_manager, coordinator --monitor, submit.sh)
- [x] us-central1-a v5e confirmed: no permission (will remove from ZONES)

---

## Additional Items from 2026-03-11 Analysis

### Offline wheels layout broken (root cause of READY-but-no-heartbeat VMs)
- `deploy_babysitter.sh` references `wheels/torch_tpu_wheels.tar.gz` but GCS has `tpu_wheels.tar.gz`
- `wheels/all_wheels.tar.gz` exists in us-east1 but NOT in europe-west4 or us-central2
- Net effect: no-internet VMs (us-east1-d, us-central2-b) fail preflight → never heartbeat
- Fix: standardize all 3 buckets with identical wheel artifacts + consistent filenames in script

### Dead VMs squatting health-check quota
- READY VMs with 0 heartbeats consume ~5 health checks each doing nothing
- Need "READY + N deploy attempts + still no heartbeat → delete" policy in vm_requester
- Recovering v6e-ew4a-1 + v6e-ew4a-3 = +16 v6e chips for free

### deploy_babysitter: fast path vs full bootstrap
- Full redeploy (installs + downloads) is wasteful if env is already set up
- Fast path: if preflight passes + code/data/model exist → only restart babysitter
- Full path: only when preflight fails or assets missing

### vm_requester: quota guard before VM creation
- Check `HEALTH_CHECKS usage` before each create attempt
- If within 5 of limit: skip creates, log "blocked by HEALTH_CHECKS quota"
- Saves wasted API calls + provides clear signal in logs

### pip consistency
- Replace all bare `pip install` with `python3 -m pip install`
- Log `python3 -V`, pip version, and site-packages path to deploy log
- Eliminates "installed but not importable" class of failures

---

## Evidence-Based V2 Design (2026-03-12, from live exp13 retrospective)

### Numbers at time of analysis
- exp13: 61/120 validated (51%), 50% done, ETA ~6h
- exp12_1: 177/185 validated (96%)
- Fleet: 3 alive v6e VMs (ew4a-2 stale, ew4a-6, ew4a-8d alive), 7 v4 VMs (uc2b-1..6, spot3), 68 chips total
- 62 completed tasks loaded, 0 in failed/ dir, 90 total retries across all task states

---

### A. Task Lifecycle — Retry and Failure Analysis

**Retry distribution (all task states):**
- 0 retries: 62 completed tasks, 18 running, 11 pending = ~82 tasks (71%)
- 1-2 retries: 19 running + 6 pending tasks
- 3-6 retries: 11 running tasks (worst: exp12_1__v3_c5e-06_k0.5_lr0.001_uf10 at retries=6)
- Total retries across all states: **90**
- Tasks with >=1 retry: **33/115 = 29%**

**Failure types (running tasks only, 30 tasks with retries):**
- exit_code=1: 18 tasks (60%) — v4: 14, v6e: 7
- heartbeat_stale: 7 tasks (23%) — stale TTL exceeded (900s monitor TTL)
- worker_moved_on: 2 tasks (7%) — task reclaimed while worker still held it

**Root cause of exit_code=1:** Mostly v4 VMs early in exp13 when env setup was unreliable. v4 tasks account for 14/18 rc=1 failures despite having fewer tasks. v6e rc=1 failures were from WANDB_MODE=online episode (mass crash, now fixed).

**Key insight:** The 90 retries × ~85min avg wasted per retry = ~128 estimated wasted VM-hours of compute. Most retries are from the first few sessions before the v4 env was stable and before WANDB_MODE=disabled was enforced.

---

### B. VM Efficiency — v4 vs v6e

**Actual measured task duration (from result.total_sec):**

| VM type | Chips/VM | Tasks completed | Avg duration | Steps/sec | Tasks/VM/day |
|---------|----------|----------------|--------------|-----------|--------------|
| v6e-8   | 8        | 40             | 96.5 min     | 0.178     | 119          |
| v4-8    | 4        | 22             | 170.1 min    | 0.087     | 34           |

**Throughput ratio: v6e delivers 3.5× more tasks/VM/day than v4.**

v6e completions by VM (over whole exp so far):
- v6e-ew4a-8d: 15 tasks (most productive single VM)
- v6e-ew4a-2: 12 tasks (preempted multiple times)
- v6e-ew4a-6: 7 tasks (live, active)

v4 completions (7 VMs combined): 22 tasks = ~3 each.

**V2 fleet recommendation:** Prioritize v6e-8 VMs. v4 contributes ~18% of throughput at ~3.5× the time cost per task. v4 is worth keeping when v6e quota is exhausted but should not be primary fleet.

**v5e:** Impractical. ~1 task/VM/day at ~100s/step (20× slower than v6e). Do not include in v2 fleet unless no v6e quota.

---

### C. Preemption Analysis

**Total preemption events logged in vm_requester.log:** 15 lines (some duplicate VM names across cycles)

**Unique VMs preempted (from log):**
- europe-west4-a: v6e-ew4a-1, 2, 3, 4, 5, 10 (6 unique VMs)
- us-east1-d: v6e-ue1d-1, 2, 3 (3 unique VMs, ue1d-1 preempted 4× — very unstable)
- europe-west4-b: v5e-ew4b-1 (1 VM)

**ue1d-1 preempted 4 times** = worst preemption offender. us-east1-d zone is highly unreliable for spot VMs.

**Preemption waste:** 10 preemptions × 48min avg = ~8 wasted VM-hours. Minor compared to retry waste.

**vm_requester response:** Auto-detected PREEMPTED state + deleted + recreated. Worked correctly. Startup script ensures fresh babysitter on reboot without SSH deploy.

---

### D. Completion Timeline — Gaps and Valleys

**Hourly completions (exp13):**
```
2026-03-11 15:00:  2 tasks (very first completions — v6e quick tasks)
2026-03-11 21:00:  3 tasks
2026-03-11 22:00: 18 tasks  ← bulk v6e wave 1
2026-03-11 23:00: 17 tasks  ← bulk v6e wave 2
2026-03-12 00:00: 20 tasks  ← mixed v6e + v4
2026-03-12 01:00:  1 task   ← (session in progress)
```

**Gap from 15:00 to 21:00 (6 hours, only 2 completions):** This is the startup/relaunch period. v6e tasks take ~96min. First completions at 15:00 were the 2 adamw baseline tasks (shorter). Bulk v6e wave started ~22:00 = ~7h after launch at 15:00. Consistent with: 5min XLA + 96min training + relaunch disruptions.

**No valley from "relaunch disruption"** — the completion rate was smooth. The gap was just normal startup latency.

---

### E. Reconcile Loop False Alarm — Root Cause

**Symptom:** Every watchdog cycle sees ~170 "tasks missing from all queues!" for exp12_1 and triggers populate.py. Logs show this every ~30 min.

**Root cause:** The reconciler counts:
`pending + running + completed(GCS) + failed + invalidated == expected_total`

But `expected_total` for exp12_1 = 185. After validation, tasks are moved from `GCS completed/` to `local validated/`. The 177 validated tasks are gone from GCS completed/. Reconciler sees only ~8 in GCS, adds up to ~13, sees 185-13=172 "missing".

populate.py correctly handles this by also checking `local validated/` dir and skipping those. So populate fires but correctly skips 177 and only re-queues the 8 truly pending ones. **No data corruption, but wasted 6 gsutil ls calls + populate scan every 30 min.**

**V2 fix:** Reconciler must count local validated/ in the accounted total, OR maintain a GCS-level `validated/` prefix (move JSON there instead of deleting it). Local validated/ is already accessible via the mounted filesystem.

---

### F. Infrastructure Waste Summary

| Category | Quantity | Impact |
|----------|----------|--------|
| Task retries | 90 retries | ~128 wasted VM-hours |
| Preemptions | ~10 unique events | ~8 wasted VM-hours |
| XLA compile overhead | ~5% per task | ~6 min/task (acceptable) |
| Reconcile false alarms | every 30-min cycle | Noise only, no data loss |
| us-central1-a API calls | 16 failures/cycle × 21 cycles | 336 wasted API calls |
| v6e env_fail episodes | 24 VMs logged env_fail | Multiple sessions; now fixed |
| WANDB_MODE=online crash | 22+ tasks hit max_retries | ~1 session; fixed |

**Biggest waste by far: retries (128h) >> preemptions (8h) >> everything else.**

---

### G. Priority-Ordered V2 Improvements (evidence-based)

#### P0 — Data Quality (blocks science)

**G1. Per-step training loss** (existing item #1)
- Evidence: Only 8-9 eval points per run. Convergence curve invisible between evals.
- Implementation: append-only `train_loss.jsonl` every step in `train_v2_tpu.py`, upload to GCS at end. 889 lines @ ~40 bytes = 35KB — trivial.
- File: `~/sf_bema/experiments/shared/train/train_v2_tpu.py`

**G2. LAUNCH_MODE pjrt** (existing item #2)
- Evidence: Current `LAUNCH_MODE='single'` uses `debug_single_process=True` — not production path.
- File: `~/distributed_tpu_training/pull/babysitter.py` line 219.

#### P1 — Retry Rate Reduction (biggest waste: 128h)

**G3. Retry classification: infra vs code errors**
- Evidence: 18 rc=1 (code errors), 7 heartbeat_stale (infra), 2 worker_moved_on (race condition).
- Fix: parse task's last log line for known crash signatures before incrementing retries.
  - `exit_code=-9` or `Killed` → infra crash (SIGKILL) → re-queue immediately, max_retries=10
  - `exit_code=1` + `Traceback` in log → code error → max_retries=3, alert after 2
  - `heartbeat_stale` → infra → re-queue with delay, max_retries=10
- File: `~/distributed_tpu_training/pull/gcs.py` `reclaim_stale()` + `~/distributed_tpu_training/pull/babysitter.py` task upload

**G4. Per-task log upload on failure**
- Evidence: When task exits rc=1, we only know the exit code. Can't diagnose without logs.
- Fix: babysitter.py uploads last 100 lines of training stdout to `coord_v2/logs/task_fail_{task_id}.log` when exit_code != 0.
- File: `~/distributed_tpu_training/pull/babysitter.py` in `chip_worker()` after proc.wait()

**G5. Heartbeat staleness causes premature reclaim (7 events)**
- Evidence: 7 tasks got `last_error=heartbeat_stale` from monitor/watchdog while they were still actually training. All 3 spot-checked high-retry tasks were alive (e.g., exp12_1__v3_c5e-06 at step 1200, retries=6).
- Root cause: heartbeat fires every 300s (5 min). If monitor TTL=900s and GCS write latency is high, 2 missed heartbeats = reclaim.
- Fix options: (a) increase HEARTBEAT_INTERVAL to 120s for more frequent writes, or (b) increase stale TTL to 1800s (30 min). Option (b) is safer — a training task won't go silent for 30 min unless truly dead.
- File: `~/distributed_tpu_training/pull/babysitter.py` line 37 (`HEARTBEAT_INTERVAL`), `~/distributed_tpu_training/pull/monitor.py` `--stale-ttl` default

#### P2 — Fleet Reliability

**G6. Reconcile loop fix** (new finding)
- Evidence: exp12_1 reconciler fires every cycle with 172 "missing" tasks. 177 validated tasks gone from GCS, so accounted=13 << expected=185.
- Fix: in `overnight_watchdog.sh` `reconcile_invariant()`, add `local validated=$(ls ~/sf_bema/results/${exp}/validated/*.json 2>/dev/null | wc -l)` to the accounted sum.
- File: `~/distributed_tpu_training/pull/overnight_watchdog.sh` lines 122-127

**G7. us-central1-a removal** (existing item #4, already done in ZONES but keep confirmed)
- Evidence: 100 API failure calls logged (grep shows 174 "permission" lines, ~16/cycle × 21 cycles).
- Status: Already removed from ZONES per "Done this session" list. Confirm it stays removed.

**G8. HEALTH_CHECKS quota** (existing item #3)
- Evidence: 70/75 at session start, capping fleet at ~14 VMs.
- Action: Request 300-500 at GCP console before v2 launch.

**G9. v6e env_fail recovery loop**
- Evidence: 24 env_fail log files in GCS. Current env_fail = torch/hydra/omegaconf/transformers missing. This happens on fresh recreated VMs before packages install.
- Pattern: vm_requester detects dead babysitter → re-deploys → deploy installs packages → babysitter starts. But if pip install fails (e.g., torch_xla missing and wheel download fails), babysitter exits env_fail → vm_requester re-deploys again → infinite loop every 2 min.
- Fix: after N consecutive deploy attempts with env_fail, mark VM as "broken" for 30 min before retrying. Log `ENV_FAIL_LOCKOUT: {vm}` to /tmp/vm_requester.log.
- File: `~/distributed_tpu_training/pull/vm_requester.sh`

**G10. Fast-path babysitter deploy**
- Evidence: Full deploy (pip install + gsutil download + tar extract) takes 3-8 min. For a healthy VM that just needs a babysitter restart, this is all wasted time.
- Fix: deploy_babysitter.sh should check preflight BEFORE installing packages. If preflight passes (torch importable, model/code present), skip to babysitter launch.
- Current code: installs packages first, then runs preflight. Invert: preflight → if passes, skip install; if fails, install then re-preflight.
- File: `~/distributed_tpu_training/pull/deploy_babysitter.sh`

#### P3 — Observability

**G11. Per-VM throughput in dashboard**
- Evidence: vm_status.py shows step/task/zone but not tasks_completed or steps_per_sec.
- Fix: add `tasks_completed` counter to heartbeat. babysitter increments when task moves to completed/. Dashboard shows per-VM productivity.
- File: `~/distributed_tpu_training/pull/babysitter.py` heartbeat dict, `~/distributed_tpu_training/pull/dashboard.py`

**G12. ETA display in watchdog log**
- Evidence: watchdog logs VALIDATED counts but no ETA. Have to manually calculate.
- Fix: in `report_progress()`, compute rate over last N cycles → ETA. `log "ETA: exp13 ~{eta_hours}h"`.
- File: `~/distributed_tpu_training/pull/overnight_watchdog.sh`

---

### H. V2 Pre-Launch Checklist (do ALL before starting v2)

1. [ ] Request HEALTH_CHECKS quota 300-500
2. [ ] Implement per-step training loss in train_v2_tpu.py
3. [ ] Change LAUNCH_MODE to pjrt and test on 1 v6e chip
4. [ ] Increase heartbeat stale TTL to 1800s OR decrease HEARTBEAT_INTERVAL to 120s
5. [ ] Add task log upload on failure (last 100 lines to GCS)
6. [ ] Fix reconcile loop to count local validated/ dir
7. [ ] Verify us-central1-a removed from vm_requester ZONES
8. [ ] Add env_fail lockout (max 3 consecutive deploys, then 30-min pause)
9. [ ] Test fast-path deploy on a live v6e VM

---

## I. V2 Fleet Robustness Design — "How to Maintain 20+ v6e VMs"

*Added 2026-03-12 after session 10/11 fleet depletion analysis.*

### Root Cause Analysis: Why Fleet Keeps Depleting

**Primary blockers (in order of severity):**

1. **HEALTH_CHECKS quota = 75 (project-wide, shared).**
   - Every TPU VM consumes 1 HEALTH_CHECK upon creation.
   - ~60 used by other GCP resources (load balancers, backend services, old VMs).
   - Only ~15 free for TPU VMs at any time.
   - This hard-caps the fleet. No code fix helps. **Must request 5000.**
   - Quota guard in vm_requester.sh was BROKEN (`--format='value(quotas[METRIC=X].limit)'` returns empty → fail-open). Fixed in session 11: use `--format='json(quotas)'` + python3 parse.

2. **Dead VMs hold quota for hours.**
   - Before: deploy attempt counter added (auto-delete after 5×30min = 2.5h).
   - After: manual deletion freed 3 slots immediately. But still ~2.5h delay in steady state.
   - V2 fix: reduce MAX_DEPLOY_ATTEMPTS to 3, DEPLOY_COOLDOWN_S to 900 (15 min). Dead VMs gone in 45 min max.

3. **torch install bug keeps VMs in env_fail indefinitely.**
   - Root cause: `pip install torch --index-url libtpu-releases` fails silently because libtpu-releases is a find-links page, not a PEP 503 index. `--index-url` can't parse it.
   - Fix applied in session 11: changed to `-f` (find-links) + added post-cleanup re-check before pip install (clearing ~/.local shadow often sufficient).
   - V2 fix: also add pre-flight fast-path: if torch importable BEFORE package install, skip install entirely.

4. **Spot VM capacity in eu-w4a intermittent.**
   - "internal error has occurred" = GCP has no spot v6e capacity in that zone.
   - No fix — this is GCP capacity. Mitigation: spread across multiple zones (eu-w4a + us-e1d).
   - At-capacity periods block fleet growth until GCP releases capacity.

5. **torch env_fail on fresh VMs corrupted by `pip install torch` (CUDA version).**
   - Previously: `pip install torch` without `--index-url` installed CUDA torch 2.10.0, shadowing system TPU torch 2.9.0. ALL subsequent imports fail.
   - Fixed: deploy script now uses `-f https://storage.googleapis.com/libtpu-releases/index.html` for torch, and checks if system torch is accessible after clearing ~/.local shadow.

### V2 Fleet Design Requirements

**To reliably maintain 20+ v6e VMs:**

1. **HEALTH_CHECKS quota = 5000** (pending approval). Non-negotiable for 20 VMs.

2. **Fast dead-VM reclamation** (max 45 min from death to deletion):
   - MAX_DEPLOY_ATTEMPTS=3, DEPLOY_COOLDOWN_S=900
   - On FAILED_ENV telemetry from GCS: immediate delete (don't wait for N attempts)

3. **Idempotent, fast-path deploy_babysitter.sh:**
   - If preflight passes → skip ALL package install → launch babysitter directly.
   - Reduce deploy time from 5-8 min to <30s for healthy VMs.

4. **Correct torch detection on v6e:**
   - Clear ~/.local shadow → check system torch → ONLY install if still missing.
   - Use `-f` (find-links) not `--index-url`.

5. **Reliable quota guard:**
   - Use `--format='json(quotas)'` + python3 parse (not gcloud format filter which returns empty).
   - Block creates when remaining ≤ 5 AND log clearly which quota is the bottleneck.

6. **Multi-zone v6e fleet** (eu-w4a + us-e1d):
   - eu-w4a: spot v6e, up to 8 VMs, internet access.
   - us-e1d: spot v6e, up to 8 VMs, no internet (wheels bundle required).
   - Together: 16 v6e VMs = 128 chips when both zones have capacity.

### V2 Sweep Completion Guarantee

To "finish" a sweep without human intervention:

1. **Permanent failure after K retries** (K=8, current default):
   - After 8 retries, move to `failed_permanent/` — never requeue.
   - Watchdog reports permanently_failed tasks at end.
   - Completion = `validated + permanently_failed == expected_total`.

2. **Atomic task claiming** (prevents ghost tasks):
   - Use GCS object generation-match for atomic `pending/ → running/` transition.
   - OR: monitor deduper kills `pending/` copy when `running/` exists.

3. **Task identity = task_id not run_name:**
   - Checkpoint path = `/tmp/ckpts/{exp}/{task_id}.pt`
   - GCS checkpoint = `{GCS_BASE}/{exp}/{task_id}.pt`
   - This prevents cross-experiment checkpoint contamination for same-hyperparams configs.

4. **step/total_steps in heartbeat** (not hardcoded /889):
   - babysitter.py passes `total_steps` from task config to heartbeat.
   - Dashboard reads it: no hardcoded denominator.

5. **GCS health summary** (`coord_v2/health/system.json`):
   - Written every watchdog cycle: fleet state, validated counts, ETA.
   - Allows external monitoring without SSH.

---

## Session 10 Additions (2026-03-12)

### flock guard is broken for long-running scripts
**Root cause**: `exec 9>lock; flock -n 9` — FD 9 is inherited by ALL background children. When parent dies, children hold lock, new parent can't start.

**Confirmed broken in**: overnight_watchdog.sh (caused 90-min outage session 10), vm_requester.sh (caused multiple instances in earlier sessions).

**V2 rule**: ALWAYS use PID file + kill -0 for single-instance guards:
```bash
_PID_FILE="$HOME/.locks/script.pid"
if [ -f "$_PID_FILE" ]; then
  _OLD_PID=$(cat "$_PID_FILE" 2>/dev/null)
  if [ -n "$_OLD_PID" ] && kill -0 "$_OLD_PID" 2>/dev/null; then
    echo "already running (PID $_OLD_PID) — exiting"; exit 0
  fi
fi
echo $$ > "$_PID_FILE"
trap 'rm -f "$_PID_FILE"' EXIT
```

### MAX_RETRIES_REQUEUE too low
- Current: 8. Tasks with retries>8 permanently stuck (exp13__v4_c1.0 had retries=11+).
- Fix: raise to 20, OR separate infra failures (unlimited retry) from code failures (limited retry).
- Operator reset: `requeue_all_failed.sh` with retries=0 reset.

### Operator reset capability
- Manual requeue command needed (currently requires ssh into blocklab + raw GCS operations).
- V2: add `bash ~/distributed_tpu_training/pull/ops.sh requeue-all-failed` command.

---

## Session 10 Lessons (2026-03-12, exp13 final stretch + infrastructure cleanup)

### Checkpoint device mismatch (root cause of 120+ retries on exp13 final task)
- **Bug**: `load_checkpoint()` in `train_v2_tpu.py` uses `map_location='cpu'`. Optimizer state (ema, theta, m, v) lands on CPU. First `optimizer.step()` fails: XLA params + CPU state → device mismatch → exit_code=1 in ~9s.
- **V2 fix** (required — not optional):
  ```python
  # In train_v2_tpu.py load_checkpoint(), after optimizer.load_state_dict():
  for state in optimizer.state.values():
      for k, v in state.items():
          if isinstance(v, torch.Tensor):
              state[k] = v.to(dev)
  ```
- **Files**: `~/sf_bema/experiments/shared/train/train_v2_tpu.py`, `shared/optimizers/ema_regularized_adamw_v4_tpu.py`

### Storage lifecycle (459 GiB wasted on checkpoints after experiment)
- **Problem**: checkpoints never cleaned up. 3-bucket replication × 100+ configs = ~459 GiB persisting forever.
- **V2 design**: checkpoint auto-delete in babysitter.py after successful validation. Keep at most 1 checkpoint per in-flight task.
- **Script**: `~/distributed_tpu_training/cleanup_gcs.sh` — run after every experiment. Usage: `bash cleanup_gcs.sh --dry-run` then `bash cleanup_gcs.sh`.

### GCS heartbeat cleanup on VM deletion (stale state accumulation)
- **Problem**: vm_requester.sh never deleted GCS heartbeat/telemetry files when deleting VMs → 119 stale files accumulated.
- **V2 design**: `cleanup_vm_gcs()` helper in vm_requester.sh, called at all 3 deletion points (PREEMPTED, stuck-CREATING, dead-VM).
- **Fixed in v1**: added to `~/distributed_tpu_training/pull/vm_requester.sh`, uploaded to all 3 GCS buckets.

### Regional data locality (cross-region egress cost)
- **Problem**: XLA cache only in europe-west4 bucket. v4 VMs (us-central2) pull it cross-region.
- **V2 design**: At experiment setup, sync all assets (model, data, XLA cache, wheels) to all regional buckets. `populate.py` or `sync_assets.sh` runs once before distributing tasks.
- **Temporary fix**: `gsutil -m cp -r gs://gcp-researchcredits-blocklab-europe-west4/xla_cache/ gs://gcp-researchcredits-blocklab-1-us-central2/`

### Monitor.py stall detection (gets stuck every ~3-4h)
- **Problem**: monitor.py process stays alive but stops logging. overnight_watchdog doesn't detect this.
- **V2 fix**: write `LAST_ACTIVE` timestamp file on each monitor cycle. overnight_watchdog kills+restarts monitor.py if file is >600s old.

### Finishing report
- **V2**: monitor.py should emit a structured finishing report when `validated == total`:
  - Wall-clock duration, tasks/hour, VM-hours, retry/preemption breakdown
  - Storage sizes (prompt to run cleanup_gcs.sh)
  - Automatically copy results to local `experiments/<exp>/results/`

