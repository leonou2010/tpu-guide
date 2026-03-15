# exp13_rerun Test Run Findings

Track issues observed during the run. Fix before next experiment.

Categories: **[ROBUSTNESS]** **[STABILITY]** **[SPEED]** **[CORRECTNESS]**

---

## Pre-Launch Findings (before exp13_rerun)

### Known issues going in

| # | Category | Issue | Priority |
|---|----------|-------|----------|
| 1 | ROBUSTNESS | HEALTH_CHECKS quota still 75 (need 5000) — caps fleet at ~15 VMs | P0 |
| 2 | ROBUSTNESS | Stale processes from old experiment must be manually killed before launch | P1 |
| 3 | SPEED | No task failure log upload — rc=1 diagnosis requires SSH | P1 |
| 4 | SPEED | MAX_DEPLOY_ATTEMPTS=5, DEPLOY_COOLDOWN_S=1800 — dead VMs tie up quota for 2.5h | P1 |
| 5 | ROBUSTNESS | No retry classification (infra vs code error) — code bugs exhaust 3 retries uselessly | P1 |
| 6 | CORRECTNESS | Per-step JSONL not uploaded to GCS results path — lost after cleanup_gcs.sh | P1 |
| 7 | CORRECTNESS | monitor.py doesn't download JSONL during validation — not local after cleanup | P2 |

---

## Stability Watch List (monitor actively)

Things known to be unstable — record each occurrence with timestamp + task_id:

| # | Category | What to watch for | How to detect |
|---|----------|--------------------|---------------|
| S1 | STABILITY | VM restarts mid-training (process crash, not preemption) | heartbeat step resets to 1 without PREEMPTED state change |
| S2 | STABILITY | XLA recompile on every babysitter restart | log `[preflight]` lines — should only appear once per fresh VM |
| S3 | STABILITY | Spot preemption rate (VMs leaving) | count PREEMPTED deletions in vm_requester.log |
| S4 | STABILITY | Checkpoint resume working after VM recreate | step should continue from last checkpoint, not reset to 0 |
| S5 | STABILITY | GCS checkpoint save lag — if VM deleted before checkpoint saved, full restart | compare GCS checkpoint step vs last heartbeat step |
| S6 | SPEED | XLA cache hit rate — fresh compile = 5min overhead per restart | watch for 5min+ gap at step 1 in heartbeat timing |
| S7 | ROBUSTNESS | Tasks stuck in running/ after VM deletion | running/ count stays high after vm_requester deletes VM |
| S8 | SPEED | Deploy time per VM — target <5min from VM ready to babysitter training | vm_requester.log timestamps |
| S9 | STABILITY | Multiple babysitter processes competing on same VM | `pgrep -c babysitter` > 1 on any VM |

## During Run Findings

<!-- Format: [TIMESTAMP UTC] [CATEGORY] description -->

- [2026-03-12 ~19:30 UTC] [STABILITY] us-east1-d v6e zone exhausted — all 8 creation attempts returned "no more capacity". Zero ue1d VMs for entire run. **Fix needed**: vm_requester should back off faster on repeated RESOURCE_EXHAUSTED (currently retries every cycle).
- [2026-03-12 ~19:30 UTC] [ROBUSTNESS] 27 tasks failed permanently (retries=7) on v5e VMs: `OSError: libtpu not found`. v5e torch_xla install broken. All failed tasks manually re-queued with retries=0. **Fix needed**: v5e should be excluded from fleet or libtpu path fixed before next run.
- [2026-03-12 ~19:30 UTC] [SPEED] All VMs in xla_compile at launch — expected, but ~5-10 min dead time per VM. XLA cache being used (downloaded from GCS).
- [2026-03-12 ~19:45 UTC] [STABILITY] **ROOT CAUSE OF ALL FAILURES**: babysitter.py had `LAUNCH_MODE='pjrt'` instead of `'single'`. PJRT spawns child processes via ProcessPoolExecutor; child processes can't find libtpu.so → `OSError: libtpu not found` on ALL VM types (v6e, v4, v5e). 71 tasks failed. Fix: revert to `LAUNCH_MODE='single'` (debug_single_process=True — no child process spawning). **V3 rule: LAUNCH_MODE=single is the ONLY confirmed working mode. Never switch to pjrt without testing libtpu in child processes first.**
- [2026-03-12 ~19:45 UTC] [ROBUSTNESS] **Agent intervention required to fix failures** — static scripts did not detect or recover from the LAUNCH_MODE bug. vm_requester only handles VM-level failures (preemption, deploy fails), not application-level crashes. Monitor requeues but doesn't diagnose root cause. **V3 requirement: system must be fully autonomous — no human/agent babysitting assumed.**

---

## exp13_rerun Relaunch Findings (2026-03-12 ~22:30 UTC)

### Bugs Found During Relaunch

**BUG-R1: Root babysitter survives FORCE_REDEPLOY [CRITICAL]**
- `pkill -9 -f babysitter.py` from kwokchunau SSH user can't kill root-owned processes
- Root babysitter was started in a previous session (sudo/root session unknown origin)
- Old root babysitter keeps claiming tasks → exit_code=2 → 79 tasks burned to failed/ in minutes
- **Fix**: Deploy script now uses `sudo pkill` with fallback. Uploaded to all 3 GCS buckets.

**BUG-R2: `/tmp/babysitter_chip*.log` root-owned, blocks new babysitter**
- Old root babysitter creates `/tmp/babysitter_chip0_*.log` etc. owned by root
- New kwokchunau babysitter: `PermissionError: Permission denied: '/tmp/babysitter_chip1_*.log'`
- Training completely blocked
- **Fix**: Deploy script now does `sudo rm -f /tmp/babysitter_chip*.log` before launch

**BUG-R3: `/tmp/boot_state.json` root-owned, blocks telemetry**
- `report_phase()` can't write to root-owned `/tmp/boot_state.json`
- Non-fatal (script continues), but telemetry broken
- **Fix**: `report_phase()` now does `sudo rm -f /tmp/boot_state.json || rm -f` before writing

**BUG-R4: Wrong GCS bucket for v4 VMs in SSH commands**
- Correct bucket: `gs://gcp-researchcredits-blocklab-1-us-central2` (not `us-central2`)
- Note: deploy_babysitter.sh auto-detects correctly from ZONE — only SSH copy command was wrong
- **Fix**: Used correct bucket in SSH commands

**BUG-R5: populate --clear races with old babysitters**
- Running populate --clear while broken babysitters are active → burned retries
- By the time populate finishes, failed/ already has 79 entries (5-11 retries each)
- **v3 Fix**: Drain mode — signal babysitters to stop claiming before running populate

### deploy_babysitter.sh Changes Made
1. `report_phase()`: `sudo rm -f /tmp/boot_state.json` before writing
2. Kill section: `sudo pkill` + fallback to user pkill for all process kills
3. Added `sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log /tmp/boot_state.json`

---

## exp13_rerun Re-Relaunch Findings (2026-03-12 ~23:00 UTC)

**ROOT CAUSE: v6e VMs have CUDA torch (`2.9.0+cu128`), not TPU torch**
- System-wide torch (`/usr/local/lib/python3.10`) is CUDA-enabled (no libtpu.so)
- pip install from libtpu-releases index does NOT install torch — falls back to PyPI CUDA version
- Fix: install from GCS `tpu_core/` wheels: `libtpu-0.0.2.whl`, `torch-2.9.0.whl` (TPU), `torch_xla-2.9.0.whl`
- **GCS location**: `gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_core/`

**BUG-R6: libtpu.so not on C extension lookup path — TPU_LIBRARY_PATH needed**
- After installing `libtpu-0.0.2.whl`, `python3 -c "import torch_xla; print(_found_libtpu)"` returns True
- But `torch_xla._XLAC._xla_get_default_device()` (C extension) still fails: `libtpu not found`
- Root cause: Python's `libtpu.get_library_path()` returns the .so path, but _XLAC reads TPU_LIBRARY_PATH env var
- Fix: After install, detect path and `export TPU_LIBRARY_PATH=$(python3 -c "import libtpu; print(libtpu.get_library_path())")`
- Confirmed working: `TPU_LIBRARY_PATH=~/.local/.../libtpu/libtpu.so python3 -c "import torch_xla; print(torch_xla.device())"` → `xla:0`

**BUG-R7: deploy_babysitter.sh exec redirect blocked by existing file**
- Script starts with `exec >> /tmp/deploy_babysitter.log 2>&1` (line 15)
- If file was created by a previous deploy but is now stale or locked, exec fails silently (exits with set -uo pipefail)
- The `sudo rm -f /tmp/deploy_babysitter.log` in the kill section comes AFTER the exec redirect
- Fix needed: Add `sudo rm -f /tmp/deploy_babysitter.log` BEFORE `exec` on line ~14

**BUG-R8: retry classifier marks libtpu env errors as permanent code_error**
- `libtpu not found` shows as rc=1 with traceback → classified as `code_error_permanent` after 3 retries
- Burns tasks permanently for what is actually a deploy/infra failure
- Fix: Added `libtpu not found`, `libtpu.so`, `OSError: [Errno`, `No module named` to `infra_fail` patterns in babysitter.py

**BUG-R9: vm_requester stale heartbeat threshold allows crash-looping babysitters to block deploys**
- Crash-looping babysitters write heartbeats (status=xla_compile/idle) even when training fails
- vm_requester treats fresh heartbeat as "healthy" → skips deploy → broken babysitter never replaced
- Threshold: 2700s — crash loops must stop running for 45min before vm_requester redeploys
- Fix needed: v3 should check heartbeat STATUS (not just freshness) — if status=xla_compile for >15min, force redeploy

### deploy_babysitter.sh Changes Made (all versions)
1. `report_phase()`: `sudo rm -f /tmp/boot_state.json` before writing
2. Kill section: `sudo pkill` + fallback to user pkill for all process kills
3. Added `sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log /tmp/boot_state.json`
4. Removed broken v6e-specific torch install (libtpu-releases pip) — replaced with GCS tpu_core wheels
5. Added unified `_NEEDS_TPU_TORCH` check for ALL VM types — installs from GCS tpu_core
6. Added `TPU_LIBRARY_PATH` auto-detection from libtpu Python package after install

---

## exp13_rerun Session 3 Findings (2026-03-13 ~02:00–03:20 UTC)

**BUG-S1: FAILED_ENV torch/torch_xla on new v6e VMs**
- New v6e VMs (created from fresh image) have system CUDA torch at `/usr/local/lib/python3.10/dist-packages/` that fails to import
- libtpu-releases pip index contains torch_xla but NOT torch itself (falls back to PyPI)
- `--no-index -f libtpu-releases`: `ERROR: Could not find a version that satisfies the requirement torch==2.9.0 (from versions: none)`
- Root cause (initial wrong fix): `pip install --user --ignore-installed` installs CUDA torch (`2.9.0+cu128`) to `~/.local/`, which shadows system torch and can break torch_xla/libtpu
- **CORRECT Fix**: detect `+cu` in version OR path under `~/.local/` → `rm -rf ~/.local/.../site-packages/torch* torch_xla* libtpu*` → re-check → if still missing, `sudo pip install` (system-wide, no --user)
- **Rule**: NEVER `pip install --user torch` on v6e. User-site CUDA wheel shadow breaks torch_xla/libtpu even if `import torch` succeeds.

**BUG-S2: `USER: unbound variable` crash in deploy_babysitter.sh**
- Script runs with `set -uo pipefail`; in startup SSH context `${USER}` env var is not set
- `sudo chown -R "${USER}:${USER}" /tmp/ckpts/` → immediate exit
- **Fix**: `_cur_user=$(id -un 2>/dev/null || echo "kwokchunau")` + use `$_cur_user`

**BUG-S3: tpu_core wheels (GCS) are v4-only, break v6e**
- `torch-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl` in GCS `wheels/tpu_core/` is a non-CUDA build for v4 VMs
- Installing it on v6e: `ValueError: libcublas.so.*[0-9] not found`
- **Rule**: tpu_core wheels = v4 ONLY. v6e always uses PyPI torch + libtpu-releases torch_xla.

**BUG-S4: CUDA detection direction corrected**
- Initially removed `+cu` check thinking CUDA torch was correct for v6e — WRONG
- `+cu` in version OR path under `~/.local/` means a PyPI CUDA wheel is shadowing system torch
- **Correct behaviour**: treat both as failure → purge user-site → rely on system torch
- **Fix**: purge `~/.local/.../torch* torch_xla* libtpu*`, then re-check; only install if still missing (system-wide via `sudo pip`)

### deploy_babysitter.sh Final State (Session 3)
- v6e torch section: detects if torch/torch_xla importable; if not, installs from PyPI + libtpu-releases with `--user --ignore-installed`
- Preflight: checks `import torch; import torch_xla` and `import transformers`, `import hydra` (no CUDA version check)
- USER variable: uses `id -un` fallback everywhere (safe in minimal SSH env)
- PYTHON_BIN: resolved at start, used consistently throughout
- Confirmed working: ew4a-4 and ew4a-5 both reached IDLE_AWAITING_WORK and started claiming tasks

### v3 Requirements Added (Session 3)
8. v6e torch install MUST use PyPI (not --no-index) for torch; libtpu-releases only for torch_xla
9. `--user --ignore-installed` ensures user-site (~/.local/) shadows system site regardless of what's pre-installed
10. Never install tpu_core GCS wheels on v6e VMs
11. USER env var cannot be assumed in startup SSH scripts; always use `id -un` fallback

### v3 Requirements Added
1. Deploy must verify babysitter user is NOT root after launch
2. Deploy must clean all root-owned temp files at start (comprehensive list) — INCLUDING deploy_babysitter.log BEFORE exec
3. Populate must have drain-and-wait before clearing
4. Babysitter per-chip logs should use PID-namespaced names to avoid conflicts
5. vm_requester must check heartbeat STATUS (xla_compile>15min = stuck) not just freshness
6. Retry classifier: env/infra errors must never be permanent (libtpu, missing modules, OSError)
7. GCS tpu_core wheels must be kept in sync with deployed torch_xla version

---

## exp13_rerun Overnight Session 4 Findings (2026-03-13 05:07–06:00 UTC)

**BUG-O1: monitor.py stuck/silent for 4+ hours (01:04–05:11 UTC)**
- PID 352404 showed `S (sleeping)` state but no log output for 4+ hours
- Likely stuck in a GCS HTTP call with no timeout
- Impact: 0 tasks validated, 0 stale tasks reclaimed during this window
- Fix applied 05:11 UTC: killed + restarted with correct `--exp exp13_rerun:120`
- **v3 fix**: monitor.py must set per-call timeout on all gsutil subprocess calls; add watchdog restart

**BUG-O2: libtpu not found on fresh v6e VMs (CRITICAL — caused 50+ task failures)**
- Root cause: deploy_babysitter.sh skips torch/torch_xla install if system versions are importable
- But system torch_xla from PyPI does NOT bundle libtpu.so — needs separate install
- When system torch_xla loads, `OSError: libtpu not found` on every training attempt
- Affected VMs: ew4a-4, ew4a-7 (both newly deployed without libtpu)
- ew4a-4 burned tasks for ~40 min before identified; ew4a-7 burned ~25 tasks before deleted
- Pattern: status=xla_compile with rapidly-cycling ages (failed and reclaimed new task in <30s)
- Fix applied to deploy_babysitter.sh (05:25 UTC): added libtpu-specific check:
  ```bash
  if ! "${PYTHON_BIN}" -c "import libtpu" 2>/dev/null; then
      sudo pip install libtpu -f https://storage.googleapis.com/libtpu-releases/index.html
  fi
  ```
- Uploaded to all 3 GCS buckets
- **v3 fix**: check libtpu importability AND set TPU_LIBRARY_PATH on ALL v6e/v5e VMs during deploy

**BUG-O3: Spot preemption spike (05:30–05:46 UTC)**
- ew4a-3, ew4a-6, ew4a-7, ew4a-8 all PREEMPTED in short window
- Preemption rate much higher than overnight average
- Only 2 VMs reliably working: ew4a-1 (8 chips, training) + ew4a-2 (8 chips, mixed)
- v4 VMs (uc2b-1..5): all 20 chips healthy and training throughout
- **Note**: ew4a-1 first completions imminent (step 1600/1778 at 05:46 UTC)

**BUG-O4: pgrep -af regex failure (cosmetic)**
- `pgrep -af 'name1\|name2'` uses POSIX BRE; `\|` is literal, not alternation
- Returns exit 1 even when processes exist → agent spawned duplicate processes
- Fix: `pgrep -af 'name1' || pgrep -af 'name2'` or use `ps aux | grep`

### Tasks Burned by libtpu Bug (BUG-O2)

~50 tasks moved to failed/ between 05:13–05:45 UTC from ew4a-4 and ew4a-7. All requeued.

Approximate breakdown:
- ew4a-7 (8 chips × ~3 tasks/chip before deletion): ~25 tasks
- ew4a-4 (8 chips × ~3 tasks/chip during failure window): ~25 tasks
- All had retries=11 at time of detection (accumulated from many cycles)

### Fixes Applied in Session 4

1. Killed stuck monitor.py (PID 352404), restarted with correct params
2. Deleted ew4a-7 (burning tasks with libtpu error) — vm_requester will recreate
3. Deleted ew4a-4 (same issue) — vm_requester will recreate with fixed script
4. Fixed deploy_babysitter.sh: add `libtpu` install for v6e/v5e when missing
5. Uploaded fixed deploy script to all 3 GCS buckets
6. Bulk-requeued 50+ failed tasks
7. Cleaned 14 stale t1v-n-* heartbeat files

### v3 Requirements Added (Session 4)

12. monitor.py must have per-call timeout on all GCS operations (max 30s per gsutil call)
13. monitor.py watchdog: if no log in >5 min, restart automatically
14. deploy_babysitter.sh must check `import libtpu` separately from `import torch_xla`
15. libtpu install is REQUIRED on v6e/v5e regardless of torch_xla import status
16. Rapid task failure detection: if >5 tasks from same VM fail in <10 min → VM alarm

---

## Post-Run Summary

<!-- Fill in after run completes -->

### What broke / was unstable

### Speed bottlenecks

### What to fix before next run

- [2026-03-13 03:42 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-5 is PREEMPTED — deleting...

- [2026-03-13 03:52 UTC] [STABILITY] FAILED exp13_rerun__ema_lr0.005_k0.5 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 03:52:23 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:52 UTC] [CORRECTNESS] FAILED exp13_rerun__ema_lr0.005_k0.7 (worker=v6e-ew4a-2_chip1, retries=6) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id POST_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 03:52 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.001_k0.3 (worker=v6e-ew4a-4_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip7 @ 03:52:44 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:53 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.001_k0.5 (worker=v6e-ew4a-4_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip2 @ 03:52:52 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:53 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.003_k0.7 (worker=v6e-ew4a-4_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip5 @ 03:53:04 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:53 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.001_k0.3 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 03:53:18 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:53 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.003_k0.5 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 03:53:17 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:53 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c0.005_lr0.001_k0.3 (worker=v6e-ew4a-2_chip2, retries=6) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.005_k0.3 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 03:53:21 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.005_k0.5 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 03:53:36 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.003_k0.3 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 03:53:47 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.003_k0.5 (worker=v6e-ew4a-4_chip6, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip6 @ 03:53:55 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c1.0_lr0.001_k0.5 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 03:54:05 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:54 UTC] [STABILITY] FAILED exp13_rerun__v4_c1e-05_lr0.003_k0.3 (worker=v6e-ew4a-5_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip0 @ 03:54:36 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:55 UTC] [STABILITY] FAILED exp13_rerun__v4_c1e-05_lr0.003_k0.7 (worker=v6e-ew4a-4_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip5 @ 03:54:37 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 03:55 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c2e-05_lr0.003_k0.5 (worker=v6e-ew4a-2_chip7, retries=7) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 03:55 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.003_k0.7 (worker=v6e-ew4a-4_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip1 @ 03:54:47 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:09 UTC] [STABILITY] FAILED exp13_rerun__adamw_lr0.001 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 04:09:03 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:09 UTC] [STABILITY] FAILED exp13_rerun__adamw_lr0.005 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 04:09:03 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:09 UTC] [STABILITY] FAILED exp13_rerun__ema_lr0.003_k0.7 (worker=v6e-ew4a-5_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip5 @ 04:05:13 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:09 UTC] [STABILITY] FAILED exp13_rerun__ema_lr0.005_k0.3 (worker=v6e-ew4a-4_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip2 @ 04:09:36 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.001_k0.7 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.003_k0.3 (worker=v6e-ew4a-5_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip5 @ 04:05:13 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.005_k0.3 (worker=v6e-ew4a-4_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip3 @ 04:10:14 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.001_k0.5 (worker=v6e-ew4a-4_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip5 @ 04:10:05 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.001_k0.7 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:10 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.003_k0.3 (worker=v6e-ew4a-5_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip7 @ 04:05:07 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0005_lr0.005_k0.7 (worker=v6e-ew4a-4_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip1 @ 04:10:57 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.001_k0.3 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.001_k0.5 (worker=v6e-ew4a-5_chip6, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip6 @ 04:05:06 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.001_k0.7 (worker=v6e-ew4a-4_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip5 @ 04:10:52 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.003_k0.7 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.005_k0.3 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 04:11:22 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:11 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.002_lr0.005_k0.7 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:12 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.005_lr0.001_k0.7 (worker=v6e-ew4a-4_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip2 @ 04:12:05 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:12 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c0.005_lr0.003_k0.3 (worker=v6e-ew4a-2_chip1, retries=6) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 04:12 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.005_lr0.003_k0.7 (worker=v6e-ew4a-5_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip4 @ 04:05:03 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:12 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.005_lr0.005_k0.3 (worker=v6e-ew4a-5_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip4 @ 04:05:03 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:12 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.005_lr0.005_k0.5 (worker=v6e-ew4a-4_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip2 @ 04:12:36 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.001_k0.5 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.003_k0.5 (worker=v6e-ew4a-4_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip7 @ 04:13:04 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.003_k0.7 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.005_k0.5 (worker=v6e-ew4a-5_chip6, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip6 @ 04:05:06 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.02_lr0.005_k0.7 (worker=v6e-ew4a-4_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip1 @ 04:13:14 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.001_k0.5 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.001_k0.7 (worker=v6e-ew4a-5_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip7 @ 04:05:07 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.003_k0.3 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.003_k0.7 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 04:14:14 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.001_k0.5 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.001_k0.7 (worker=v6e-ew4a-4_chip5, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip5 @ 04:14:09 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:14 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.003_k0.3 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.003_k0.5 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.003_k0.7 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 04:15:03 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.005_k0.3 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.2_lr0.005_k0.7 (worker=v6e-ew4a-5_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip0 @ 04:05:04 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.001_k0.3 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:15 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.001_k0.5 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:16 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.001_k0.7 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:16 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.005_k0.3 (worker=v6e-ew4a-5_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip3 @ 04:05:32 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:16 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.5_lr0.005_k0.7 (worker=v6e-ew4a-4_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip3 @ 04:16:16 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:16 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c1.0_lr0.001_k0.3 (worker=v6e-ew4a-3_chip4, retries=7) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-13 04:16 UTC] [STABILITY] FAILED exp13_rerun__v4_c1.0_lr0.001_k0.7 (worker=v6e-ew4a-4_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip2 @ 04:16:43 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:16 UTC] [STABILITY] FAILED exp13_rerun__v4_c1.0_lr0.003_k0.7 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 04:16:37 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c1.0_lr0.005_k0.7 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c1e-05_lr0.001_k0.5 (worker=v6e-ew4a-4_chip6, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip6 @ 04:17:12 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c1e-05_lr0.001_k0.7 (worker=v6e-ew4a-5_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip0 @ 04:05:04 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c1e-05_lr0.005_k0.7 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 04:17:24 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.001_k0.7 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 04:17:16 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:17 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.005_k0.3 (worker=v6e-ew4a-5_chip2, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip2 @ 04:05:00 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:18 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.005_k0.5 (worker=v6e-ew4a-4_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip7 @ 04:17:28 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:18 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.005_k0.7 (worker=v6e-ew4a-5_chip1, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-5_chip1 @ 04:05:12 ===
  log> /usr/bin/python3: can't open file '/root/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:18 UTC] [STABILITY] FAILED exp13_rerun__v4_c5e-05_lr0.003_k0.5 (worker=v6e-ew4a-4_chip4, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip4 @ 04:18:06 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:18 UTC] [STABILITY] FAILED exp13_rerun__v4_c5e-05_lr0.003_k0.7 (worker=v6e-ew4a-4_chip6, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip6 @ 04:18:25 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:32 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.0002_lr0.005_k0.5 (worker=v6e-ew4a-4_chip7, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip7 @ 04:31:41 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:32 UTC] [STABILITY] FAILED exp13_rerun__v4_c0.05_lr0.005_k0.3 (worker=v6e-ew4a-4_chip0, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip0 @ 04:32:17 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:32 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c0.2_lr0.005_k0.5 (worker=v6e-ew4a-2_chip7, retries=11) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-13 04:33 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c1.0_lr0.003_k0.5 (worker=v6e-ew4a-3_chip5, retries=7) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-13 04:33 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c2e-05_lr0.001_k0.3 (worker=v6e-ew4a-2_chip1, retries=8) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 04:33 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c5e-05_lr0.003_k0.3 (worker=v6e-ew4a-2_chip2, retries=7) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 04:33 UTC] [STABILITY] FAILED exp13_rerun__v4_c5e-05_lr0.005_k0.5 (worker=v6e-ew4a-4_chip3, retries=11) — rc=? exit_code=2
  log> === v6e-ew4a-4_chip3 @ 04:33:02 ===
  log> /usr/bin/python3: can't open file '/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/train_tpu.py': [Errno 2] No such file or directory
  -> requeued

- [2026-03-13 04:33 UTC] [CORRECTNESS] FAILED exp13_rerun__v4_c5e-05_lr0.005_k0.7 (worker=v6e-ew4a-3_chip0, retries=10) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-13 04:37 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-5 is PREEMPTED — deleting...

- [2026-03-13 04:25 UTC] [CORRECTNESS] **BUG-S5: Code bundle `sf_bema_code_exp13.tar.gz` missing `exp13_tpu_rerun/`** — bundle only had `exp13_tpu/train_tpu.py`, but tasks specify `train_script=exp13_tpu_rerun/train_tpu.py`. All 53+ tasks claimed by ew4a-4 and ew4a-5 (recreated from preemptions) failed with exit_code=2 `No such file or directory`. **Fix**: rebuilt bundle to include both `exp13_tpu/` and `exp13_tpu_rerun/`, uploaded to all 3 GCS buckets. **v3 rule**: code bundle must be verified after any task path changes — run `tar tzf bundle.tar.gz | grep train_tpu` before upload.
- [2026-03-13 04:25 UTC] [STABILITY] ew4a-5 root babysitter: failure log showed `/root/sf_bema/` path (HOME=/root) — babysitter ran as root. Preemption then deleted VM; ew4a-6 created as replacement and will get fixed deploy script.

- [2026-03-13 05:03 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-6 is PREEMPTED — deleting...

- [2026-03-13 05:13 UTC] [STABILITY] FAILED exp13_rerun__v4_c2e-05_lr0.001_k0.5 (worker=v6e-ew4a-7_chip7, retries=11) — rc=? infra_crash rc=1
  log> (no log in GCS)
  -> requeued

- [2026-03-13 05:26 UTC] [ROBUSTNESS] FAILED exp13_rerun__adamw_lr0.003 (worker=v6e-ew4a-7_chip7, retries=11) — env/torch error
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/tpu.py", line 359, in library_path
  log>     raise EnvironmentError('libtpu not found')
  log> OSError: libtpu not found
  -> requeued

- [2026-03-13 05:26 UTC] [ROBUSTNESS] FAILED exp13_rerun__v4_c1.0_lr0.005_k0.5 (worker=v6e-ew4a-8_chip2, retries=11) — env/torch error
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/tpu.py", line 359, in library_path
  log>     raise EnvironmentError('libtpu not found')
  log> OSError: libtpu not found
  -> requeued

- [2026-03-13 05:26 UTC] [ROBUSTNESS] FAILED exp13_rerun__v4_c1e-05_lr0.003_k0.5 (worker=v6e-ew4a-4_chip2, retries=11) — env/torch error
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/tpu.py", line 359, in library_path
  log>     raise EnvironmentError('libtpu not found')
  log> OSError: libtpu not found
  -> requeued

- [2026-03-13 05:29 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-5 is PREEMPTED — deleting...

- [2026-03-13 05:39 UTC] [ROBUSTNESS] FAILED exp13_rerun__ema_lr0.001_k0.7 (worker=v6e-ew4a-8_chip5, retries=11) — env/torch error
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/tpu.py", line 359, in library_path
  log>     raise EnvironmentError('libtpu not found')
  log> OSError: libtpu not found
  -> requeued

- [2026-03-13 05:42 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 05:55 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 06:08 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-7 is PREEMPTED — deleting...

- [2026-03-13 06:08 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-6 is PREEMPTED — deleting...

- [2026-03-13 06:08 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-3 is PREEMPTED — deleting...

- [2026-03-13 06:20 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-8 is PREEMPTED — deleting...

- [2026-03-13 06:34 UTC] [ROBUSTNESS] **BUG-O2 libtpu fix CONFIRMED WORKING**: ew4a-3/4/5 deployed with fixed deploy_babysitter.sh (adds `import libtpu` check + install). All 24 chips training at steps 1-100. Zero libtpu errors. Fix uploaded to all 3 GCS buckets at 05:25 UTC.

- [2026-03-13 06:34 UTC] [STABILITY] Fleet recovery after preemption wave: 5 v6e VMs recreated by vm_requester (ew4a-1..5 all READY), 5 v4 VMs stable. 57 chips training, 21/120 validated. Run proceeding normally.

- [2026-03-13 07:50 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 07:50 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

### BUG-O5: Heartbeat-stale false reclaim (11x) — 2026-03-13 08:08 UTC
**Symptom**: `exp13_rerun__v4_c0.0005_lr0.001_k0.3` sent to failed/ with retries=11, last_error=heartbeat_stale. Meanwhile, v4-uc2b-5_chip2 was at step 300/1778 (actively training).
**Root cause**: uc2b-5_chip2 was not writing heartbeats frequently enough (or GCS writes were failing intermittently). Monitor stale-ttl=1800s exceeded → reclaim. But chip was alive. Pattern repeated 11 times.
**Impact**: Wasted 11× partial runs on uc2b-5. Task was in failed/ but chip still running. Double-run result when chip completes.
**Fix for v3**: 
1. Heartbeat frequency must be <stale-ttl/3 (babysitter should heartbeat every 300s when stale-ttl=1800).
2. Monitor reclaim should check if chip's heartbeat shows a *different* task_id before reclaiming (orphan detection). If chip still shows same task_id, don't reclaim even if slightly stale.
3. Consider exponential backoff on retries for heartbeat_stale (vs code errors).
4. Add alert when retries > 5 for same task — manual intervention needed.

- [2026-03-13 08:42 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 08:42 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 09:32 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-2 is PREEMPTED — deleting...

- [2026-03-13 09:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 09:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 09:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

### BUG-O6: User-site torch_xla affects v4 VMs too — 2026-03-13 10:00 UTC
**Symptom**: `exp13_rerun__v4_c2e-05_lr0.005_k0.7` failed 8× on v4-uc2b-2_chip2 with:
`RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted`
Error occurs at first batch in training loop (torch_xla.sync() call).
**Root cause**: uc2b-2 has user-site torch_xla at `~/.local/lib/python3.10/site-packages/torch_xla/`. Same root cause as ew4a-2 PRE_START_SESSION_BARRIER but different error on v4 hardware.
**Pattern**: User-site torch_xla causes:
  - v6e: PRE_START_SESSION_BARRIER (initialization failure)
  - v4: buffer deleted error (runtime failure during first batch)
**Impact**: 8 wasted task attempts on uc2b-2_chip2.
**Fix for v3**: deploy_babysitter.sh must purge user-site torch_xla on ALL VM types (currently only does it on v6e). Check `~/.local/lib/python3.10/site-packages/torch_xla/` before launch; rm -rf if present on v4 VMs too.
**v3 requirement**: Add `PURGE_USER_SITE_TORCH_XLA=1` flag in deploy script, apply on all platforms.

- [2026-03-13 10:24 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 10:24 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 11:16 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-3 is PREEMPTED — deleting...

- [2026-03-13 11:16 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-2 is PREEMPTED — deleting...

- [2026-03-13 11:16 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-5 is PREEMPTED — deleting...

- [2026-03-13 11:16 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-4 is PREEMPTED — deleting...

- [2026-03-13 11:16 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

### BUG-O7: Zombie running tasks after force-redeploy — 2026-03-13 11:20 UTC
**Symptom**: 5 tasks stuck in running/ for 7-9 hours with no results anywhere:
  - exp13_rerun__v4_c0.0005_lr0.005_k0.3 (claimed 560 min ago)
  - exp13_rerun__v4_c0.005_lr0.005_k0.7 (claimed 482 min ago)
  - exp13_rerun__v4_c0.02_lr0.001_k0.3 (claimed 487 min ago)
  - exp13_rerun__v4_c0.05_lr0.003_k0.3 (claimed 414 min ago)
  - exp13_rerun__v4_c1e-05_lr0.001_k0.3 (claimed 485 min ago)
**Root cause**: Tasks were claimed by ew4a-1/2 chips around 02:00-03:00 UTC. Chips likely completed training ~04:30-05:00 UTC but the force-redeploy at 05:25 UTC killed the babysitters mid-result-upload. running/ was never moved to completed/. No result was uploaded.
**Why monitor didn't catch it**: Monitor was stuck 01:04-05:11 UTC (BUG-O1). After restart, orphan detection should have triggered when chips moved to new tasks — but the `claimed_at` in running/ was still old (no update). The orphan detection may have a bug: it reclaims if worker heartbeat's task_id != running task's task_id, but if orphan detection was not consistently checking across all running tasks, these slipped through.
**Fix applied**: Manually moved running/ → pending/ with retries=0. Tasks now running.
**Fix for v3**: 
1. Monitor must reclaim running tasks if claimed_at > 2× task_duration (e.g., >6h for 3h tasks).
2. Add a cap: if task has been in running/ > max_task_hours × 2, force reclaim regardless of heartbeat.
3. After force-redeploy, scan running/ for tasks with old claimed_at and reclaim/requeue them.
4. deploy_babysitter.sh should requeue orphaned running/ tasks before deploying new babysitter.

### BUG-O8: Empty/0-byte pending task files — 2026-03-13 11:20 UTC
**Symptom**: 4 tasks had 0-byte JSON in pending/ for entire run duration, never claimed:
  - exp13_rerun__v4_c0.0005_lr0.005_k0.5
  - exp13_rerun__v4_c0.005_lr0.003_k0.5
  - exp13_rerun__v4_c0.02_lr0.001_k0.7
  - exp13_rerun__v4_c5e-05_lr0.005_k0.3
**Root cause**: populate.py wrote 0-byte files for these 4 tasks. Likely a GCS write failure that silently created empty objects. babysitter.py's claim logic (gsutil mv) succeeded (moves the empty file) but JSON parse failed → babysitter skipped task or crashed claim silently.
**Impact**: 4 tasks unclaimed for 11+ hours (entire run). Discovered only at 11:15 UTC.
**Fix applied**: Reconstructed correct JSON from experiment module, re-uploaded to pending/. Tasks immediately claimed by idle chips.
**Fix for v3**:
1. populate.py must verify each written file is non-empty (stat or cat+len after write).
2. babysitter.py's claim logic must validate JSON after mv → if invalid, move back to pending/ and log error.
3. populate.py should run validation pass after all writes.

- [2026-03-13 12:08 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 12:08 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 13:00 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 13:00 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 13:00 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 13:50 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-1 is PREEMPTED — deleting...

- [2026-03-13 13:50 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-3 is PREEMPTED — deleting...

- [2026-03-13 13:50 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-2 is PREEMPTED — deleting...

- [2026-03-13 13:50 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-5 is PREEMPTED — deleting...

- [2026-03-13 13:50 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-4 is PREEMPTED — deleting...

- [2026-03-13 15:06 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-2 is PREEMPTED — deleting...

- [2026-03-13 15:06 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 15:06 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 15:55 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-1 is PREEMPTED — deleting...

- [2026-03-13 15:55 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-2 is PREEMPTED — deleting...

- [2026-03-13 15:55 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-8: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 15:55 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 16:43 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 16:43 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 17:19 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-1 is PREEMPTED — deleting...

- [2026-03-13 17:19 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-5 is PREEMPTED — deleting...

- [2026-03-13 17:19 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-6 is PREEMPTED — deleting...

- [2026-03-13 17:19 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 18:54 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-1 is PREEMPTED — deleting...

- [2026-03-13 18:54 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 18:54 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:06 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:06 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:06 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:29 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-1 is PREEMPTED — deleting...

- [2026-03-13 19:29 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:29 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:29 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:29 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:52 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:52 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:52 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 19:52 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 20:04 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-7: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 20:16 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 20:16 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 20:51 UTC] [STABILITY] Preemption:   us-east1-d: v6e-ue1d-3 is PREEMPTED — deleting...

- [2026-03-13 21:02 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 21:02 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 21:02 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 21:49 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 21:49 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:01 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-4 is PREEMPTED — deleting...

- [2026-03-13 22:12 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-3 is PREEMPTED — deleting...

- [2026-03-13 22:22 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c0.002_lr0.005_k0.7 (worker=v6e-ew4a-5_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-13 22:23 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-7: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:34 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c1.0_lr0.001_k0.7 (worker=v6e-ew4a-5_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-13 22:34 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c5e-05_lr0.005_k0.7 (worker=v6e-ew4a-2_chip5, retries=4) — python exception
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-13 22:35 UTC] [STABILITY] Preemption:   europe-west4-a: v6e-ew4a-3 is PREEMPTED — deleting...

- [2026-03-13 22:35 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:46 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:46 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:46 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 22:46 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-7: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

## ue1d FAILED_ENV Root Cause Analysis (2026-03-13)

**Root cause**: us-east1-d `v2-alpha-tpuv6e` VM image is a NEWER version that does NOT pre-install torch/torch_xla/libtpu/nvidia (unlike europe-west4-a). PyPI `torch==2.9.0` requires `nvidia-*-cu12` packages that only support Python 3.11+ (can't install on Python 3.10).

**Key facts**:
- ew4a image: pre-installs torch 2.9.0 + nvidia CUDA packages (4.3GB) + torch_xla in /usr/local/
- ue1d image: only basic Python packages (78 total: setuptools, requests, etc.)
- `pypi.org` unreachable from ue1d (no internet) but `storage.googleapis.com` works
- libtpu-releases index only has `libtpu-nightly` (no torch/torch_xla wheels there)
- PyPI torch 2.9.0 cp310 wheel is 900MB and works with --no-deps
- CUDA stubs approach works after torch is installed (DT_RPATH matches)

**Fix applied (2026-03-13)**:
1. Pre-downloaded torch 2.9.0 cp310 + torch_xla 2.9.0 cp310 + libtpu-0.0.2 + small deps from PyPI
2. Uploaded to `gs://<BUCKET>/wheels/torch_v6e_cp310/` in all regional buckets
3. Updated deploy_babysitter.sh: v6e else block now downloads from GCS and installs via pip --no-deps

**Additional finding**: snap-based gsutil (`/snap/bin/gsutil`) fails with "cannot create user data directory: /home/kwokchunau/snap/google-cloud-cli/273: Permission denied" when run as non-root via SSH (startup script runs as root, creates root-owned snap dir). Fix: use `sudo gsutil` or chown the snap dir.

**V3 requirement**: Support ue1d zone. Wheels bundle is set up; startup script will auto-apply fix on new VMs.

- [2026-03-13 22:58 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:09 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:20 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:30 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c5e-05_lr0.005_k0.5 (worker=v6e-ew4a-2_chip2, retries=6) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Socket closed
  -> requeued

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ew4a-4: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-1: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-3: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-2: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:32 UTC] [ROBUSTNESS] VM deploy FAILED_ENV:   v6e-ue1d-5: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=3 applies

- [2026-03-13 23:53 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c0.005_lr0.001_k0.7 (worker=v6e-ew4a-5_chip3, retries=7) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-13 23:53 UTC] [CORRECTNESS] FAILED exp13_rerun2__v4_c0.5_lr0.001_k0.3 (worker=v6e-ew4a-2_chip3, retries=7) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 05:13 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.001_k0.3 (worker=v6e-ew4a-2_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 05:47 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.002_lr0.005_k0.5 (worker=v6e-ew4a-4_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 05:59 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.003_k0.7 (worker=v6e-ew4a-1_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 05:59 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.005_k0.7 (worker=v6e-ew4a-1_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 06:10 UTC] [CORRECTNESS] FAILED exp13_rerun3__adamw_lr0.001 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 06:33 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1e-05_lr0.001_k0.5 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 06:45 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.003_k0.5 (worker=v6e-ew4a-1_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 06:45 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.002_lr0.003_k0.5 (worker=v6e-ew4a-4_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 06:56 UTC] [CORRECTNESS] FAILED exp13_rerun3__ema_lr0.005_k0.5 (worker=v6e-ew4a-4_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 06:56 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.001_k0.7 (worker=v6e-ew4a-4_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 06:56 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.2_lr0.005_k0.3 (worker=v6e-ew4a-5_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 07:08 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.001_k0.3 (worker=v6e-ew4a-4_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 07:19 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1e-05_lr0.003_k0.3 (worker=v6e-ew4a-4_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 07:31 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.001_k0.7 (worker=v6e-ew4a-1_chip3, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-14 07:42 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.005_k0.3 (worker=v6e-ew4a-3_chip7, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 07:54 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.05_lr0.005_k0.7 (worker=v6e-ew4a-4_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:05 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.003_k0.3 (worker=v6e-ew4a-1_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:17 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.001_k0.5 (worker=v6e-ew4a-4_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 08:28 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.001_k0.7 (worker=v6e-ew4a-1_chip3, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:28 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.005_lr0.001_k0.3 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:28 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.003_k0.5 (worker=v6e-ew4a-3_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:29 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.001_k0.7 (worker=v6e-ew4a-4_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 08:40 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.001_k0.3 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 08:52 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.001_k0.3 (worker=v6e-ew4a-1_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 08:52 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.005_k0.7 (worker=v6e-ew4a-5_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 09:03 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.003_k0.5 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 09:26 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.05_lr0.003_k0.7 (worker=v6e-ew4a-3_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 09:26 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.001_k0.5 (worker=v6e-ew4a-2_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 09:37 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.005_k0.5 (worker=v6e-ew4a-4_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 09:38 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.005_k0.3 (worker=v6e-ew4a-5_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 09:38 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.005_k0.7 (worker=v6e-ew4a-5_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 10:00 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.001_k0.7 (worker=v6e-ew4a-5_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 10:35 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.003_k0.7 (worker=v6e-ew4a-3_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 10:46 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.005_k0.5 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 11:09 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.2_lr0.005_k0.7 (worker=v6e-ew4a-2_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 11:21 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.2_lr0.003_k0.7 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 11:44 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.001_k0.5 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 11:55 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.003_k0.3 (worker=v6e-ew4a-1_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 12:18 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.2_lr0.005_k0.5 (worker=v6e-ew4a-2_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 12:30 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.005_k0.3 (worker=v6e-ew4a-5_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 12:41 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.002_lr0.005_k0.7 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 12:53 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.003_k0.7 (worker=v6e-ew4a-2_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 13:39 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.005_lr0.003_k0.5 (worker=v6e-ew4a-4_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 14:02 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.001_k0.5 (worker=v6e-ew4a-2_chip2, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 14:02 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.001_k0.5 (worker=v6e-ew4a-5_chip2, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Socket closed
  -> requeued

- [2026-03-14 14:14 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.003_k0.7 (worker=v6e-ew4a-2_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 14:14 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.005_k0.3 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 14:37 UTC] [CORRECTNESS] FAILED exp13_rerun3__adamw_lr0.003 (worker=v6e-ew4a-1_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 14:37 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.001_k0.3 (worker=v6e-ew4a-2_chip3, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 14:37 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.003_k0.5 (worker=v6e-ew4a-5_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-14 14:37 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.001_k0.3 (worker=v6e-ew4a-5_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Cancelling all calls
  -> requeued

- [2026-03-14 14:49 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.005_k0.3 (worker=v6e-ew4a-1_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 14:49 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.05_lr0.003_k0.3 (worker=v6e-ew4a-2_chip2, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Socket closed
  -> requeued

- [2026-03-14 14:49 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1e-05_lr0.005_k0.7 (worker=v6e-ew4a-5_chip3, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 15:00 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.05_lr0.003_k0.5 (worker=v6e-ew4a-5_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 15:01 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.005_k0.5 (worker=v6e-ew4a-4_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 15:35 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.003_k0.5 (worker=v6e-ew4a-2_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 15:58 UTC] [CORRECTNESS] FAILED exp13_rerun3__adamw_lr0.005 (worker=v6e-ew4a-4_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 15:58 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0005_lr0.005_k0.7 (worker=v6e-ew4a-1_chip5, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 16:10 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.2_lr0.003_k0.3 (worker=v6e-ew4a-1_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline Exceeded
  -> requeued

- [2026-03-14 16:21 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.02_lr0.001_k0.5 (worker=v6e-ew4a-2_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 16:44 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.002_lr0.003_k0.3 (worker=v6e-ew4a-5_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 16:44 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.005_lr0.001_k0.5 (worker=v6e-ew4a-1_chip6, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 16:45 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.005_k0.5 (worker=v6e-ew4a-5_chip0, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 17:07 UTC] [CORRECTNESS] FAILED exp13_rerun3__ema_lr0.003_k0.5 (worker=v6e-ew4a-5_chip3, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 17:08 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.002_lr0.001_k0.7 (worker=v6e-ew4a-2_chip6, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 17:20 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.001_k0.5 (worker=v6e-ew4a-4_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 17:45 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.0005_lr0.005_k0.5 (worker=v4-uc2b-2_chip1, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 17:45 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.5_lr0.005_k0.7 (worker=v4-uc2b-1_chip0, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 17:58 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1.0_lr0.003_k0.5 (worker=v6e-ew4a-3_chip6, retries=6) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 17:58 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.005_k0.3 (worker=v6e-ew4a-4_chip1, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Socket closed
  -> requeued

- [2026-03-14 18:37 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.002_lr0.001_k0.5 (worker=v6e-ue1d-2_chip0, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 18:51 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.2_lr0.001_k0.7 (worker=v6e-ue1d-2_chip6, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 18:51 UTC] [STABILITY] FAILED exp13_rerun3__v4_c2e-05_lr0.003_k0.5 (worker=v6e-ue1d-5_chip5, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:04 UTC] [CORRECTNESS] FAILED exp13_rerun3__ema_lr0.001_k0.7 (worker=v6e-ue1d-3_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 19:04 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.0002_lr0.001_k0.7 (worker=v6e-ue1d-4_chip2, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:04 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.005_lr0.005_k0.3 (worker=v6e-ue1d-2_chip0, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:04 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.005_lr0.005_k0.7 (worker=v4-uc2b-3_chip3, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:04 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.05_lr0.001_k0.3 (worker=v6e-ue1d-2_chip6, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:04 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.05_lr0.005_k0.5 (worker=v6e-ew4a-3_chip3, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 19:04 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.5_lr0.001_k0.7 (worker=v6e-ue1d-5_chip6, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:42 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c1e-05_lr0.003_k0.5 (worker=v6e-ue1d-1_chip1, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id POST_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 19:54 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.002_lr0.003_k0.7 (worker=v4-uc2b-3_chip3, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 19:54 UTC] [STABILITY] FAILED exp13_rerun3__v4_c0.2_lr0.003_k0.5 (worker=v6e-ue1d-5_chip6, retries=4) — rc=? code_error_permanent rc=1 retries=3
  log>   File "/home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu_rerun/../../shared/train/train_v2_tpu.py", line 245, in run_training
  log>     train_ds = load_from_disk(os.path.join(data_path, "train"))
  log>   File "/home/kwokchunau/.local/lib/python3.10/site-packages/datasets/load.py", line 1559, in load_from_disk
  log>     raise FileNotFoundError(f"Directory {dataset_path} not found")
  log> FileNotFoundError: Directory /home/kwokchunau/sf_bema/experiments/exp13_smollm2_smoltalk/data/train not found
  -> requeued

- [2026-03-14 20:19 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c2e-05_lr0.003_k0.7 (worker=v6e-ew4a-5_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-14 21:35 UTC] [CORRECTNESS] FAILED exp13_rerun3__ema_lr0.001_k0.5 (worker=v6e-ue1d-1_chip6, retries=5) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Timeout while waiting for barrier id PRE_START_SESSION_BARRIER broadcasted by barrier master
  -> requeued

- [2026-03-14 22:12 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.0002_lr0.003_k0.3 (worker=v6e-ew4a-2_chip3, retries=6) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/distributed/parallel_loader.py", line 46, in next
  log>     torch_xla.sync()
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 87, in sync
  log>     torch_xla._XLAC._xla_step_marker(
  log> RuntimeError: Check failed: !buffer->IsDeleted(): buffer with shape bf16[49152,576] on device TPU:0 is deleted
  -> requeued

- [2026-03-14 23:02 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c0.5_lr0.003_k0.3 (worker=v6e-ew4a-1_chip7, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Deadline exceeded
  -> requeued

- [2026-03-14 23:40 UTC] [CORRECTNESS] FAILED exp13_rerun3__v4_c5e-05_lr0.003_k0.7 (worker=v6e-ue1d-1_chip4, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id POST_START_SESSION_BARRIER is not ready
  -> requeued

- [2026-03-15 02:34 UTC] [CORRECTNESS] FAILED exp13_rerun3__ema_lr0.001_k0.3 (worker=v6e-ew4a-1_chip0, retries=4) — python exception
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/_internal/pjrt.py", line 107, in initialize_singleprocess
  log>     xm.set_replication(torch_xla.device(), [])
  log>   File "/usr/local/lib/python3.10/dist-packages/torch_xla/torch_xla.py", line 40, in device
  log>     return torch.device(torch_xla._XLAC._xla_get_default_device())
  log> RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready
  -> requeued
