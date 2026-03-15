# V3 Development Plan — exp13_rerun3 Test

**Date**: 2026-03-14
**Test experiment**: exp13_rerun3 (120 tasks, 5×v6e-ew4a + 5×v6e-ue1d + 5×v4-uc2b)

---

## 5 Success Metrics

| # | Metric | Target | Current |
|---|--------|--------|---------|
| 1 | **VM usage ratio** | 100% requested VMs runnable | ew4a ✓, ue1d ?, uc2b ? |
| 2 | **Getting VM ratio** | All QRs created + ACTIVE | Not started (vm_manager not launched) |
| 3 | **Running indicator** | max_step monotonically increasing | Not started |
| 4 | **All VMs ratio** | v4 + v6e all training | v6e ew4a ✓, ue1d ?, v4 pending |
| 5 | **Healthy queuing** | Dashboard shows QR states + VM phases | Dashboard written (v3/dashboard.py) |

---

## Current Status (2026-03-14)

### Completed
- ✅ Step A: CUDA stubs built and uploaded to all 3 GCS buckets
- ✅ Step B: v3 code (babysitter.py, gcs.py, deploy_babysitter.sh) in all 3 buckets
- ✅ Step B.5: v3/dashboard.py written with VM Boot State + QR State panels
- ✅ Step D: exp13_rerun3 queue populated (120 tasks, pending=115)
- ✅ Step C (ew4a-1): deploy reaches IDLE_AWAITING_WORK after libtpu fix

### In Progress
- 🔄 Step C (uc2b-1): deploy launched via nohup, awaiting telemetry

### Pending
- ⏳ Step C (ue1d): all ue1d VMs preempted, test via vm_manager QR
- ⏳ Step E: delete old-style VMs before vm_manager launch
- ⏳ Step F: launch vm_manager + monitor

---

## Bug Fixes Applied to deploy_babysitter.sh

### Fix 1: ew4a torch check (no +cu version rejection)
**Problem**: `_needs_torch()` rejected ew4a's `2.9.0+cu128` as CUDA torch.
**Root cause**: ew4a system torch is correctly CUDA build + torch_xla coexist. Should not check version string.
**Fix**: Replaced version check with direct importability test in ew4a branch.

### Fix 2: v4 sudo pip (system site-packages)
**Problem**: v4 torch installed to user-site, but `PYTHONNOUSERSITE=1` in hard gate blocked it.
**Fix**: Changed all v4 tpu_core installs to `sudo pip install` (system-wide).

### Fix 3: v4 deps before torch (typing_extensions etc.)
**Problem**: `torch --no-deps` failed because typing_extensions not in system python.
**Fix**: Added pre-install loop for all non-torch wheels from tpu_core.

### Fix 4: PJRT_DEVICE gate instead of torch_xla.device()
**Problem**: `torch_xla.device()` → `OSError: libtpu not found` even when libtpu IS loaded (torch_xla 2.9.0 quirk).
**Root cause**: `library_path` property triggers a separate libtpu import that fails even after successful dlopen.
**Fix**: Check `PJRT_DEVICE=TPU` after `import torch_xla` — set at import time, reliable proxy.

### Fix 5: libtpu not installed on ew4a (PJRT defaults to CPU)
**Problem**: `torch_xla._found_libtpu = False` on ew4a. System torch_xla 2.9.0 has no bundled libtpu.so AND `libtpu` Python package not installed.
**Root cause**: `_setup_tpu_vm_library_path()` in torch_xla/__init__.py checks: (1) TPU_LIBRARY_PATH, (2) torch_xla/lib/libtpu.so, (3) `import libtpu`. All 3 fail → `PJRT_DEVICE=CPU`.
**Fix**: Download `libtpu-0.0.2-py3-none-any.whl` from GCS `wheels/tpu_core/` and install when `import libtpu` fails. The wheel contains `libtpu/libtpu.so`.
**Result**: ew4a-1 now reaches IDLE_AWAITING_WORK ✓

---

## Next Steps (ordered)

### Step C-final: Verify uc2b-1 reaches IDLE_AWAITING_WORK
Watch telemetry for `gs://...us-central2.../coord_v2/telemetry/v4-uc2b-1_boot.json`.
v4 installs take ~5-10 min (tpu_core wheel download + install).

### Step E: Delete old-style VMs
```bash
GCLOUD=~/google-cloud-sdk/bin/gcloud
# ew4a (4 VMs - but ew4a-1 was just tested, might skip if still IDLE)
for vm in v6e-ew4a-1 v6e-ew4a-2 v6e-ew4a-3 v6e-ew4a-5; do
    $GCLOUD alpha compute tpus tpu-vm delete $vm --zone=europe-west4-a \
        --project=gcp-research-credits-489020 --quiet &
done
# ue1d (already preempted: -2, -5, -6 gone; check if any remain)
# uc2b (5 VMs)
for vm in v4-uc2b-1 v4-uc2b-2 v4-uc2b-3 v4-uc2b-4 v4-uc2b-5; do
    $GCLOUD alpha compute tpus tpu-vm delete $vm --zone=us-central2-b \
        --project=gcp-research-credits-489020 --quiet &
done
wait && echo "All VMs deleted"
```

### Step F: Launch vm_manager + monitor
```bash
nohup python3 -u ~/distributed_tpu_training/v3/vm_manager.py \
    >> /tmp/vm_manager_v3.log 2>&1 &
echo "vm_manager PID: $!"

nohup python3 -u ~/distributed_tpu_training/v3/monitor.py \
    --exp exp13_rerun3:120 --interval 60 --stale-ttl 1800 \
    >> /tmp/monitor_v3.log 2>&1 &
echo "monitor PID: $!"
```

---

## Architecture: Lessons from levanter/xpk/tpunanny

See `V3_LESSONS.md` for full details. Key patterns applied:

| Source | Pattern | Applied in v3 |
|--------|---------|---------------|
| tpunanny | QueuedResource API | vm_manager.py uses QR API |
| tpunanny | State polling loop | vm_manager thread-per-VM |
| levanter | Health check timeout | monitor.py stale-ttl=1800s |
| levanter | Process group kill | deploy_babysitter.sh SIGKILL + killpg |
| xpk | Exponential backoff | vm_manager retry with backoff (TODO) |
| tpunanny | Rich TUI dashboard | dashboard.py with VM Boot State panel |
| levanter | Libtpu lock cleanup | deploy_babysitter.sh fuser /dev/vfio/* |

---

## Dashboard Usage

```bash
# Watch live (recommended):
python3 ~/distributed_tpu_training/v3/dashboard.py --exp exp13_rerun3:120 --interval 30

# One-shot:
python3 ~/distributed_tpu_training/v3/dashboard.py --exp exp13_rerun3:120 --once
```

Shows: Queue stats, VM Boot State per VM, QueuedResource states, Active workers, Idle workers.

---

## Metric Improvement Roadmap

### Metric 1: VM Usage Ratio (100% runnable)
- **Current blockers**: libtpu missing (Fixed ✓), torch not found (Fixed ✓), wrong CUDA torch (Fixed ✓)
- **Next**: Verify ue1d deploy via QR test. ue1d needs `torch_v6e_cp310/` + CUDA stubs + libtpu.
- **TODO**: Add redeploy retry in vm_manager if telemetry shows FAILED_ENV_* after 10 min.

### Metric 2: Getting VM Ratio (all QRs → ACTIVE)
- **Current**: vm_manager uses QR API (tpunanny pattern). Should handle WAITING_FOR_RESOURCES.
- **TODO**: Dashboard QR panel shows WAITING count. Alert if stuck >30 min.
- **TODO**: vm_manager should delete FAILED QRs and recreate.

### Metric 3: Running Indicator (step monotonic)
- **Current**: heartbeat tracks `{step, task_id}`. Monitor reclaims stale after 1800s.
- **TODO**: Dashboard panel: "max step seen" per VM type, with delta since last refresh.
- **TODO**: Alert if no step progress across ALL VMs for >30 min (fleet stall).

### Metric 4: All VMs Ratio (v4 + v6e running)
- **Current**: ew4a ✓ (confirmed). ue1d: needs verification. v4: pending.
- **TODO**: Add v5e support. ue1d uses torch_v6e_cp310 wheels — verify CUDA stubs are extracted correctly.

### Metric 5: Healthy Queuing (dashboard)
- **Current**: QR State panel added to dashboard.py.
- **TODO**: Show QR state transitions (WAITING → ACTIVE transition rate).
- **TODO**: Show age of each task in queue (detect stuck pending tasks).
