# V3 TPU Training System — Comprehensive Design Plan

> Test experiment: **exp13_rerun3** (120 tasks, same sweep as rerun2)
> Goal: 100% completion, zero manual intervention, all VM types participating

---

## 1. What We Learned From All Three Projects

### tpunanny (martin-marek/tpunanny)
~300 lines. The cleanest approach we've seen. Key insight:

```python
# Thread-per-TPU, each running this loop:
while not stop_event.is_set():
    status = _recreate(tpu_id, ...)  # check/delete/create via QueuedResource API
    if status != 'exists': ran_ssh_script = False
    if not ran_ssh_script:
        # poll until ACTIVE, then SSH deploy
        wait_for_active()
        _run(tpu_id, zone, project, ssh_script)
        ran_ssh_script = True
    stop_event.wait(60)  # re-check every 60s
```

**What we steal:** QueuedResource API (GCP queues spot requests, no "internal error" loops),
thread-per-TPU model (no global cycle blocking), simple 3-state machine.

### levanter (marin-community/levanter)
Full training framework. Key patterns:
- `queued-resources create` then poll until state == `ACTIVE` (not just READY)
- Per-step retries (5x with 5s backoff) at every setup phase
- Explicit `capacity_type → gcloud flags` mapping (no string munging)
- Clean VM type → runtime version mapping

**What we steal:** Capacity type abstraction, polling pattern for QueuedResource.

### Our v2 System
Strong core (GCS pull queue, heartbeats, monitor). 17 bugs found in exp13_rerun.
The coordination layer is good. The VM management layer is what's broken.

---

## 2. Root Causes of All Problems

### A. ue1d VMs never work (FAILED_ENV_TPU_INIT)
**Root cause chain:**
1. Newer v2-alpha-tpuv6e image on ue1d has NO torch pre-installed (ew4a has torch 2.9.0+cu128)
2. torch 2.9.0+cu128 needs 25+ CUDA .so files with exact sonames + versioned symbols
3. Our GCC stub approach used wrong sonames (libcudnn.so.**12** instead of .so.**9**, etc.)
4. Even with correct sonames, versioned symbol `NVSHMEM` in libnvshmem requires a version script
5. Result: `import torch` fails → FAILED_ENV_TPU_INIT → delete → recreate → repeat

**Fix:** Build comprehensive versioned CUDA stubs once on blocklab → tar → GCS → extract on ue1d.
See Section 4A.

### B. Health check quota blocks all VM creates
- HC quota = 75 at last check (need 5000 for 20+ VM fleet)
- vm_requester.sh blocks ALL zone creation if global HC count near limit
- Single exhausted zone poisons the whole fleet
- **Fix:** Request quota increase + per-zone HC tracking in v3.

### C. vm_requester.sh is too slow and fragile
- Dead VM detection: 45 min (3 attempts × 15 min cooldown)
- No queued resources: "internal error" on capacity exhaustion → 10 min backoff
- Single global loop: one hung SSH blocks all zone processing
- `gcloud tpu-vm create` races with GCP capacity → needs client-side retry logic
- **Fix:** Replace with `vm_manager.py` using tpunanny's thread-per-VM + QueuedResource.

### D. v5e / ue1d torch setup fails consistently
- v5e: libtpu.so path not set when LAUNCH_MODE='pjrt' → OSError in child process
- ue1d: CUDA stubs wrong (see A above)
- **Fix:** Strict pre-flight test with PYTHONNOUSERSITE=1 before babysitter launch.
  Fail fast and report clearly instead of burning retries.

### E. Root processes survive kill / user-site torch shadows system torch
- Root babysitter survives `pkill` from user account
- pip install --user creates `~/.local/torch` shadowing `/usr/local/torch`
- **Fix:** Always install to system (`sudo pip3`), always kill with `sudo pkill`.

### F. No autonomous alarm — requires human/agent to detect code bugs
- Monitor requeues failures but doesn't diagnose root cause
- If all tasks on a VM fail within 10 min, no alert — just retries
- **Fix:** Rapid failure detector (5 fails/VM/10min → delete VM, log alert).

---

## 3. V3 Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────┐
│  blocklab (coordinator)                                     │
│                                                             │
│  vm_manager.py          monitor.py          dashboard.py   │
│  ┌─────────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Thread per VM   │    │ Validate     │    │ GCS queue │  │
│  │ QueuedResource  │    │ Reclaim      │    │ VM states │  │
│  │ deploy loop     │    │ Rapid-fail   │    │ Boot tele │  │
│  └────────┬────────┘    └──────────────┘    └───────────┘  │
│           │ SSH + GCS                                       │
└───────────┼─────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────┐
│  TPU VMs  │                  │
│           ▼                  │
│  deploy_babysitter.sh        │
│    (strict phase checking)   │
│           │                  │
│  babysitter.py (per-chip)    │
│    → GCS task queue          │
│    → heartbeats              │
│    → training subprocess     │
└──────────────────────────────┘
            │
    gs://bucket/coord_v2/
    pending/ running/ completed/ failed/ heartbeats/ telemetry/
```

### What changes vs v2

| Component | v2 | v3 |
|-----------|----|----|
| VM creation | `gcloud tpu-vm create` (fragile) | QueuedResource API (GCP queues) |
| VM manager | `vm_requester.sh` (bash, global loop) | `vm_manager.py` (Python, per-VM threads) |
| Death detection | 45 min | Immediate on FAILED_ENV |
| ue1d torch | GCC stubs (wrong versions) | Pre-built versioned stub tarball from GCS |
| v5e support | Broken (PJRT mode) | Fixed (single mode + OOM guard) |
| Dashboard | Heartbeats only | + VM states + boot telemetry panel |
| Alarm | None | Rapid failure detector |
| Drain mode | Abort on populate --clear | Graceful drain signal |

---

## 4. Implementation Plan

### 4A. Build ue1d CUDA Stubs (one-time, do first)
**File:** `v3/scripts/build_cuda_stubs.sh` — run once on blocklab

Problem: ue1d has no CUDA libs. torch 2.9.0+cu128 loads 25+ `.so` files at import.
We build minimal stub `.so` files that satisfy the dynamic linker without real CUDA.

Key facts from ew4a (confirmed):
```
libcudnn.so.9          (not .12 — cuDNN version 9.x)
libcudnn_adv.so.9      (+ 6 more cudnn_* variants)
libcufft.so.11         (not .12 — cufft version 11.x)
libcurand.so.10        (not .12)
libcusolver.so.11      (not .12)
libcufile.so.0         (not .12)
libcusparseLt.so.0     (not .12)
libnccl.so.2           (not .12)
libnvshmem_host.so.3   (not .12 — AND needs versioned symbol NVSHMEM)
libnvToolsExt.so.1     (not .12)
libcublas.so.12        (correct)
libcudart.so.12        (correct — but needs versioned symbol `libcudart.so.12`)
... etc
```

Build each stub with its correct soname + version script:
```bash
# Example for nvshmem (needs versioned symbols)
cat > nvshmem.map << 'EOF'
NVSHMEM {
  global: nvshmem_malloc; nvshmem_free; nvshmem_init; ...17 symbols...;
  local: *;
};
EOF
gcc -shared -fPIC -Wl,-soname,libnvshmem_host.so.3 \
    -Wl,--version-script,nvshmem.map \
    -o libnvshmem_host.so.3 nvshmem_stub.c

# Example for cudart (needs version tag `libcudart.so.12`)
cat > cudart.map << 'EOF'
libcudart.so.12 {
  global: cudaMalloc; cudaFree; cudaMemcpy; ...all cudart exports...;
  local: *;
};
EOF
gcc -shared -fPIC -Wl,-soname,libcudart.so.12 \
    -Wl,--version-script,cudart.map \
    -o libcudart.so.12 cudart_stub.c
```

**To get exact version tags:** `readelf -V /usr/local/lib/python3.10/dist-packages/nvidia/*/lib/*.so*` on ew4a.

Output: `v3/stubs/nvidia_stubs_v6e.tar.gz` (~1MB) → upload to all 3 GCS buckets.

On ue1d deploy:
```bash
gsutil cp ${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz /tmp/
tar xzf /tmp/nvidia_stubs_v6e.tar.gz -C /usr/local/lib/python3.10/dist-packages/
```

### 4B. vm_manager.py (replaces vm_requester.sh)
**File:** `v3/vm_manager.py`

Inspired by tpunanny but adapted for our pull-based multi-VM fleet.

```python
# Core: one thread per configured VM slot
class VMSlot:
    def __init__(self, name, zone, tpu_type, bucket, project):
        ...

    def run(self, stop_event):
        """Main babysit loop for this VM slot."""
        while not stop_event.is_set():
            state = self._ensure_queued_resource()  # create/recreate if needed
            if state == 'ACTIVE':
                if not self._babysitter_healthy():
                    self._deploy_babysitter()  # SSH deploy
            stop_event.wait(60)

    def _ensure_queued_resource(self):
        """Use QueuedResource API. Returns state: ACTIVE/WAITING/FAILED/SUSPENDED."""
        try:
            info = tpu_client.get_queued_resource(name=self.qr_name)
            state = info.state.state.name
            if state in ('FAILED', 'SUSPENDED'):
                # Delete and recreate immediately
                self._delete_queued_resource()
                self._create_queued_resource()
                return 'WAITING'
            return state  # ACTIVE or WAITING_FOR_RESOURCES
        except NotFound:
            self._create_queued_resource()
            return 'WAITING'

    def _create_queued_resource(self):
        """tpunanny's _create() pattern."""
        tpu_client.create_queued_resource(
            parent=f'projects/{PROJECT}/locations/{self.zone}',
            queued_resource_id=self.name,
            queued_resource=QueuedResource(
                tpu=QueuedResource.Tpu(node_spec=[NodeSpec(
                    parent=..., node_id=self.name,
                    node=Node(
                        accelerator_type=self.tpu_type,
                        runtime_version=get_runtime(self.tpu_type),
                        network_config=NetworkConfig(enable_external_ips=False),
                        metadata={'startup-script': ''},  # no startup script — SSH deploy
                    )
                )]),
                spot=QueuedResource.Spot(),
            )
        )

    def _babysitter_healthy(self):
        """Check GCS heartbeat. True if fresh heartbeat from this VM."""
        hb = read_gcs_heartbeat(self.name)
        if not hb: return False
        age = time.time() - hb.get('timestamp', 0)
        return age < 300  # 5 min

    def _deploy_babysitter(self):
        """SSH deploy with 5x retry (levanter pattern)."""
        script = f'gs://{self.bucket}/pull_code/deploy_babysitter.sh'
        for attempt in range(5):
            result = subprocess.run([
                'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
                self.name, f'--zone={self.zone}', '--tunnel-through-iap',
                '--command', f'gsutil cp {script} /tmp/deploy.sh && bash /tmp/deploy.sh',
                '--worker=0',
            ], timeout=180, capture_output=True)
            if result.returncode == 0: return
            time.sleep(5 * (attempt + 1))
        log(f'[{self.name}] deploy failed after 5 attempts')

    def _check_rapid_failure(self):
        """Detect VMs that fail tasks too quickly. Delete and recreate."""
        # Check GCS failed/ for tasks from this VM in last 10 min
        # If > 5 failures, delete VM
        ...


def main():
    # Fleet config (from vm_configs/)
    slots = [
        VMSlot('v6e-ew4a-1', 'europe-west4-a', 'v6e-8', BUCKET_EW4A, PROJECT),
        VMSlot('v6e-ew4a-2', 'europe-west4-a', 'v6e-8', BUCKET_EW4A, PROJECT),
        VMSlot('v6e-ew4a-3', 'europe-west4-a', 'v6e-8', BUCKET_EW4A, PROJECT),
        VMSlot('v6e-ue1d-1', 'us-east1-d',     'v6e-8', BUCKET_UE1D, PROJECT),
        VMSlot('v6e-ue1d-2', 'us-east1-d',     'v6e-8', BUCKET_UE1D, PROJECT),
        VMSlot('v4-uc2b-1',  'us-central2-b',  'v4-8',  BUCKET_UC2B, PROJECT),
        # ... etc
    ]

    stop_event = threading.Event()
    threads = [threading.Thread(target=s.run, args=(stop_event,), daemon=True) for s in slots]
    for i, t in enumerate(threads):
        t.start()
        stop_event.wait(2)  # stagger (tpunanny pattern)

    try:
        while True: stop_event.wait(10)
    except KeyboardInterrupt:
        stop_event.set()
        for t in threads: t.join(timeout=10)
```

**Key improvements over vm_requester.sh:**
- Each VM has its own thread → no global blocking
- QueuedResource API → GCP handles capacity queuing
- FAILED/SUSPENDED → immediate delete+recreate (not 45 min wait)
- 5x SSH retry per attempt (not 1x)
- Python → easier to test, easier to read

### 4C. deploy_babysitter.sh — Strict Phase Verification
**File:** `v3/deploy_babysitter.sh`

Current problem: silent failures at every phase. New rule: **verify everything before proceeding.**

```bash
# Phase 1: Detect VM type
IS_V6E_EW4A=false; IS_V6E_UE1D=false; IS_V4=false; IS_V5E=false
if [ "$ZONE" = "europe-west4-a" ] && ls /dev/vfio/ 2>/dev/null | grep -q .; then
    IS_V6E_EW4A=true
elif ls /dev/vfio/ 2>/dev/null | grep -q .; then
    IS_V6E_UE1D=true  # Any other zone with vfio = ue1d-type
elif ls /dev/accel* 2>/dev/null | grep -q .; then
    IS_V4=true
fi

# Phase 2: Install torch (per type, with HARD VERIFICATION)
if [ "$IS_V6E_EW4A" = true ]; then
    # torch pre-installed — just verify
    PYTHONNOUSERSITE=1 python3 -c "import torch; import torch_xla" || {
        report_phase FAILED_ENV_TORCH_IMPORT; exit 1
    }

elif [ "$IS_V6E_UE1D" = true ]; then
    # Step 1: Install torch + torch_xla from GCS wheels
    install_v6e_wheels_from_gcs
    # Step 2: Install CUDA stubs tarball
    gsutil cp ${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz /tmp/
    sudo tar xzf /tmp/nvidia_stubs_v6e.tar.gz -C /usr/local/lib/python3.10/dist-packages/
    # Step 3: HARD VERIFY
    PYTHONNOUSERSITE=1 python3 -c "import torch; import torch_xla" || {
        report_phase FAILED_ENV_TORCH_IMPORT; exit 1
    }

elif [ "$IS_V4" = true ]; then
    install_v4_tpu_core_wheels
    python3 -c "import torch; import torch_xla; assert torch_xla._found_libtpu" || {
        report_phase FAILED_ENV_TORCH_IMPORT; exit 1
    }
fi

# Phase 3: TPU device init (HARD GATE — exit if fails)
TPU_OUT=$(python3 -c "
import torch_xla
d = torch_xla.device()
print('OK', d)
" 2>&1)
echo "$TPU_OUT" | grep -q "^OK" || { report_phase FAILED_ENV_TPU_INIT; exit 1; }

# Phase 4: Install training deps
# Check each import individually, install only what's missing
for pkg in hydra transformers sympy antlr4 datasets wandb; do
    python3 -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
done
[ -n "$MISSING" ] && install_from_gcs_or_pip "$MISSING"
# HARD VERIFY
python3 -c "import hydra, transformers, datasets, wandb" || {
    report_phase FAILED_ENV_PACKAGES; exit 1
}

# Phase 5: Download code/model/data (idempotent checks)
# Phase 6: Launch babysitter (with sudo process cleanup first)
```

**Key changes:**
- `IS_V6E_UE1D` detected by zone, not just device type
- HARD EXIT at each verification step (not `|| echo "WARNING"`)
- CUDA stubs from tarball (not runtime GCC compilation)
- All installs verified before proceeding

### 4D. Dashboard — VM State Panel
**File:** `v3/dashboard.py`

Add a new panel that fetches:
1. GCS `telemetry/{TPU_NAME}_boot.json` — boot phase per VM
2. `gcloud alpha compute tpus queued-resources list` — WAITING/ACTIVE/FAILED

```python
def fetch_vm_states():
    """Fetch boot telemetry for all known VMs."""
    telemetry = {}
    paths = gcs_list(f"{CONTROL_PLANE}/telemetry")
    raw = gcs_read_batch(paths, max_workers=20)
    for path, data in raw.items():
        vm = os.path.basename(path).replace('_boot.json', '')
        try:
            telemetry[vm] = json.loads(data)
        except:
            pass
    return telemetry

# New panel: VM Boot Status
# Shows each VM: zone | phase | age | status
# phase: BOOTING → INSTALLING_TPU_TORCH → TESTING_TPU_INIT → LAUNCHING_BABYSITTER → training
# Phases in red: FAILED_ENV_TPU_INIT, FAILED_ENV_PACKAGES, FAILED_NO_DEVICES
```

This makes every VM visible: you can see which ones are WAITING_FOR_RESOURCES,
which are BOOTING, which failed at torch install, and which are training.

### 4E. Rapid Failure Detector (in monitor.py)
Add to monitor's reclaim loop:

```python
def check_rapid_failures(window_secs=600, threshold=5):
    """Delete VMs that fail too many tasks too fast — likely broken environment."""
    failed_paths = gcs_list(f"{CONTROL_PLANE}/failed")
    recent_by_vm = defaultdict(list)
    cutoff = time.time() - window_secs

    for path in failed_paths:
        task = json.loads(gcs_read(path))
        if task.get('failed_at', 0) > cutoff:
            vm = task.get('last_worker', '').rsplit('_chip', 1)[0]
            if vm: recent_by_vm[vm].append(task)

    for vm, failures in recent_by_vm.items():
        if len(failures) >= threshold:
            log(f'RAPID FAILURE: {vm} failed {len(failures)} tasks in {window_secs}s — flagging for redeploy')
            # Write a GCS flag that vm_manager picks up
            gcs_write(f"{CONTROL_PLANE}/flags/redeploy_{vm}.json",
                      json.dumps({'reason': 'rapid_failure', 'count': len(failures)}))
```

`vm_manager.py` checks for redeploy flags each cycle and force-redeploys flagged VMs.

### 4F. Drain Mode (populate.py)
```bash
# Before populate --clear:
python3 populate.py --drain  # sets a GCS drain flag, waits for running/ to empty
python3 populate.py --clear --force  # only after drain completes
```

```python
def drain(timeout_secs=3600):
    """Signal all babysitters to stop claiming. Wait for running/ to empty."""
    gcs_write(f"{CONTROL_PLANE}/flags/drain.json", json.dumps({'until': time.time() + timeout_secs}))
    while True:
        running = gcs_list(f"{CONTROL_PLANE}/running")
        if not running: break
        print(f"Draining... {len(running)} tasks still running")
        time.sleep(30)
    gcs_delete(f"{CONTROL_PLANE}/flags/drain.json")
```

In `babysitter.py`, check drain flag before claiming new tasks:
```python
def claim_task(worker_id):
    if gcs_exists(f"{CONTROL_PLANE}/flags/drain.json"):
        return None  # Don't claim during drain
    ...
```

---

## 5. VM Fleet Configuration for v3

### Zones and Quota
| Zone | Type | Quota | Target VMs | Notes |
|------|------|-------|-----------|-------|
| europe-west4-a | v6e-8 | 64c | 5 VMs | Most reliable, has internet |
| us-east1-d | v6e-8 | 64c | 5 VMs | Fixed by CUDA stubs (4B) |
| us-central2-b | v4-8 | 64c | 5 VMs | Working, no internet |
| europe-west4-b | v5e-4 | 64c | 0 initially | Skip until v6e/v4 stable |
| us-central1-a | v5e-4 | 64c | 0 initially | Skip |

**Total v3 fleet:** 15 VMs, 88 chips (5×8 v6e-ew4a + 5×8 v6e-ue1d + 5×4 v4-uc2b? Wait, v4-8 = 4 chips)
Actually: 5×8=40 v6e-ew4a + 5×8=40 v6e-ue1d + 5×4=20 v4-uc2b = **100 chips**

### HC Quota
**BLOCKER:** Request HC quota increase to 5000 before v3 launch.
Manual: GCP Console → IAM & Admin → Quotas → search "HEALTH_CHECKS" → request 5000.

---

## 6. File Structure

```
distributed_tpu_training/
├── v2/                        # Current (keep for reference)
└── v3/                        # New
    ├── vm_manager.py          # Replaces vm_requester.sh (tpunanny-inspired)
    ├── babysitter.py          # Minor changes: drain mode, TPU_NAME guard
    ├── gcs.py                 # From v2 (all 7 fixes applied)
    ├── monitor.py             # + rapid failure detector
    ├── populate.py            # + drain mode
    ├── dashboard.py           # + VM state panel + boot telemetry
    ├── deploy_babysitter.sh   # Rewritten: strict verification, CUDA stubs from GCS
    ├── check_progress.py      # Unchanged
    ├── upload_wandb.py        # Unchanged
    └── scripts/
        ├── build_cuda_stubs.sh    # One-time: build ue1d CUDA stubs on blocklab
        └── upload_wheels.sh       # Ensure all GCS wheel buckets are populated
```

---

## 7. Implementation Sequence

### Step 0 — Build CUDA stubs (1-2 hours)
1. SSH to ew4a-1, run `readelf -V nvidia/*/lib/*.so*` to get exact version tags
2. Write `build_cuda_stubs.sh` on blocklab — compile all stubs with correct version scripts
3. Tar → `nvidia_stubs_v6e.tar.gz` → upload to all 3 GCS buckets
4. Test on ue1d-2: `sudo tar xzf ... && python3 -c "import torch; import torch_xla"`

### Step 1 — deploy_babysitter.sh rewrite (2-3 hours)
1. Rewrite v3/deploy_babysitter.sh with strict phase verification
2. Test on all 3 VM types (ew4a, ue1d, uc2b)
3. Upload to GCS when all types pass

### Step 2 — vm_manager.py (3-4 hours)
1. Write vm_manager.py with QueuedResource API + thread-per-VM
2. Test in dry-run mode (log what it would do without actually creating VMs)
3. Test on 1 VM per zone before full fleet

### Step 3 — dashboard.py additions (1 hour)
1. Add `fetch_vm_states()` using GCS telemetry
2. Add VM Boot Status panel
3. Test with `--once`

### Step 4 — monitor.py + populate.py (1 hour)
1. Add rapid failure detector to monitor
2. Add drain mode to populate + babysitter

### Step 5 — exp13_rerun3 test
1. Run `populate.py --exp exp13_rerun3` (120 tasks, same sweep)
2. Start vm_manager.py (creates VMs via QueuedResource)
3. Start monitor.py
4. Watch dashboard — should see all 3 zones come online
5. Zero manual intervention target

---

## 8. Success Criteria for exp13_rerun3

| Metric | v2 actual | v3 target |
|--------|-----------|-----------|
| Completion | 97/120 (test run stopped) | 120/120 |
| VM types working | ew4a + uc2b only | ew4a + ue1d + uc2b |
| Manual interventions | ~5 (this session alone) | 0 |
| Dead VM detection | 45 min | <5 min |
| Fleet efficiency | 54% (24/44 chips training) | >80% |
| Tasks failed permanently | ~10 per run | 0 (code bugs retry 3x, then pause) |
| ue1d participation | 0% | ≥30% |

---

## 9. Open Questions / Decisions Needed

1. **HC quota:** Has the 5000 quota request been submitted? This is a hard blocker for >15 VMs.

2. **v5e now or later?** v5e is 20x slower than v6e (~100s/step). Only worth it if v6e quota exhausted.
   Recommendation: **skip v5e for exp13_rerun3**, add in v3.1 after v6e/v4 stable.

3. **QueuedResource vs direct tpu-vm create for v4?** v4 uses `tpu-ubuntu2204-base` runtime.
   QueuedResource API should work for v4 too — same `accelerator_type=v4-8`.
   Need to test: does `--spot` work for v4 in us-central2-b?

4. **Number of VM slots per zone:** Currently 5 per zone. With HC quota at 75, can support
   ~15 VMs (5 HC per VM). At 5000, can support the full 320 chip quota (40 VMs).
   Recommendation: start with 5 per zone for exp13_rerun3.

5. **Drain mode timeout:** How long to wait for running tasks to finish before clearing?
   Recommendation: 4h (max task runtime for a single config on v4).

---

## 10. Key Rules Carried Forward from v2

These are confirmed working — do NOT change in v3:

- `LAUNCH_MODE=single` — ONLY confirmed working mode. Never pjrt without testing.
- Pull-based GCS queue — correct design, keep exactly.
- `gcs_read_batch()` with 20 workers — fast parallel GCS reads.
- `reclaim_stale()` with `startup_grace_s=2700` — prevents false reclaim during XLA compile.
- Heartbeat every `min(eval_interval, stale_ttl//3)` seconds.
- Rolling `/tmp/ckpt_*.pt` checkpoints — survive preemption.
- `TPU_VISIBLE_CHIPS=str(chip_idx)` for per-chip isolation.
- Thread-per-chip with 45s stagger to prevent simultaneous vfio contention.
- `os.killpg(pgid, SIGKILL)` for clean subprocess termination.
- XLA cache per-version in GCS — download at deploy time.
- `PYTHONNOUSERSITE=1` in babysitter to prevent user-site shadowing.
