# TPU Guide

## Mental Model

**Each VM is independent.** One VM = one experiment in production. Parallel = within a single VM (maximize chip usage with 1 proc per chip).

**3-phase workflow**: Explore all VMs → optimize each independently → pick best → kill rest.

## Generic Coordinator System

New experiments only need a 6-line `.env` config + a Python module with `build_configs()` and `run_single()`. See `COORDINATOR.md` for the full design.

```bash
# Generic commands (replace submit_tpu_job_12c_v2.sh)
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --setup
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --auto
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --sweep
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --cancel
EXP=<name> bash ~/tpu_guide/watch.sh
```

Experiment configs: `~/tpu_guide/experiments/<exp>.env`
Coordinator: `~/tpu_guide/coordinator.py` (GCS-based pull model, heartbeat, auto-reclaim)

**Legacy**: `submit_tpu_job_12c_v2.sh` still works for exp12c (hardcoded, static partitioning).

## "Submit exp to TPU" — 3-Phase Workflow

### Phase 1: Prepare code for TPU
- **NEVER modify original GPU code.** Create `*_tpu.py` wrappers/copies with XLA adaptations.
- Naming: `train_v1.py` → `train_v2_tpu.py`, `run.py` → `run_tpu_v2.py`, etc.
- XLA rules: no `.item()`, use `torch._foreach_*`, static shapes
- Test locally: `PJRT_DEVICE=CPU python test_xla_full.py`

### Phase 2: Explore — test ALL VMs, optimize each independently

#### Step 1: Request all available VMs
Goal: find the best VM type. Prefer newest gen (v6e > v5e > v4), then largest (more chips).

#### Step 2: Setup each VM
```bash
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --setup
```

#### Step 3: Preflight each VM (5 min) — collect timing/memory data
```bash
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --preflight
```
- Measures: step time, peak HBM/chip, XLA compile time
- Calculate `PROCS_PER_HOST`: `floor((HBM_chip - 5GB_runtime) / HBM_per_proc)` with buffer
- **Actively manage estimates**: update this doc with every data point

#### Step 4: Launch coordinated sweep on each VM
```bash
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --sweep
```
- Workers pull configs from GCS queue (no static partitioning)
- Automatic dedup: no double work across VMs
- Dead worker detection via heartbeat (25min TTL)

#### Step 5: Babysit each VM (~10 min)
- Confirm all processes running, loss decreasing
- **Update this doc** with any new timing/memory measurements

### Phase 3: Pick best VM, kill the rest
- **Pick best by**: TPU generation (v6e > v5e > v4), then VM size (more chips), then throughput
- Cancel others: `EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --cancel`
- One VM runs the full experiment from here on
- Monitor: `EXP=<name> bash ~/tpu_guide/watch.sh`

## VMs (all Spot)

| TPU_NAME | Type | Hosts | Chips/h | Procs | Step | cfg time | cfg/h | Status |
|----------|------|-------|---------|-------|------|----------|-------|--------|
| v6e-ue1d | v6e-16 | 4 | 4 | 16 | 4.9s | ~2.4h | 6.6 | RUNNING |
| v6e-ew4a-16 | v6e-16 | 4 | 4 | 16 | 5.6s | ~2.7h | 5.9 | RUNNING |
| v6e-ew4a | v6e-8 | 1 | 8 | 8 | 4.9s | ~2.4h | 3.3 | RUNNING |
| v4-uc2b | v4-32 | 4 | 4 | 16 | 8.4s | ~4.1h | 3.9 | RUNNING |
| v4-uc2b-spot | v4-32 | 4 | 4 | 16 | 8.3s | ~4.0h | 4.0 | RUNNING |

Bucket prefix: `gs://gcp-researchcredits-blocklab-`

## Commands (generic submit.sh)

| Command | What it does |
|---------|-------------|
| `--setup` | Install packages, deploy code+data+model |
| `--preflight` | Quick run (~5 min): XLA compile + measure step time/HBM |
| `--auto` | Preflight → sweep |
| `--sweep` | Launch coordinated sweep (GCS pull model) |
| `--gen-queue` | Generate GCS work queue from build_configs() |
| `--push-code` | Deploy code only |
| `--cancel` | Kill all tmux sessions + zombie processes on all workers |
| `--pull-results` | Download results from GCS |
| `--reclaim` | Reclaim stale claimed configs |
| `--status` | Show tmux sessions + queue status |

## Monitoring

```bash
EXP=<name> bash ~/tpu_guide/watch.sh    # 5-min auto-refresh progress table
```

## Memory & Resource Estimates (actively managed)

**Model**: SmolLM2-135M (162.8M params), eff_bs=128, v3_tpu optimizer

| Metric | v6e | v4 | v5e | Source |
|--------|-----|-----|------|--------|
| HBM/chip | 32 GB | 32 GB | 16 GB | spec |
| XLA runtime overhead | ~5 GB | ~5 GB | ~5 GB | measured |
| HBM/process | ~10.8 GB | ~10.8 GB | ~10.8 GB | measured v6e preflight |
| Usable HBM/chip | 27 GB | 27 GB | 11 GB | chip - 5GB runtime |
| Procs/chip | 1 (safe, ~11GB buffer) | 1 (safe, ~11GB buffer) | **OOM** (10.8+5=15.8 > 16) | HBM math |
| XLA compile | ~5 min | ~6 min | — | measured |
| Step time | **~4.9s** | ~8.4s | — | measured (live) |
| 1 config (1778 steps) | **~2.4h** | ~4.1h | — | measured (live) |
| Checkpoint size | ~2.6 GB | ~2.6 GB | — | calculated |

Steps per config = 1778 (889 steps/epoch × 2 epochs).

### Per-VM throughput (single VM doing 225 configs alone)

| VM | Type | Procs | cfg/h | 225 cfgs | Notes |
|----|------|-------|-------|----------|-------|
| v6e-ue1d | v6e-16 | 16 | 6.6 | ~34h (~1.4d) | Fastest |
| v6e-ew4a-16 | v6e-16 | 16 | 5.9 | ~38h (~1.6d) | |
| v4-uc2b | v4-32 | 16 | 3.9 | ~58h (~2.4d) | |
| v4-uc2b-spot | v4-32 | 16 | 4.0 | ~56h (~2.3d) | |
| v6e-ew4a | v6e-8 | 8 | 3.3 | ~68h (~2.8d) | Smallest |

### Multi-VM exploration mode (current)
All 5 VMs running independently (same experiment, GCS coordination skips completed configs).

## Checkpointing & Coordination

- **Rolling checkpoint**: 1 per process (~2.6 GB), mid-config resume on preemption
- **Resume-safe**: on restart, scans local `outputs/` + GCS for completed configs
- **GCS coordination** (multi-VM only): completed configs marked on GCS (`completed/<label>.done`), all VMs skip them
- Storage cost: negligible ($0.02/GB/month)

## Gotchas

- All VMs use `LAUNCH_MODE=single` + `TPU_VISIBLE_CHIPS` for chip isolation
- v4 zombies: `fuser /dev/accel*`. v5e/v6e: `fuser /dev/vfio/*`
- No-internet VMs (v4-uc2b, v4-uc2b-spot, v6e-ue1d): `WANDB_MODE=disabled` (not `offline`)
- Loss: starts ~1.63, drops to ~1.27 by step 100
