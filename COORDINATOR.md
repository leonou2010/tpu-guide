# TPU Coordinator System — Centralized Push Architecture

Blocklab is the brain. Workers are dumb executors. Key currency: **valid result files on local disk**.

## Architecture

```
BLOCKLAB (coordinator.py)          GCS (per-VM regional bucket)       TPU VMs (workers)
  │                                  │                                  │
  │  --init: build configs,          │                                  │
  │  distribute proportional to      │                                  │
  │  estimated throughput,           │                                  │
  │  push assignments ──────────────>│  <BUCKET>/coord/<EXP>/           │
  │                                  │    assignments/<TPU>.json        │
  │                                  │                                  │
  │  submit.sh --sweep on each VM ─────────────────────────────────────>│
  │                                  │                                  │
  │                                  │   ┌─────────────────────────┐    │
  │                                  │   │ WORKER (each proc)      │    │
  │                                  │   │ Read assignments.json   │    │
  │                                  │   │ Take my_slice[i::N]     │    │
  │                                  │   │                         │    │
  │                                  │   │ For each config:        │    │
  │                                  │<──│   Clean tmp_ staging    │    │
  │                                  │   │   Run training (Popen)  │    │
  │                                  │<──│   Step-coupled heartbeat│    │
  │                                  │<──│   Upload tmp_ staging   │    │
  │                                  │<──│   Commit → results/     │    │
  │                                  │<──│   Write done/<l>.done   │    │
  │                                  │   │   Cleanup checkpoint    │    │
  │                                  │   │                         │    │
  │                                  │   │ All done? Wait 60s,     │    │
  │                                  │   │ re-read assignment      │    │
  │                                  │   │ (may have rebalanced)   │    │
  │                                  │   └─────────────────────────┘    │
  │                                  │                                  │
  │  --monitor (every 60s):          │                                  │
  │    Read heartbeats from ALL ←────│                                  │
  │    Pull new result JSONs   ←─────│  (lightweight, ~KB each)         │
  │    Validate locally              │                                  │
  │    ├─ VALID: → validated/        │                                  │
  │    └─ INVALID: retry or fail     │                                  │
  │                                  │                                  │
  │    After 30min: smart rebalance  │                                  │
  │    ├─ Keep position 0-1 per proc │                                  │
  │    └─ Move position 2+ to fast  ─>│  update assignments/*.json     │
  │                                  │                                  │
  │    Dead VM? (heartbeat > 25min)  │                                  │
  │    └─ Orphan configs → alive VMs─>│  delete dead assignment         │
  │                                  │                                  │
  │  Exit when all validated locally │                                  │
```

## Key Properties

- **Zero contention**: each VM has its own assignment file, each proc has a static slice
- **Workers only access their own regional bucket** (fast, no cross-region)
- **Coordinator reads from all buckets** (blocklab has internet)
- **Result counts when validated on local disk** (`~/sf_bema/results/<EXP>/validated/`)
- **Step-coupled heartbeat**: deadlock → no stdout → no heartbeat → detected
- **Smooth rebalancing**: only moves position 2+ tasks; current + next untouched
- **Rolling checkpoint cleanup**: checkpoint deleted after each config completes

## GCS Layout (per-VM bucket)

```
gs://<VM_BUCKET>/coord/<EXP_NAME>/
  assignments/<TPU_NAME>.json   # [{label, overrides}, ...] for this VM
  heartbeat/<worker_id>.json    # {worker_id, vm, timestamp, step, label}
  results/<label>/summary.json  # Final result (two-phase committed)
  results/tmp_<label>/          # Staging area (incomplete upload)
  done/<label>.done             # Receipt: "worker_id timestamp"
```

## Local State (ground truth)

```
~/sf_bema/results/<EXP>/
  validated/<label>.json        # Coordinator-validated result files
  state.json                    # Coordinator state: assignments, retries, failed
  tmp_pull/                     # Temp download area (cleaned after validation)
```

## Workflow

```bash
# 1. Write experiment code + run_tpu.py with build_configs(), build_command()
# 2. Create ~/distributed_tpu_training/experiments/exp13.env (6 lines)
# 3. Create ~/distributed_tpu_training/vm_configs/<vm>.env for each VM

# 4. Setup + init + sweep ALL VMs (one command each)
EXP=exp13 bash ~/distributed_tpu_training/submit.sh --setup-all
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --init
EXP=exp13 bash ~/distributed_tpu_training/submit.sh --sweep-all

# 5. Start coordinator + babysitter (both long-running)
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --monitor
EXP=exp13 TOTAL=185 bash ~/distributed_tpu_training/babysit.sh

# 6. Monitor (separate terminal)
EXP=exp13 bash ~/distributed_tpu_training/watch.sh
```

Per-VM (for debugging): `EXP=exp13 TPU_NAME=v6e-ew4a bash ~/distributed_tpu_training/submit.sh --setup`

## Rebalancing Strategy

1. **Initial**: All configs distributed proportional to estimated throughput (from STEP_S)
2. **After 30min**: First smart rebalance using actual throughput from heartbeats
3. **Rule**: For each VM, keep first `2 * num_procs` items (= current + next per proc). Move the rest.
4. **Every 1h**: Periodic rebalance to handle drifting conditions
5. **Dead VM**: Orphaned configs immediately appended to alive VMs

## Experiment Module Contract

```python
def build_configs() -> list[tuple[str, list[str]]]:
    """Return [(label, [hydra_override, ...]), ...]"""

def build_command(overrides: list[str]) -> list[str]:
    """Return subprocess command list for one config."""

def run_single(overrides: list[str], dry_run: bool = False):
    """Run one training config. Return subprocess result."""

def preflight():
    """Quick run for timing/HBM estimate."""

# Optional:
def validate_result(data: dict) -> tuple[bool, str]:
    """Experiment-specific validation. Return (is_valid, reason)."""
```

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Heartbeat interval | 5 min | Low GCS traffic |
| Stale TTL | 25 min | 5x heartbeat; no live worker goes 25min silent |
| Monitor poll | 60s | Fast rebalancing, cost-negligible |
| Rebalance delay | 30 min | Wait for real throughput data |
| Rebalance interval | 1 hour | Periodic correction |
| Max retries | 2 | Failed configs retired after 3 attempts |

## Commands

```bash
# Coordinator (blocklab)
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --init       # Distribute configs
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --monitor    # Coordination loop
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --status     # One-shot status
EXP=exp13 python3 ~/distributed_tpu_training/coordinator.py --dry-run    # Print all configs

# Per-VM (via submit.sh)
EXP=exp13 TPU_NAME=<vm> bash ~/distributed_tpu_training/submit.sh --setup
EXP=exp13 TPU_NAME=<vm> bash ~/distributed_tpu_training/submit.sh --preflight
EXP=exp13 TPU_NAME=<vm> bash ~/distributed_tpu_training/submit.sh --sweep
EXP=exp13 TPU_NAME=<vm> bash ~/distributed_tpu_training/submit.sh --status
EXP=exp13 TPU_NAME=<vm> bash ~/distributed_tpu_training/submit.sh --cancel
```

## Failure Modes

| Failure | Detection | Recovery | Waste |
|---------|-----------|----------|-------|
| Worker preempted | 25 min (TTL) | Orphan configs → alive VMs | <=25 min × 1 chip |
| Training deadlock | No stdout → no heartbeat → TTL | Same as preempted | <=25 min × 1 chip |
| Invalid result | Validation on pull | Retry up to 2x, then fail | 1 config time |
| Coordinator down | N/A | Workers keep running; restart --monitor picks up | 0 |
| GCS outage | Worker retries next loop | Self-healing | 0 |
| Old gcloud SDK | `gcloud storage` fails | Auto-fallback to `gsutil` | 0 |
| Stale GCS data | Workers skip all configs | `--init` auto-cleans; if manual: `gcloud storage rm -r .../done/` and `.../heartbeat/` | 0 (no training wasted) |

## Timing

| Phase | v6e | v4 |
|-------|-----|----|
| XLA compile (first config) | ~5-10 min | ~8-12 min |
| Monitor first cycle (many heartbeat files) | 3-5 min | 3-5 min |

## Known Issues

- **v4 VMs (us-central2-b)**: gcloud SDK 347 (2021) lacks `gcloud storage`. GCS helpers auto-detect
  via `_use_gsutil()` and fall back to `gsutil cat`, `gsutil cp`, `gsutil ls`, `gsutil rm`.
- **Dashboard**: `dashboard.py` uses `rich` Live mode → needs interactive terminal. Cannot run via nohup.
- **Stale done receipts**: Previous runs leave done/ receipts on GCS. `--init` now auto-cleans done/ and heartbeat/ dirs. Without this, workers skip all configs thinking they're completed.
- **Worker done_labels cache**: Workers only cache labels they completed themselves (my_done_labels). GCS receipts are re-checked every loop iteration. This prevents stale cached state.
- **Python stdout buffering through tee**: Workers must use `PYTHONUNBUFFERED=1` and `python3 -u`. Without this, logs appear empty even during active training/XLA compile.
- **v4 VMs need setup.sh**: v4 VMs require full `--setup` (package install from GCS wheels). Without it, hydra/torch etc are missing.
