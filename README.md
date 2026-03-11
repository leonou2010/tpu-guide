# TPU Guide

## Mental Model

- **Centralized push model**: Blocklab (coordinator.py) distributes configs to VMs. Workers execute their assignments. No races, no thundering herd.
- **Each VM is independent.** Workers only access their own regional bucket.
- **Key currency**: valid result files on local disk (`~/sf_bema/results/<EXP>/validated/`).
- **Fleet is experiment-specific.** Check `vm_scan.sh` and `vm_configs/` for current state.

Full design doc: `~/tpu_guide/COORDINATOR.md`

## Quick Start

```bash
# 1. Write experiment code (build_configs, build_command, run_single, preflight)
# 2. Create ~/tpu_guide/experiments/<exp>.env (6 lines)

# 3. Launch everything (acquires VMs, setup, init, sweep, monitor, fleet manager)
EXP=<name> bash ~/tpu_guide/run.sh

# 4. Monitor
python3 ~/tpu_guide/dashboard.py --exp <name> --interval 30   # rich TUI
tail -f /tmp/fleet_<name>.log                                   # fleet manager
tail -f /tmp/monitor_<name>.log                                 # coordinator
```

run.sh handles the full lifecycle: VM acquisition → setup → config distribution → sweep → monitor + fleet manager (background). Fleet manager auto-recovers preempted VMs, expands fleet, and copies results when done.

For debugging a single VM:
```bash
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --setup
EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh --sweep
```

## Commands

### Coordinator (blocklab)

| Command | What it does |
|---------|-------------|
| `coordinator.py --init` | Build configs, distribute to VMs proportional to throughput |
| `coordinator.py --monitor` | Coordination loop: pull results, validate, rebalance, reclaim dead |
| `coordinator.py --status` | One-shot status of all VMs and configs |
| `coordinator.py --dry-run` | Print all configs without distributing |

### Fleet (submit.sh, no TPU_NAME needed)

| Command | What it does |
|---------|-------------|
| `--setup-all` | Setup all VMs in `vm_configs/` |
| `--sweep-all` | Launch workers on all VMs |
| `--cancel-all` | Cancel all VMs |

### Per-VM (submit.sh, TPU_NAME required)

| Command | What it does |
|---------|-------------|
| `--setup` | Install packages, deploy code+data+model to all workers |
| `--preflight` | Quick run (~5 min): XLA compile + measure step time/HBM |
| `--auto` | Preflight → sweep |
| `--sweep` | Launch workers on this VM (reads assignment from GCS) |
| `--push-code` | Deploy code only (no packages/data) |
| `--status` | Show tmux sessions + coordinator status |
| `--logs` | Tail live logs from worker-0 |
| `--cancel` | Kill all tmux sessions + zombie processes on all workers |
| `--pull-results` | Download results from GCS to local |
| `--push-cache` / `--pull-cache` | Sync XLA compilation cache to/from GCS |

### Autonomous (started by run.sh)

| Command | What it does |
|---------|-------------|
| `fleet_manager.sh` | Every 10 min: recover dead VMs, expand fleet, check monitor, copy results when done |

## Monitoring

```bash
EXP=<name> bash ~/tpu_guide/watch.sh                          # 5-min auto-refresh
EXP=<name> python3 ~/tpu_guide/coordinator.py --status        # one-shot
python3 ~/tpu_guide/dashboard.py --exp <name> --interval 30   # rich TUI
bash ~/tpu_guide/vm_scan.sh                                    # fleet: VMs + quota + capacity
```

## File Layout

```
~/tpu_guide/
  run.sh                # Master script — single entry point for full experiment
  fleet_manager.sh      # Automated lifecycle: preemption, expansion, completion
  coordinator.py        # Centralized push coordinator
  submit.sh             # Per-VM + fleet submit script
  setup.sh              # VM setup (packages, code, data, model)
  watch.sh              # Auto-refresh monitor
  dashboard.py          # Rich TUI dashboard
  vm_scan.sh            # Fleet scanner
  COORDINATOR.md        # Design doc (deep dive)
  vm_configs/<TPU>.env  # Per-VM config (zone, bucket, type, chips)
  experiments/<EXP>.env # Per-experiment config (module, dirs, steps)
```

## GCS Layout (per-VM bucket)

```
gs://<VM_BUCKET>/coord/<EXP_NAME>/
  assignments/<TPU_NAME>.json   # [{label, overrides}, ...] for this VM
  heartbeat/<worker_id>.json    # {worker_id, vm, timestamp, step, label}
  results/<label>/summary.json  # Final result (two-phase committed)
  done/<label>.done             # Receipt: "worker_id timestamp"
```

## Local Results (ground truth)

```
~/sf_bema/results/<EXP>/
  validated/<label>.json   # Coordinator-validated result files
  state.json               # Coordinator state: assignments, retries, failed
```

## Memory & Resource Estimates

**Model**: SmolLM2-135M (162.8M params), eff_bs=128, v3_tpu optimizer

| Chip | HBM/chip | Procs/chip | Step time | Config time (1778 steps) |
|------|----------|-----------|-----------|--------------------------|
| v6e  | 32 GB    | 1         | ~4.9s     | ~2.4h                    |
| v4   | 32 GB    | 1         | ~8.4s     | ~4.1h                    |
| v5e  | 16 GB    | **OOM**   | —         | —                        |

HBM/proc: ~10.8 GB. Steps/config = 1778 (889 steps/epoch × 2 epochs).

## Gotchas

- **v5e**: OOM. Never request.
- **v4 GLIBC**: Ubuntu 20.04 has GLIBC 2.31, torch_xla 2.9.0 needs 2.34. v4 VMs dropped from fleet.
- **No-internet VMs**: `WANDB_MODE=disabled` (not `offline`). Only europe-west4 has internet.
- **Chip isolation**: `LAUNCH_MODE=single` + `TPU_VISIBLE_CHIPS` for 1 proc/chip.
- **Kill zombies**: v4 `fuser /dev/accel*`, v6e `fuser /dev/vfio/*`.
- **Python buffering**: `PYTHONUNBUFFERED=1` + `python3 -u` + `flush=True`.
- **Stale GCS state**: `--init` auto-cleans done/ and heartbeat/ dirs.
- **v4 old gcloud SDK**: Coordinator auto-detects and uses `gsutil` fallback.
