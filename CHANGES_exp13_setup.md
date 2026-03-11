# exp13 Setup Changes — 2026-03-10

## Overview
Setting up exp13 (120 configs, 882 steps/config) to run on full fleet: v6e, v4, v5e.

## Files Modified

### 1. coordinator.py
- **import shutil** added (for copy_validated_results)
- **load_vm_configs()**: Added `elif accel.startswith('v5'): cfg['STEP_S'] = 6.0` — v5e VMs now included
- **worker_sweep()**: v5e OOM auto-fix — detects `ACCELERATOR_TYPE` containing 'v5' or TPU name containing 'v5', halves batch_size and doubles gradient_accumulation_steps to preserve effective BS=128
- **ExperimentQueue class**: Queue management (enqueue/dequeue/mark_running/mark_done/status_report). File: ~/tpu_guide/queue.json
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

## Recovery Commands
```bash
# If orchestrator dies:
# 1. Check exp12_1 status
ls ~/sf_bema/results/exp12_1/validated/ | wc -l  # should be 185

# 2. Start exp13 manually
cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
EXP=exp13 python3 ~/tpu_guide/coordinator.py --init
# Then deploy per-VM
EXP=exp13 TPU_NAME=v6e-ew4a-4 bash ~/tpu_guide/submit.sh --sweep
# Repeat for each VM...

# 3. Monitor
EXP=exp13 python3 -u ~/tpu_guide/coordinator.py --monitor | tee /tmp/monitor_exp13.log

# 4. Dashboard
watch -c -n30 'python3 ~/tpu_guide/dashboard.py --exp exp13'

# 5. Copy results when done
mkdir -p ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
cp ~/sf_bema/results/exp13/validated/*.json ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
```
