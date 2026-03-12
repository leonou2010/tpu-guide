# V2 Launch Checklist

Pull-based TPU sweep system. Blocklab is the coordinator; VMs pull tasks from GCS.

---

## Pre-launch (one-time per experiment)

### 1. Publish artifacts to GCS

```bash
# Upload v2 scripts to all 3 regional buckets (VMs download from their region)
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4/pull_code \
  gs://gcp-researchcredits-blocklab-1-us-central2/pull_code \
  gs://gcp-researchcredits-blocklab-us-east1/pull_code; do
  gsutil -m cp ~/distributed_tpu_training/v2/{gcs.py,babysitter.py,deploy_babysitter.sh} "$b/"
done

# Rebuild and upload code bundle (if train_v2_tpu.py or configs changed)
cd ~/sf_bema/experiments
tar czf /tmp/sf_bema_code.tar.gz \
  expN_*/conf/ expN_*/expN_tpu/ expN_*/train_*.py \
  shared/optimizers/*.py shared/train/train_v2_tpu.py
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4 \
  gs://gcp-researchcredits-blocklab-1-us-central2 \
  gs://gcp-researchcredits-blocklab-us-east1; do
  gsutil cp /tmp/sf_bema_code.tar.gz "$b/code/sf_bema_code.tar.gz"
done
```

### 2. Populate task queue

```bash
# Dry-run first to verify task count
EXP=expN python3 ~/distributed_tpu_training/v2/populate.py --dry-run

# Populate
EXP=expN python3 ~/distributed_tpu_training/v2/populate.py
```

### 3. (Manual) Request HEALTH_CHECKS quota = 5000 in GCP Console
- IAM & Admin → Quotas → filter "HEALTH_CHECKS" → request 5000

---

## Launch

```bash
# Set experiment config (required by watchdog + monitor)
export EXPERIMENTS="expN:200"           # NAME:TOTAL pairs, space-separated
export RESULTS_BASE="$HOME/sf_bema/results"  # where validated/ dirs live

# Start overnight watchdog (starts monitor.py + vm_requester.sh automatically)
nohup bash ~/distributed_tpu_training/v2/overnight_watchdog.sh \
  >> /tmp/overnight_watchdog.log 2>&1 &
echo "Watchdog PID: $!"

# Verify processes started within ~30s
pgrep -af 'monitor.py'
pgrep -af 'vm_requester.sh'
```

Or start components individually:

```bash
# Monitor (validates results, reclaims stale tasks)
EXPERIMENTS="expN:200" nohup python3 -u ~/distributed_tpu_training/v2/monitor.py \
  --exp expN:200 --interval 60 --stale-ttl 1800 \
  >> /tmp/monitor_pull.log 2>&1 &

# VM requester (creates/manages TPU VMs, deploys babysitter)
nohup bash ~/distributed_tpu_training/v2/vm_requester.sh \
  >> /tmp/vm_requester.log 2>&1 &
```

---

## Monitor progress

```bash
# Quick status (parallel heartbeat reads, ~5s)
python3 ~/distributed_tpu_training/v2/check_progress.py

# Per-VM health (boot phase, step, task, env_fail)
python3 ~/distributed_tpu_training/v2/vm_status.py

# Live dashboard (refreshes every 30s)
watch -c -n30 'python3 ~/distributed_tpu_training/v2/dashboard.py --once'

# Tail logs
tail -f /tmp/monitor_pull.log
tail -f /tmp/vm_requester.log
tail -f /tmp/overnight_watchdog.log
```

---

## Post-experiment

```bash
# Upload per-step loss + eval metrics to W&B, AND copy all results to local experiment folder
python3 ~/distributed_tpu_training/v2/upload_wandb.py \
  --exp expN \
  --copy-to ~/sf_bema/experiments/expN_<name>/expN_tpu/results/

# Dry-run first to verify counts
python3 ~/distributed_tpu_training/v2/upload_wandb.py --exp expN --dry-run

# Clean GCS (~400+ GiB of checkpoints, heartbeats, coord history)
bash ~/distributed_tpu_training/v2/cleanup_gcs.sh --dry-run
bash ~/distributed_tpu_training/v2/cleanup_gcs.sh
```

After upload_wandb.py runs, the experiment folder will have:
- `results/<label>.json` — summary, config, eval_logs, hydra overrides
- `results/<label>_train_loss.jsonl` — per-step train loss (for learning curves)
- W&B runs at `wandb.ai/ka3094-columbia-university/expN-tpu`

---

## Script roles

| Script | Runs on | Role |
|--------|---------|------|
| `populate.py` | blocklab | Writes tasks to `coord_v2/pending/` |
| `vm_requester.sh` | blocklab | Creates/deletes VMs, runs `deploy_babysitter.sh` |
| `deploy_babysitter.sh` | blocklab→VM | Installs env, downloads assets, launches babysitter |
| `babysitter.py` | VM | Claims tasks, runs training, uploads results |
| `gcs.py` | VM + blocklab | GCS task queue primitives |
| `monitor.py` | blocklab | Validates results, reclaims stale tasks |
| `overnight_watchdog.sh` | blocklab | Watchdog: restarts monitor + vm_requester if dead |
| `check_progress.py` | blocklab | Quick parallel heartbeat read |
| `vm_status.py` | blocklab | Per-VM health (boot phase, step, task) |
| `dashboard.py` | blocklab | TUI fleet + queue dashboard |
| `upload_wandb.py` | blocklab | Post-experiment: JSONL → W&B runs |
| `cleanup_gcs.sh` | blocklab | Post-experiment: delete checkpoints + history |
