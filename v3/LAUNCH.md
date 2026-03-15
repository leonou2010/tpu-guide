# V3 Launch Guide — exp13_rerun3

## Pre-launch Checklist

### Step 0: Build CUDA stubs tarball (one-time, needed for ue1d VMs)
```bash
bash ~/distributed_tpu_training/v3/scripts/build_cuda_stubs.sh
# Uploads nvidia_stubs_v6e.tar.gz to all 3 GCS buckets automatically
```
Requires v6e-ew4a-1 to be running. Falls back to inline GCC if tarball missing.

### Step 1: Upload v3 code to all 3 GCS buckets
```bash
for bucket in \
    gs://gcp-researchcredits-blocklab-europe-west4 \
    gs://gcp-researchcredits-blocklab-us-east1 \
    gs://gcp-researchcredits-blocklab-1-us-central2; do
    for f in babysitter.py gcs.py deploy_babysitter.sh; do
        gsutil cp ~/distributed_tpu_training/v3/$f ${bucket}/pull_code_v3/$f
    done
done
```

### Step 2: Test deploy_babysitter.sh on each VM type
```bash
# ew4a (v6e, system torch):
gcloud alpha compute tpus tpu-vm ssh v6e-ew4a-1 --zone=europe-west4-a \
    --project=gcp-research-credits-489020 --tunnel-through-iap \
    --command='TPU_NAME=v6e-ew4a-1 ZONE=europe-west4-a FORCE_REDEPLOY=1 bash ~/pull_code/deploy_babysitter.sh'

# ue1d (v6e, no torch):
gcloud alpha compute tpus tpu-vm ssh v6e-ue1d-1 --zone=us-east1-d \
    --project=gcp-research-credits-489020 --tunnel-through-iap \
    --command='TPU_NAME=v6e-ue1d-1 ZONE=us-east1-d FORCE_REDEPLOY=1 bash ~/pull_code/deploy_babysitter.sh'

# uc2b (v4):
gcloud alpha compute tpus tpu-vm ssh v4-uc2b-1 --zone=us-central2-b \
    --project=gcp-research-credits-489020 --tunnel-through-iap \
    --command='TPU_NAME=v4-uc2b-1 ZONE=us-central2-b FORCE_REDEPLOY=1 bash ~/pull_code/deploy_babysitter.sh'
```
All 3 must reach `PHASE: IDLE_AWAITING_WORK`.

### Step 3: Request HEALTH_CHECKS quota (manual)
GCP Console → IAM & Admin → Quotas → filter HEALTH_CHECKS → Request 5000.
Without this, fleet caps at ~15 VMs.

### Step 4: Populate tasks
```bash
# Stop any existing experiment (drain + clear)
EXP=exp13_rerun2 python3 ~/distributed_tpu_training/v3/populate.py --drain
EXP=exp13_rerun2 python3 ~/distributed_tpu_training/v3/populate.py --clear --force

# Populate exp13_rerun3
EXP=exp13_rerun3 python3 ~/distributed_tpu_training/v3/populate.py --dry-run  # verify 120 tasks
EXP=exp13_rerun3 python3 ~/distributed_tpu_training/v3/populate.py
```

## Launch

```bash
# VM manager (replaces vm_requester.sh)
nohup python3 -u ~/distributed_tpu_training/v3/vm_manager.py >> /tmp/vm_manager_v3.log 2>&1 &
echo "vm_manager PID: $!"

# Monitor
nohup python3 -u ~/distributed_tpu_training/v3/monitor.py \
    --exp exp13_rerun3:120 \
    --interval 60 --stale-ttl 1800 >> /tmp/monitor_v3.log 2>&1 &
echo "monitor PID: $!"
```

## Dashboard
```bash
watch -n 60 'python3 ~/distributed_tpu_training/v2/dashboard.py --exp exp13_rerun3 --once'
```

## Status check
```bash
# Quick progress
python3 ~/distributed_tpu_training/v3/check_progress.py

# VM manager log
tail -50 /tmp/vm_manager_v3.log

# Monitor log
tail -30 /tmp/monitor_v3.log

# Deploy log on a VM (example)
gcloud alpha compute tpus tpu-vm ssh v6e-ew4a-1 --zone=europe-west4-a \
    --project=gcp-research-credits-489020 --tunnel-through-iap \
    --command='tail -30 /tmp/deploy_babysitter.log'
```

## Key improvements vs v2

| Problem | v2 | v3 |
|---------|----|----|
| ue1d torch failures | inline GCC stubs (wrong sonames) | pre-built tarball + fallback |
| VM manager blocking | single bash loop (45min dead detection) | thread-per-VM (60s detection) |
| Capacity exhaustion | "internal error" retry loops | QueuedResource (GCP handles queuing) |
| Broken VM detection | none | rapid_fail flag from monitor |
| Drain before clear | not implemented | `populate.py --drain` |

## Emergency

```bash
# Force-redeploy a specific VM immediately
python3 -c "
import json, sys
sys.path.insert(0, '$HOME/distributed_tpu_training/v3')
from gcs import gcs_write, CONTROL_PLANE
vm = 'v6e-ue1d-1'
gcs_write(f'{CONTROL_PLANE}/flags/redeploy_{vm}.json', json.dumps({'manual': True}))
print('Flag written')
"

# Stop vm_manager
kill $(pgrep -f vm_manager.py)

# Stop monitor
kill $(pgrep -f 'v3/monitor.py')
```
