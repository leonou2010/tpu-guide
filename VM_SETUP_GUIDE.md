# VM Setup Guide — Per VM Type

Each VM type has a different runtime, Python setup, and wheel bundle. This doc covers exact setup for each.

---

## Common: All VM Types

### Control plane (blocklab side)
```bash
# Monitor progress
python3 ~/distributed_tpu_training/pull/check_progress.py

# Scan all VMs
bash ~/distributed_tpu_training/vm_scan.sh

# Deploy to a specific VM
TPU_NAME=<vm-name> ZONE=<zone> WANDB_MODE=<online|disabled> bash ~/distributed_tpu_training/pull/deploy.sh

# Deploy to all READY VMs (background daemon)
nohup bash ~/distributed_tpu_training/pull/vm_requester.sh >> /tmp/vm_requester.log 2>&1 &
```

### Pull-code sync — ALWAYS upload to all 3 buckets after any code change
```bash
for f in babysitter.py gcs.py monitor.py; do
  for b in gs://gcp-researchcredits-blocklab-europe-west4 \
            gs://gcp-researchcredits-blocklab-us-east1 \
            gs://gcp-researchcredits-blocklab-1-us-central2; do
    gsutil cp ~/distributed_tpu_training/pull/$f $b/pull_code/$f
  done
done
gsutil cp /tmp/deploy_babysitter.sh gs://gcp-researchcredits-blocklab-europe-west4/pull_code/deploy_babysitter.sh
gsutil cp /tmp/deploy_babysitter.sh gs://gcp-researchcredits-blocklab-us-east1/pull_code/deploy_babysitter.sh
gsutil cp /tmp/deploy_babysitter.sh gs://gcp-researchcredits-blocklab-1-us-central2/pull_code/deploy_babysitter.sh
```

---

## v6e-8 — europe-west4-a (internet, W&B works)

### Zone: europe-west4-a
### Runtime: v2-alpha-tpuv6e
### Chips: 8 per VM
### Bucket: gs://gcp-researchcredits-blocklab-europe-west4
### WANDB_MODE: online (has internet)
### Speed: ~5s/step, ~1.2h/config (889 steps)
### XLA cache: gs://.../xla_cache_v6e_fresh/ (67 files, ~19GB)

### Create VM:
```bash
gcloud alpha compute tpus tpu-vm create v6e-ew4a-N \
  --zone=europe-west4-a --project=gcp-research-credits-489020 \
  --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot --internal-ips
```

### SSH:
```bash
gcloud alpha compute tpus tpu-vm ssh v6e-ew4a-N --zone=europe-west4-a \
  --project=gcp-research-credits-489020 --tunnel-through-iap \
  --command="<cmd>"
```

### Python setup:
- torch_xla pre-installed in system Python (v2-alpha-tpuv6e image)
- `python3 -c "import torch_xla"` should work immediately
- Missing: hydra-core, omegaconf, transformers, sympy, antlr4, datasets, wandb
- Install: `pip install hydra-core omegaconf transformers sympy antlr4-python3-runtime==4.9.3 datasets wandb`

### deploy_babysitter.sh behavior:
1. Sets PATH (conda/local/system)
2. Checks torch (already present via image)
3. pip installs missing packages (internet available)
4. Downloads model from GCS if missing
5. Downloads exp code from GCS if missing
6. Downloads XLA cache from GCS
7. Launches `flock -n /tmp/tpu_babysitter.lock python3 -u ~/pull_code/babysitter.py`

### vm_configs: ~/distributed_tpu_training/vm_configs/v6e-ew4a-{1,2,4,5,6,8d,9,10}.env

---

## v6e-8 — us-east1-d (NO internet, WANDB disabled)

### Zone: us-east1-d
### Runtime: v2-alpha-tpuv6e
### Chips: 8 per VM
### Bucket: gs://gcp-researchcredits-blocklab-us-east1
### WANDB_MODE: disabled (no internet)
### Speed: ~5s/step, ~1.2h/config

### Create VM:
```bash
gcloud alpha compute tpus tpu-vm create v6e-ue1d-N \
  --zone=us-east1-d --project=gcp-research-credits-489020 \
  --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot --internal-ips
```

### Python setup:
- Same as ew4a (torch_xla pre-installed)
- BUT pip install fails (no internet) — must use GCS wheels bundle
- Wheels bundle: gs://gcp-researchcredits-blocklab-us-east1/wheels/ue1d_all_wheels.tar.gz
- Safe install: skip torch*/nvidia*/libtpu*/jax*/triton* to avoid breaking TPU torch

### CRITICAL: Missing packages on us-east1-d
If babysitter env_fail: deploy_babysitter.sh downloads wheels from ue1d bucket and
installs only: hydra-core, omegaconf, transformers, sympy, antlr4-python3-runtime==4.9.3, datasets, wandb

### vm_configs: ~/distributed_tpu_training/vm_configs/v6e-ue1d-{1,2,3,4,5,6}.env

---

## v4-8 — us-central2-b (NO internet, WANDB disabled)

### Zone: us-central2-b
### Runtime: tpu-ubuntu2204-base
### Chips: 4 per VM (NOT 8 — v4-8 = 8 cores = 4 chips, 2 cores/chip)
### Bucket: gs://gcp-researchcredits-blocklab-1-us-central2
### WANDB_MODE: disabled
### Speed: ~12s/step, ~3h/config (889 steps)
### gcloud SDK: old v347 — use gsutil (gcloud storage broken)

### Create VM:
```bash
gcloud alpha compute tpus tpu-vm create v4-uc2b-N \
  --zone=us-central2-b --project=gcp-research-credits-489020 \
  --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --internal-ips
```

### CRITICAL: v4 chips
- CHIPS_PER_HOST=4 (NOT 8!)
- Workers 4-7 will crash: "Failed to get global TPU topology"
- Devices: /dev/accel0 through /dev/accel3

### Python setup:
- tpu-ubuntu2204-base does NOT come with torch pre-installed
- Must install from GCS tpu_core wheels:
  ```bash
  gsutil -m cp 'gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_core/*.whl' /tmp/tpu_core/
  pip install /tmp/tpu_core/libtpu-*.whl --no-deps
  pip install /tmp/tpu_core/torch-*.whl --no-deps
  pip install /tmp/tpu_core/torch_xla-*.whl --no-deps
  ```
- Then install other packages from tpu_wheels.tar.gz (no internet):
  ```bash
  gsutil cp 'gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_wheels.tar.gz' /tmp/
  tar xzf /tmp/tpu_wheels.tar.gz -C /tmp/
  # Install only safe packages (skip torch/nvidia/libtpu/jax/triton)
  for whl in /tmp/tpu_wheels/*.whl; do
    name=$(basename "$whl" | tr '-' ' ' | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
    case "$name" in
      torch*|nvidia*|libtpu*|jax*|triton*) continue ;;
      *) pip install "$whl" --no-deps ;;
    esac
  done
  ```

### XLA cache:
- Location: gs://gcp-researchcredits-blocklab-1-us-central2/xla_cache/
- Must match torch_xla version (2.9.0 = current)

### gcloud usage:
- Use `gsutil` (old SDK 347 — gcloud storage broken)
- SSH: `gcloud alpha compute tpus tpu-vm ssh --tunnel-through-iap`

### vm_configs: ~/distributed_tpu_training/vm_configs/v4-uc2b-{1,2,3,4,5,6,7,spot3}.env

---

## v5e-4 — europe-west4-b (internet, WANDB works)

### Zone: europe-west4-b
### Runtime: v2-alpha-tpuv5-lite
### Chips: 4 per VM
### Bucket: gs://gcp-researchcredits-blocklab-europe-west4
### WANDB_MODE: online (has internet)
### Speed: ~25s/step, ~6h/config — SLOW but works
### OOM fix: batch_size MUST be 4 (bs=8 OOMs on 16GB/chip HBM)

### Create VM:
```bash
gcloud alpha compute tpus tpu-vm create v5e-ew4b-N \
  --zone=europe-west4-b --project=gcp-research-credits-489020 \
  --accelerator-type=v5litepod-4 --version=v2-alpha-tpuv5-lite --spot --internal-ips
```

### Python setup:
- torch_xla pre-installed (v2-alpha-tpuv5-lite image)
- Same as v6e: pip install missing packages (internet available)
- gcloud storage works (new SDK)

### OOM fix in babysitter.py:
- Detects ACCELERATOR_TYPE=v5litepod-4 or 'v5' in TPU_NAME
- Halves batch_size + doubles gradient_accumulation_steps (preserves effective BS=128)

### vm_configs: ~/distributed_tpu_training/vm_configs/v5e-ew4b-{1,2,...}.env

---

## v5e-4 — us-central1-a (NO internet, WANDB disabled)

### Zone: us-central1-a
### Runtime: v2-alpha-tpuv5-lite
### Chips: 4 per VM
### Bucket: gs://gcp-researchcredits-blocklab-europe-west4 (no own bucket)
### WANDB_MODE: disabled
### Speed: ~25s/step, ~6h/config
### gcloud storage: works (new SDK)

### Create VM:
```bash
gcloud alpha compute tpus tpu-vm create v5e-uc1a-N \
  --zone=us-central1-a --project=gcp-research-credits-489020 \
  --accelerator-type=v5litepod-4 --version=v2-alpha-tpuv5-lite --spot --internal-ips
```

### Python setup:
- torch_xla pre-installed
- No internet — must use GCS wheels
- Use `gcloud storage cp` (NOT gsutil — broken on v5e)

### vm_configs: ~/distributed_tpu_training/vm_configs/v5e-uc1a-{1,2}.env.disabled (need to enable)

---

## Diagnosing a Dead/Stuck VM

```bash
# 1. Check heartbeat age in progress
python3 ~/distributed_tpu_training/pull/check_progress.py

# 2. SSH in and check
gcloud alpha compute tpus tpu-vm ssh <VM> --zone=<ZONE> --tunnel-through-iap \
  --project=gcp-research-credits-489020 \
  --command="tail -50 /tmp/babysitter.log; echo '---'; pgrep -fl python; echo '---'; df -h /tmp"

# 3. Check GCS logs
gsutil cat gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/logs/<worker_id>_last.log
gsutil cat gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/logs/env_fail_<vm-name>.log

# 4. Common causes:
#   - env_fail: torch/hydra missing → redeploy with fix
#   - step stuck at 1 for >30min: XLA hang → redeploy
#   - age > 3600s: stale heartbeat → monitor.py will reclaim after stale-ttl
#   - rc=1: training script error → check last.log
#   - PREEMPTED: vm_requester auto-deletes and recreates
```

---

## Starting Overnight Operation (Checklist)

```bash
# 1. Check all 3 overnight processes are running
pgrep -fla 'monitor.py|vm_requester.sh|overnight_watchdog'

# 2. Start any that are missing
pgrep -f 'monitor.py.*--interval' || \
  nohup python3 -u ~/distributed_tpu_training/pull/monitor.py --interval 60 --stale-ttl 3600 >> /tmp/monitor_pull.log 2>&1 &

pgrep -f 'vm_requester.sh' || \
  nohup bash ~/distributed_tpu_training/pull/vm_requester.sh >> /tmp/vm_requester.log 2>&1 &

pgrep -f 'overnight_watchdog.sh' || \
  nohup bash ~/distributed_tpu_training/pull/overnight_watchdog.sh >> /tmp/overnight_watchdog.log 2>&1 &

# 3. Quick status
python3 ~/distributed_tpu_training/pull/check_progress.py
bash ~/distributed_tpu_training/vm_scan.sh

# 4. Live dashboard
watch -c -n60 'python3 ~/distributed_tpu_training/pull/dashboard.py --once 2>/dev/null'
```
