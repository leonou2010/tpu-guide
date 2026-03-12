# V2 Pre-Deploy Checklist

Run through this before every experiment launch. Each item has a command to verify it.

---

## 0. TPU version of the experiment exists

Every experiment needs a `_tpu/` folder before it can run on TPU. If you're converting a GPU
experiment or creating a new one, do this first:

**Required files in `~/sf_bema/experiments/<WORK_DIR>/<expN_tpu>/`:**

| File | Purpose |
|------|---------|
| `train_tpu.py` | XLA wrapper — wraps `shared/train/train_v2_tpu.py`, sets LAUNCH_MODE |
| `run_tpu.py` | Config builder — `build_configs()` returns `[(label, overrides)]` |
| `OBJECTIVE.md` | Required before any run (what are we testing, win condition) |

```bash
# Check it exists
WORK_DIR=exp13_smollm2_smoltalk
EXP_TPU_DIR=exp13_tpu   # or exp13_tpu_rerun, etc.
ls ~/sf_bema/experiments/${WORK_DIR}/${EXP_TPU_DIR}/{train_tpu.py,run_tpu.py,OBJECTIVE.md}

# Verify run_tpu.py has build_configs() and the right SCRIPT path
python3 -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/sf_bema/experiments/${WORK_DIR}'))
from ${EXP_TPU_DIR}.run_tpu import build_configs
configs = build_configs()
print(f'{len(configs)} configs')
print(f'first label: {configs[0][0]}')
print(f'last label:  {configs[-1][0]}')
"

# Dry-run to confirm count matches expected
python3 ~/sf_bema/experiments/${WORK_DIR}/${EXP_TPU_DIR}/run_tpu.py --dry-run
```

**How to create train_tpu.py** (copy from nearest experiment, update optimizer import):

```bash
# Copy from exp13_tpu (uses v4 optimizer + train_v2_tpu)
cp ~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/train_tpu.py \
   ~/sf_bema/experiments/<WORK_DIR>/<expN_tpu>/train_tpu.py
# Then edit the optimizer yaml name and docstring as needed
```

The train_tpu.py wrapper must:
- Import `train_v2_tpu` from `shared/train/`
- Support `LAUNCH_MODE=pjrt` (default) — do NOT call `torch_xla.launch()` directly from run_tpu.py
- Accept Hydra overrides as CLI args

---

## 1. Experiment config exists

```bash
cat ~/distributed_tpu_training/experiments/${EXP}.env
# Must have: EXP_NAME, EXP_MODULE, WORK_DIR
```

**Critical:** `EXP_NAME` must be unique per run. For reruns, use `exp13_rerun` not `exp13`.
Reusing the same `EXP_NAME` will silently skip tasks already in `completed/` or local `validated/`.

---

## 2. Code bundle is current and named correctly

```bash
# Rebuild (always do this if train_v2_tpu.py or any expN_tpu/ file changed)
cd ~/sf_bema/experiments
tar czf /tmp/sf_bema_code.tar.gz \
  expN_*/conf/ expN_*/expN_tpu/ expN_*/expN_tpu_rerun/ \
  shared/optimizers/*.py shared/train/train_v2_tpu.py

# Upload to all 3 buckets (MUST be named sf_bema_code.tar.gz)
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4 \
  gs://gcp-researchcredits-blocklab-us-east1 \
  gs://gcp-researchcredits-blocklab-1-us-central2; do
  gsutil cp /tmp/sf_bema_code.tar.gz "$b/code/sf_bema_code.tar.gz"
done

# Verify
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4 \
  gs://gcp-researchcredits-blocklab-us-east1 \
  gs://gcp-researchcredits-blocklab-1-us-central2; do
  gsutil ls -l "$b/code/sf_bema_code.tar.gz"
done
```

---

## 3. Babysitter scripts uploaded to all 3 buckets

```bash
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4/pull_code \
  gs://gcp-researchcredits-blocklab-us-east1/pull_code \
  gs://gcp-researchcredits-blocklab-1-us-central2/pull_code; do
  gsutil -m cp \
    ~/distributed_tpu_training/v2/gcs.py \
    ~/distributed_tpu_training/v2/babysitter.py \
    ~/distributed_tpu_training/v2/deploy_babysitter.sh \
    "$b/"
done

# Verify all 3 have same timestamp
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4/pull_code \
  gs://gcp-researchcredits-blocklab-us-east1/pull_code \
  gs://gcp-researchcredits-blocklab-1-us-central2/pull_code; do
  echo "$b:"
  gsutil ls -l "$b/deploy_babysitter.sh" "$b/babysitter.py" "$b/gcs.py" | awk '{print $2, $3, $NF}'
done
```

---

## 4. GCS state is clean for this EXP_NAME

```bash
CTRL=gs://gcp-researchcredits-blocklab-europe-west4/coord_v2
EXP_NAME=expN  # replace with actual

# Check if any old tasks exist for this experiment
gsutil ls "${CTRL}/pending/${EXP_NAME}__*.json" 2>/dev/null | wc -l
gsutil ls "${CTRL}/running/${EXP_NAME}__*.json" 2>/dev/null | wc -l
gsutil ls "${CTRL}/completed/${EXP_NAME}__*.json" 2>/dev/null | wc -l
gsutil ls "${CTRL}/failed/${EXP_NAME}__*.json" 2>/dev/null | wc -l

# Check local validated dir
ls ~/sf_bema/results/${EXP_NAME}/validated/*.json 2>/dev/null | wc -l
```

If any counts are non-zero and this is a fresh run (not a resume): either use a new EXP_NAME, or clear with `--clear` in populate.py.

---

## 5. Populate task queue (dry-run first)

```bash
# Dry-run: verify correct task count
EXP=${EXP_NAME} python3 ~/distributed_tpu_training/v2/populate.py --dry-run

# Populate
EXP=${EXP_NAME} python3 ~/distributed_tpu_training/v2/populate.py
```

---

## 6. HEALTH_CHECKS quota

Target: **5000** (not the default 75).

```bash
# Check current quota
gcloud compute project-info describe --project=gcp-research-credits-489020 \
  --format="json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'HEALTH_CHECK' in q.get('metric',''):
        print(q)
"
```

If < 5000: GCP Console → IAM & Admin → Quotas → filter "HEALTH_CHECKS" → Request 5000.

---

## 7. No zombie processes from previous run

```bash
pgrep -af 'monitor.py'
pgrep -af 'vm_requester.sh'
pgrep -af 'overnight_watchdog.sh'

# If stale processes from a different experiment are running, kill them:
pkill -f 'distributed_tpu_training/v2/monitor.py'
pkill -f 'vm_requester.sh'
pkill -f 'overnight_watchdog.sh'
rm -f ~/.locks/vm_requester.pid ~/.locks/overnight_watchdog.pid
```

---

## 8. Launch and verify

```bash
export EXPERIMENTS="${EXP_NAME}:TOTAL"
export RESULTS_BASE="$HOME/sf_bema/results"

nohup bash ~/distributed_tpu_training/v2/overnight_watchdog.sh \
  >> /tmp/overnight_watchdog.log 2>&1 &

# Verify within 30s
sleep 5
pgrep -af 'monitor.py'
pgrep -af 'vm_requester.sh'
tail -5 /tmp/overnight_watchdog.log
```

---

## Known Hazards (from post-mortems)

| Hazard | Rule |
|--------|------|
| Rerunning same `EXP_NAME` | Always use a new name (e.g. `exp13_rerun`). Old `completed/` causes silent skips. |
| Code bundle wrong name | Must be `sf_bema_code.tar.gz` — not `sf_bema_code_exp13.tar.gz` |
| Not syncing all 3 buckets | VMs in us-east1/us-central2 download from their own bucket |
| `min_steps=400` validation | All experiments must reach > 400 steps or results are invalidated + requeued |
| `pip install torch` on v6e | Installs CUDA torch, breaks TPU. Never do this. |
| `--force-reinstall` all wheels | Overwrites TPU torch. Never do this. |
| `--init` on running experiment | Clears GCS heartbeats + done/. Never call on a live run. |
