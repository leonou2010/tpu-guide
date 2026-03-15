# Report: US v6e (us-east1-d) Bring-Up and Proof via Returned Loss Data

Date: 2026-03-14 (UTC)
Owner: kwokchunau + Codex pairing
Scope: Validate that a US v6e TPU VM can be set up correctly and prove success by producing training loss data that is uploaded to GCS and fetched back.

## Summary
US v6e VM `v6e-ue1d-2` initially failed to train due to a broken dataset path on the VM. After fixing the dataset directory and running a short single-chip TPU smoke job, the VM uploaded per-step training loss JSONL to the US bucket, and the loss file was fetched successfully from GCS. This proves end-to-end: VM setup -> TPU training -> returned loss artifact.

## Environment
- VM: `v6e-ue1d-2`
- Zone: `us-east1-d`
- Project: `gcp-research-credits-489020`
- Bucket (proof artifacts): `gs://gcp-researchcredits-blocklab-us-east1`
- Experiment code: `~/sf_bema/experiments/exp13_smollm2_smoltalk`
- Access: `gcloud alpha compute tpus tpu-vm ssh ... --tunnel-through-iap`

## Initial Failure
Symptom:
- Training failed with `FileNotFoundError: .../exp13_smollm2_smoltalk/data/train not found`.

Root cause:
- `~/sf_bema/experiments/exp13_smollm2_smoltalk/data` was a symlink to `/scratch/...`.
- ue1d VMs do not have `/scratch`, so the symlink target was missing.

## Fix Applied (on the VM)
### 1) SSH (IAP)
Run from the control machine:
```bash
gcloud alpha compute tpus tpu-vm ssh v6e-ue1d-2 --zone=us-east1-d \
  --project=gcp-research-credits-489020 --tunnel-through-iap
```

### 2) Replace broken `data/` symlink with real directory
Run on the VM:
```bash
DATA=~/sf_bema/experiments/exp13_smollm2_smoltalk/data
[ -L "$DATA" ] && rm -f "$DATA"
mkdir -p "$DATA"
```

### 3) Download dataset locally (large transfer)
Run on the VM:
```bash
gsutil -m cp -r gs://gcp-researchcredits-blocklab-us-east1/data/smoltalk_full/data/train "$DATA/"
gsutil -m cp -r gs://gcp-researchcredits-blocklab-us-east1/data/smoltalk_full/data/val   "$DATA/"
```

### 4) Ensure model is present offline
Run on the VM:
```bash
mkdir -p /tmp/SmolLM2-135M
gsutil -m cp gs://gcp-researchcredits-blocklab-us-east1/models/SmolLM2-135M/* /tmp/SmolLM2-135M/ || true
test -f /tmp/SmolLM2-135M/config.json
```

## Proof Run (TPU training + returned loss data)
Goal:
- Run a short TPU smoke job.
- Upload per-step training loss JSONL to GCS.
- Fetch the JSONL from outside the VM.

Key settings:
- Set `TPU_NUM_WORKERS=1` to avoid multi-host GCS barrier waiting.
- Use `LAUNCH_MODE=single` and `TPU_VISIBLE_CHIPS=0` for a single-chip smoke run.
- Set `TRAIN_LOSS_JSONL` and `GCS_CHECKPOINT_DIR` to ensure loss JSONL is uploaded.

Command (run on the VM):
```bash
cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
RUNSTAMP=20260314_211607

export BUCKET=gs://gcp-researchcredits-blocklab-us-east1
export TPU_NUM_WORKERS=1
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=0
export LAUNCH_MODE=single
export CHIPS_PER_HOST=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache
export XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache
export TRAIN_LOSS_JSONL=/tmp/ue1d_smoke_${RUNSTAMP}.jsonl
export GCS_CHECKPOINT_DIR=${BUCKET}/checkpoints/ue1d_smoke_${RUNSTAMP}

python3 -u exp13_tpu_rerun/train_tpu.py \
  optimizer=adamw_ema_pullback_v4_tpu optimizer.lr=0.001 optimizer.lambda_pullback=0.01 optimizer.clamp_pullback=true \
  optimizer.ema_kappa=0.3 optimizer.use_ema_eval=false optimizer.ema_update_freq=5 +warmup_steps=50 \
  model.name=/tmp/SmolLM2-135M \
  +training.max_steps=5 +training.max_train_seconds=240 training.eval_interval=5
```

Observed completion signal:
- `DONE: adamw_ema_pullback_v4_lr0.001_bs128_lp0.01_clamp_w50_uf5_s42`
- Printed best validation loss: `best_val_loss = 1.5940`
- Printed JSONL upload destination (below).

### Evidence: returned loss artifact
GCS path uploaded by the VM:
- `gs://gcp-researchcredits-blocklab-us-east1/checkpoints/ue1d_smoke_20260314_211607/adamw_ema_pullback_v4_lr0.001_bs128_lp0.01_clamp_w50_uf5_s42_train_loss.jsonl`

Fetch command (from control machine):
```bash
gsutil cat gs://gcp-researchcredits-blocklab-us-east1/checkpoints/ue1d_smoke_20260314_211607/adamw_ema_pullback_v4_lr0.001_bs128_lp0.01_clamp_w50_uf5_s42_train_loss.jsonl
```

Fetched contents:
- `{"step": 1, "loss": 1.6380615234375}`

## Lessons Learned
- Do not rely on `/scratch` on ue1d; ensure `.../data` is a real directory (or a symlink to a mount that actually exists on ue1d).
- For single-VM validation, set `TPU_NUM_WORKERS=1` or training can block on multi-host rendezvous/barrier logic.
- Dataset download is multi-GB; first-time bring-up will be slow.
- `--tunnel-through-iap` is required for reliable SSH in this environment (direct internal-IP SSH can time out).

## Reporting Template (copy/paste)
Use this for Slack/email:
- Goal: Bring up us-east1-d v6e VM and prove TPU training by returning loss data.
- VM/zone: `v6e-ue1d-2` / `us-east1-d` (2026-03-14 UTC).
- Issue: training failed due to broken `data` symlink to `/scratch`.
- Fix: replaced symlink with real local `data/` directory and downloaded smoltalk_full train/val.
- Proof: loss JSONL uploaded and fetched from `gs://gcp-researchcredits-blocklab-us-east1/checkpoints/ue1d_smoke_20260314_211607/..._train_loss.jsonl` containing `{"step": 1, "loss": 1.6380615}`.
