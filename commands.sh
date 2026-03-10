#!/bin/bash
# TPU VM Command Templates — copy-paste ready
# Usage: source this file for reference, or copy individual commands

# ══════════════════════════════════════════════════════════════
# VM 1: v6e-ew4a (europe-west4-a, v6e-8, 1 host, 8 chips)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v6e-ew4a ZONE=europe-west4-a TPU_NUM_WORKERS=1 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight (5 min):
# TPU_NAME=v6e-ew4a ZONE=europe-west4-a TPU_NUM_WORKERS=1 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v6e-ew4a ZONE=europe-west4-a TPU_NUM_WORKERS=1 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online LAUNCH_MODE=single PROCS_PER_HOST=8 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v6e-ew4a ZONE=europe-west4-a TPU_NUM_WORKERS=1 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# VM 2: v6e-ue1d (us-east1-d, v6e-16, 4 hosts, 8 chips/host)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v6e-ue1d ZONE=us-east1-d TPU_NUM_WORKERS=4 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-us-east1 WANDB_MODE=disabled bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight:
# TPU_NAME=v6e-ue1d ZONE=us-east1-d TPU_NUM_WORKERS=4 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-us-east1 WANDB_MODE=disabled LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v6e-ue1d ZONE=us-east1-d TPU_NUM_WORKERS=4 CHIPS_PER_HOST=8 BUCKET=gs://gcp-researchcredits-blocklab-us-east1 WANDB_MODE=disabled LAUNCH_MODE=single PROCS_PER_HOST=8 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v6e-ue1d ZONE=us-east1-d TPU_NUM_WORKERS=4 BUCKET=gs://gcp-researchcredits-blocklab-us-east1 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# VM 3: v4-uc2b (us-central2-b, v4-32, 4 hosts, 4 chips/host)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v4-uc2b ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight:
# TPU_NAME=v4-uc2b ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v4-uc2b ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled LAUNCH_MODE=single PROCS_PER_HOST=4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v4-uc2b ZONE=us-central2-b TPU_NUM_WORKERS=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# VM 4: v4-uc2b-spot (us-central2-b, v4-32, 4 hosts, 4 chips/host)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v4-uc2b-spot ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight:
# TPU_NAME=v4-uc2b-spot ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v4-uc2b-spot ZONE=us-central2-b TPU_NUM_WORKERS=4 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 WANDB_MODE=disabled LAUNCH_MODE=single PROCS_PER_HOST=4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v4-uc2b-spot ZONE=us-central2-b TPU_NUM_WORKERS=4 BUCKET=gs://gcp-researchcredits-blocklab-1-us-central2 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# VM 5: v5e-uc1a (us-central1-a, v5litepod-64, 16 hosts, 4 chips/host)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v5e-uc1a ZONE=us-central1-a TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-us-central1 WANDB_MODE=disabled bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight:
# TPU_NAME=v5e-uc1a ZONE=us-central1-a TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-us-central1 WANDB_MODE=disabled LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v5e-uc1a ZONE=us-central1-a TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-us-central1 WANDB_MODE=disabled LAUNCH_MODE=single PROCS_PER_HOST=4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v5e-uc1a ZONE=us-central1-a TPU_NUM_WORKERS=16 BUCKET=gs://gcp-researchcredits-blocklab-us-central1 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# VM 6: v5e-ew4b (europe-west4-b, v5litepod-64, 16 hosts, 4 chips/host)
# ══════════════════════════════════════════════════════════════
# Setup:
# TPU_NAME=v5e-ew4b ZONE=europe-west4-b TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --setup

# Preflight:
# TPU_NAME=v5e-ew4b ZONE=europe-west4-b TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight

# Parallel sweep:
# TPU_NAME=v5e-ew4b ZONE=europe-west4-b TPU_NUM_WORKERS=16 CHIPS_PER_HOST=4 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 WANDB_MODE=online LAUNCH_MODE=single PROCS_PER_HOST=4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --sweep

# Cancel:
# TPU_NAME=v5e-ew4b ZONE=europe-west4-b TPU_NUM_WORKERS=16 BUCKET=gs://gcp-researchcredits-blocklab-europe-west4 bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel

# ══════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ══════════════════════════════════════════════════════════════

# Cancel ALL VMs:
cancel_all() {
  for env in ~/tpu_guide/vm_configs/*.env; do
    source "$env"
    echo "Cancelling $TPU_NAME..."
    bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --cancel &
  done
  wait
  echo "All cancelled."
}

# Push code to ALL VMs:
push_code_all() {
  for env in ~/tpu_guide/vm_configs/*.env; do
    source "$env"
    echo "Pushing code to $TPU_NAME..."
    bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --push-code &
  done
  wait
  echo "Code pushed to all VMs."
}

# Preflight ALL VMs (parallel):
preflight_all() {
  for env in ~/tpu_guide/vm_configs/*.env; do
    source "$env"
    echo "Preflight $TPU_NAME..."
    LAUNCH_MODE=single bash ~/tpu_guide/submit_tpu_job_12c_v2.sh --preflight &
  done
  wait
  echo "All preflights launched. Monitor: bash ~/tpu_guide/monitor_v2.sh"
}
