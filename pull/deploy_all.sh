#!/bin/bash
# deploy_all.sh — Deploy pull-based babysitter to all VMs sequentially
# Usage: bash ~/distributed_tpu_training/pull/deploy_all.sh [vm_name_filter]
set -uo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
CONTROL_PLANE=gs://gcp-researchcredits-blocklab-europe-west4/coord_v2
MODEL_PATH=/tmp/SmolLM2-135M

log() { echo "[$(date '+%H:%M:%S')] $*"; }

FILTER="${1:-}"

for cfg_file in ~/distributed_tpu_training/vm_configs/*.env; do
    [ -f "$cfg_file" ] || continue
    source "$cfg_file"
    PPH=${PROCS_PER_HOST:-0}
    [ "$PPH" -eq 0 ] && continue

    # Apply filter if given
    if [ -n "$FILTER" ] && [[ ! "$TPU_NAME" == *"$FILTER"* ]]; then
        continue
    fi

    log "=== Deploying $TPU_NAME ($ZONE, $ACCELERATOR_TYPE) ==="

    $GCLOUD alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" --project=$PROJECT --tunnel-through-iap \
        --worker=all --command="
            mkdir -p ~/distributed_tpu_training/pull
            gsutil -m cp $BUCKET/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || \
                gcloud storage cp $BUCKET/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || true

            for s in \$(tmux list-sessions 2>/dev/null | cut -d: -f1); do
                tmux kill-session -t \"\$s\" 2>/dev/null
            done
            { fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
            sleep 1
            rm -f /tmp/ckpt_*.pt 2>/dev/null
            mkdir -p /tmp/xla_cache

            tmux new-session -d -s babysitter \
                \"export PATH=\\\$HOME/miniconda3/bin:\\\$HOME/.local/bin:\\\$PATH; \
                export CONTROL_PLANE=$CONTROL_PLANE \
                BUCKET=$BUCKET \
                TPU_NAME=$TPU_NAME \
                ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-} \
                MODEL_PATH=$MODEL_PATH \
                WANDB_MODE=${WANDB_MODE:-online} \
                PJRT_DEVICE=TPU \
                XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache \
                XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache \
                LLVM_NUM_THREADS=32 \
                HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1; \
                flock -n /tmp/tpu_babysitter.lock python3 -u ~/distributed_tpu_training/pull/babysitter.py \
                2>&1 | tee /tmp/babysitter.log; echo BABYSITTER_EXITED\"

            tmux list-sessions 2>/dev/null
        " 2>&1 | tail -5

    if [ $? -eq 0 ]; then
        log "  ✓ $TPU_NAME done"
    else
        log "  ✗ $TPU_NAME FAILED"
    fi
    echo ""
done

log "All deployments complete."
