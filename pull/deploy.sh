#!/bin/bash
# deploy.sh — Deploy pull-based babysitter to TPU VMs.
#
# Usage:
#   TPU_NAME=v6e-ew4a-2 bash ~/distributed_tpu_training/pull/deploy.sh           # single VM
#   bash ~/distributed_tpu_training/pull/deploy.sh --all                          # all VMs
#   bash ~/distributed_tpu_training/pull/deploy.sh --all --dry-run                # preview
#
# Env vars (from vm_config or manual):
#   TPU_NAME, ZONE, BUCKET, ACCELERATOR_TYPE, WANDB_MODE, MODEL_PATH

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
CONTROL_PLANE=${CONTROL_PLANE:-gs://gcp-researchcredits-blocklab-europe-west4/coord_v2}
MODEL_PATH=${MODEL_PATH:-/tmp/SmolLM2-135M}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── SSH helper ───────────────────────────────────────────────────────────

ssh_vm() {
    local vm=$1 zone=$2
    shift 2
    $GCLOUD alpha compute tpus tpu-vm ssh "$vm" \
        --zone="$zone" --project=$PROJECT --tunnel-through-iap \
        --worker=all --command="$*" 2>&1
}

# ── Deploy to one VM ─────────────────────────────────────────────────────

deploy_one() {
    local vm=$1 zone=$2 bucket=$3 accel=${4:-} wandb_mode=${5:-online}

    log "Deploying babysitter -> $vm ($zone)"

    # 1. Upload pull/ code to GCS
    log "  Uploading code to $bucket/code/pull/"
    gsutil -m cp "$SCRIPT_DIR"/*.py "$bucket/code/pull/" 2>/dev/null || \
        gcloud storage cp "$SCRIPT_DIR"/*.py "$bucket/code/pull/"

    # 2. Deploy to VM
    log "  Deploying to VM..."
    ssh_vm "$vm" "$zone" "
        set -e

        # Pull babysitter code from GCS
        mkdir -p ~/distributed_tpu_training/pull
        gsutil -m cp $bucket/code/pull/*.py ~/distributed_tpu_training/pull/ 2>/dev/null || \
            gcloud storage cp $bucket/code/pull/*.py ~/distributed_tpu_training/pull/

        # Kill existing training sessions
        echo 'Killing existing sessions...'
        for s in \$(tmux list-sessions 2>/dev/null | cut -d: -f1); do
            tmux kill-session -t \"\$s\" 2>/dev/null
        done
        { fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
        sleep 1

        # NOTE: Keep checkpoints for resume — babysitter will resume from /tmp/ckpt_*.pt
        # rm -f /tmp/ckpt_*.pt  # DO NOT DELETE — causes restart from scratch

        # Ensure XLA cache dir exists
        mkdir -p /tmp/xla_cache

        # Launch babysitter in tmux with flock
        echo 'Launching babysitter...'
        tmux new-session -d -s babysitter \\
            \"export PATH=\\\$HOME/miniconda3/bin:\\\$HOME/.local/bin:\\\$PATH; \\
            export CONTROL_PLANE=$CONTROL_PLANE \\
            BUCKET=$bucket \\
            TPU_NAME=$vm \\
            ACCELERATOR_TYPE=$accel \\
            MODEL_PATH=$MODEL_PATH \\
            WANDB_MODE=$wandb_mode \\
            PJRT_DEVICE=TPU \\
            XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache \\
            XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache \\
            LLVM_NUM_THREADS=32 \\
            HF_HUB_OFFLINE=1 \\
            TRANSFORMERS_OFFLINE=1 \\
            HYDRA_FULL_ERROR=1 \\
            PYTHONUNBUFFERED=1; \\
            flock -n /tmp/tpu_babysitter.lock python3 -u ~/distributed_tpu_training/pull/babysitter.py \\
            2>&1 | tee /tmp/babysitter.log; \\
            echo BABYSITTER_EXITED\"

        echo 'Babysitter launched.'
        tmux list-sessions 2>/dev/null
    "

    if [ $? -eq 0 ]; then
        log "  ✓ $vm deployed"
    else
        log "  ✗ $vm FAILED"
    fi
}

# ── Deploy all VMs ───────────────────────────────────────────────────────

deploy_all() {
    local dry_run=${1:-false}
    local vm_dir="$SCRIPT_DIR/../vm_configs"

    for cfg_file in "$vm_dir"/*.env; do
        [ -f "$cfg_file" ] || continue

        # Source config
        source "$cfg_file"
        local pph=${PROCS_PER_HOST:-0}
        [ "$pph" -eq 0 ] && continue

        if [ "$dry_run" = "true" ]; then
            log "[dry-run] Would deploy to $TPU_NAME ($ZONE, $ACCELERATOR_TYPE)"
        else
            deploy_one "$TPU_NAME" "$ZONE" "$BUCKET" "${ACCELERATOR_TYPE:-}" "${WANDB_MODE:-online}" &
        fi
    done

    if [ "$dry_run" != "true" ]; then
        log "Waiting for all deployments..."
        wait
        log "All deployments done."
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-}" in
    --all)
        deploy_all "${2:-false}"
        ;;
    --dry-run)
        deploy_all "true"
        ;;
    *)
        if [ -z "${TPU_NAME:-}" ]; then
            echo "Usage: TPU_NAME=<vm> bash deploy.sh   OR   bash deploy.sh --all"
            exit 1
        fi
        # Auto-source vm config if it exists
        vm_cfg="$SCRIPT_DIR/../vm_configs/${TPU_NAME}.env"
        if [ -f "$vm_cfg" ]; then
            source "$vm_cfg"
        fi
        deploy_one "$TPU_NAME" "${ZONE:?}" "${BUCKET:?}" "${ACCELERATOR_TYPE:-}" "${WANDB_MODE:-online}"
        ;;
esac
