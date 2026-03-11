#!/bin/bash
# vm_scanner.sh — Periodically scan for new/recreated VMs and add them to the running experiment
# Runs alongside the main orchestrator. Creates VM configs for new VMs, sets them up, deploys sweep.
# Usage: EXP=exp13 bash ~/tpu_guide/vm_scanner.sh 2>&1 | tee /tmp/vm_scanner.log
set -uo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCAN_INTERVAL=${SCAN_INTERVAL:-600}  # 10 min default

EXP=${EXP:?'EXP env var required'}

log() { echo "[$(date '+%H:%M:%S')] [scanner] $*"; }

# Zone → TPU type → bucket mapping
declare -A ZONE_TYPE=(
  [europe-west4-a]="v6e"
  [us-east1-d]="v6e"
  [us-central2-b]="v4"
  [europe-west4-b]="v5e"
)
declare -A ZONE_BUCKET=(
  [europe-west4-a]="gs://gcp-researchcredits-blocklab-europe-west4"
  [us-east1-d]="gs://gcp-researchcredits-blocklab-us-east1"
  [us-central2-b]="gs://gcp-researchcredits-blocklab-1-us-central2"
  [europe-west4-b]="gs://gcp-researchcredits-blocklab-europe-west4"
)
declare -A ZONE_WANDB=(
  [europe-west4-a]="online"
  [us-east1-d]="disabled"
  [us-central2-b]="disabled"
  [europe-west4-b]="online"
)
declare -A ZONE_RUNTIME=(
  [europe-west4-a]="v2-alpha-tpuv6e"
  [us-east1-d]="v2-alpha-tpuv6e"
  [us-central2-b]="tpu-ubuntu2204-base"
  [europe-west4-b]="v2-alpha-tpuv5-lite"
)

ZONES="europe-west4-a us-east1-d us-central2-b europe-west4-b"

create_vm_config() {
  local name=$1 zone=$2 accel=$3
  local cfg_file="$SCRIPT_DIR/vm_configs/${name}.env"

  if [ -f "$cfg_file" ]; then
    return 0  # already exists
  fi

  local bucket=${ZONE_BUCKET[$zone]}
  local wandb=${ZONE_WANDB[$zone]}
  local runtime=${ZONE_RUNTIME[$zone]}
  local tpu_type=${ZONE_TYPE[$zone]}

  # Determine chips and procs
  local chips=8
  local procs=8
  if [[ "$accel" == *"v5litepod-4"* ]]; then
    chips=4
    procs=4
  fi

  cat > "$cfg_file" << EOF
# VM: $name — $zone — $accel (auto-created by vm_scanner)
TPU_NAME=$name
ZONE=$zone
TPU_NUM_WORKERS=1
CHIPS_PER_HOST=$chips
BUCKET=$bucket
ACCELERATOR_TYPE=$accel
RUNTIME_VERSION=$runtime
WANDB_MODE=$wandb
PROCS_PER_HOST=$procs
EOF
  log "Created VM config: $cfg_file"
  return 1  # new config created
}

setup_and_deploy() {
  local name=$1
  local cfg_file="$SCRIPT_DIR/vm_configs/${name}.env"
  source "$cfg_file"
  local zone=${ZONE}
  local bucket=${BUCKET}

  # Check torch_xla
  log "  Checking $name for torch_xla..."
  local check
  check=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" \
    --zone="$zone" --project=$PROJECT --tunnel-through-iap \
    --command="python3 -c 'import torch_xla; print(\"OK\")' 2>/dev/null || echo NEED_SETUP" 2>&1 | grep -E 'OK|NEED_SETUP' | head -1)

  if [ "$check" != "OK" ]; then
    log "  Setting up $name..."
    EXP=$EXP TPU_NAME=$name bash "$SCRIPT_DIR/submit.sh" --setup 2>&1 | tail -3 || {
      log "  WARNING: Setup failed for $name"
      return 1
    }
  fi

  # Deploy code + start sweep
  log "  Deploying $EXP -> $name..."
  EXP=$EXP TPU_NAME=$name bash "$SCRIPT_DIR/submit.sh" --sweep 2>&1 | tail -3 || {
    log "  WARNING: Deploy failed for $name"
    return 1
  }
  log "  $name: deployed and sweeping!"
}

log "=== VM Scanner started for $EXP (interval=${SCAN_INTERVAL}s) ==="

while true; do
  log "--- Scanning all zones ---"
  NEW_VMS=0

  for zone in $ZONES; do
    # List all READY VMs in this zone
    VMS=$($GCLOUD alpha compute tpus tpu-vm list \
      --zone="$zone" --project=$PROJECT \
      --format='csv[no-heading](name,acceleratorType,state)' 2>/dev/null | grep ',READY$' || true)

    while IFS=',' read -r name accel state; do
      [ -z "$name" ] && continue

      cfg_file="$SCRIPT_DIR/vm_configs/${name}.env"

      # Create config if missing
      if [ ! -f "$cfg_file" ]; then
        create_vm_config "$name" "$zone" "$accel"
        log "New VM discovered: $name ($accel) in $zone"
        setup_and_deploy "$name" || true
        NEW_VMS=$((NEW_VMS + 1))
      else
        # Config exists — check if VM has an active assignment
        # (if not, it might need re-deployment after preemption recovery)
        source "$cfg_file"
        PPH=${PROCS_PER_HOST:-0}
        if [ "$PPH" -gt 0 ]; then
          # Quick liveness check via heartbeat age would be ideal,
          # but for now just check if tmux sessions exist
          HAS_SESSIONS=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" \
            --zone="$zone" --project=$PROJECT --tunnel-through-iap \
            --command="tmux list-sessions 2>/dev/null | grep -c '${EXP}' || echo 0" 2>&1 | grep -oE '[0-9]+' | tail -1 || echo 0)

          if [ "${HAS_SESSIONS:-0}" -eq 0 ]; then
            log "  $name: READY but no active sessions — re-deploying..."
            setup_and_deploy "$name" || true
            NEW_VMS=$((NEW_VMS + 1))
          fi
        fi
      fi
    done <<< "$VMS"
  done

  if [ "$NEW_VMS" -gt 0 ]; then
    log "Scan complete: $NEW_VMS VMs (re)deployed"
    # Re-init coordinator to include new VMs in distribution
    cd ~/sf_bema/experiments/$(grep WORK_DIR "$SCRIPT_DIR/experiments/${EXP}.env" | cut -d= -f2)
    EXP=$EXP python3 ~/tpu_guide/coordinator.py --init 2>&1 | tail -3
  else
    log "Scan complete: no changes"
  fi

  sleep $SCAN_INTERVAL
done
