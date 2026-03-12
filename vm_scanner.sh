#!/bin/bash
# vm_scanner.sh — Periodically scan for new/recreated VMs and add them to the running experiment
# Runs alongside the main orchestrator. Creates VM configs for new VMs, sets them up, deploys sweep.
# Usage: EXP=exp13 bash ~/distributed_tpu_training/vm_scanner.sh 2>&1 | tee /tmp/vm_scanner.log
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
  [us-central1-a]="v5e"
)
declare -A ZONE_BUCKET=(
  [europe-west4-a]="gs://gcp-researchcredits-blocklab-europe-west4"
  [us-east1-d]="gs://gcp-researchcredits-blocklab-us-east1"
  [us-central2-b]="gs://gcp-researchcredits-blocklab-1-us-central2"
  [europe-west4-b]="gs://gcp-researchcredits-blocklab-europe-west4"
  [us-central1-a]="gs://gcp-researchcredits-blocklab-europe-west4"
)
declare -A ZONE_WANDB=(
  [europe-west4-a]="online"
  [us-east1-d]="disabled"
  [us-central2-b]="disabled"
  [europe-west4-b]="online"
  [us-central1-a]="online"
)
declare -A ZONE_RUNTIME=(
  [europe-west4-a]="v2-alpha-tpuv6e"
  [us-east1-d]="v2-alpha-tpuv6e"
  [us-central2-b]="tpu-ubuntu2204-base"
  [europe-west4-b]="v2-alpha-tpuv5-lite"
  [us-central1-a]="v2-alpha-tpuv5-lite"
)

declare -A ZONE_ACCEL=(
  [europe-west4-a]="v6e-8"
  [us-east1-d]="v6e-8"
  [us-central2-b]="v4-8"
  [europe-west4-b]="v5litepod-4"
  [us-central1-a]="v5litepod-4"
)
declare -A ZONE_MAX_CHIPS=(
  [europe-west4-a]=64
  [us-east1-d]=64
  [us-central2-b]=64
  [europe-west4-b]=64
  [us-central1-a]=64
)

ZONES="europe-west4-a us-east1-d us-central2-b europe-west4-b us-central1-a"

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
          # Check for ANY active tmux sessions (don't kill other experiments)
          ALL_SESSIONS=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" \
            --zone="$zone" --project=$PROJECT --tunnel-through-iap \
            --command="tmux list-sessions 2>/dev/null | wc -l || echo 0" 2>&1 | grep -oE '[0-9]+' | tail -1 || echo 0)

          if [ "${ALL_SESSIONS:-0}" -eq 0 ]; then
            log "  $name: READY, no sessions at all — deploying $EXP..."
            setup_and_deploy "$name" || true
            NEW_VMS=$((NEW_VMS + 1))
          else
            # Has sessions — check if they're for OUR experiment
            OUR_SESSIONS=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" \
              --zone="$zone" --project=$PROJECT --tunnel-through-iap \
              --command="tmux list-sessions 2>/dev/null | grep -c '${EXP}' || echo 0" 2>&1 | grep -oE '[0-9]+' | tail -1 || echo 0)
            if [ "${OUR_SESSIONS:-0}" -eq 0 ]; then
              log "  $name: has sessions from another experiment — skipping (let babysitter handle transition)"
            fi
          fi
        fi
      fi
    done <<< "$VMS"
  done

  # --- Phase 2: Try to create new VMs in zones with spare quota ---
  log "--- Trying to expand fleet ---"
  for zone in $ZONES; do
    accel=${ZONE_ACCEL[$zone]:-""}
    [ -z "$accel" ] && continue
    max_chips=${ZONE_MAX_CHIPS[$zone]:-64}
    runtime=${ZONE_RUNTIME[$zone]:-""}
    chip_per_vm=8
    if [[ "$accel" == *"v5litepod-4"* ]]; then
      chip_per_vm=4
    fi

    # Count current chips in zone
    CURRENT=$($GCLOUD alpha compute tpus tpu-vm list \
      --zone="$zone" --project=$PROJECT \
      --format='csv[no-heading](acceleratorType)' 2>/dev/null | while read a; do
        echo "$a" | grep -oP '\d+$'
      done | paste -sd+ | bc 2>/dev/null || echo 0)
    CURRENT=${CURRENT:-0}

    REMAINING=$((max_chips - CURRENT))
    if [ "$REMAINING" -lt "$chip_per_vm" ]; then
      continue  # zone full
    fi

    # How many VMs can we create?
    NUM_NEW=$((REMAINING / chip_per_vm))
    [ "$NUM_NEW" -gt 3 ] && NUM_NEW=3  # max 3 per cycle to avoid hammering API

    log "  $zone: ${CURRENT}/${max_chips} chips, trying $NUM_NEW new VMs..."

    for i in $(seq 1 $NUM_NEW); do
      # Generate unique name
      PREFIX="${ZONE_TYPE[$zone]}-$(echo $zone | sed 's/[^a-z0-9]//g' | head -c8)"
      # Find next available number
      EXISTING=$($GCLOUD alpha compute tpus tpu-vm list \
        --zone="$zone" --project=$PROJECT \
        --format='value(name)' 2>/dev/null || true)
      for n in $(seq 1 99); do
        CANDIDATE="${ZONE_TYPE[$zone]}-$(echo $zone | cut -d- -f1-2 | head -c3)$(echo $zone | grep -oP '\d+[a-z]$')-${n}"
        # Make more readable names
        case "$zone" in
          europe-west4-a) CANDIDATE="v6e-ew4a-${n}" ;;
          us-east1-d)     CANDIDATE="v6e-ue1d-${n}" ;;
          us-central2-b)  CANDIDATE="v4-uc2b-${n}" ;;
          europe-west4-b) CANDIDATE="v5e-ew4b-${n}" ;;
          us-central1-a)  CANDIDATE="v5e-uc1a-${n}" ;;
        esac
        if ! echo "$EXISTING" | grep -qx "$CANDIDATE"; then
          break
        fi
      done

      log "  Creating $CANDIDATE ($accel) in $zone..."
      $GCLOUD alpha compute tpus tpu-vm create "$CANDIDATE" \
        --zone="$zone" --project=$PROJECT \
        --accelerator-type="$accel" --version="$runtime" --spot --internal-ips 2>&1 | tail -2 &
    done
    wait  # wait for all creates in this zone
  done

  if [ "$NEW_VMS" -gt 0 ]; then
    log "Scan complete: $NEW_VMS VMs (re)deployed"
    # NOTE: Do NOT run --init here — it clears heartbeats and done receipts!
    # New VMs will pick up work from existing assignments via the coordinator.
  else
    log "Scan complete: no changes"
  fi

  sleep $SCAN_INTERVAL
done
