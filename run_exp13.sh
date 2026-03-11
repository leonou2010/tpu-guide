#!/bin/bash
# run_exp13.sh — Autonomous orchestrator for exp13
# Handles: exp12_1 completion, VM setup, exp13 deploy+monitor, VM scanning
# Usage: bash ~/tpu_guide/run_exp13.sh 2>&1 | tee /tmp/run_exp13.log
set -euo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
GSUTIL=~/google-cloud-sdk/bin/gsutil
PROJECT=gcp-research-credits-489020
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 0: Wait for exp12_1 to finish ──────────────────────────────────
log "=== Phase 0: Checking exp12_1 status ==="
EXP12_VALIDATED=$(ls ~/sf_bema/results/exp12_1/validated/*.json 2>/dev/null | wc -l)
log "exp12_1: $EXP12_VALIDATED/185 validated"

if [ "$EXP12_VALIDATED" -lt 185 ]; then
  log "exp12_1 not done yet ($EXP12_VALIDATED/185). Waiting for completion..."
  while true; do
    EXP12_VALIDATED=$(ls ~/sf_bema/results/exp12_1/validated/*.json 2>/dev/null | wc -l)
    if [ "$EXP12_VALIDATED" -ge 185 ]; then
      log "exp12_1 COMPLETE: $EXP12_VALIDATED/185 validated!"
      break
    fi
    # Check if monitor is alive
    if ! pgrep -f "coordinator.*--monitor" > /dev/null 2>&1; then
      log "WARNING: exp12_1 monitor dead. Restarting..."
      cd ~/sf_bema/experiments/exp10_smollm2_smoltalk
      EXP=exp12_1 python3 -u ~/tpu_guide/coordinator.py --monitor >> /tmp/monitor_exp12_1.log 2>&1 &
      log "Monitor restarted (pid=$!)"
    fi
    log "exp12_1: $EXP12_VALIDATED/185 validated. Waiting 120s..."
    sleep 120
  done
fi

# Copy exp12_1 results
log "Copying exp12_1 validated results..."
DST12=~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu/results/
mkdir -p "$DST12"
cp ~/sf_bema/results/exp12_1/validated/*.json "$DST12/" 2>/dev/null || true
COPIED12=$(ls "$DST12"/*.json 2>/dev/null | wc -l)
log "exp12_1 results: $COPIED12 files copied to $DST12"

# Kill exp12_1 babysitter if still running
pkill -f "babysit_exp12_1" 2>/dev/null || true

# ── Phase 1: Cancel exp12_1 workers on all VMs ──────────────────────────
log "=== Phase 1: Cancelling exp12_1 workers on fleet ==="
for cfg in "$SCRIPT_DIR"/vm_configs/*.env; do
  vm=$(basename "$cfg" .env)
  EXP=exp12_1 TPU_NAME=$vm bash "$SCRIPT_DIR/submit.sh" --cancel 2>/dev/null || true
done
log "All exp12_1 workers cancelled."

# ── Phase 2: Scan and set up VMs ────────────────────────────────────────
log "=== Phase 2: VM fleet scan + setup ==="

setup_vm() {
  local tpu_name=$1
  local zone=$2
  local bucket=$3
  log "  [setup] $tpu_name ($zone) — checking if already set up..."

  # Quick check: can we import torch_xla?
  CHECK=$($GCLOUD alpha compute tpus tpu-vm ssh "$tpu_name" \
    --zone="$zone" --project=$PROJECT --tunnel-through-iap \
    --command="python3 -c 'import torch_xla; print(\"OK\")' 2>/dev/null || echo NEED_SETUP" 2>&1 | grep -E 'OK|NEED_SETUP' | head -1)

  if [ "$CHECK" = "OK" ]; then
    log "  [setup] $tpu_name: torch_xla OK, skipping setup"
    return 0
  fi

  log "  [setup] $tpu_name: needs setup, running..."
  EXP=exp13 TPU_NAME=$tpu_name ZONE=$zone BUCKET=$bucket \
    bash "$SCRIPT_DIR/submit.sh" --setup 2>&1 | tail -5
  return $?
}

# Check each VM config and setup if needed
READY_VMS=0
SETUP_FAILED=()
for cfg in "$SCRIPT_DIR"/vm_configs/*.env; do
  vm=$(basename "$cfg" .env)
  source "$cfg"
  VM_ZONE=${ZONE:-europe-west4-a}
  VM_BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}

  # Check if VM is READY
  STATUS=$($GCLOUD alpha compute tpus tpu-vm describe "$vm" \
    --zone="$VM_ZONE" --project=$PROJECT --format='value(state)' 2>/dev/null || echo "NOT_FOUND")

  if [ "$STATUS" != "READY" ]; then
    log "  $vm: status=$STATUS — skipping"
    continue
  fi

  if setup_vm "$vm" "$VM_ZONE" "$VM_BUCKET"; then
    READY_VMS=$((READY_VMS + 1))
  else
    log "  WARNING: $vm setup failed"
    SETUP_FAILED+=("$vm")
  fi
done
log "Fleet: $READY_VMS VMs ready. Failed: ${SETUP_FAILED[*]:-none}"

# ── Phase 3: Init + deploy exp13 ────────────────────────────────────────
log "=== Phase 3: Init + deploy exp13 ==="

cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
export EXP=exp13

# Init: distribute configs to VMs
log "Initializing exp13 (distribute configs)..."
python3 -u ~/tpu_guide/coordinator.py --init 2>&1

# Deploy and sweep on all VMs
log "Deploying code and starting sweep on all VMs..."
for cfg in "$SCRIPT_DIR"/vm_configs/*.env; do
  vm=$(basename "$cfg" .env)
  source "$cfg"
  VM_ZONE=${ZONE:-europe-west4-a}
  VM_BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
  PPH=${PROCS_PER_HOST:-0}

  # Skip VMs with 0 procs (disabled)
  if [ "$PPH" -eq 0 ]; then
    log "  $vm: PROCS_PER_HOST=0, skipping"
    continue
  fi

  # Check if VM is READY
  STATUS=$($GCLOUD alpha compute tpus tpu-vm describe "$vm" \
    --zone="$VM_ZONE" --project=$PROJECT --format='value(state)' 2>/dev/null || echo "NOT_FOUND")
  if [ "$STATUS" != "READY" ]; then
    log "  $vm: status=$STATUS — skipping deploy"
    continue
  fi

  log "  Deploying exp13 -> $vm ($PPH procs)..."
  EXP=exp13 TPU_NAME=$vm bash "$SCRIPT_DIR/submit.sh" --sweep 2>&1 | tail -3 || {
    log "  WARNING: Deploy to $vm failed"
  }
done

log "=== Phase 3 complete: exp13 deployed ==="

# ── Phase 4: Monitor until complete ─────────────────────────────────────
log "=== Phase 4: Monitoring exp13 (120 configs, 882 steps each) ==="
log "Dashboard: watch -c -n30 'python3 ~/tpu_guide/dashboard.py --exp exp13'"

# Start monitor (blocks until all 120 validated)
cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
EXP=exp13 python3 -u ~/tpu_guide/coordinator.py --monitor 2>&1 | tee -a /tmp/monitor_exp13.log

# ── Phase 5: Copy results ───────────────────────────────────────────────
log "=== Phase 5: exp13 COMPLETE — copying results ==="
DST13=~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
mkdir -p "$DST13"
cp ~/sf_bema/results/exp13/validated/*.json "$DST13/" 2>/dev/null || true
COPIED13=$(ls "$DST13"/*.json 2>/dev/null | wc -l)
log "exp13 results: $COPIED13 files copied to $DST13"
log "=== ALL DONE ==="
