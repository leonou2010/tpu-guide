#!/bin/bash
# babysit_exp13.sh — Autonomous babysitter for exp13
# Handles: monitor keepalive, exp12_1 transition, VM recovery, result copy
# Usage: nohup bash ~/tpu_guide/babysit_exp13.sh >> /tmp/babysit_exp13.log 2>&1 &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
EXP=exp13
TARGET=120

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Helper: deploy exp13 to a single VM ---
deploy_vm() {
  local vm=$1
  log "  Deploying exp13 -> $vm"
  EXP=exp13 TPU_NAME=$vm bash "$SCRIPT_DIR/submit.sh" --sweep 2>&1 | tail -3 || {
    log "  WARNING: Deploy to $vm failed"
  }
}

# --- Helper: count validated results ---
count_validated() {
  ls ~/sf_bema/results/exp13/validated/*.json 2>/dev/null | wc -l
}

# --- Phase 0: Wait for exp12_1 last configs + transition VMs ---
EXP12_DONE=false
EXP12_VMS="v6e-ew4a-4 v6e-ew4a-5 v6e-ew4a-6 v6e-ew4a-8d"

log "=== babysit_exp13 started (target: $TARGET configs) ==="

# --- Main monitoring loop ---
while true; do
  VALIDATED=$(count_validated)
  log "Progress: $VALIDATED/$TARGET validated"

  # Check exp12_1 transition
  if [ "$EXP12_DONE" = "false" ]; then
    EXP12_COUNT=$(ls ~/sf_bema/results/exp12_1/validated/*.json 2>/dev/null | wc -l)
    if [ "$EXP12_COUNT" -ge 185 ]; then
      log "exp12_1 COMPLETE ($EXP12_COUNT/185). Transitioning VMs to exp13..."
      EXP12_DONE=true
      # Copy exp12_1 results
      mkdir -p ~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu/results/
      cp ~/sf_bema/results/exp12_1/validated/*.json ~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu/results/ 2>/dev/null
      log "exp12_1 results copied."
      # Deploy exp13 to freed VMs
      for vm in $EXP12_VMS; do
        deploy_vm "$vm" &
      done
      wait
      log "exp12_1 VMs transitioned to exp13"
    else
      log "  exp12_1: $EXP12_COUNT/185 (waiting for completion)"
    fi
  fi

  # Check if exp13 is complete
  if [ "$VALIDATED" -ge "$TARGET" ]; then
    log "=== exp13 COMPLETE: $VALIDATED/$TARGET validated ==="
    # Copy results to experiment folder
    DST=~/sf_bema/experiments/exp13_smollm2_smoltalk/exp13_tpu/results/
    mkdir -p "$DST"
    cp ~/sf_bema/results/exp13/validated/*.json "$DST/" 2>/dev/null
    COPIED=$(ls "$DST"/*.json 2>/dev/null | wc -l)
    log "Results copied: $COPIED files -> $DST"
    log "=== ALL DONE ==="
    break
  fi

  # Check if monitor is alive
  if ! pgrep -f "coordinator.*--monitor" > /dev/null 2>&1; then
    log "WARNING: exp13 monitor dead. Restarting..."
    cd ~/sf_bema/experiments/exp13_smollm2_smoltalk
    EXP=exp13 python3 -u ~/tpu_guide/coordinator.py --monitor >> /tmp/monitor_exp13.log 2>&1 &
    MONITOR_PID=$!
    log "Monitor restarted (pid=$MONITOR_PID)"
  fi

  # Periodic VM check — look for VMs with no active sessions
  if [ $((RANDOM % 6)) -eq 0 ]; then  # ~1 in 6 cycles (~every 30 min)
    log "--- Periodic VM health check ---"
    for cfg in "$SCRIPT_DIR"/vm_configs/*.env; do
      vm=$(basename "$cfg" .env)
      source "$cfg"
      PPH=${PROCS_PER_HOST:-0}
      [ "$PPH" -eq 0 ] && continue

      # Check if VM is READY
      STATUS=$($GCLOUD alpha compute tpus tpu-vm describe "$vm" \
        --zone="${ZONE}" --project=$PROJECT --format='value(state)' 2>/dev/null || echo "NOT_FOUND")

      if [ "$STATUS" != "READY" ]; then
        continue
      fi

      # Check for active tmux sessions
      HAS_SESSIONS=$($GCLOUD alpha compute tpus tpu-vm ssh "$vm" \
        --zone="${ZONE}" --project=$PROJECT --tunnel-through-iap \
        --command="tmux list-sessions 2>/dev/null | grep -c exp13 || echo 0" 2>&1 | grep -oE '[0-9]+' | tail -1 || echo 0)

      if [ "${HAS_SESSIONS:-0}" -eq 0 ]; then
        log "  $vm: READY but no exp13 sessions — re-deploying..."
        deploy_vm "$vm" &
      fi
    done
    wait
  fi

  sleep 300  # Check every 5 min
done
