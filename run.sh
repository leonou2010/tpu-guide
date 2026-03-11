#!/bin/bash
# run.sh — Single command to run a full TPU experiment end-to-end.
#
# Usage: EXP=<name> bash ~/tpu_guide/run.sh
#
# What it does:
#   1. Acquire VMs (create up to quota in all managed zones)
#   2. Setup all VMs (packages, code, data, model)
#   3. Distribute configs (coordinator --init)
#   4. Sweep all VMs (launch workers)
#   5. Start coordinator monitor (background)
#   6. Start fleet manager (background — handles preemption, expansion, completion)
#
# Prerequisites:
#   - ~/tpu_guide/experiments/${EXP}.env exists
#   - Experiment code is in place (build_configs, build_command, run_single)
#   - GCS buckets have wheels, data, model

set -euo pipefail

EXP=${EXP:?'EXP required (e.g. EXP=exp12_1)'}

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

source "$SCRIPT_DIR/experiments/${EXP}.env"

LOG=/tmp/run_${EXP}.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Zone config ────────────────────────────────────────────────────────────

declare -A ZONE_MAX_VMS ZONE_BUCKET ZONE_WANDB ZONE_SHORT ZONE_RUNTIME
ZONE_MAX_VMS[europe-west4-a]=8
ZONE_BUCKET[europe-west4-a]=gs://gcp-researchcredits-blocklab-europe-west4
ZONE_WANDB[europe-west4-a]=online
ZONE_SHORT[europe-west4-a]=ew4a
ZONE_RUNTIME[europe-west4-a]=v2-alpha-tpuv6e

ZONE_MAX_VMS[us-east1-d]=8
ZONE_BUCKET[us-east1-d]=gs://gcp-researchcredits-blocklab-us-east1
ZONE_WANDB[us-east1-d]=disabled
ZONE_SHORT[us-east1-d]=ue1d
ZONE_RUNTIME[us-east1-d]=v2-alpha-tpuv6e

MANAGED_ZONES=("europe-west4-a" "us-east1-d")

# ── Helper: find next VM number for a zone ─────────────────────────────────

next_vm_number() {
    local zone_short=$1 max=0
    for cfg in "$SCRIPT_DIR"/vm_configs/v6e-${zone_short}-*.env 2>/dev/null; do
        [ -f "$cfg" ] || continue
        local num
        num=$(basename "$cfg" .env | sed "s/v6e-${zone_short}-//" | sed 's/[^0-9]//g')
        [ -n "$num" ] && [ "$num" -gt "$max" ] && max=$num
    done
    echo $((max + 1))
}

count_vms_in_zone() {
    local zone_short=$1 count=0
    for cfg in "$SCRIPT_DIR"/vm_configs/v6e-${zone_short}-*.env 2>/dev/null; do
        [ -f "$cfg" ] && count=$((count + 1))
    done
    echo $count
}

write_vm_config() {
    local name=$1 zone=$2 bucket=$3 wandb=$4
    cat > "$SCRIPT_DIR/vm_configs/${name}.env" << EOF
# VM: $name — $zone — v6e-8 SPOT (8 chips, 1 host)
TPU_NAME=$name
ZONE=$zone
TPU_NUM_WORKERS=1
CHIPS_PER_HOST=8
BUCKET=$bucket
ACCELERATOR_TYPE=v6e-8
RUNTIME_VERSION=v2-alpha-tpuv6e
WANDB_MODE=$wandb
PROCS_PER_HOST=8
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Acquire VMs
# ══════════════════════════════════════════════════════════════════════════════

log "=== Phase 1: Acquiring VMs ==="

for zone in "${MANAGED_ZONES[@]}"; do
    zone_short=${ZONE_SHORT[$zone]}
    max_vms=${ZONE_MAX_VMS[$zone]}
    bucket=${ZONE_BUCKET[$zone]}
    wandb=${ZONE_WANDB[$zone]}
    runtime=${ZONE_RUNTIME[$zone]}
    current=$(count_vms_in_zone "$zone_short")

    log "  $zone: $current/$max_vms VMs"

    while [ "$current" -lt "$max_vms" ]; do
        num=$(next_vm_number "$zone_short")
        name="v6e-${zone_short}-${num}"
        log "  Creating $name..."

        if timeout 180 $GCLOUD alpha compute tpus tpu-vm create "$name" \
            --zone="$zone" --project=$PROJECT \
            --accelerator-type=v6e-8 --version="$runtime" \
            --spot --internal-ips 2>&1 | tee -a "$LOG"; then
            write_vm_config "$name" "$zone" "$bucket" "$wandb"
            log "  $name: CREATED"
            current=$((current + 1))
        else
            log "  $name: no capacity. Stopping for $zone."
            break
        fi
    done
done

vm_count=$(ls "$SCRIPT_DIR"/vm_configs/v6e-*.env 2>/dev/null | wc -l)
log "Total VMs: $vm_count"

if [ "$vm_count" -eq 0 ]; then
    log "FATAL: No VMs available. Exiting."
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Setup all VMs
# ══════════════════════════════════════════════════════════════════════════════

log "=== Phase 2: Setting up all VMs ==="
EXP=$EXP bash "$SCRIPT_DIR/submit.sh" --setup-all 2>&1 | tee -a "$LOG"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Init (distribute configs)
# ══════════════════════════════════════════════════════════════════════════════

log "=== Phase 3: Distributing configs ==="
cd ~/sf_bema/experiments/$WORK_DIR
TOTAL=$(EXP=$EXP python3 ~/tpu_guide/coordinator.py --dry-run 2>/dev/null | grep -oP '^\d+ configs' | grep -oP '^\d+' || echo 0)
if [ "$TOTAL" -eq 0 ]; then
    # Fallback: count from init output
    init_out=$(EXP=$EXP python3 ~/tpu_guide/coordinator.py --init 2>&1)
    echo "$init_out" | tee -a "$LOG"
    TOTAL=$(echo "$init_out" | grep -oP '\d+ configs from' | grep -oP '^\d+' || echo 0)
else
    EXP=$EXP python3 ~/tpu_guide/coordinator.py --init 2>&1 | tee -a "$LOG"
fi
log "Total configs: $TOTAL"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Sweep all VMs
# ══════════════════════════════════════════════════════════════════════════════

log "=== Phase 4: Sweeping all VMs ==="
EXP=$EXP bash "$SCRIPT_DIR/submit.sh" --sweep-all 2>&1 | tee -a "$LOG"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Start monitor + fleet manager (background)
# ══════════════════════════════════════════════════════════════════════════════

log "=== Phase 5: Starting monitor + fleet manager ==="

MONITOR_LOG=/tmp/monitor_${EXP}.log
nohup env EXP=$EXP PYTHONUNBUFFERED=1 python3 -u ~/tpu_guide/coordinator.py --monitor >> "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
log "Monitor PID: $MONITOR_PID"

nohup env EXP=$EXP TOTAL=$TOTAL bash ~/tpu_guide/fleet_manager.sh >> /tmp/fleet_${EXP}.log 2>&1 &
FLEET_PID=$!
log "Fleet manager PID: $FLEET_PID"

log ""
log "══════════════════════════════════════════════════════════════"
log "  EXPERIMENT $EXP LAUNCHED"
log "  $vm_count VMs | $TOTAL configs | fleet manager running"
log ""
log "  Dashboard:     python3 ~/tpu_guide/dashboard.py --exp $EXP --interval 30"
log "  Monitor log:   tail -f $MONITOR_LOG"
log "  Fleet log:     tail -f /tmp/fleet_${EXP}.log"
log "  Run log:       tail -f $LOG"
log ""
log "  Fleet manager handles everything from here:"
log "    - Preemption recovery"
log "    - VM expansion"
log "    - Monitor restarts"
log "    - Result collection when done"
log "══════════════════════════════════════════════════════════════"
