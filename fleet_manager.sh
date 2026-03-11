#!/bin/bash
# fleet_manager.sh — Automated TPU VM lifecycle manager
# Replaces babysit.sh. Runs alongside coordinator.py --monitor.
#
# Usage: EXP=<name> TOTAL=<N> bash ~/tpu_guide/fleet_manager.sh
#
# Every 10 minutes:
#   1. Check/recover dead VMs (preempted or crashed workers)
#   2. Try to acquire more VMs up to quota
#   3. Ensure coordinator monitor is alive
#   4. Check completion, copy results when done

set -euo pipefail

EXP=${EXP:?'EXP required'}
TOTAL=${TOTAL:?'TOTAL required (number of configs)'}

# ── Paths & constants ────────────────────────────────────────────────────────

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VM_CONFIGS_DIR="$SCRIPT_DIR/vm_configs"

source "$SCRIPT_DIR/experiments/${EXP}.env"

RESULTS_DIR=~/sf_bema/results/$EXP/validated
EXP_DIR=~/sf_bema/experiments/$WORK_DIR/${EXP}_tpu
LOG=/tmp/fleet_${EXP}.log
MONITOR_LOG=/tmp/monitor_${EXP}.log
CHECK_INTERVAL=600  # 10 minutes

# ── Zone config ──────────────────────────────────────────────────────────────
# zone -> max_vms (quota/8), bucket, wandb_mode, zone_short

declare -A ZONE_MAX_VMS ZONE_BUCKET ZONE_WANDB ZONE_SHORT
ZONE_MAX_VMS[europe-west4-a]=8
ZONE_BUCKET[europe-west4-a]=gs://gcp-researchcredits-blocklab-europe-west4
ZONE_WANDB[europe-west4-a]=online
ZONE_SHORT[europe-west4-a]=ew4a

ZONE_MAX_VMS[us-east1-d]=8
ZONE_BUCKET[us-east1-d]=gs://gcp-researchcredits-blocklab-us-east1
ZONE_WANDB[us-east1-d]=disabled
ZONE_SHORT[us-east1-d]=ue1d

MANAGED_ZONES=("europe-west4-a" "us-east1-d")

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

count_validated() {
    ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l
}

restart_monitor() {
    log "RESTARTING coordinator monitor..."
    cd ~/sf_bema/experiments/$WORK_DIR
    nohup env EXP=$EXP PYTHONUNBUFFERED=1 python3 -u ~/tpu_guide/coordinator.py --monitor >> "$MONITOR_LOG" 2>&1 &
    log "New monitor PID: $!"
}

copy_results() {
    log "Copying validated results to exp folder..."
    mkdir -p "$EXP_DIR/results"
    cp "$RESULTS_DIR"/*.json "$EXP_DIR/results/"
    local count
    count=$(ls "$EXP_DIR/results/"*.json 2>/dev/null | wc -l)
    log "DONE: $count results copied to $EXP_DIR/results/"
}

# List TPU VMs in a zone. Output: "name state" per line.
list_vms_in_zone() {
    local zone=$1
    $GCLOUD alpha compute tpus tpu-vm list \
        --zone="$zone" --project=$PROJECT \
        --format="value(name,state)" 2>/dev/null || true
}

# Check if a specific VM is READY. Returns 0 if READY, 1 otherwise.
# Sets VM_STATE to the state string (or "NOT_FOUND").
check_vm_state() {
    local name=$1 zone=$2
    local listing
    listing=$(list_vms_in_zone "$zone")
    VM_STATE=$(echo "$listing" | awk -v n="$name" '$1==n {print $2}')
    if [ -z "$VM_STATE" ]; then
        VM_STATE="NOT_FOUND"
        return 1
    elif [ "$VM_STATE" = "READY" ]; then
        return 0
    else
        return 1
    fi
}

# Check if tmux sweep sessions are running on a VM
check_workers_alive() {
    local name=$1 zone=$2
    local result
    result=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" \
        --zone="$zone" --project=$PROJECT --tunnel-through-iap \
        --worker=0 --command="tmux list-sessions 2>/dev/null | grep -c '$EXP_NAME' || echo 0" \
        2>/dev/null) || { echo "0"; return; }
    echo "$result" | grep -oP '^\d+$' | head -1 || echo "0"
}

# Delete a VM (non-blocking wait, with timeout)
delete_vm() {
    local name=$1 zone=$2
    log "  Deleting $name in $zone..."
    timeout 120 $GCLOUD alpha compute tpus tpu-vm delete "$name" \
        --zone="$zone" --project=$PROJECT --quiet 2>&1 | tee -a "$LOG" || \
        log "  WARN: delete $name may have failed or timed out"
}

# Create a VM
create_vm() {
    local name=$1 zone=$2 accel=$3 runtime=$4
    log "  Creating $name in $zone ($accel)..."
    timeout 180 $GCLOUD alpha compute tpus tpu-vm create "$name" \
        --zone="$zone" --project=$PROJECT \
        --accelerator-type="$accel" --version="$runtime" \
        --spot --internal-ips 2>&1 | tee -a "$LOG"
}

# Setup + sweep a VM
setup_and_sweep() {
    local name=$1
    log "  Running setup on $name..."
    EXP=$EXP TPU_NAME="$name" bash ~/tpu_guide/submit.sh --setup 2>&1 | tee -a "$LOG" || {
        log "  WARN: setup failed for $name"
        return 1
    }
    log "  Running sweep on $name..."
    EXP=$EXP TPU_NAME="$name" bash ~/tpu_guide/submit.sh --sweep 2>&1 | tee -a "$LOG" || {
        log "  WARN: sweep failed for $name"
        return 1
    }
    return 0
}

# Write a vm_config file
write_vm_config() {
    local name=$1 zone=$2 bucket=$3 wandb=$4
    cat > "$VM_CONFIGS_DIR/${name}.env" << EOF
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
    log "  Wrote config: $VM_CONFIGS_DIR/${name}.env"
}

# Count VMs we manage in a given zone (based on vm_configs)
count_managed_vms_in_zone() {
    local zone=$1
    local count=0
    for cfg in "$VM_CONFIGS_DIR"/*.env; do
        [ -f "$cfg" ] || continue
        local cfg_zone
        cfg_zone=$(grep '^ZONE=' "$cfg" | cut -d= -f2)
        if [ "$cfg_zone" = "$zone" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Find the next available VM number for a zone
next_vm_number() {
    local zone_short=$1
    local max=0
    for cfg in "$VM_CONFIGS_DIR"/v6e-${zone_short}-*.env; do
        [ -f "$cfg" ] || continue
        local num
        num=$(basename "$cfg" .env | sed "s/v6e-${zone_short}-//" | sed 's/[^0-9]//g')
        if [ -n "$num" ] && [ "$num" -gt "$max" ]; then
            max=$num
        fi
    done
    echo $((max + 1))
}

# ── Phase 1: Check and recover dead VMs ─────────────────────────────────────

check_dead_vms() {
    log "--- Phase 1: Checking existing VMs ---"
    local recovered=0 resweep=0

    for cfg in "$VM_CONFIGS_DIR"/*.env; do
        [ -f "$cfg" ] || continue
        local name zone
        name=$(grep '^TPU_NAME=' "$cfg" | cut -d= -f2)
        zone=$(grep '^ZONE=' "$cfg" | cut -d= -f2)
        local accel runtime
        accel=$(grep '^ACCELERATOR_TYPE=' "$cfg" | cut -d= -f2)
        runtime=$(grep '^RUNTIME_VERSION=' "$cfg" | cut -d= -f2)

        # Check VM state
        if check_vm_state "$name" "$zone"; then
            # VM is READY — check if workers are alive
            local alive
            alive=$(check_workers_alive "$name" "$zone")
            if [ "$alive" = "0" ]; then
                log "  $name: READY but workers dead. Re-sweeping..."
                EXP=$EXP TPU_NAME="$name" bash ~/tpu_guide/submit.sh --sweep 2>&1 | tee -a "$LOG" || \
                    log "  WARN: re-sweep failed for $name"
                resweep=$((resweep + 1))
            else
                log "  $name: OK ($alive sessions)"
            fi
        else
            # VM is PREEMPTED/NOT_FOUND/other
            log "  $name: $VM_STATE — recovering..."

            # Delete if it still exists (PREEMPTED state)
            if [ "$VM_STATE" != "NOT_FOUND" ]; then
                delete_vm "$name" "$zone"
                # Wait a moment for deletion to propagate
                sleep 5
            fi

            # Recreate
            if create_vm "$name" "$zone" "$accel" "$runtime"; then
                log "  $name: recreated. Setting up..."
                if setup_and_sweep "$name"; then
                    log "  $name: RECOVERED successfully"
                    recovered=$((recovered + 1))
                else
                    log "  $name: recreated but setup/sweep failed"
                fi
            else
                log "  $name: FAILED to recreate (no capacity?)"
            fi
        fi
    done

    log "  Phase 1 done: $recovered recovered, $resweep re-swept"
}

# ── Phase 2: Try to acquire more VMs ────────────────────────────────────────

try_acquire_vms() {
    log "--- Phase 2: Checking for expansion opportunities ---"
    local acquired=0

    for zone in "${MANAGED_ZONES[@]}"; do
        local max_vms=${ZONE_MAX_VMS[$zone]}
        local bucket=${ZONE_BUCKET[$zone]}
        local wandb=${ZONE_WANDB[$zone]}
        local zone_short=${ZONE_SHORT[$zone]}

        local current
        current=$(count_managed_vms_in_zone "$zone")

        if [ "$current" -ge "$max_vms" ]; then
            log "  $zone: $current/$max_vms VMs (at quota)"
            continue
        fi

        local slots=$((max_vms - current))
        log "  $zone: $current/$max_vms VMs ($slots slots available)"

        # Try one VM at a time per cycle (avoid hammering API)
        local num
        num=$(next_vm_number "$zone_short")
        local new_name="v6e-${zone_short}-${num}"

        log "  Trying to create $new_name in $zone..."
        if create_vm "$new_name" "$zone" "v6e-8" "v2-alpha-tpuv6e"; then
            # Write config
            write_vm_config "$new_name" "$zone" "$bucket" "$wandb"

            # Setup + sweep
            if setup_and_sweep "$new_name"; then
                log "  $new_name: ACQUIRED and launched"
                acquired=$((acquired + 1))
            else
                log "  $new_name: created but setup/sweep failed — config saved, will retry next cycle"
            fi
        else
            log "  $new_name: no capacity in $zone (normal)"
        fi
    done

    log "  Phase 2 done: $acquired new VMs acquired"
}

# ── Phase 3: Check monitor ──────────────────────────────────────────────────

check_monitor() {
    log "--- Phase 3: Checking coordinator monitor ---"
    local monitor_pid
    monitor_pid=$(pgrep -f "coordinator.py --monitor" || true)

    if [ -z "$monitor_pid" ]; then
        local validated
        validated=$(count_validated)
        if [ "$validated" -ge "$TOTAL" ]; then
            log "  Monitor not running, but all $TOTAL configs done. OK."
        else
            log "  Monitor DEAD ($validated/$TOTAL done). Restarting..."
            restart_monitor
        fi
    else
        log "  Monitor alive (PID $monitor_pid)"
    fi
}

# ── Phase 4: Check completion ────────────────────────────────────────────────

check_completion() {
    local validated
    validated=$(count_validated)
    local last_line
    last_line=$(tail -1 "$MONITOR_LOG" 2>/dev/null || echo "no log")

    log "--- Progress: $validated/$TOTAL validated | $last_line ---"

    if [ "$validated" -ge "$TOTAL" ]; then
        log "ALL $TOTAL CONFIGS VALIDATED!"
        copy_results
        log "=== Fleet manager complete. Results in $EXP_DIR/results/ ==="
        return 0
    fi
    return 1
}

# ── Main loop ────────────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"
log "=== Fleet manager started for $EXP (target: $TOTAL configs) ==="
log "    Zones: ${MANAGED_ZONES[*]}"
log "    Check interval: ${CHECK_INTERVAL}s"
log "    VM configs: $VM_CONFIGS_DIR"

while true; do
    log "========== Fleet check cycle =========="

    check_dead_vms
    try_acquire_vms
    check_monitor

    if check_completion; then
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
