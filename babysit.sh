#!/bin/bash
# Generic overnight babysitter
# Usage: EXP=<name> TOTAL=<N> bash ~/distributed_tpu_training/babysit.sh
# Checks every 10 min: monitor alive? results accumulating?
# When done: copies results to exp folder.

set -euo pipefail

EXP=${EXP:?'EXP required'}
TOTAL=${TOTAL:?'TOTAL required (number of configs)'}

# Load experiment config for WORK_DIR
source ~/distributed_tpu_training/experiments/${EXP}.env

RESULTS_DIR=~/sf_bema/results/$EXP/validated
EXP_DIR=~/sf_bema/experiments/$WORK_DIR/${EXP}_tpu
LOG=/tmp/babysit_${EXP}.log
MONITOR_LOG=/tmp/monitor_${EXP}.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

restart_monitor() {
    log "RESTARTING coordinator monitor..."
    cd ~/sf_bema/experiments/$WORK_DIR
    nohup env EXP=$EXP PYTHONUNBUFFERED=1 python3 -u ~/distributed_tpu_training/coordinator.py --monitor >> "$MONITOR_LOG" 2>&1 &
    log "New monitor PID: $!"
}

copy_results() {
    log "Copying validated results to exp folder..."
    mkdir -p "$EXP_DIR/results"
    cp "$RESULTS_DIR"/*.json "$EXP_DIR/results/"
    local count=$(ls "$EXP_DIR/results/"*.json 2>/dev/null | wc -l)
    log "DONE: $count results copied to $EXP_DIR/results/"
}

log "=== Babysitter started for $EXP (target: $TOTAL configs) ==="

while true; do
    # Count validated results
    validated=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)

    # Check monitor alive
    monitor_pid=$(pgrep -f "coordinator.py --monitor" || true)
    if [ -z "$monitor_pid" ]; then
        log "WARNING: Monitor not running!"
        if [ "$validated" -ge "$TOTAL" ]; then
            log "All $TOTAL configs validated. No restart needed."
        else
            restart_monitor
        fi
    fi

    # Get last monitor line for status
    last_line=$(tail -1 "$MONITOR_LOG" 2>/dev/null || echo "no log")

    log "Progress: $validated/$TOTAL validated | monitor_pid=${monitor_pid:-DEAD} | $last_line"

    # Check if done
    if [ "$validated" -ge "$TOTAL" ]; then
        log "ALL $TOTAL CONFIGS VALIDATED!"
        copy_results
        log "=== Babysitter complete. Results in $EXP_DIR/results/ ==="
        exit 0
    fi

    sleep 600  # 10 min
done
