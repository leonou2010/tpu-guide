#!/bin/bash
# Overnight babysitter for exp12_1
# Checks every 10 min: monitor alive? results accumulating? VMs healthy?
# When done: copies results to exp folder.

set -euo pipefail

EXP=exp12_1
RESULTS_DIR=~/sf_bema/results/$EXP/validated
EXP_DIR=~/sf_bema/experiments/exp10_smollm2_smoltalk/exp12_1_tpu
TOTAL=185
LOG=/tmp/babysit_${EXP}.log
MONITOR_LOG=/tmp/monitor_${EXP}.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

restart_monitor() {
    log "RESTARTING coordinator monitor..."
    cd ~/sf_bema/experiments/exp10_smollm2_smoltalk
    nohup env EXP=$EXP PYTHONUNBUFFERED=1 python3 -u ~/tpu_guide/coordinator.py --monitor >> "$MONITOR_LOG" 2>&1 &
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
