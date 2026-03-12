#!/bin/bash
# overnight_watchdog.sh — Autonomous overnight babysitter for pull-based TPU experiments
# Watches: monitor.py, vm_requester.sh, task completion, failed tasks
#
# Usage:
#   EXPERIMENTS="exp14:200 exp15:300" nohup bash ~/distributed_tpu_training/pull/overnight_watchdog.sh \
#     >> /tmp/overnight_watchdog.log 2>&1 &
#
# Config via env vars:
#   EXPERIMENTS    — space-separated NAME:TOTAL pairs (required)
#   RESULTS_BASE   — local results directory (default: ~/sf_bema/results)
#   CHECK_INTERVAL — poll interval in seconds (default: 600)

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────

GCLOUD=~/google-cloud-sdk/bin/gcloud
CTRL=gs://gcp-researchcredits-blocklab-europe-west4/coord_v2
CHECK_INTERVAL=${CHECK_INTERVAL:-600}  # 10 min
MAX_RETRIES_REQUEUE=20  # only re-queue if retries < this
RESULTS_BASE=${RESULTS_BASE:-"$HOME/sf_bema/results"}

# Parse EXPERIMENTS env var into arrays
# Format: "exp14:200 exp15:300"
if [ -z "${EXPERIMENTS:-}" ]; then
  echo "ERROR: EXPERIMENTS env var required. Example: EXPERIMENTS=\"exp14:200 exp15:300\""
  exit 1
fi

declare -A EXP_TOTALS
EXP_NAMES=()
for spec in $EXPERIMENTS; do
  name="${spec%%:*}"
  total="${spec##*:}"
  EXP_NAMES+=("$name")
  EXP_TOTALS["$name"]="$total"
done

# ── Single-instance guard ─────────────────────────────────────────────────

mkdir -p "$HOME/.locks"
_WD_PID_FILE="$HOME/.locks/overnight_watchdog.pid"
if [ -f "$_WD_PID_FILE" ]; then
  _OLD_PID=$(cat "$_WD_PID_FILE" 2>/dev/null)
  if [ -n "$_OLD_PID" ] && kill -0 "$_OLD_PID" 2>/dev/null; then
    echo "overnight_watchdog already running (PID $_OLD_PID) — exiting"
    exit 0
  fi
fi
echo $$ > "$_WD_PID_FILE"
trap 'rm -f "$_WD_PID_FILE"' EXIT

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

# ── Process watchdog ──────────────────────────────────────────────────────

ensure_monitor() {
  local last_active_age=0
  if [ -f /tmp/monitor_last_active ]; then
    last_active_age=$(( $(date +%s) - $(cat /tmp/monitor_last_active | cut -d. -f1) ))
  fi
  if ! pgrep -f 'distributed_tpu_training/pull/monitor.py' >/dev/null 2>&1 || [ "$last_active_age" -gt 600 ]; then
    if [ "$last_active_age" -gt 600 ]; then
      log "WATCHDOG: monitor.py STALLED (last active ${last_active_age}s ago) — killing and restarting"
      pkill -f 'distributed_tpu_training/pull/monitor.py' 2>/dev/null || true
      sleep 2
    else
      log "WATCHDOG: monitor.py died! Restarting..."
    fi
    # Build --exp args from EXPERIMENTS
    local exp_args=""
    for spec in $EXPERIMENTS; do
      exp_args="$exp_args --exp $spec"
    done
    RESULTS_BASE="$RESULTS_BASE" python3 -u ~/distributed_tpu_training/pull/monitor.py \
      $exp_args --interval 60 --stale-ttl 1800 --results-base "$RESULTS_BASE" \
      >> /tmp/monitor_pull.log 2>&1 &
    log "WATCHDOG: monitor.py restarted (PID $!) tracking: $EXPERIMENTS"
  fi
}

ensure_vm_requester() {
  local pid_file="$HOME/.locks/vm_requester.pid"
  local pid=""
  [ -f "$pid_file" ] && pid=$(cat "$pid_file" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    return 0  # already running
  fi
  log "WATCHDOG: vm_requester.sh died! Restarting..."
  bash ~/distributed_tpu_training/pull/vm_requester.sh >> /tmp/vm_requester.log 2>&1 &
  log "WATCHDOG: vm_requester.sh restarted (PID $!)"
}

# ── Task management ──────────────────────────────────────────────────────

requeue_failed() {
  # Move failed tasks back to pending (if retries < MAX_RETRIES_REQUEUE)
  local failed_list=$($GCLOUD storage ls ${CTRL}/failed/ 2>/dev/null)
  [ -z "$failed_list" ] && return

  local count=0
  for path in $failed_list; do
    local task_id=$(basename "$path" .json)
    local raw=$($GCLOUD storage cat "$path" 2>/dev/null)
    [ -z "$raw" ] && continue

    local retries=$(echo "$raw" | python3 -c "import sys,json; print(json.load(sys.stdin).get('retries',0))" 2>/dev/null)
    if [ "${retries:-99}" -lt "$MAX_RETRIES_REQUEUE" ]; then
      # Reset and move to pending
      echo "$raw" | python3 -c "
import sys, json
d = json.load(sys.stdin)
d['retries'] = 0
d.pop('failed_at', None)
d.pop('worker_id', None)
d.pop('claimed_at', None)
print(json.dumps(d))
" | $GCLOUD storage cp - "${CTRL}/pending/${task_id}.json" 2>/dev/null
      $GCLOUD storage rm "$path" 2>/dev/null
      count=$((count + 1))
    fi
  done
  [ $count -gt 0 ] && log "Re-queued $count failed tasks"
}

# ── Dedup check ──────────────────────────────────────────────────────────

dedup_queue() {
  # Remove pending tasks that are already completed
  local completed=$($GCLOUD storage ls ${CTRL}/completed/ 2>/dev/null | xargs -I{} basename {} .json)
  [ -z "$completed" ] && return

  local deduped=0
  for tid in $completed; do
    if $GCLOUD storage ls "${CTRL}/pending/${tid}.json" >/dev/null 2>&1; then
      $GCLOUD storage rm "${CTRL}/pending/${tid}.json" 2>/dev/null
      deduped=$((deduped + 1))
    fi
  done
  [ $deduped -gt 0 ] && log "Deduped $deduped pending tasks (already completed)"
}

# ── Stall repair ─────────────────────────────────────────────────────────

trigger_repair() {
  log "STALL DETECTED (no active workers) — triggering repair..."

  # 1. Re-populate missing tasks (skips already completed/validated)
  for exp in "${EXP_NAMES[@]}"; do
    log "  Repopulating $exp..."
    EXP=$exp python3 ~/distributed_tpu_training/pull/populate.py 2>&1 | tail -5
  done

  # 2. Re-queue permanently failed tasks
  requeue_failed

  # NOTE: do NOT call reclaim_stale with a tight TTL here — that would kill
  # running tasks mid-training (tasks take 70+ min). Monitor.py handles
  # reclaim with the proper stale_ttl (--stale-ttl 1800).
}

# ── Reconciler invariant ──────────────────────────────────────────────────

reconcile_invariant() {
  # Assert pending + running + completed + failed + invalidated + validated == expected_total per exp.
  # If tasks are missing from all queues, repopulate.
  for exp in "${EXP_NAMES[@]}"; do
    local expected=${EXP_TOTALS[$exp]:-0}
    [ "$expected" -eq 0 ] && continue

    local pending=$(gsutil ls "${CTRL}/pending/${exp}__*.json" 2>/dev/null | wc -l)
    local running=$(gsutil ls "${CTRL}/running/${exp}__*.json" 2>/dev/null | wc -l)
    local completed=$(gsutil ls "${CTRL}/completed/${exp}__*.json" 2>/dev/null | wc -l)
    local failed=$(gsutil ls "${CTRL}/failed/${exp}__*.json" 2>/dev/null | wc -l)
    local invalidated=$(gsutil ls "${CTRL}/invalidated/${exp}__*.json" 2>/dev/null | wc -l)
    # Count locally validated results — these are not in GCS after monitor downloads them
    local validated=$(ls "${RESULTS_BASE}/${exp}/validated/"*.json 2>/dev/null | wc -l)
    local accounted=$((pending + running + completed + failed + invalidated + validated))

    log "RECONCILE $exp: pending=$pending running=$running completed=$completed failed=$failed invalidated=$invalidated validated=$validated accounted=$accounted expected=$expected"

    if [ "$accounted" -lt "$expected" ]; then
      local missing=$((expected - accounted))
      log "RECONCILE: $missing tasks missing from all queues! Auto-populating..."
      EXP=$exp python3 ~/distributed_tpu_training/pull/populate.py 2>&1 | tail -5
    fi
  done
}

# ── Progress report ──────────────────────────────────────────────────────

report_progress() {
  local pending=$($GCLOUD storage ls ${CTRL}/pending/ 2>/dev/null | wc -l)
  local running=$($GCLOUD storage ls ${CTRL}/running/ 2>/dev/null | wc -l)
  local completed=$($GCLOUD storage ls ${CTRL}/completed/ 2>/dev/null | wc -l)
  local failed=$($GCLOUD storage ls ${CTRL}/failed/ 2>/dev/null | wc -l)
  local invalidated=$($GCLOUD storage ls ${CTRL}/invalidated/ 2>/dev/null | wc -l)

  log "PROGRESS: pending=$pending running=$running completed=$completed failed=$failed invalidated=$invalidated"

  # Per-experiment validated counts
  local validated_summary=""
  for exp in "${EXP_NAMES[@]}"; do
    local total=${EXP_TOTALS[$exp]:-0}
    local val=$(ls "${RESULTS_BASE}/${exp}/validated/"*.json 2>/dev/null | wc -l)
    validated_summary="$validated_summary ${exp}=${val}/${total}"
  done
  log "VALIDATED:$validated_summary"

  # Stall detection: only trigger repair if NO active workers AND completions flat.
  # Tasks take 70+ min — normal to have no new completions for multiple cycles.
  local active_workers=0
  active_workers=$(python3 ~/distributed_tpu_training/pull/check_progress.py 2>/dev/null | \
    grep -oP 'training=\K\d+' | head -1 || echo 0)
  active_workers=${active_workers:-0}

  if [ -f /tmp/watchdog_last_completed ]; then
    local last=$(cat /tmp/watchdog_last_completed)
    if [ "$completed" = "$last" ] && [ "$pending" -gt 0 ] && [ "$active_workers" -eq 0 ]; then
      log "WARNING: No active workers AND no completions (pending=$pending). Triggering repair..."
      trigger_repair
    elif [ "$completed" = "$last" ] && [ "$pending" -gt 0 ]; then
      log "INFO: No new completions, but $active_workers workers active — normal, tasks take 70+ min"
    fi
  fi
  echo "$completed" > /tmp/watchdog_last_completed

  # Check per-experiment completion
  for exp in "${EXP_NAMES[@]}"; do
    local total=${EXP_TOTALS[$exp]:-0}
    local val=$(ls "${RESULTS_BASE}/${exp}/validated/"*.json 2>/dev/null | wc -l)
    if [ "$val" -ge "$total" ]; then
      log "=== ${exp^^} COMPLETE! ${val}/${total} validated ==="
    fi
  done
}

# ── Main loop ────────────────────────────────────────────────────────────

log "=== OVERNIGHT WATCHDOG STARTED ==="
log "Check interval: ${CHECK_INTERVAL}s"
log "Control plane: ${CTRL}"
log "Experiments: $EXPERIMENTS"
log "Results base: $RESULTS_BASE"

cycle=0
while true; do
  cycle=$((cycle + 1))
  log "--- Cycle $cycle ---"

  # 1. Ensure critical processes are alive
  ensure_monitor
  ensure_vm_requester

  # 2. Re-queue failed tasks
  requeue_failed

  # 3. Dedup + reconciler (every 3 cycles)
  if [ $((cycle % 3)) -eq 0 ]; then
    dedup_queue
    reconcile_invariant
  fi

  # 4. Report progress (includes stall detection + trigger_repair)
  report_progress

  sleep $CHECK_INTERVAL
done
