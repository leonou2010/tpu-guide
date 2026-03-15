#!/bin/bash
# Persistent VM requester — keeps trying to create VMs in all zones
# Also: deletes PREEMPTED VMs, deploys babysitter to READY VMs without one
# Usage: nohup bash ~/distributed_tpu_training/pull/vm_requester.sh >> /tmp/vm_requester.log 2>&1 &

# Single-instance guard (lock first, then pidfile; avoids stale pidfile races)
mkdir -p "$HOME/.locks"
_VM_LOCK_FILE="$HOME/.locks/vm_requester.lock"
exec 9>"$_VM_LOCK_FILE"
if ! flock -n 9; then
  echo "vm_requester already running (lock busy: $_VM_LOCK_FILE) — exiting"
  exit 0
fi

_VM_PID_FILE="$HOME/.locks/vm_requester.pid"
echo $$ > "$_VM_PID_FILE"
trap 'rm -f "$_VM_PID_FILE"' EXIT

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
INTERVAL=${INTERVAL:-120}  # 2 min cycle
CTRL="gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"

# Zone configs: zone accelerator_type version max_vms name_prefix type
# type: primary = v6e (preferred), fallback = v4/v5e (only created when v6e is full/exhausted)
ZONES=(
  "europe-west4-a v6e-8 v2-alpha-tpuv6e 8 v6e-ew4a primary"
  "us-east1-d v6e-8 v2-alpha-tpuv6e 8 v6e-ue1d primary"
  "us-central2-b v4-8 tpu-ubuntu2204-base 8 v4-uc2b fallback"
  "europe-west4-b v5litepod-4 v2-alpha-tpuv5-lite 16 v5e-ew4b fallback"
  # us-central1-a removed: permission denied for v5litepod-4 (wastes 16 API calls/cycle)
)

# ── Helpers ──────────────────────────────────────────────────────────────────

get_bucket() {
  local zone=$1
  case "$zone" in
    europe-west4*) echo "gs://gcp-researchcredits-blocklab-europe-west4" ;;
    us-east1*)     echo "gs://gcp-researchcredits-blocklab-us-east1" ;;
    us-central2*)  echo "gs://gcp-researchcredits-blocklab-1-us-central2" ;;
    *)             echo "gs://gcp-researchcredits-blocklab-europe-west4" ;;
  esac
}

get_wandb() {
  # W&B API key not deployed to VMs — use disabled everywhere.
  # Results are saved to GCS (GCS_CHECKPOINT_DIR), W&B is non-critical for sweep.
  echo "disabled"
}

# Per-VM deploy cooldown — don't deploy more than once per 30 min
DEPLOY_COOLDOWN_S=900
DEPLOY_CACHE_DIR=/tmp/vm_deploy_cache
mkdir -p "$DEPLOY_CACHE_DIR"

deploy_cooldown_ok() {
  local name=$1
  local f="${DEPLOY_CACHE_DIR}/${name}"
  [ -f "$f" ] || { return 0; }  # never deployed — OK
  local age=$(( $(date +%s) - $(cat "$f") ))
  [ "$age" -ge "$DEPLOY_COOLDOWN_S" ]  # returns 0 (true) if cooldown elapsed
}

mark_deployed() {
  local name=$1
  date +%s > "${DEPLOY_CACHE_DIR}/${name}"
}

# Clean up GCS state for a deleted VM (heartbeats + telemetry)
cleanup_vm_gcs() {
  local name=$1
  gsutil -m rm -f \
    "${CTRL}/heartbeats/${name}_chip0.json" \
    "${CTRL}/heartbeats/${name}_chip1.json" \
    "${CTRL}/heartbeats/${name}_chip2.json" \
    "${CTRL}/heartbeats/${name}_chip3.json" \
    "${CTRL}/heartbeats/${name}_chip4.json" \
    "${CTRL}/heartbeats/${name}_chip5.json" \
    "${CTRL}/heartbeats/${name}_chip6.json" \
    "${CTRL}/heartbeats/${name}_chip7.json" \
    "${CTRL}/telemetry/${name}_boot.json" \
    "${CTRL}/telemetry/${name}_preempted.json" 2>/dev/null || true
}

# Delete a VM and clean up ALL associated resources:
#   1. Capture internal VM ID (needed for health check cleanup — names use numeric ID not friendly name)
#   2. Delete the VM
#   3. Delete orphaned health checks (GCP creates 5/VM, never auto-deletes them)
#   4. Clean up GCS heartbeats + telemetry
#
# GCP health check naming: tpu-{project_number}-{vm_internal_id}-{event_type}
# Without explicit cleanup these accumulate against HEALTH_CHECKS quota (default 75).
delete_vm() {
  local name=$1 zone=$2
  local region=${zone%-*}  # europe-west4-a → europe-west4

  # Step 1: capture internal VM ID BEFORE deletion
  local vm_id
  vm_id=$($GCLOUD alpha compute tpus tpu-vm describe "$name" --zone="$zone" \
    --project=$PROJECT --format="value(id)" 2>/dev/null || true)

  # Step 2: delete the VM
  $GCLOUD alpha compute tpus tpu-vm delete "$name" --zone="$zone" \
    --project=$PROJECT --quiet 2>&1 | tail -1

  # Step 3: delete orphaned health checks + health check services
  # GCP creates 5 health checks + 5 health check services per VM, never auto-deletes them.
  # Health check services must be deleted first (they reference the health checks).
  # gcloud SDK lacks 'health-check-services' subcommand — use REST API.
  if [ -n "$vm_id" ]; then
    local token
    token=$($GCLOUD auth print-access-token 2>/dev/null || true)
    if [ -n "$token" ]; then
      local base="https://compute.googleapis.com/compute/v1/projects/${PROJECT}/regions/${region}"
      # Delete health check services referencing this VM's health checks
      local svc_names
      svc_names=$(curl -s -H "Authorization: Bearer $token" \
        "${base}/healthCheckServices" 2>/dev/null | \
        python3 -c "
import sys,json
d=json.load(sys.stdin)
vid='${vm_id}'
for item in d.get('items',[]):
    hcs=item.get('healthChecks',[])
    if any(vid in hc for hc in hcs):
        print(item['name'])
" 2>/dev/null || true)
      if [ -n "$svc_names" ]; then
        echo "$svc_names" | while read -r svc; do
          curl -s -X DELETE -H "Authorization: Bearer $($GCLOUD auth print-access-token 2>/dev/null)" \
            "${base}/healthCheckServices/${svc}" >/dev/null 2>&1 || true
        done
        echo "  $name: deleted health check services for vm_id=$vm_id"
        sleep 3  # wait for services to be removed before deleting health checks
      fi
      # Delete the 5 health checks
      local hcs
      hcs=$($GCLOUD compute health-checks list \
        --project=$PROJECT --filter="name~${vm_id}" \
        --format="value(name)" 2>/dev/null || true)
      if [ -n "$hcs" ]; then
        local count; count=$(echo "$hcs" | grep -c .)
        echo "  $name: deleting $count health checks (vm_id=$vm_id)..."
        echo "$hcs" | while read -r hc; do
          $GCLOUD compute health-checks delete "$hc" \
            --region="$region" --project=$PROJECT --quiet 2>/dev/null || true
        done
      fi
    fi
  fi

  # Step 4: clean up GCS state
  cleanup_vm_gcs "$name"
}

# Per-VM deploy attempt counter — track consecutive failed deploy attempts
# After MAX_DEPLOY_ATTEMPTS with no heartbeat, delete the dead VM (frees health-check quota)
MAX_DEPLOY_ATTEMPTS=3
DEPLOY_ATTEMPT_DIR=/tmp/vm_deploy_attempts
mkdir -p "$DEPLOY_ATTEMPT_DIR"

# Per-zone internal-error counter — GCP "internal error" on create means no capacity.
# After MAX_INTERNAL_ERRORS consecutive failures, skip zone for INTERNAL_ERROR_BACKOFF_CYCLES cycles.
MAX_INTERNAL_ERRORS=3
INTERNAL_ERROR_BACKOFF_CYCLES=5
INTERNAL_ERROR_DIR=/tmp/vm_internal_error
mkdir -p "$INTERNAL_ERROR_DIR"

get_internal_errors() { local z=$1; local f="${INTERNAL_ERROR_DIR}/${z//\//_}"; [ -f "$f" ] && cat "$f" || echo "0 0"; }
set_internal_errors() { local z=$1 count=$2 since=$3; echo "$count $since" > "${INTERNAL_ERROR_DIR}/${z//\//_}"; }
clear_internal_errors() { local z=$1; rm -f "${INTERNAL_ERROR_DIR}/${z//\//_}"; }

zone_internal_error_blocked() {
  local z=$1; local now cycle
  now=$(date +%s); cycle=$(( now / INTERVAL ))
  read -r count since <<< "$(get_internal_errors "$z")"
  [ "${count:-0}" -lt "$MAX_INTERNAL_ERRORS" ] && return 1  # not blocked
  local cycles_elapsed=$(( cycle - ${since:-0} ))
  [ "$cycles_elapsed" -lt "$INTERNAL_ERROR_BACKOFF_CYCLES" ] && return 0  # blocked
  clear_internal_errors "$z"; return 1  # backoff expired
}

record_internal_error() {
  local z=$1; local cycle
  cycle=$(( $(date +%s) / INTERVAL ))
  read -r count _since <<< "$(get_internal_errors "$z")"
  count=$(( ${count:-0} + 1 ))
  set_internal_errors "$z" "$count" "$cycle"
  echo "$count"
}

get_deploy_attempts() {
  local name=$1
  local f="${DEPLOY_ATTEMPT_DIR}/${name}"
  [ -f "$f" ] && cat "$f" || echo 0
}

incr_deploy_attempts() {
  local name=$1
  local count
  count=$(get_deploy_attempts "$name")
  echo $((count + 1)) > "${DEPLOY_ATTEMPT_DIR}/${name}"
}

clear_deploy_attempts() {
  local name=$1
  rm -f "${DEPLOY_ATTEMPT_DIR}/${name}"
}

# Check HEALTH_CHECKS quota — returns "ok" or "blocked: <reason>"
# NOTE: --format='value(quotas[METRIC=X].limit)' returns empty (gcloud filter bug).
# Use JSON parse instead.
check_health_checks_quota() {
  local json result
  json=$($GCLOUD compute project-info describe --project=$PROJECT \
    --format='json(quotas)' 2>/dev/null)
  if [ -z "$json" ]; then
    echo "ok (quota API unavailable — fail open)"
    return
  fi
  result=$(echo "$json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for q in data.get('quotas', []):
        if q.get('metric') == 'HEALTH_CHECKS':
            limit = int(q['limit'])
            usage = int(q['usage'])
            remaining = limit - usage
            if remaining <= 5:
                print(f'blocked: HEALTH_CHECKS usage={usage}/{limit} remaining={remaining}')
            else:
                print(f'ok (HEALTH_CHECKS usage={usage}/{limit})')
            sys.exit(0)
    print('ok (HEALTH_CHECKS metric not found)')
except Exception as e:
    print(f'ok (parse error: {e})')
" 2>/dev/null || echo "ok (python error)")
  echo "$result"
}

# GCS hostname cache — map friendly VM name to internal GCP hostname
HOSTNAME_CACHE_DIR=/tmp/vm_hostname_cache
mkdir -p "$HOSTNAME_CACHE_DIR"

get_vm_hostname() {
  # Returns cached hostname for VM, or empty string if unknown
  local name=$1
  local f="${HOSTNAME_CACHE_DIR}/${name}"
  [ -f "$f" ] && cat "$f" || true
}

cache_vm_hostname() {
  local name=$1 zone=$2
  local hn
  if command -v timeout >/dev/null 2>&1; then
    hn=$(timeout 60s $GCLOUD alpha compute tpus tpu-vm ssh "$name" --zone="$zone" --project=$PROJECT \
      --tunnel-through-iap --command="hostname -s" 2>/dev/null | tr -d '[:space:]' || true)
  else
    hn=$($GCLOUD alpha compute tpus tpu-vm ssh "$name" --zone="$zone" --project=$PROJECT \
      --tunnel-through-iap --command="hostname -s" 2>/dev/null | tr -d '[:space:]' || true)
  fi
  [ -n "$hn" ] && echo "$hn" > "${HOSTNAME_CACHE_DIR}/${name}"
  echo "$hn"
}

# Check GCS heartbeat age for a VM (in seconds). Returns 9999 if no heartbeat.
# Checks by friendly name AND cached hostname (for VMs that fell back to internal names).
vm_heartbeat_age() {
  local name=$1
  local now best_age=9999
  now=$(date +%s)

  _scan_hb_pattern() {
    local pattern=$1
    local hb_paths
    hb_paths=$(gsutil ls "${CTRL}/heartbeats/${pattern}_chip*.json" 2>/dev/null || true)
    [ -z "$hb_paths" ] && return
    while IFS= read -r hb_path; do
      [ -z "$hb_path" ] && continue
      local ts
      ts=$(gsutil cat "$hb_path" 2>/dev/null | \
        python3 -c "import sys,json; print(int(json.load(sys.stdin).get('timestamp',0)))" 2>/dev/null || echo 0)
	      if [ "${ts:-0}" -gt 0 ]; then
	        local age=$(( now - ts ))
	        [ "$age" -lt 0 ] && age=0  # guard against clock skew
	        [ "$age" -lt "$best_age" ] && best_age=$age
	      fi
	    done <<< "$hb_paths"
	  }

  # Fast path: friendly name
  _scan_hb_pattern "$name"

  # Fallback: check cached hostname (internal GCP name)
  if [ "$best_age" -eq 9999 ]; then
    local hn
    hn=$(get_vm_hostname "$name")
    [ -n "$hn" ] && _scan_hb_pattern "$hn"
  fi

  echo $best_age
}

# Check the STATUS field from the freshest chip heartbeat for a VM.
# Returns: training|xla_compile|idle|uploading|unknown
vm_heartbeat_status() {
  local name=$1
  local best_status="unknown" best_age=9999
  local now
  now=$(date +%s)

  _scan_status_pattern() {
    local pattern=$1
    local hb_paths
    hb_paths=$(gsutil ls "${CTRL}/heartbeats/${pattern}_chip*.json" 2>/dev/null || true)
    [ -z "$hb_paths" ] && return
    while IFS= read -r hb_path; do
      [ -z "$hb_path" ] && continue
      local hb_json
      hb_json=$(gsutil cat "$hb_path" 2>/dev/null || true)
      [ -z "$hb_json" ] && continue
      local ts status
      ts=$(echo "$hb_json" | python3 -c "import sys,json; print(int(json.load(sys.stdin).get('timestamp',0)))" 2>/dev/null || echo 0)
      status=$(echo "$hb_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo unknown)
	      if [ "${ts:-0}" -gt 0 ]; then
	        local age=$(( now - ts ))
	        [ "$age" -lt 0 ] && age=0  # guard against clock skew
	        if [ "$age" -lt "$best_age" ]; then
	          best_age=$age
	          best_status=$status
	        fi
	      fi
    done <<< "$hb_paths"
  }

  _scan_status_pattern "$name"
  if [ "$best_status" = "unknown" ]; then
    local hn
    hn=$(get_vm_hostname "$name")
    [ -n "$hn" ] && _scan_status_pattern "$hn"
  fi

  echo "$best_status"
}

# Fetch best heartbeat info for a VM (from the freshest chip heartbeat).
# Prints: "<age_s> <status> <step>"
vm_heartbeat_best() {
  local name=$1
  local now best_age=9999 best_status="unknown" best_step=0
  now=$(date +%s)

  _scan_best_pattern() {
    local pattern=$1
    local hb_paths
    hb_paths=$(gsutil ls "${CTRL}/heartbeats/${pattern}_chip*.json" 2>/dev/null || true)
    [ -z "$hb_paths" ] && return
    while IFS= read -r hb_path; do
      [ -z "$hb_path" ] && continue
      local hb_json
      hb_json=$(gsutil cat "$hb_path" 2>/dev/null || true)
      [ -z "$hb_json" ] && continue
      local ts status step
      read -r ts status step <<<"$(echo "$hb_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(int(d.get('timestamp',0)), d.get('status','unknown'), int(d.get('step',0)))" 2>/dev/null || echo "0 unknown 0")"
      [ "${ts:-0}" -le 0 ] && continue
      local age=$(( now - ts ))
      [ "$age" -lt 0 ] && age=0
      if [ "$age" -lt "$best_age" ]; then
        best_age=$age
        best_status=$status
        best_step=$step
      fi
    done <<< "$hb_paths"
  }

  _scan_best_pattern "$name"
  if [ "$best_status" = "unknown" ]; then
    local hn
    hn=$(get_vm_hostname "$name")
    [ -n "$hn" ] && _scan_best_pattern "$hn"
  fi

  echo "$best_age $best_status $best_step"
}

# Status/step tracking for "stuck-but-heartbeating" detection (no VM changes required).
STATUS_TRACK_DIR=/tmp/vm_status_track
STEP_TRACK_DIR=/tmp/vm_step_track
mkdir -p "$STATUS_TRACK_DIR" "$STEP_TRACK_DIR"

status_duration_s() {
  local name=$1 status=$2 now=$3
  local f="${STATUS_TRACK_DIR}/${name}"
  local last_status="" seen_at=0
  if [ -f "$f" ]; then
    read -r last_status seen_at < "$f" 2>/dev/null || true
  fi
  if [ "$status" != "$last_status" ] || [ "${seen_at:-0}" -le 0 ]; then
    echo "$status $now" > "$f"
    echo 0
    return
  fi
  echo $(( now - seen_at ))
}

step_stale_s() {
  local name=$1 step=$2 now=$3
  local f="${STEP_TRACK_DIR}/${name}"
  local last_step=-1 changed_at=0
  if [ -f "$f" ]; then
    read -r last_step changed_at < "$f" 2>/dev/null || true
  fi
  if [ "$step" -ne "${last_step:--1}" ] || [ "${changed_at:-0}" -le 0 ]; then
    echo "$step $now" > "$f"
    echo 0
    return
  fi
  echo $(( now - changed_at ))
}

deploy_babysitter() {
  local name=$1 zone=$2 bucket=$3 wandb_mode=${4:-disabled} force_redeploy=${5:-0}
  echo "  Deploying babysitter to $name..."
  local env_prefix=""
  if [ "$force_redeploy" = "1" ]; then
    env_prefix="FORCE_REDEPLOY=1 "
    echo "  $name: FORCE_REDEPLOY=1 (override deploy guard)"
  fi
  # timeout 120s: prevents a stuck SSH from blocking the calling process indefinitely.
  # The deploy runs in background (&) so this timeout only kills the one stuck SSH,
  # not the rest of the cycle. The global `wait` after the deploy loop is bounded by
  # the worst-case SSH timeout (120s × concurrent deploys) rather than being unbounded.
  timeout 120s $GCLOUD alpha compute tpus tpu-vm ssh $name --zone=$zone --project=$PROJECT \
    --tunnel-through-iap --command="
      mkdir -p ~/pull_code
      gcloud storage cp '${bucket}/pull_code/*' ~/pull_code/ 2>/dev/null || \
      gsutil -m cp '${bucket}/pull_code/*' ~/pull_code/ 2>/dev/null
      chmod +x ~/pull_code/deploy_babysitter.sh
      # deploy_babysitter.sh handles its own health check and kill logic
      ${env_prefix}TPU_NAME=${name} ZONE=${zone} WANDB_MODE=${wandb_mode} bash ~/pull_code/deploy_babysitter.sh
    " 2>&1 | tail -5
  local rc=${PIPESTATUS[0]}
  if [ $rc -eq 124 ]; then
    echo "  $name: deploy SSH timed out after 120s — will retry next cycle"
  fi
}

# ── Orphan health check sweep ────────────────────────────────────────────────
# GCP auto-creates 5 health checks + 5 health-check services per TPU VM.
# They are NOT auto-deleted on VM deletion — they accumulate against quota (default 75).
# delete_vm() handles cleanup for VMs we explicitly delete; this sweep handles
# VMs deleted outside vm_requester (direct GCP preemption, manual gcloud delete).
#
# Run every HC_CLEANUP_INTERVAL cycles (~20 min). Safe: only deletes health checks
# whose VM ID does not appear in any live TPU VM across all watched zones.
HC_CLEANUP_INTERVAL=10
_hc_cleanup_counter=0

cleanup_orphan_health_checks() {
  echo "  [HC cleanup] Scanning for orphaned health checks..."
  local token
  token=$($GCLOUD auth print-access-token 2>/dev/null || true)
  [ -z "$token" ] && echo "  [HC cleanup] No auth token — skipping" && return

  # Collect all live VM IDs across all zones
  local live_ids=""
  for zoneconf in "${ZONES[@]}"; do
    read -r zone _rest <<< "$zoneconf"
    local ids
    ids=$($GCLOUD alpha compute tpus tpu-vm list --zone="$zone" --project=$PROJECT \
      --format="value(id)" 2>/dev/null || true)
    live_ids="$live_ids $ids"
  done

  local total_deleted=0
  for region in europe-west4 us-east1 us-central2; do
    local base="https://compute.googleapis.com/compute/v1/projects/${PROJECT}/regions/${region}"

    # List all health check names in this region
    local hc_names
    hc_names=$($GCLOUD compute health-checks list --project=$PROJECT \
      --filter="region:$region" --format="value(name)" 2>/dev/null || true)
    [ -z "$hc_names" ] && continue

    # Extract unique VM IDs that have no live VM
    local orphan_ids
    orphan_ids=$(echo "$hc_names" | \
      python3 -c "
import sys, re
live = set('${live_ids}'.split())
seen = set()
for name in sys.stdin.read().splitlines():
    m = re.search(r'tpu-\d+-(\d+)-', name)
    if m:
        vid = m.group(1)
        if vid not in live and vid not in seen:
            seen.add(vid)
            print(vid)
" 2>/dev/null || true)

    [ -z "$orphan_ids" ] && echo "  [HC cleanup] $region: no orphans" && continue

    local orphan_count; orphan_count=$(echo "$orphan_ids" | grep -c .)
    echo "  [HC cleanup] $region: found $orphan_count orphan VM IDs — cleaning up..."

    # Refresh token per region (long sweep can exceed 1h token lifetime)
    token=$($GCLOUD auth print-access-token 2>/dev/null || true)

    while IFS= read -r vm_id; do
      [ -z "$vm_id" ] && continue

      # Step 1: delete health-check services referencing this VM
      local svc_names
      svc_names=$(curl -s -H "Authorization: Bearer $token" \
        "${base}/healthCheckServices" 2>/dev/null | \
        python3 -c "
import sys,json
d=json.load(sys.stdin)
vid='${vm_id}'
for item in d.get('items',[]):
    if any(vid in hc for hc in item.get('healthChecks',[])):
        print(item['name'])
" 2>/dev/null || true)
      if [ -n "$svc_names" ]; then
        local svc_count; svc_count=$(echo "$svc_names" | grep -c .)
        echo "$svc_names" | while IFS= read -r svc; do
          curl -s -X DELETE -H "Authorization: Bearer $token" \
            "${base}/healthCheckServices/${svc}" >/dev/null 2>&1 || true
        done
        echo "  [HC cleanup] $region: deleted $svc_count services for vm=$vm_id"
        sleep 2  # let services clear before deleting checks
      fi

      # Step 2: delete the health checks themselves
      local hcs
      hcs=$($GCLOUD compute health-checks list \
        --project=$PROJECT --filter="name~${vm_id} AND region:${region}" \
        --format="value(name)" 2>/dev/null || true)
      if [ -n "$hcs" ]; then
        local hc_count; hc_count=$(echo "$hcs" | grep -c .)
        echo "$hcs" | while IFS= read -r hc; do
          $GCLOUD compute health-checks delete "$hc" \
            --region="$region" --project=$PROJECT --quiet 2>/dev/null || true
        done
        echo "  [HC cleanup] $region: deleted $hc_count health checks for vm=$vm_id"
        total_deleted=$((total_deleted + hc_count))
      fi
    done <<< "$orphan_ids"
  done

  if [ "$total_deleted" -gt 0 ]; then
    echo "  [HC cleanup] Done — deleted $total_deleted orphan health checks"
  else
    echo "  [HC cleanup] Done — no orphans found"
  fi
}

# ── CREATING grace tracking ───────────────────────────────────────────────────
# Per-VM: record when we first saw it CREATING (or tried to create it).
# If < CREATING_GRACE_S, don't attempt to describe/create again (avoids API spam).
CREATING_GRACE_S=1200   # 20 min grace for VM creation
CREATING_CACHE_DIR=/tmp/vm_creating_cache
mkdir -p "$CREATING_CACHE_DIR"

mark_creating() {
  local name=$1
  local f="${CREATING_CACHE_DIR}/${name}"
  [ -f "$f" ] || date +%s > "$f"  # only set once (first seen)
}

creating_age_s() {
  local name=$1
  local f="${CREATING_CACHE_DIR}/${name}"
  [ -f "$f" ] || { echo 9999; return; }
  echo $(( $(date +%s) - $(cat "$f") ))
}

clear_creating() {
  local name=$1
  rm -f "${CREATING_CACHE_DIR}/${name}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

while true; do
  echo "[$(date -u '+%H:%M:%S')] === VM Request Cycle ==="

  # Periodic orphan health check sweep (every HC_CLEANUP_INTERVAL cycles)
  _hc_cleanup_counter=$(( _hc_cleanup_counter + 1 ))
  if [ "$(( _hc_cleanup_counter % HC_CLEANUP_INTERVAL ))" -eq 0 ]; then
    cleanup_orphan_health_checks
  fi

  # ETA estimate from queue state
  _pending=$(gsutil ls "${CTRL}/pending/" 2>/dev/null | wc -l)
  _running=$(gsutil ls "${CTRL}/running/" 2>/dev/null | wc -l)
  _completed=$(gsutil ls "${CTRL}/completed/" 2>/dev/null | wc -l)
  _total=$(( _pending + _running + _completed ))
  _remaining=$(( _pending + _running ))
  if [ "$_completed" -gt 0 ] && [ "$_remaining" -gt 0 ]; then
    # Rate from ETA state file
    _eta_file="/tmp/vm_requester_eta.json"
    _now_ts=$(date +%s)
    _prev_completed=0; _prev_ts=$_now_ts
    [ -f "$_eta_file" ] && read -r _prev_completed _prev_ts < "$_eta_file"
    _delta_tasks=$(( _completed - _prev_completed ))
    _delta_secs=$(( _now_ts - _prev_ts ))
    if [ "$_delta_secs" -gt 60 ] && [ "$_delta_tasks" -gt 0 ]; then
      _rate_per_hr=$(echo "scale=1; $_delta_tasks * 3600 / $_delta_secs" | bc 2>/dev/null || echo "?")
      _eta_mins=$(echo "scale=0; $_remaining * $_delta_secs / $_delta_tasks / 60" | bc 2>/dev/null || echo "?")
      echo "  ETA: ~${_eta_mins}min | rate: ${_rate_per_hr} tasks/hr | ${_completed}/${_total} done, ${_remaining} remaining"
      echo "$_completed $_now_ts" > "$_eta_file"
    else
      echo "  Progress: ${_completed}/${_total} done, ${_remaining} remaining (ETA available after first completions)"
      [ ! -f "$_eta_file" ] && echo "$_completed $_now_ts" > "$_eta_file"
    fi
  else
    echo "  Progress: ${_completed}/${_total:-?} done, ${_remaining} remaining"
  fi

  # Check HEALTH_CHECKS quota once per cycle — skip all creates if near limit
  HC_STATUS=$(check_health_checks_quota)
  if [[ "$HC_STATUS" == blocked* ]]; then
    echo "  QUOTA GUARD: $HC_STATUS — VM creates disabled this cycle"
    HC_BLOCKED=true
  else
    echo "  $HC_STATUS"
    HC_BLOCKED=false
  fi

  # Track v6e capacity across all primary zones this cycle.
  # Fallback (v4/v5e) VM creation is gated on v6e being full or resource-exhausted.
  v6e_has_capacity=false  # true = at least one primary zone can still take more VMs

  for zoneconf in "${ZONES[@]}"; do
    read -r zone accel version max_vms prefix vm_type <<< "$zoneconf"
    BUCKET=$(get_bucket "$zone")
    WANDB=$(get_wandb "$zone")

    # List VMs with state
    vm_list=$($GCLOUD alpha compute tpus tpu-vm list --zone=$zone --project=$PROJECT \
      --format='csv[no-heading](name,state)' 2>/dev/null)

    ready_count=0
    creating_count=0

    # Phase 1: Handle PREEMPTED and count states
    while IFS=',' read -r name state; do
      [ -z "$name" ] && continue
      case "$state" in
        PREEMPTED)
          echo "  $zone: $name is PREEMPTED — deleting..."
          delete_vm "$name" "$zone"
          clear_creating "$name"
          ;;
        READY)
          ready_count=$((ready_count + 1))
          clear_creating "$name"  # no longer creating
          ;;
        CREATING)
          creating_count=$((creating_count + 1))
          mark_creating "$name"
          age=$(creating_age_s "$name")
          if [ "$age" -gt "$CREATING_GRACE_S" ]; then
            echo "  $zone: $name stuck CREATING for ${age}s (>20m) — deleting and recreating"
            delete_vm "$name" "$zone"
            clear_creating "$name"
            creating_count=$((creating_count - 1))
          fi
          ;;
      esac
    done <<< "$vm_list"

    total=$((ready_count + creating_count))
    echo "  $zone: ${ready_count} ready, ${creating_count} creating (${total}/${max_vms})"

    # Phase 2: Deploy babysitter to READY VMs with stale/missing heartbeats.
    # Dead VM detection: if no heartbeat after MAX_DEPLOY_ATTEMPTS, delete the VM.
    # Use GCS heartbeat as truth source (survives local /tmp resets).
	    while IFS=',' read -r name state; do
	      [ -z "$name" ] && continue
	      [ "$state" != "READY" ] && continue
	      _now=$(date +%s)
	      read -r hb_age hb_status hb_step <<<"$(vm_heartbeat_best "$name")"
	      force_deploy=0

	      # Stuck-but-heartbeating detection:
	      # The old check used hb_age, but heartbeats refresh frequently so hb_age stays small.
	      # Track status duration and step progress locally across cycles instead.
	      XLA_STUCK_THRESHOLD=1200      # 20 min (compile/cache warm can be slow)
	      NO_PROGRESS_THRESHOLD=1800    # 30 min (avoid false positives if logs are sparse)
	      status_dur=$(status_duration_s "$name" "$hb_status" "$_now")
	      step_stale=$(step_stale_s "$name" "${hb_step:-0}" "$_now")
	      if [ "$hb_status" = "xla_compile" ] && [ "$hb_age" -lt 2700 ] && [ "$status_dur" -gt "$XLA_STUCK_THRESHOLD" ]; then
	        echo "  $name: status=xla_compile for ${status_dur}s (> ${XLA_STUCK_THRESHOLD}s) — treating as stale (force redeploy)"
	        hb_age=9999  # Force redeploy path
	        force_deploy=1
	      elif [ "$hb_status" = "training" ] && [ "$hb_age" -lt 2700 ] && [ "$step_stale" -gt "$NO_PROGRESS_THRESHOLD" ]; then
	        echo "  $name: status=training but step=${hb_step} unchanged for ${step_stale}s (> ${NO_PROGRESS_THRESHOLD}s) — treating as stale (force redeploy)"
	        hb_age=9999  # Force redeploy path
	        force_deploy=1
	      fi
	      if [ "$hb_age" -gt 2700 ]; then
	        attempts=$(get_deploy_attempts "$name")
	        # Check boot telemetry for FAILED_ENV — log clearly so it's obvious in output
	        _boot_phase=$(gsutil cat "${CTRL}/telemetry/${name}_boot.json" 2>/dev/null | \
          python3 -c "import sys,json; print(json.load(sys.stdin).get('phase',''))" 2>/dev/null || true)
        if [[ "$_boot_phase" == FAILED_ENV* ]]; then
          echo "  $name: FAILED_ENV in boot telemetry — env broken, MAX_DEPLOY_ATTEMPTS=$MAX_DEPLOY_ATTEMPTS applies"
        fi
        if [ "$attempts" -ge "$MAX_DEPLOY_ATTEMPTS" ]; then
          # VM has been deployed to N times with no heartbeat — it's dead (env_fail loop, broken VM)
          echo "  $name: NO heartbeat after $attempts deploy attempts — deleting dead VM (frees HC quota)"
          delete_vm "$name" "$zone"
          clear_deploy_attempts "$name"
          clear_creating "$name"
          # Remove from deploy cooldown cache so fresh VM creation is allowed
          rm -f "${DEPLOY_CACHE_DIR}/${name}"
          ready_count=$((ready_count - 1))
        elif deploy_cooldown_ok "$name"; then
          echo "  $name: heartbeat age=${hb_age}s (>2700s) — deploying babysitter (attempt $((attempts+1))/$MAX_DEPLOY_ATTEMPTS)"
          mark_deployed "$name"
          incr_deploy_attempts "$name"
          # Cache hostname before deploy so next cycle can check hostname-based heartbeats
          [ -z "$(get_vm_hostname "$name")" ] && cache_vm_hostname "$name" "$zone" &
	          deploy_babysitter "$name" "$zone" "$BUCKET" "$WANDB" "$force_deploy" &
	        else
	          echo "  $name: heartbeat age=${hb_age}s but deploy cooldown active — skipping (attempt ${attempts}/$MAX_DEPLOY_ATTEMPTS)"
	        fi
      else
        # VM is healthy and training — reset deploy attempt counter
        clear_deploy_attempts "$name"
        echo "  $name: heartbeat age=${hb_age}s status=$hb_status — healthy, skipping deploy"
      fi
    done <<< "$vm_list"
    # DON'T wait here — let deploys run concurrently while we create VMs

    # Phase 3: Create new VMs if under quota (up to 3 per zone per cycle)
    # Fallback zones (v4/v5e) only create VMs when all primary (v6e) zones are full/exhausted.
    # Skip if HEALTH_CHECKS quota is near limit
    if [ "$vm_type" = "fallback" ] && $v6e_has_capacity; then
      echo "  $zone: SKIPPING creation — v6e has capacity (prioritising v6e)"
    elif zone_internal_error_blocked "$zone"; then
      echo "  $zone: SKIPPING creation — repeated internal errors (backoff active)"
    elif [ "$total" -lt "$max_vms" ] && ! $HC_BLOCKED; then
      # Track v6e capacity for this cycle
      [ "$vm_type" = "primary" ] && v6e_has_capacity=true
      created=0
      zone_exhausted=false
      for i in $(seq 1 $max_vms); do
        [ "$created" -ge 3 ] && break  # max 3 new VMs per zone per cycle
        $zone_exhausted && break

        name="${prefix}-${i}"

        # Skip if this VM is in CREATING grace window (don't race describe API)
        cage=$(creating_age_s "$name")
        if [ "$cage" -lt "$CREATING_GRACE_S" ]; then
          echo "  $zone: $name still in CREATING grace (${cage}s) — skipping"
          continue
        fi

        if ! $GCLOUD alpha compute tpus tpu-vm describe $name --zone=$zone --project=$PROJECT &>/dev/null; then
          echo "  $zone: trying to create $name..."
          # FORCE_REDEPLOY=1: on fresh boot, always deploy (stale heartbeats from preempted VM must be ignored)
          # Note: startup scripts run as root with HOME unset — use HOME=/root explicitly.
          # Use bash -c with semicolons (not &&) for compatibility with gcloud metadata parsing.
          STARTUP_META="startup-script=#!/bin/bash
export HOME=/root
gsutil cp ${BUCKET}/pull_code/deploy_babysitter.sh /tmp/deploy_startup.sh
TPU_NAME=${name} ZONE=${zone} WANDB_MODE=${WANDB} FORCE_REDEPLOY=1 bash /tmp/deploy_startup.sh"
          create_out=$($GCLOUD alpha compute tpus tpu-vm create $name \
            --zone=$zone --project=$PROJECT \
            --accelerator-type=$accel --version=$version --spot \
            --internal-ips \
            --metadata="$STARTUP_META" 2>&1)
          create_rc=$?
          echo "$create_out" | tail -2

          if echo "$create_out" | grep -qi 'ALREADY_EXISTS'; then
            echo "  $name: ALREADY_EXISTS — skipping increment"
            mark_creating "$name"
            clear_internal_errors "$zone"
          elif echo "$create_out" | grep -qi 'RESOURCE_EXHAUSTED\|quota\|insufficient'; then
            echo "  $zone: RESOURCE_EXHAUSTED — backing off zone for this cycle"
            zone_exhausted=true
          elif echo "$create_out" | grep -qi 'internal error\|code.*13'; then
            _err_count=$(record_internal_error "$zone")
            echo "  $zone: internal error on $name (${_err_count}/${MAX_INTERNAL_ERRORS}) — GCP capacity issue"
            if [ "$_err_count" -ge "$MAX_INTERNAL_ERRORS" ]; then
              echo "  $zone: ${MAX_INTERNAL_ERRORS}+ internal errors — backing off for ${INTERNAL_ERROR_BACKOFF_CYCLES} cycles"
              zone_exhausted=true
            fi
          elif [ $create_rc -eq 0 ]; then
            echo "  CREATED $name!"
            clear_internal_errors "$zone"
            mark_creating "$name"
            created=$((created + 1))
            sleep 15  # brief pause before next creation
          fi
        fi
      done
      # Wait for background deploys before next zone
      wait
    fi
  done

  echo "[$(date -u '+%H:%M:%S')] Sleeping ${INTERVAL}s..."
  sleep $INTERVAL
done
