#!/bin/bash
# monitor.sh — Generic TPU sweep dashboard (parameterized by EXP)
# Design: all SSH + GCS queries run in parallel, static info prints immediately
# Updated for centralized push coordinator (reads state.json + per-VM buckets)
#
# Usage: EXP=exp12c bash ~/tpu_guide/monitor.sh

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020

# ── Validate ────────────────────────────────────────────────────────────────

EXP=${EXP:?'EXP env var required (e.g. EXP=exp12c)'}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_CONFIG="$SCRIPT_DIR/experiments/${EXP}.env"
if [ ! -f "$EXP_CONFIG" ]; then
  echo "ERROR: experiment config not found: $EXP_CONFIG" >&2
  exit 1
fi
source "$EXP_CONFIG"

# ── KNOWN DATA (from experiment config + actively managed) ──

TOTAL_CONFIGS=${TOTAL_CONFIGS:-0}
STEPS_PER_CONFIG=${STEPS_PER_CONFIG:-1778}
HBM_PER_PROC=10.8     # GB, measured
XLA_OVERHEAD=5         # GB, measured
CKPT_SIZE=2.6          # GB, calculated

# Get total configs from experiment module if not set
if [ "$TOTAL_CONFIGS" = "0" ]; then
  TOTAL_CONFIGS=$(cd ~/sf_bema/experiments/$WORK_DIR 2>/dev/null && EXP=$EXP python3 -c "
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~/sf_bema/experiments'))
m = __import__('${EXP_MODULE}', fromlist=['build_configs'])
print(len(m.build_configs()))
" 2>/dev/null || echo 0)
fi

# Dynamic VM list from vm_configs/
VMS=()
for cfg_file in "$SCRIPT_DIR"/vm_configs/*.env; do
  [ -f "$cfg_file" ] || continue
  eval "$(grep -E '^(TPU_NAME|ZONE|ACCELERATOR_TYPE|TPU_NUM_WORKERS|PROCS_PER_HOST|BUCKET|WANDB_MODE)=' "$cfg_file")"
  case "$ACCELERATOR_TYPE" in
    v6e-*) STEP_S=4.9; HBM_CHIP=32 ;;
    v4-*)  STEP_S=8.4; HBM_CHIP=32 ;;
    v5*)   STEP_S=0;   HBM_CHIP=16; PROCS_PER_HOST=0 ;;  # OOM
    *)     STEP_S=5.0; HBM_CHIP=32 ;;
  esac
  VMS+=("$TPU_NAME $ZONE $ACCELERATOR_TYPE $TPU_NUM_WORKERS $PROCS_PER_HOST $BUCKET $STEP_S $HBM_CHIP $WANDB_MODE")
done

B='\033[1m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; D='\033[90m'; C='\033[36m'; N='\033[0m'

echo -e "${B}══════════════════════════════════════════════════════════════════════════════════════════════════════${N}"
echo -e "${B}  ${EXP_NAME} SWEEP  │  $(date '+%Y-%m-%d %H:%M:%S')  │  ${TOTAL_CONFIGS} configs × ${STEPS_PER_CONFIG} steps${N}"
echo -e "${B}══════════════════════════════════════════════════════════════════════════════════════════════════════${N}"
echo ""

tmpdir=$(mktemp -d)

# ── Phase 0: Read local state.json + validated count ────────────────────────

STATE_FILE="$HOME/sf_bema/results/${EXP_NAME}/state.json"
VALIDATED_DIR="$HOME/sf_bema/results/${EXP_NAME}/validated"
VALIDATED_COUNT=0
if [ -d "$VALIDATED_DIR" ]; then
  VALIDATED_COUNT=$(ls "$VALIDATED_DIR"/*.json 2>/dev/null | wc -l)
fi

# Read state.json for assignment info
STATE_ASSIGNMENTS=""
STATE_FAILED=0
if [ -f "$STATE_FILE" ]; then
  STATE_ASSIGNMENTS=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
a = s.get('assignments', {})
for vm, labels in sorted(a.items()):
    print(f'{vm}:{len(labels)}')
" 2>/dev/null)
  STATE_FAILED=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(len(s.get('failed', [])))
" 2>/dev/null || echo 0)
fi

# ── Phase 0b: GCS done receipts from each VM's bucket (parallel) ───────────

for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"
  [ "$PPH" = "0" ] && continue
  COORD_PREFIX="$BUCKET/coord/$EXP_NAME"
  (
    d=$(${GCLOUD} storage ls "$COORD_PREFIX/done/" 2>/dev/null | wc -l)
    a=0
    raw=$(${GCLOUD} storage cat "$COORD_PREFIX/assignments/${TPU}.json" 2>/dev/null)
    if [ -n "$raw" ]; then
      a=$(echo "$raw" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
    fi
    echo "$d $a" > "$tmpdir/gcs_${TPU}"
  ) &
done

# ── Phase 1: Launch ALL SSH queries in parallel ─────────────────────────────

idx=0
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"
  [ "$PPH" = "0" ] && { idx=$((idx+1)); continue; }
  ( timeout 45 $GCLOUD alpha compute tpus tpu-vm ssh "$TPU" \
      --zone="$ZONE" --project=$PROJECT --tunnel-through-iap \
      --worker=0 --command="
        echo PROCS_START
        tmux list-sessions 2>/dev/null | grep -c ${EXP_NAME} || echo 0
        echo PROCS_END
        echo LOGSIZE_START
        wc -l /tmp/${EXP_NAME}_0.log 2>/dev/null | cut -d' ' -f1 || echo 0
        echo LOGSIZE_END
        echo LOG_START
        tail -20 /tmp/${EXP_NAME}.log 2>/dev/null
        for f in /tmp/${EXP_NAME}_*.log; do [ -f \"\$f\" ] && tail -10 \"\$f\" 2>/dev/null; done
        echo LOG_END
        echo WANDB_START
        grep -oP 'https://wandb.ai/[^ ]+' /tmp/${EXP_NAME}_0.log 2>/dev/null | head -1
        echo WANDB_END
      " > "$tmpdir/ssh_${idx}" 2>/dev/null ) &
  idx=$((idx+1))
done

# ── Phase 2: Print static info immediately while queries run ─────────────

idx=0
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"

  if [ "$PPH" = "0" ]; then
    echo -e "  ${B}${TPU}${N} (${ACCEL}) — ${R}OOM${N}: ${HBM_CHIP}GB chip, need ${HBM_PER_PROC}+${XLA_OVERHEAD}GB"
    echo ""
    idx=$((idx+1))
    continue
  fi

  echo -e "  ${B}${TPU}${N} (${ACCEL}, ${ZONE}) — ${D}fetching live data...${N}"

  idx=$((idx+1))
done

# ── Phase 3: Wait for all background queries ────────────────────────────────

wait

# ── Phase 4: Print coordinator state + live data ────────────────────────────

echo ""
echo -e "${B}── COORDINATOR STATUS ──${N}"
echo -e "  Validated (local): ${G}${VALIDATED_COUNT}${N}/${TOTAL_CONFIGS}  │  Failed: ${R}${STATE_FAILED}${N}"
if [ -n "$STATE_ASSIGNMENTS" ]; then
  echo -e "  Initial assignments:"
  echo "$STATE_ASSIGNMENTS" | while IFS=: read vm count; do
    echo -e "    ${vm}: ${count} configs"
  done
fi

# Per-VM GCS stats
echo ""
echo -e "  GCS done receipts:"
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"
  [ "$PPH" = "0" ] && continue
  if [ -f "$tmpdir/gcs_${TPU}" ]; then
    read -r GCS_DONE GCS_ASSIGNED < "$tmpdir/gcs_${TPU}"
    GCS_DONE=${GCS_DONE:-0}; GCS_ASSIGNED=${GCS_ASSIGNED:-0}
    echo -e "    ${TPU}: done=${G}${GCS_DONE}${N}  assigned=${GCS_ASSIGNED}"
  fi
done

echo ""
echo -e "${B}── LIVE STATUS ──${N}"
echo ""

# Accumulators for aggregate stats
agg_total_procs=0
agg_total_tput="0"
agg_active_vms=0

idx=0
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"
  [ "$PPH" = "0" ] && { idx=$((idx+1)); continue; }

  procs=$((HOSTS * PPH))
  cfg_h=$(echo "scale=2; $STEPS_PER_CONFIG * $STEP_S / 3600" | bc)
  tput=$(echo "scale=2; $procs / $cfg_h" | bc)

  combined=$(cat "$tmpdir/ssh_${idx}" 2>/dev/null)
  log=$(echo "$combined" | sed -n '/LOG_START/,/LOG_END/p' | grep -v 'LOG_')
  n_procs=$(echo "$combined" | sed -n '/PROCS_START/,/PROCS_END/p' | grep -v 'PROCS_' | head -1)
  n_procs=${n_procs:-0}
  log_size=$(echo "$combined" | sed -n '/LOGSIZE_START/,/LOGSIZE_END/p' | grep -v 'LOGSIZE_' | head -1)
  log_size=${log_size:-0}
  wb_url=$(echo "$combined" | sed -n '/WANDB_START/,/WANDB_END/p' | grep -v 'WANDB_' | head -1)

  # Config progress
  cur_config=0
  if echo "$log" | grep -q "\[sweep\] === Starting\|\[sweep\] Completed\|\[sweep\] Claimed"; then
    cur_config=$(echo "$log" | grep -oP '(Starting:|Completed:|Claimed:) \K[^ ]+' | tail -1)
    if ! [[ "$cur_config" =~ ^[0-9]+$ ]]; then
      cur_config=0
    fi
  fi
  done_in_log=$(echo "$log" | grep -oP 'done=\K[0-9]+' | tail -1)
  done_in_log=${done_in_log:-0}

  # Step + loss + time-left
  cur_step="—"; train_loss="—"; best_val="—"; time_left="—"
  if echo "$log" | grep "step " | grep -v "Steps/" | grep -q "/"; then
    line=$(echo "$log" | grep "step " | grep -v "Steps/" | tail -1 | sed 's/^[[:space:]]*//')
    cur_step=$(echo "$line" | grep -oP 'step \K[0-9]+')
    t=$(echo "$line" | grep -oP 'train=\K[0-9.]+'); [ -n "$t" ] && train_loss="$t"
    b=$(echo "$line" | grep -oP 'best=\K[0-9.]+'); [ -n "$b" ] && [ "$b" != "0.0000" ] && best_val="$b"
    tl=$(echo "$line" | grep -oP '~\K[0-9.]+h'); [ -n "$tl" ] && time_left="${tl}"
  fi

  # Live step time
  live_step_s=""; live_cfg_h=""; live_tput=""
  if [ "$cur_step" != "—" ] && [ "$time_left" != "—" ]; then
    remaining_steps=$((STEPS_PER_CONFIG - cur_step))
    if [ "$remaining_steps" -gt 0 ]; then
      tl_num=$(echo "$time_left" | sed 's/h//')
      live_step_s=$(echo "scale=1; $tl_num * 3600 / $remaining_steps" | bc)
      live_cfg_h=$(echo "scale=1; $STEPS_PER_CONFIG * $live_step_s / 3600" | bc)
      live_tput=$(echo "scale=1; $procs / $live_cfg_h" | bc)
    fi
  fi

  display_step_s="${live_step_s:-$STEP_S}s"
  display_cfg_h="${live_cfg_h:-$cfg_h}h"
  display_tput="${live_tput:-$tput}"
  actual_tput="${live_tput:-$tput}"

  # Per-VM solo ETA
  solo_eta="—"
  if [ "$TOTAL_CONFIGS" -gt 0 ] && [ "$(echo "$actual_tput > 0" | bc)" = "1" ]; then
    solo_h=$(echo "scale=1; $TOTAL_CONFIGS / $actual_tput" | bc)
    solo_d=$(echo "scale=1; $solo_h / 24" | bc)
    solo_eta="${solo_h}h (~${solo_d}d)"
  fi

  # Status
  status=""
  is_active=0
  if [ -z "$combined" ]; then
    status="${R}OFFLINE${N}"
  elif [ "$n_procs" = "0" ]; then
    if echo "$combined" | grep -q "${EXP_NAME}"; then n_procs=1
    else status="${D}IDLE${N}"; fi
  fi
  if [ -z "$status" ]; then
    # XLA compile: procs running but log is empty (new session, not stale log)
    if [ "$n_procs" -gt 0 ] 2>/dev/null && [ "$log_size" = "0" ]; then
      status="${Y}XLA COMPILE${N}"; is_active=1
    elif echo "$log" | grep -q "SWEEP_DONE\|auto_sweep_done\|Worker.*finished\|Assignment file gone"; then status="${G}DONE${N}"
    elif echo "$log" | grep -q "Traceback\|Error\|FAILED\|OOM"; then status="${R}ERROR${N}"
    elif echo "$log" | grep -q "compiling\|Compiling"; then status="${Y}COMPILE${N}"; is_active=1
    elif echo "$log" | grep -q "\[sweep\]\|step "; then status="${G}RUNNING${N}"; is_active=1
    elif echo "$log" | grep -q "PREFLIGHT\|preflight"; then status="${Y}PREFLIGHT${N}"; is_active=1
    elif echo "$log" | grep -q "Waiting for rebalanced"; then status="${Y}WAITING${N}"; is_active=1
    else status="${Y}STARTING${N}"; is_active=1; fi
  fi

  # Accumulate for aggregate (only active VMs)
  if [ "$is_active" = "1" ]; then
    agg_total_procs=$((agg_total_procs + procs))
    agg_total_tput=$(echo "scale=2; $agg_total_tput + $actual_tput" | bc)
    agg_active_vms=$((agg_active_vms + 1))
  fi

  # GCS assignment/done for this VM
  gcs_info=""
  if [ -f "$tmpdir/gcs_${TPU}" ]; then
    read -r GCS_DONE GCS_ASSIGNED < "$tmpdir/gcs_${TPU}"
    gcs_info="assigned=${GCS_ASSIGNED:-0} done=${GCS_DONE:-0}"
  fi

  # Print per-VM block
  echo -ne "  ${B}${TPU}${N}: "; echo -e "${status}"
  printf "    Parallel: ${C}%d hosts × %d procs/host = %d procs${N}\n" "$HOSTS" "$PPH" "$procs"
  printf "    Memory:   %dGB/chip, %sGB usable, %sGB/proc, buffer=%.1fGB, ckpt=%sGB/proc\n" \
    "$HBM_CHIP" "$(echo "$HBM_CHIP - $XLA_OVERHEAD" | bc)" "$HBM_PER_PROC" \
    "$(echo "$HBM_CHIP - $XLA_OVERHEAD - $HBM_PER_PROC" | bc)" "$CKPT_SIZE"
  printf "    Timing:   step=%s  config=%s  throughput=%s cfg/h\n" "$display_step_s" "$display_cfg_h" "$display_tput"
  printf "    Solo ETA: %s (if only this VM)  │  GCS: %s\n" "$solo_eta" "$gcs_info"
  printf "    Procs: w0=%d visible  │  Done (this VM): %s\n" "$n_procs" "$done_in_log"
  if [ "$cur_step" != "—" ]; then
    printf "    Progress: step %s/%d  train=%s  best=%s  ~%s left (current cfg)\n" \
      "$cur_step" "$STEPS_PER_CONFIG" "$train_loss" "$best_val" "$time_left"
  fi
  if [ "$WANDB" = "online" ] && [ -n "$wb_url" ]; then
    echo -e "    W&B:      ${C}${wb_url}${N}"
  fi
  echo ""

  idx=$((idx+1))
done

# ── Phase 5: Aggregate stats ────────────────────────────────────────────────

echo -e "${B}── AGGREGATE ──${N}"

# Remaining = total - validated
remaining=$((TOTAL_CONFIGS - VALIDATED_COUNT))

# Overall ETA
if [ "$(echo "$agg_total_tput > 0" | bc)" = "1" ] && [ "$TOTAL_CONFIGS" -gt 0 ]; then
  overall_eta_h=$(echo "scale=1; $remaining / $agg_total_tput" | bc)
  overall_eta_d=$(echo "scale=1; $overall_eta_h / 24" | bc)
  done_pct=0
  if [ "$TOTAL_CONFIGS" -gt 0 ]; then
    done_pct=$(echo "scale=0; $VALIDATED_COUNT * 100 / $TOTAL_CONFIGS" | bc)
  fi

  printf "  Active VMs: %d  │  Total procs: %d  │  Total throughput: %s cfg/h\n" \
    "$agg_active_vms" "$agg_total_procs" "$agg_total_tput"
  printf "  Progress:   %s/%s validated (%s%%)  │  Remaining: %s  │  Failed: %s\n" \
    "$VALIDATED_COUNT" "$TOTAL_CONFIGS" "$done_pct" "$remaining" "$STATE_FAILED"
  printf "  Overall ETA: ${B}%sh (~%sd)${N}\n" "$overall_eta_h" "$overall_eta_d"
else
  printf "  Active VMs: %d  │  Total procs: %d  │  No throughput data yet\n" \
    "$agg_active_vms" "$agg_total_procs"
  printf "  Validated: %s/%s  │  Remaining: %s  │  Failed: %s\n" \
    "$VALIDATED_COUNT" "$TOTAL_CONFIGS" "$remaining" "$STATE_FAILED"
fi
echo ""

rm -rf "$tmpdir"
echo -e "${D}Auto-refresh: EXP=$EXP bash ~/tpu_guide/watch.sh${N}"
