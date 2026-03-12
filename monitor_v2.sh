#!/bin/bash
# monitor_v2.sh — TPU sweep dashboard (comprehensive per-VM stats)
# Design: all SSH + GCS queries run in parallel, static info prints immediately
GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020

# ── KNOWN DATA (actively managed — update after every run) ──
TOTAL_CONFIGS=225
STEPS_PER_CONFIG=1778  # 889 steps/epoch × 2 epochs
HBM_PER_PROC=10.8     # GB, measured
XLA_OVERHEAD=5         # GB, measured
CKPT_SIZE=2.6          # GB, calculated

VMS=(
  # TPU          ZONE               ACCEL         HOSTS  PPH  BUCKET                                                     STEP_S  HBM_CHIP  WANDB
  "v4-uc2b       us-central2-b      v4-32         4      4    gs://gcp-researchcredits-blocklab-1-us-central2             8.4     32        disabled"
  "v4-uc2b-spot  us-central2-b      v4-32         4      4    gs://gcp-researchcredits-blocklab-1-us-central2             8.3     32        disabled"
  "v6e-ue1d      us-east1-d         v6e-16        4      4    gs://gcp-researchcredits-blocklab-us-east1                  4.9     32        disabled"
  "v6e-ew4a      europe-west4-a     v6e-8         1      8    gs://gcp-researchcredits-blocklab-europe-west4              4.9     32        online"
  "v6e-ew4a-16   europe-west4-a     v6e-16        4      4    gs://gcp-researchcredits-blocklab-europe-west4              5.6     32        online"
)

B='\033[1m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; D='\033[90m'; C='\033[36m'; N='\033[0m'

echo -e "${B}══════════════════════════════════════════════════════════════════════════════════════════════════════${N}"
echo -e "${B}  exp12c_tpu_v2 SWEEP  │  $(date '+%Y-%m-%d %H:%M:%S')  │  ${TOTAL_CONFIGS} configs × ${STEPS_PER_CONFIG} steps${N}"
echo -e "${B}══════════════════════════════════════════════════════════════════════════════════════════════════════${N}"
echo ""

tmpdir=$(mktemp -d)

# ── Phase 1: Launch ALL queries in parallel ──

# GCS storage queries (parallel, per unique bucket)
declare -A BUCKET_SIZE
unique_buckets=()
for vm_line in "${VMS[@]}"; do
  bkt=$(echo "$vm_line" | awk '{print $6}')
  found=0; for u in "${unique_buckets[@]}"; do [ "$u" = "$bkt" ] && found=1; done
  [ "$found" = "0" ] && unique_buckets+=("$bkt")
done
for bkt in "${unique_buckets[@]}"; do
  ( $GCLOUD storage du -s "$bkt/" 2>/dev/null | awk '{print $1}' > "$tmpdir/bkt_$(echo "$bkt" | md5sum | cut -c1-8)" ) &
done

# SSH queries (parallel, all VMs at once)
idx=0
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"
  [ "$PPH" = "0" ] && { idx=$((idx+1)); continue; }
  ( timeout 45 $GCLOUD alpha compute tpus tpu-vm ssh "$TPU" \
      --zone="$ZONE" --project=$PROJECT --tunnel-through-iap \
      --worker=0 --command="
        echo PROCS_START
        tmux list-sessions 2>/dev/null | grep -c exp12c || echo 0
        echo PROCS_END
        echo LOG_START
        tail -20 /tmp/exp12c.log 2>/dev/null
        for f in /tmp/exp12c_*.log; do [ -f \"\\$f\" ] && tail -10 \"\\$f\" 2>/dev/null; done
        echo LOG_END
        echo WANDB_START
        grep -oP 'https://wandb.ai/[^ ]+' /tmp/exp12c_0.log 2>/dev/null | head -1
        echo WANDB_END
      " > "$tmpdir/ssh_${idx}" 2>/dev/null ) &
  idx=$((idx+1))
done

# ── Phase 2: Print static info immediately while queries run ──
idx=0
for vm_line in "${VMS[@]}"; do
  read -r TPU ZONE ACCEL HOSTS PPH BUCKET STEP_S HBM_CHIP WANDB <<< "$vm_line"

  if [ "$PPH" = "0" ]; then
    echo -e "  ${B}${TPU}${N} (${ACCEL}) — ${R}OOM${N}: ${HBM_CHIP}GB chip, need ${HBM_PER_PROC}+${XLA_OVERHEAD}GB"
    echo ""
    idx=$((idx+1))
    continue
  fi

  procs=$((HOSTS * PPH))
  cfg_h=$(echo "scale=2; $STEPS_PER_CONFIG * $STEP_S / 3600" | bc)
  tput=$(echo "scale=2; $procs / $cfg_h" | bc)

  echo -e "  ${B}${TPU}${N} (${ACCEL}, ${ZONE})"
  printf "    Parallel: ${C}%d hosts × %d procs/host = %d procs${N}\n" "$HOSTS" "$PPH" "$procs"
  printf "    Memory:   %dGB/chip, %sGB usable, %sGB/proc, buffer=%.1fGB, ckpt=%sGB/proc\n" \
    "$HBM_CHIP" "$(echo "$HBM_CHIP - $XLA_OVERHEAD" | bc)" "$HBM_PER_PROC" \
    "$(echo "$HBM_CHIP - $XLA_OVERHEAD - $HBM_PER_PROC" | bc)" "$CKPT_SIZE"
  printf "    Timing:   step=%ss  config=%sh  throughput=%s cfg/h (est)\n" "$STEP_S" "$cfg_h" "$tput"
  echo -e "    ${D}... fetching live data ...${N}"

  idx=$((idx+1))
done

# ── Phase 3: Wait for all background queries ──
wait

# ── Phase 4: Print live data ──
echo ""
echo -e "${B}── LIVE STATUS ──${N}"
echo ""

# Load bucket sizes
for bkt in "${unique_buckets[@]}"; do
  key=$(echo "$bkt" | md5sum | cut -c1-8)
  BUCKET_SIZE["$bkt"]=$(cat "$tmpdir/bkt_$key" 2>/dev/null)
  BUCKET_SIZE["$bkt"]=${BUCKET_SIZE["$bkt"]:-0}
done

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
  wb_url=$(echo "$combined" | sed -n '/WANDB_START/,/WANDB_END/p' | grep -v 'WANDB_' | head -1)

  # Config progress
  cur_config=0
  if echo "$log" | grep -q "\[sweep\] Config"; then
    cur_config=$(echo "$log" | grep "\[sweep\] Config" | tail -1 | grep -oP 'Config \K[0-9]+' | head -1)
    cur_config=${cur_config:-0}
  fi

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

  # VM-ETA
  vm_eta="—"
  remaining_cfgs=$((TOTAL_CONFIGS - cur_config))
  actual_tput="${live_tput:-$tput}"
  if [ "$(echo "$actual_tput > 0" | bc)" = "1" ]; then
    eta_h=$(echo "scale=1; $remaining_cfgs / $actual_tput" | bc)
    [ "$cur_config" -gt 0 ] && vm_eta="${eta_h}h (${remaining_cfgs} left)" || vm_eta="${eta_h}h"
  fi

  # Status
  status=""
  if [ -z "$combined" ]; then
    status="${R}OFFLINE${N}"
  elif [ "$n_procs" = "0" ]; then
    if echo "$combined" | grep -q "exp12c"; then n_procs=1
    else status="${D}IDLE${N}"; fi
  fi
  if [ -z "$status" ]; then
    if echo "$log" | grep -q "SWEEP_DONE\|auto_sweep_done"; then status="${G}DONE${N}"
    elif echo "$log" | grep -q "Traceback\|Error\|FAILED\|OOM"; then status="${R}ERROR${N}"
    elif echo "$log" | grep -q "compiling\|Compiling"; then status="${Y}COMPILE${N}"
    elif echo "$log" | grep -q "\[sweep\] Config\|step "; then status="${G}RUNNING${N}"
    elif echo "$log" | grep -q "PREFLIGHT\|preflight"; then status="${Y}PREFLIGHT${N}"
    else status="${Y}STARTING${N}"; fi
  fi

  # GCS storage
  bkt_bytes=${BUCKET_SIZE["$BUCKET"]:-0}
  if [ "$bkt_bytes" -gt 1073741824 ] 2>/dev/null; then
    storage_str="$(echo "scale=1; $bkt_bytes / 1073741824" | bc)GB"
  elif [ "$bkt_bytes" -gt 1048576 ] 2>/dev/null; then
    storage_str="$(echo "scale=1; $bkt_bytes / 1048576" | bc)MB"
  else
    storage_str="${bkt_bytes}B"
  fi

  # Print
  echo -ne "  ${B}${TPU}${N}: "; echo -e "${status}"
  printf "    Timing:   step=%s  config=%s  throughput=%s cfg/h  │  ETA: %s\n" "$display_step_s" "$display_cfg_h" "$display_tput" "$vm_eta"
  printf "    Storage:  %s  │  Procs: w0=%d visible\n" "$storage_str" "$n_procs"
  if [ "$cur_step" != "—" ]; then
    printf "    Progress: cfg %d/%d  step %s/%d  train=%s  best=%s  ~%s left\n" \
      "$cur_config" "$TOTAL_CONFIGS" "$cur_step" "$STEPS_PER_CONFIG" "$train_loss" "$best_val" "$time_left"
  fi
  if [ "$WANDB" = "online" ] && [ -n "$wb_url" ]; then
    echo -e "    W&B:      ${C}${wb_url}${N}"
  fi
  echo ""

  idx=$((idx+1))
done

rm -rf "$tmpdir"
echo -e "${D}Auto-refresh: bash ~/distributed_tpu_training/watch.sh${N}"
