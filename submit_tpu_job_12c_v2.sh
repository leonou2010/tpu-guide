#!/bin/bash
# submit_tpu_job_12c_v2.sh — XLA-optimized TPU job workflow for exp12c_tpu
#
# Same as submit_tpu_job_12c_v2.sh but uses:
#   - run_tpu_v2.py (train_v2_tpu — no .item(), foreach optimizer)
#   - adamw_ema_pullback_v3_tpu Hydra config
#
# Usage:
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --setup
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --auto
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --preflight
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --sweep
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --status
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --logs
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --cancel
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --pull-results
#   bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --push-code
#
# Override via env var:
#   TPU_NAME=v5e-uc1a ZONE=us-central1-a bash submit_tpu_job_12c_v2.sh --auto

GCLOUD=~/google-cloud-sdk/bin/gcloud
GSUTIL=~/google-cloud-sdk/bin/gsutil
PROJECT=gcp-research-credits-489020
BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
RESULTS_GCS=$BUCKET/results/exp12c_tpu_v2
SESSION=exp12c
MODE=${1:---status}

# Auto-source vm_config if TPU_NAME is set and config exists
TPU_NAME=${TPU_NAME:-v5e-ew4b}
VM_CONFIG=~/distributed_tpu_training/vm_configs/${TPU_NAME}.env
if [ -f "$VM_CONFIG" ]; then
  source "$VM_CONFIG"
fi

# Per-VM settings (env vars override vm_config)
ZONE=${ZONE:-europe-west4-b}
TPU_NUM_WORKERS=${TPU_NUM_WORKERS:-16}
SSH_HOST=${SSH_HOST:-tpu-v6e}
LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-'--xla_tpu_use_enhanced_launch_barrier=true --xla_latency_hiding_scheduler_rerun=1 --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_flash_attention=false --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true'}

# Launch mode
LAUNCH_MODE=${LAUNCH_MODE:-single}
CHIPS_PER_HOST=${CHIPS_PER_HOST:-8}
PROCS_PER_HOST=${PROCS_PER_HOST:-1}
# Backwards compat
[ "${USE_XMP_SPAWN:-0}" = "1" ] && LAUNCH_MODE=xmp

echo "[vm] $TPU_NAME  zone=$ZONE  workers=$TPU_NUM_WORKERS  launch=$LAUNCH_MODE  chips/host=$CHIPS_PER_HOST  procs/host=$PROCS_PER_HOST"

# Run a command on ALL workers simultaneously
all_workers() {
  $GCLOUD alpha compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
    --worker=all --command="$1" 2>&1 | grep -v WARNING | grep -v tunnel | grep -v "please see"
}

push_code() {
  echo "[code 1/3] Packing..."
  tar --exclude='__pycache__' --exclude='*.pyc' -czf /tmp/sf_bema_code_12c.tar.gz \
    -C ~/sf_bema/experiments \
    shared exp10_smollm2_smoltalk/conf exp10_smollm2_smoltalk/exp12c_tpu
  echo "[code 2/3] Uploading to GCS..."
  $GSUTIL cp /tmp/sf_bema_code_12c.tar.gz $BUCKET/code/sf_bema_code_12c.tar.gz
  echo "[code 3/3] Pulling on ALL $TPU_NUM_WORKERS workers..."
  all_workers "mkdir -p ~/sf_bema/experiments && gcloud storage cp $BUCKET/code/sf_bema_code_12c.tar.gz /tmp/sf_bema_code_12c.tar.gz && tar -xz -C ~/sf_bema/experiments/ -f /tmp/sf_bema_code_12c.tar.gz"
  echo "Code deployed to all workers."
}

pull_xla_cache() {
  echo "[xla-cache] Pulling cached XLA graphs from GCS to all workers..."
  all_workers "mkdir -p /tmp/xla_cache && gcloud storage cp -r '$BUCKET/xla_cache/*' /tmp/xla_cache/ 2>/dev/null; echo XLA_PULL_\$(ls /tmp/xla_cache/ 2>/dev/null | wc -l)_files"
}

push_xla_cache() {
  echo "[xla-cache] Pushing XLA cache from worker 0 to GCS..."
  $GCLOUD alpha compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
    --worker=0 --command="gcloud storage cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_PUSH_\$(ls /tmp/xla_cache/ 2>/dev/null | wc -l)_files" 2>/dev/null
}

launch_all() {
  local cmd=$1
  local logfile=$2
  local barrier_id=$(date +%s)

  # Build tmux env vars — XLA cache uses LOCAL path, synced to/from GCS separately
  local TMUX_ENVS="-e PJRT_DEVICE=TPU -e BUCKET=$BUCKET -e XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache -e LIBTPU_INIT_ARGS='$LIBTPU_INIT_ARGS' -e LLVM_NUM_THREADS=32 -e BARRIER_RUN_ID=$barrier_id -e TPU_NUM_WORKERS=$TPU_NUM_WORKERS -e LAUNCH_MODE=$LAUNCH_MODE -e CHIPS_PER_HOST=$CHIPS_PER_HOST -e WANDB_MODE=${WANDB_MODE:-online} -e MODEL_PATH=/tmp/SmolLM2-135M -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HYDRA_FULL_ERROR=1"
  if [ "$LAUNCH_MODE" = "single" ]; then
    TMUX_ENVS="$TMUX_ENVS -e TPU_PROCESS_BOUNDS=1,1,1"
  fi

  # Pull XLA cache from GCS before launch (reuse previous compilations)
  pull_xla_cache

  all_workers "
    tmux kill-session -t $SESSION 2>/dev/null || true
    { fuser /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
    sleep 1
    tmux new-session -d \
      $TMUX_ENVS \
      -s $SESSION \
      \"cd ~/sf_bema/experiments/exp10_smollm2_smoltalk && $cmd 2>&1 | tee $logfile\"
  "
  echo "Launched on all $TPU_NUM_WORKERS workers (barrier_id=$barrier_id). tmux session: $SESSION"
  echo "[xla-cache] Local path: /tmp/xla_cache → GCS: $BUCKET/xla_cache/"
  echo "[xla-cache] After training, run: bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --push-cache"
}

launch_parallel() {
  local total_procs=$((TPU_NUM_WORKERS * PROCS_PER_HOST))

  pull_xla_cache

  echo "[parallel] $PROCS_PER_HOST procs/host × $TPU_NUM_WORKERS hosts = $total_procs total workers"

  # Create launcher script locally
  cat > /tmp/tpu_launcher.sh << 'LAUNCHER_HEREDOC'
#!/bin/bash
# tpu_launcher.sh — Launches N parallel sweep processes on one TPU host
# Usage: bash tpu_launcher.sh <procs_per_host> <total_procs> <wandb_mode> <bucket> [libtpu_args...]
PROCS=$1; TOTAL=$2; WM=$3; BKT=$4; shift 4; LIBTPU="$*"

# Get this host's worker index
WH=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" -H "Metadata-Flavor: Google" 2>/dev/null || echo 0)

# Kill existing exp12c sessions and zombie TPU processes
for s in $(tmux list-sessions 2>/dev/null | grep exp12c | cut -d: -f1); do
  tmux kill-session -t "$s" 2>/dev/null
done
{ fuser /dev/vfio/* 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
sleep 1

for p in $(seq 0 $((PROCS - 1))); do
  WID=$((WH * PROCS + p))
  tmux new-session -d -s "exp12c_${p}" \
    -e PJRT_DEVICE=TPU \
    -e TPU_PROCESS_BOUNDS=1,1,1 \
    -e TPU_NUM_WORKERS=1 \
    -e "TPU_VISIBLE_CHIPS=${p}" \
    -e XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache \
    -e CHECKPOINT_DIR=/tmp \
    -e "WANDB_MODE=${WM}" \
    -e MODEL_PATH=/tmp/SmolLM2-135M \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HYDRA_FULL_ERROR=1 \
    -e "BUCKET=${BKT}" \
    -e "LIBTPU_INIT_ARGS=${LIBTPU}" \
    -e LLVM_NUM_THREADS=32 \
    "cd ~/sf_bema/experiments/exp10_smollm2_smoltalk && python3 exp12c_tpu/run_tpu_v2.py --sweep --worker-id=${WID} --num-workers=${TOTAL} 2>&1 | tee /tmp/exp12c_${p}.log; gcloud storage cp -r outputs/ ${BKT}/results/exp12c_tpu_v2/outputs/ 2>/dev/null; echo SWEEP_DONE_${p}"
done
echo "Host ${WH}: launched ${PROCS} procs (WIDs $((WH*PROCS))-$((WH*PROCS+PROCS-1)) of ${TOTAL})"
LAUNCHER_HEREDOC
  chmod +x /tmp/tpu_launcher.sh

  # Upload to GCS and pull on all workers
  $GSUTIL cp /tmp/tpu_launcher.sh $BUCKET/code/tpu_launcher.sh
  all_workers "gcloud storage cp $BUCKET/code/tpu_launcher.sh /tmp/tpu_launcher.sh && chmod +x /tmp/tpu_launcher.sh"

  # Execute launcher on all workers
  all_workers "bash /tmp/tpu_launcher.sh $PROCS_PER_HOST $total_procs '${WANDB_MODE:-online}' '$BUCKET' '$LIBTPU_INIT_ARGS'"

  echo "[parallel] $total_procs processes launched. Monitor: bash ~/distributed_tpu_training/monitor_v2.sh"
}

case $MODE in

  --setup)
    echo "=== VM SETUP ($TPU_NUM_WORKERS workers) ==="
    $GSUTIL cp ~/distributed_tpu_training/secrets.env $BUCKET/config/secrets.env
    $GSUTIL cp ~/distributed_tpu_training/setup.sh $BUCKET/config/setup.sh
    all_workers "BUCKET=$BUCKET bash <(gcloud storage cat $BUCKET/config/setup.sh)"
    echo "Setup complete on all workers."
    ;;

  --push-code)
    echo "=== PUSH CODE ==="
    push_code
    ;;

  --preflight)
    echo "=== PREFLIGHT ==="
    push_code
    launch_all "python3 exp12c_tpu/run_tpu_v2.py --preflight" \
      "/tmp/exp12c.log"
    echo "Monitor: bash ~/distributed_tpu_training/submit_tpu_job_12c_v2.sh --logs"
    ;;

  --sweep)
    echo "=== PARALLEL SWEEP ($PROCS_PER_HOST procs/host × $TPU_NUM_WORKERS hosts) ==="
    push_code
    if [ "$PROCS_PER_HOST" -gt 1 ]; then
      launch_parallel
    else
      launch_all "python3 exp12c_tpu/run_tpu_v2.py --sweep; \
        gcloud storage cp -r outputs/ $RESULTS_GCS/${TPU_NAME}/outputs/ 2>/dev/null; echo SWEEP_DONE" \
        "/tmp/exp12c.log"
    fi
    echo "Monitor: bash ~/distributed_tpu_training/monitor_v2.sh"
    ;;

  --auto)
    echo "=== AUTO: preflight → sweep ($PROCS_PER_HOST procs/host × $TPU_NUM_WORKERS hosts) ==="
    push_code
    if [ "$PROCS_PER_HOST" -gt 1 ]; then
      # Preflight with 1 process first, then parallel sweep via launcher script
      launch_all "echo auto_preflight_start && python3 exp12c_tpu/run_tpu_v2.py --preflight && echo auto_preflight_done && { gcloud storage cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_CACHE_PUSHED; } && bash /tmp/tpu_launcher.sh $PROCS_PER_HOST $((TPU_NUM_WORKERS * PROCS_PER_HOST)) '${WANDB_MODE:-online}' '$BUCKET' '$LIBTPU_INIT_ARGS'" \
        "/tmp/exp12c.log"
    else
      launch_all "echo auto_preflight_start && python3 exp12c_tpu/run_tpu_v2.py --preflight && echo auto_preflight_done && { gcloud storage cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_CACHE_PUSHED; } && python3 exp12c_tpu/run_tpu_v2.py --sweep && { gcloud storage cp -r outputs/ $RESULTS_GCS/${TPU_NAME}/outputs/ 2>/dev/null || true; } && echo auto_sweep_done || echo auto_FAILED" \
        "/tmp/exp12c.log"
    fi
    echo "Monitor: bash ~/distributed_tpu_training/monitor_v2.sh"
    ;;

  --status)
    echo "=== JOB STATUS ($SSH_HOST w-0) ==="
    ssh $SSH_HOST "
      echo '--- tmux sessions ---'
      tmux list-sessions 2>/dev/null || echo 'none'
      echo '--- python3 processes ---'
      ps aux | grep python3 | grep -v grep | wc -l | xargs echo 'count:'
      echo '--- last 5 log lines ---'
      tail -5 /tmp/${SESSION}.log 2>/dev/null || echo 'no log yet'
    " 2>&1 | grep -v WARNING | grep -v tunnel
    ;;

  --logs)
    echo "=== LIVE LOGS from $SSH_HOST (Ctrl+C to stop) ==="
    ssh $SSH_HOST "tail -f /tmp/${SESSION}.log" 2>&1 | grep -v WARNING | grep -v tunnel
    ;;

  --cancel)
    echo "=== CANCELLING all workers ==="
    all_workers "for s in \$(tmux list-sessions 2>/dev/null | grep exp12c | cut -d: -f1); do tmux kill-session -t \"\$s\" 2>/dev/null; done; { fuser /dev/vfio/* 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null; echo cancelled"
    ;;

  --pull-results)
    echo "=== PULLING RESULTS ==="
    mkdir -p ~/sf_bema/results/exp12c_tpu/${TPU_NAME}
    $GSUTIL -m cp -r $RESULTS_GCS/${TPU_NAME}/ ~/sf_bema/results/exp12c_tpu/${TPU_NAME}/
    echo "Results saved to ~/sf_bema/results/exp12c_tpu/${TPU_NAME}/"
    ;;

  --push-cache)
    echo "=== PUSH XLA CACHE to GCS ==="
    push_xla_cache
    ;;

  --pull-cache)
    echo "=== PULL XLA CACHE from GCS ==="
    pull_xla_cache
    ;;

  *)
    echo "Usage: bash submit_tpu_job_12c_v2.sh [--setup|--auto|--preflight|--sweep|--status|--logs|--cancel|--pull-results|--push-code|--push-cache|--pull-cache]"
    echo "  Parallel: PROCS_PER_HOST=4 bash submit_tpu_job_12c_v2.sh --sweep"
    ;;
esac
