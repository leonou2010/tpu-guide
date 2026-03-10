#!/bin/bash
# submit.sh — Generic TPU job workflow (parameterized by EXP + TPU_NAME)
#
# Centralized push model: coordinator distributes, workers execute.
#
# Usage:
#   EXP=exp12c TPU_NAME=v6e-ew4a bash ~/tpu_guide/submit.sh --setup
#   EXP=exp12c TPU_NAME=v6e-ew4a bash ~/tpu_guide/submit.sh --sweep
#   EXP=exp12c TPU_NAME=v6e-ew4a bash ~/tpu_guide/submit.sh --preflight
#   EXP=exp12c TPU_NAME=v6e-ew4a bash ~/tpu_guide/submit.sh --status
#   EXP=exp12c TPU_NAME=v6e-ew4a bash ~/tpu_guide/submit.sh --cancel
#   EXP=exp12c python3 ~/tpu_guide/coordinator.py --init      # distribute configs (blocklab)
#   EXP=exp12c python3 ~/tpu_guide/coordinator.py --monitor   # coordination loop (blocklab)

set -euo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
GSUTIL=~/google-cloud-sdk/bin/gsutil
PROJECT=gcp-research-credits-489020
MODE=${1:---status}

# ── Validate required env vars ──────────────────────────────────────────────

EXP=${EXP:?'EXP env var required (e.g. EXP=exp12c)'}
TPU_NAME=${TPU_NAME:?'TPU_NAME env var required (e.g. TPU_NAME=v6e-ew4a)'}

# ── Load experiment config ──────────────────────────────────────────────────

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_CONFIG="$SCRIPT_DIR/experiments/${EXP}.env"
if [ ! -f "$EXP_CONFIG" ]; then
  echo "ERROR: experiment config not found: $EXP_CONFIG" >&2
  exit 1
fi
source "$EXP_CONFIG"
# EXP_NAME, EXP_MODULE, WORK_DIR, CODE_DIRS, MODEL_PATH, STEPS_PER_CONFIG now set

# ── Load VM config ──────────────────────────────────────────────────────────

VM_CONFIG="$SCRIPT_DIR/vm_configs/${TPU_NAME}.env"
if [ -f "$VM_CONFIG" ]; then
  source "$VM_CONFIG"
fi

# Per-VM settings (env vars override vm_config)
ZONE=${ZONE:-europe-west4-b}
TPU_NUM_WORKERS=${TPU_NUM_WORKERS:-1}
SSH_HOST=${SSH_HOST:-$TPU_NAME}
BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-'--xla_tpu_use_enhanced_launch_barrier=true --xla_latency_hiding_scheduler_rerun=1 --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_flash_attention=false --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true'}
LAUNCH_MODE=${LAUNCH_MODE:-single}
CHIPS_PER_HOST=${CHIPS_PER_HOST:-8}
PROCS_PER_HOST=${PROCS_PER_HOST:-1}
MODEL_PATH=${MODEL_PATH:-/tmp/SmolLM2-135M}

# Derived
SESSION=${EXP_NAME}
RESULTS_GCS=$BUCKET/results/${EXP_NAME}
CODE_TAR="sf_bema_code_${EXP_NAME}.tar.gz"

echo "[vm] $TPU_NAME  zone=$ZONE  workers=$TPU_NUM_WORKERS  procs/host=$PROCS_PER_HOST  exp=$EXP_NAME"

# ── Helpers ─────────────────────────────────────────────────────────────────

all_workers() {
  $GCLOUD alpha compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
    --worker=all --command="$1" 2>&1 | grep -v WARNING | grep -v tunnel | grep -v "please see"
}

push_code() {
  echo "[code 1/3] Packing..."
  # shellcheck disable=SC2086
  tar --exclude='__pycache__' --exclude='*.pyc' -czf /tmp/$CODE_TAR \
    -C ~/sf_bema/experiments \
    $CODE_DIRS
  echo "[code 2/3] Uploading to GCS..."
  $GSUTIL cp /tmp/$CODE_TAR $BUCKET/code/$CODE_TAR

  # Also bundle coordinator.py + experiment config
  $GSUTIL cp "$SCRIPT_DIR/coordinator.py" $BUCKET/code/coordinator.py
  $GSUTIL cp "$EXP_CONFIG" $BUCKET/code/${EXP}.env

  echo "[code 3/3] Pulling on ALL $TPU_NUM_WORKERS workers..."
  all_workers "mkdir -p ~/sf_bema/experiments && gsutil cp $BUCKET/code/$CODE_TAR /tmp/$CODE_TAR && tar -xz -C ~/sf_bema/experiments/ -f /tmp/$CODE_TAR && gsutil cp $BUCKET/code/coordinator.py ~/coordinator.py && gsutil cp $BUCKET/code/${EXP}.env ~/experiments_${EXP}.env"
  echo "Code deployed to all workers."
}

pull_xla_cache() {
  echo "[xla-cache] Pulling cached XLA graphs from GCS to all workers..."
  all_workers "mkdir -p /tmp/xla_cache && gsutil -m cp -r '$BUCKET/xla_cache/*' /tmp/xla_cache/ 2>/dev/null; echo XLA_PULL_\$(ls /tmp/xla_cache/ 2>/dev/null | wc -l)_files"
}

push_xla_cache() {
  echo "[xla-cache] Pushing XLA cache from worker 0 to GCS..."
  $GCLOUD alpha compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
    --worker=0 --command="gsutil -m cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_PUSH_\$(ls /tmp/xla_cache/ 2>/dev/null | wc -l)_files" 2>/dev/null
}

launch_all() {
  local cmd=$1
  local logfile=$2
  local barrier_id=$(date +%s)

  # Build env exports inline (compatible with all tmux versions, avoids -e flag)
  local ENV_EXPORTS="export PJRT_DEVICE=TPU BUCKET=$BUCKET XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache LLVM_NUM_THREADS=32 BARRIER_RUN_ID=$barrier_id TPU_NUM_WORKERS=$TPU_NUM_WORKERS LAUNCH_MODE=$LAUNCH_MODE CHIPS_PER_HOST=$CHIPS_PER_HOST WANDB_MODE=${WANDB_MODE:-online} MODEL_PATH=$MODEL_PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 EXP=$EXP TPU_NAME=$TPU_NAME"
  # LIBTPU_INIT_ARGS needs special quoting (contains spaces)
  ENV_EXPORTS="$ENV_EXPORTS; export LIBTPU_INIT_ARGS='$LIBTPU_INIT_ARGS'"
  if [ "$LAUNCH_MODE" = "single" ]; then
    ENV_EXPORTS="$ENV_EXPORTS; export TPU_PROCESS_BOUNDS=1,1,1"
  fi

  pull_xla_cache

  all_workers "
    tmux kill-session -t $SESSION 2>/dev/null || true
    { fuser /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
    sleep 1
    tmux new-session -d \
      -s $SESSION \
      \"$ENV_EXPORTS; cd ~/sf_bema/experiments/$WORK_DIR && $cmd 2>&1 | tee $logfile\"
  "
  echo "Launched on all $TPU_NUM_WORKERS workers (barrier_id=$barrier_id). tmux session: $SESSION"
}

launch_parallel() {
  local total_procs=$((TPU_NUM_WORKERS * PROCS_PER_HOST))

  pull_xla_cache

  echo "[parallel] $PROCS_PER_HOST procs/host x $TPU_NUM_WORKERS hosts = $total_procs total workers"

  # Create launcher script locally
  cat > /tmp/tpu_launcher_${EXP_NAME}.sh << LAUNCHER_HEREDOC
#!/bin/bash
# tpu_launcher.sh — Launches N parallel coordinator workers on one TPU host
# Usage: bash tpu_launcher.sh <procs_per_host> <total_procs> <wandb_mode> <bucket> <exp> <tpu_name> <work_dir> <model_path> [libtpu_args...]
PROCS=\$1; TOTAL=\$2; WM=\$3; BKT=\$4; EXP_ARG=\$5; TPUNAME=\$6; WORKDIR=\$7; MDLPATH=\$8; shift 8; LIBTPU="\$*"

# Get this host's worker index
WH=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" -H "Metadata-Flavor: Google" 2>/dev/null || echo 0)

# Kill existing sessions and zombie TPU processes
for s in \$(tmux list-sessions 2>/dev/null | grep ${EXP_NAME} | cut -d: -f1); do
  tmux kill-session -t "\$s" 2>/dev/null
done
{ fuser /dev/vfio/* 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null || true
sleep 1

# Create experiments dir for coordinator config
mkdir -p ~/tpu_guide/experiments

# Copy coordinator + config into place
cp ~/coordinator.py ~/tpu_guide/coordinator.py 2>/dev/null || true
cp ~/experiments_\${EXP_ARG}.env ~/tpu_guide/experiments/\${EXP_ARG}.env 2>/dev/null || true

# Compute this proc's global index for --proc-idx
# proc_idx = worker_host_idx * procs_per_host + local_proc_idx
for p in \$(seq 0 \$((\$PROCS - 1))); do
  WID="\${TPUNAME}_\${WH}_\${p}"
  PROC_IDX=\$((\${WH} * \${PROCS} + \${p}))
  tmux new-session -d -s "${EXP_NAME}_\${p}" \
    "export PATH=\$HOME/miniconda3/bin:\$HOME/.local/bin:\$PATH; export PJRT_DEVICE=TPU TPU_PROCESS_BOUNDS=1,1,1 TPU_NUM_WORKERS=1 TPU_VISIBLE_CHIPS=\${p} XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache CHECKPOINT_DIR=/tmp WANDB_MODE=\${WM} MODEL_PATH=\${MDLPATH} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 BUCKET=\${BKT} LLVM_NUM_THREADS=32 EXP=\${EXP_ARG} TPU_NAME=\${TPUNAME} WORKER_IDX=\${WH}; export LIBTPU_INIT_ARGS='\${LIBTPU}'; cd ~/sf_bema/experiments/\${WORKDIR} && EXP=\${EXP_ARG} python3 -u ~/tpu_guide/coordinator.py --sweep --worker-id=\${WID} --proc-idx=\${PROC_IDX} --num-procs=\${TOTAL} 2>&1 | tee /tmp/${EXP_NAME}_\${p}.log; echo SWEEP_DONE_\${p}"
done
echo "Host \${WH}: launched \${PROCS} procs (coordinator workers)"
LAUNCHER_HEREDOC
  chmod +x /tmp/tpu_launcher_${EXP_NAME}.sh

  # Upload to GCS and pull on all workers
  $GSUTIL cp /tmp/tpu_launcher_${EXP_NAME}.sh $BUCKET/code/tpu_launcher_${EXP_NAME}.sh
  all_workers "gsutil cp $BUCKET/code/tpu_launcher_${EXP_NAME}.sh /tmp/tpu_launcher_${EXP_NAME}.sh && chmod +x /tmp/tpu_launcher_${EXP_NAME}.sh"

  # Execute launcher on all workers
  all_workers "bash /tmp/tpu_launcher_${EXP_NAME}.sh $PROCS_PER_HOST $total_procs '${WANDB_MODE:-online}' '$BUCKET' '$EXP' '$TPU_NAME' '$WORK_DIR' '$MODEL_PATH' '$LIBTPU_INIT_ARGS'"

  echo "[parallel] $total_procs coordinator workers launched. Monitor: EXP=$EXP bash ~/tpu_guide/monitor.sh"
}

# ── Commands ────────────────────────────────────────────────────────────────

case $MODE in

  --setup)
    echo "=== VM SETUP ($TPU_NUM_WORKERS workers) ==="
    $GSUTIL cp ~/tpu_guide/secrets.env $BUCKET/config/secrets.env
    $GSUTIL cp ~/tpu_guide/setup.sh $BUCKET/config/setup.sh
    all_workers "BUCKET=$BUCKET bash <(gsutil cat $BUCKET/config/setup.sh)"
    echo "Setup complete on all workers."
    ;;

  --push-code)
    echo "=== PUSH CODE ==="
    push_code
    ;;

  --init)
    echo "=== INIT: Distribute configs to all VMs ==="
    cd ~/sf_bema/experiments/$WORK_DIR
    EXP=$EXP python3 ~/tpu_guide/coordinator.py --init
    ;;

  --preflight)
    echo "=== PREFLIGHT ==="
    push_code
    launch_all "python3 ${EXP_MODULE//.//}.py --preflight" \
      "/tmp/${EXP_NAME}.log"
    echo "Monitor: EXP=$EXP TPU_NAME=$TPU_NAME bash ~/tpu_guide/submit.sh --logs"
    ;;

  --sweep)
    echo "=== SWEEP ($PROCS_PER_HOST procs/host x $TPU_NUM_WORKERS hosts) ==="
    push_code

    if [ "$PROCS_PER_HOST" -gt 1 ]; then
      launch_parallel
    else
      launch_all "mkdir -p ~/tpu_guide/experiments && cp ~/coordinator.py ~/tpu_guide/coordinator.py 2>/dev/null; cp ~/experiments_${EXP}.env ~/tpu_guide/experiments/${EXP}.env 2>/dev/null; EXP=$EXP python3 ~/tpu_guide/coordinator.py --sweep --worker-id=${TPU_NAME}_\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google' 2>/dev/null || echo 0)_0 --proc-idx=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google' 2>/dev/null || echo 0) --num-procs=${TPU_NUM_WORKERS}; echo SWEEP_DONE" \
        "/tmp/${EXP_NAME}.log"
    fi
    echo "Monitor: EXP=$EXP bash ~/tpu_guide/monitor.sh"
    ;;

  --auto)
    echo "=== AUTO: preflight → sweep ($PROCS_PER_HOST procs/host x $TPU_NUM_WORKERS hosts) ==="
    push_code

    if [ "$PROCS_PER_HOST" -gt 1 ]; then
      # Preflight with 1 process first, then parallel sweep
      launch_all "echo auto_preflight_start && python3 ${EXP_MODULE//.//}.py --preflight && echo auto_preflight_done && { gsutil -m cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_CACHE_PUSHED; } && bash /tmp/tpu_launcher_${EXP_NAME}.sh $PROCS_PER_HOST $((TPU_NUM_WORKERS * PROCS_PER_HOST)) '${WANDB_MODE:-online}' '$BUCKET' '$EXP' '$TPU_NAME' '$WORK_DIR' '$MODEL_PATH' '$LIBTPU_INIT_ARGS'" \
        "/tmp/${EXP_NAME}.log"
    else
      launch_all "echo auto_preflight_start && python3 ${EXP_MODULE//.//}.py --preflight && echo auto_preflight_done && { gsutil -m cp -r /tmp/xla_cache/* $BUCKET/xla_cache/ 2>/dev/null; echo XLA_CACHE_PUSHED; } && mkdir -p ~/tpu_guide/experiments && cp ~/coordinator.py ~/tpu_guide/coordinator.py 2>/dev/null; cp ~/experiments_${EXP}.env ~/tpu_guide/experiments/${EXP}.env 2>/dev/null; EXP=$EXP python3 ~/tpu_guide/coordinator.py --sweep --worker-id=${TPU_NAME}_\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google' 2>/dev/null || echo 0)_0 --proc-idx=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google' 2>/dev/null || echo 0) --num-procs=${TPU_NUM_WORKERS} && echo auto_sweep_done || echo auto_FAILED" \
        "/tmp/${EXP_NAME}.log"
    fi
    echo "Monitor: EXP=$EXP bash ~/tpu_guide/monitor.sh"
    ;;

  --monitor)
    echo "=== MONITOR: Start coordination loop ==="
    cd ~/sf_bema/experiments/$WORK_DIR
    EXP=$EXP python3 ~/tpu_guide/coordinator.py --monitor
    ;;

  --status)
    echo "=== JOB STATUS ($SSH_HOST w-0) ==="
    ssh $SSH_HOST "
      echo '--- tmux sessions ---'
      tmux list-sessions 2>/dev/null || echo 'none'
      echo '--- python3 processes ---'
      ps aux | grep python3 | grep -v grep | wc -l | xargs echo 'count:'
      echo '--- last 5 log lines ---'
      tail -5 /tmp/${EXP_NAME}.log 2>/dev/null || echo 'no log yet'
    " 2>&1 | grep -v WARNING | grep -v tunnel

    echo ""
    echo "=== COORDINATOR STATUS ==="
    EXP=$EXP python3 ~/tpu_guide/coordinator.py --status
    ;;

  --logs)
    echo "=== LIVE LOGS from $SSH_HOST (Ctrl+C to stop) ==="
    ssh $SSH_HOST "tail -f /tmp/${EXP_NAME}.log" 2>&1 | grep -v WARNING | grep -v tunnel
    ;;

  --cancel)
    echo "=== CANCELLING all workers ==="
    all_workers "for s in \$(tmux list-sessions 2>/dev/null | grep ${EXP_NAME} | cut -d: -f1); do tmux kill-session -t \"\$s\" 2>/dev/null; done; { fuser /dev/vfio/* 2>/dev/null; fuser /dev/accel* 2>/dev/null; } | xargs -r kill -9 2>/dev/null; echo cancelled"
    ;;

  --pull-results)
    echo "=== PULLING RESULTS ==="
    mkdir -p ~/sf_bema/results/${EXP_NAME}/${TPU_NAME}
    $GSUTIL -m cp -r $RESULTS_GCS/${TPU_NAME}/ ~/sf_bema/results/${EXP_NAME}/${TPU_NAME}/
    echo "Results saved to ~/sf_bema/results/${EXP_NAME}/${TPU_NAME}/"
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
    echo "Usage: EXP=<name> TPU_NAME=<vm> bash ~/tpu_guide/submit.sh <command>"
    echo ""
    echo "Commands:"
    echo "  --setup        Install packages, deploy code+data+model"
    echo "  --preflight    Quick run (~5 min): XLA compile + measure step time/HBM"
    echo "  --auto         Preflight → sweep"
    echo "  --init         Distribute configs to VMs (blocklab, run once)"
    echo "  --sweep        Launch workers on this VM (reads assignment from GCS)"
    echo "  --monitor      Start coordination loop (blocklab, long-running)"
    echo "  --push-code    Deploy code only"
    echo "  --status       Show tmux sessions + coordinator status"
    echo "  --logs         Tail live logs from worker-0"
    echo "  --cancel       Kill all tmux sessions + zombie processes"
    echo "  --pull-results Download results from GCS"
    echo "  --push-cache   Push XLA cache to GCS"
    echo "  --pull-cache   Pull XLA cache from GCS"
    ;;
esac
