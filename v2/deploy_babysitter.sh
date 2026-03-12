#!/bin/bash
# deploy_babysitter.sh — Idempotent VM setup script for TPU workers.
#
# Golden Rules:
#   1. FAIL FAST: set -uo pipefail (careful -e not used — some steps are optional)
#   2. GCS TELEMETRY: report_phase() at each major step for visibility
#   3. IDEMPOTENCY: check before install, clean caches to prevent disk OOM
#   4. PRE-FLIGHT ASSERTION: verify environment before launching babysitter
#
# Uploads telemetry to: gs://.../coord_v2/telemetry/<TPU_NAME>_boot.json
# Logs to: /tmp/babysitter.log (babysitter process) + /tmp/deploy_babysitter.log (this script)
set -uo pipefail

# ── Redirect own output to deploy log ────────────────────────────────────────
exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_babysitter.sh started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# ── Ensure HOME is set (startup scripts run as root with minimal env) ────────
export HOME=${HOME:-/root}

# ── PATH ─────────────────────────────────────────────────────────────────────
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH

# ── Early var detection (before TPU_NAME/ZONE are resolved) ──────────────────
# Auto-detect ZONE first (needed for BUCKET)
if [ -z "${ZONE:-}" ]; then
    ZONE=$(curl -s --connect-timeout 5 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
        | awk -F/ '{print $NF}')
fi
# Auto-detect BUCKET from zone
case "${ZONE:-}" in
    europe-west4*) BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4} ;;
    us-east1*)     BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-us-east1} ;;
    us-central2*)  BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-1-us-central2} ;;
    *)             BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4} ;;
esac
# Use configured TPU_NAME if provided, else fall back to GCP metadata
if [ -z "${TPU_NAME:-}" ]; then
    TPU_NAME=$(curl -s --connect-timeout 5 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || hostname)
fi
CTRL=${CONTROL_PLANE:-gs://gcp-researchcredits-blocklab-europe-west4/coord_v2}

echo "TPU_NAME=$TPU_NAME ZONE=${ZONE:-} BUCKET=$BUCKET"

# ── GCS Telemetry helper ──────────────────────────────────────────────────────
report_phase() {
    local phase=$1
    local ts
    ts=$(date +%s)
    printf '{"tpu_name":"%s","zone":"%s","phase":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "${ZONE:-}" "$phase" "$ts" > /tmp/boot_state.json
    # Best-effort upload — don't fail if GCS unreachable
    gsutil cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json" 2>/dev/null || true
    echo "[$(date -u '+%H:%M:%S')] PHASE: $phase"
}

report_phase "BOOTING"

# ── Safety guard: non-destructive if any chip is actively training ────────────
# Reads GCS heartbeats for all chips. If any chip has a fresh heartbeat
# (age < 2700s) with status training/xla_compile/uploading, exit safely.
# Override with FORCE_REDEPLOY=1 to skip this check.
GUARD_TTL=2700
_check_heartbeat_pattern() {
  # Check if any chip matching $1 pattern has fresh active heartbeat
  local pattern=$1
  local now_s=$2
  for _hb_path in $(gsutil ls "${CTRL}/heartbeats/${pattern}_chip*.json" 2>/dev/null || true); do
    _hb_json=$(gsutil cat "$_hb_path" 2>/dev/null || true)
    [ -z "$_hb_json" ] && continue
    _ts=$(echo "$_hb_json" | python3 -c "import sys,json; print(int(json.load(sys.stdin).get('timestamp',0)))" 2>/dev/null || echo 0)
    _status=$(echo "$_hb_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    _age=$(( now_s - _ts ))
    case "$_status" in
      training|xla_compile|uploading)
        if [ "$_age" -lt "$GUARD_TTL" ]; then
          echo "$_hb_path (age=${_age}s, status=$_status)"
          return
        fi
        ;;
    esac
  done
}
if [ -z "${FORCE_REDEPLOY:-}" ]; then
  _now_s=$(date +%s)
  _active_chip=""
  # Check by TPU_NAME (friendly name set by vm_requester)
  _active_chip=$(_check_heartbeat_pattern "$TPU_NAME" "$_now_s")
  # Also check by hostname (internal GCP name — babysitter may have fallen back to this)
  if [ -z "$_active_chip" ]; then
    _hostname=$(hostname -s 2>/dev/null || true)
    [ -n "$_hostname" ] && _active_chip=$(_check_heartbeat_pattern "$_hostname" "$_now_s")
  fi
  if [ -n "$_active_chip" ]; then
    echo "[deploy_babysitter] GUARD: active chip detected — $_active_chip"
    echo "[deploy_babysitter] GUARD: skipping destructive deploy. Set FORCE_REDEPLOY=1 to override."
    exit 0
  fi
fi

# ── Kill existing processes ───────────────────────────────────────────────────
echo "Killing existing processes..."
tmux kill-server 2>/dev/null || true
pkill -9 -f babysitter.py 2>/dev/null || true
pkill -9 -f train_tpu.py 2>/dev/null || true
# Kill training python (but not system python)
for pid in $(pgrep -f 'python3.*run_tpu\|python3.*train_tpu\|python3.*babysitter' 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done
# Kill the flock wrapper (holds /tmp/tpu_babysitter.lock even after babysitter.py dies)
pkill -9 -f 'flock.*tpu_babysitter' 2>/dev/null || true
# Remove stale lock file so flock --timeout=30 doesn't wait unnecessarily
rm -f /tmp/tpu_babysitter.lock
sleep 3

report_phase "RELEASING_DEVICES"

# Release TPU device locks
for dev in /dev/vfio/[0-9]* /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 2
for dev in /dev/vfio/[0-9]* /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
done
sleep 2

# ── Package installation ──────────────────────────────────────────────────────
report_phase "CHECKING_PACKAGES"

# Detect VM type from devices
# v4: /dev/accel[0-9]*, v6e: /dev/vfio/devices/vfio*, v5e: /dev/vfio/[0-9]* (no devices/ subdir)
IS_V4=false
IS_V6E=false
IS_V5E=false
if ls /dev/accel[0-9]* >/dev/null 2>&1; then
    IS_V4=true
    echo "Detected: v4 VM (/dev/accel devices)"
elif ls /dev/vfio/devices/vfio* >/dev/null 2>&1; then
    # v6e has /dev/vfio/devices/ subdirectory — use zone to distinguish v6e vs legacy v5e
    case "${ZONE:-}" in
        europe-west4-a*|us-east1*) IS_V6E=true; echo "Detected: v6e VM" ;;
        europe-west4-b*|us-central1*) IS_V5E=true; echo "Detected: v5e VM" ;;
        *) IS_V6E=true; echo "Detected: v6e/v5e VM (zone unknown)" ;;
    esac
elif ls /dev/vfio/[0-9]* >/dev/null 2>&1; then
    # v5e litepod: /dev/vfio/0, /dev/vfio/1, etc. (no devices/ subdir)
    IS_V5E=true
    echo "Detected: v5e VM (/dev/vfio/[0-9]* devices)"
fi

# For v4 VMs: install torch+torch_xla from GCS tpu_core wheels if missing
if $IS_V4 && ! python3 -c "import torch" 2>/dev/null; then
    report_phase "INSTALLING_V4_TORCH"
    echo "v4 VM: torch missing — installing from tpu_core wheels..."
    mkdir -p /tmp/tpu_core
    gsutil -m cp "gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || true
    pip install /tmp/tpu_core/libtpu-*.whl --no-deps 2>/dev/null || true
    pip install /tmp/tpu_core/torch-*.whl --no-deps 2>/dev/null || true
    pip install /tmp/tpu_core/torch_xla-*.whl --no-deps 2>/dev/null || true
    python3 -c "import torch" 2>/dev/null && echo "v4 torch OK" || echo "ERROR: v4 torch still missing"
fi

# For v6e/v5e VMs: torch+torch_xla should be pre-installed at system level.
# First clear any user-local shadow installs, then check if system torch is visible.
# Only attempt pip install if clearing the shadow wasn't sufficient.
if ($IS_V6E || $IS_V5E) && ! python3 -c "import torch; import torch_xla" 2>/dev/null; then
    echo "v6e/v5e VM: torch/torch_xla not importable — clearing user-local shadow..."
    # IMPORTANT: only remove user-local installs, not system torch
    rm -rf ~/.local/lib/python3.10/site-packages/torch* 2>/dev/null || true
    rm -rf ~/.local/lib/python3.10/site-packages/nvidia* 2>/dev/null || true
    rm -rf ~/.local/lib/python3.10/site-packages/triton* 2>/dev/null || true
    rm -rf ~/.local/lib/python3.10/site-packages/libtpu* 2>/dev/null || true
    rm -rf ~/.cache/pip 2>/dev/null || true

    # Re-check: clearing the shadow may be enough if system torch is intact
    if python3 -c "import torch; import torch_xla" 2>/dev/null; then
        echo "v6e torch OK (system install restored after clearing user shadow)"
    else
        report_phase "INSTALLING_V6E_TORCH"
        echo "v6e/v5e VM: system torch missing — installing from libtpu-releases..."
        # CRITICAL: use -f (find-links), NOT --index-url.
        # libtpu-releases is a find-links page (not PEP 503 index) — --index-url silently fails.
        # -f lets pip use both PyPI and libtpu-releases; libtpu torch wheel wins over CUDA torch
        # because it matches the platform (no CUDA on TPU hosts).
        pip install torch==2.9.0 torch_xla==2.9.0 \
            -f https://storage.googleapis.com/libtpu-releases/index.html 2>&1 | tail -5 || {
        echo "pip failed (no internet?) — trying GCS torch_tpu_wheels bundle..."
        # Fallback for no-internet zones (us-east1-d, us-central1-a)
        mkdir -p /tmp/torch_tpu_wheels
        gsutil cp "${BUCKET}/wheels/torch_tpu_wheels.tar.gz" /tmp/torch_tpu_wheels.tar.gz 2>/dev/null || \
            gsutil cp "gs://gcp-researchcredits-blocklab-us-east1/wheels/torch_tpu_wheels.tar.gz" \
                /tmp/torch_tpu_wheels.tar.gz 2>/dev/null || true
        if [ -f /tmp/torch_tpu_wheels.tar.gz ]; then
            tar xzf /tmp/torch_tpu_wheels.tar.gz -C /tmp/torch_tpu_wheels/ 2>/dev/null || true
            # Archive may unpack to a subdirectory — find wheels recursively
            # Install torch+torch_xla+libtpu (order: libtpu→torch→torch_xla)
            # Step 1: Install libtpu from GCS pkg (no wheel needed — direct copy to site-packages)
            if ! python3 -c "import libtpu" 2>/dev/null; then
                mkdir -p /tmp/libtpu_install
                gsutil cp "${BUCKET}/wheels/libtpu_pkg.tar.gz" /tmp/libtpu_pkg.tar.gz 2>/dev/null || true
                if [ -f /tmp/libtpu_pkg.tar.gz ]; then
                    SITE_PKG=$(python3 -c "import site; print(site.getusersitepackages())")
                    mkdir -p "$SITE_PKG"
                    tar xzf /tmp/libtpu_pkg.tar.gz -C "$SITE_PKG" 2>/dev/null || true
                    python3 -c "import libtpu" 2>/dev/null && echo "libtpu installed from pkg" || echo "libtpu still missing"
                fi
            fi
            # Step 2: torch and torch_xla (--no-deps to avoid CUDA deps from PyPI)
            find /tmp/torch_tpu_wheels -name 'torch-*.whl' | while read whl; do
                pip install "$whl" --no-deps 2>/dev/null || true
            done
            find /tmp/torch_tpu_wheels -name 'torch_xla*.whl' | while read whl; do
                pip install "$whl" --no-deps 2>/dev/null || true
            done
        fi
    }
        python3 -c "import torch; import torch_xla" 2>/dev/null && echo "v6e torch OK" || \
            echo "ERROR: v6e torch still missing after install attempt"
    fi  # end: system torch missing branch
fi  # end: v6e/v5e torch check

# Install training-specific packages (all VM types)
MISSING=""
python3 -c "import hydra" 2>/dev/null || MISSING="$MISSING hydra"
python3 -c "import transformers" 2>/dev/null || MISSING="$MISSING transformers"
python3 -c "import sympy" 2>/dev/null || MISSING="$MISSING sympy"
python3 -c "import antlr4" 2>/dev/null || MISSING="$MISSING antlr4"
python3 -c "import datasets" 2>/dev/null || MISSING="$MISSING datasets"
python3 -c "import wandb" 2>/dev/null || MISSING="$MISSING wandb"

if [ -n "$MISSING" ]; then
    report_phase "INSTALLING_PACKAGES"
    echo "Missing:$MISSING — installing..."
    # Try pip first (works on internet VMs)
    # antlr4-python3-runtime==4.9.3 needs setuptools<70 (newer setuptools breaks setup.py)
    pip install 'setuptools<70' 2>/dev/null || true
    pip install antlr4-python3-runtime==4.9.3 2>/dev/null || true
    pip install hydra-core omegaconf transformers sympy datasets wandb 2>/dev/null || true
    # Fallback: install from GCS wheels (no-internet VMs)
    python3 -c "import transformers" 2>/dev/null || {
        echo "pip failed — installing from GCS wheels bundle..."
        cd /tmp
        gcloud storage cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
            gsutil cp "${BUCKET}/wheels/tpu_wheels.tar.gz" \
                /tmp/all_wheels.tar.gz 2>/dev/null || true
        if [ -f /tmp/all_wheels.tar.gz ]; then
            tar xzf all_wheels.tar.gz -C /tmp/ 2>/dev/null || true
            # SAFE: skip torch/nvidia/libtpu/jax/triton wheels to avoid breaking TPU torch
            for whl in /tmp/all_wheels/*.whl; do
                [ -f "$whl" ] || continue
                name=$(basename "$whl" | tr '-' ' ' | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
                case "$name" in
                    torch*|nvidia*|libtpu*|jax*|triton*) continue ;;
                    *) pip install "$whl" --no-deps 2>/dev/null || true ;;
                esac
            done
        fi
        # Install standalone critical WHL files (hydra, omegaconf, wandb)
        # These are uploaded individually to ${BUCKET}/wheels/ for reliability
        for pkg in hydra_core omegaconf wandb; do
            python3 -c "import ${pkg//_core/}" 2>/dev/null && continue
            whl=$(gcloud storage ls "${BUCKET}/wheels/${pkg}-*.whl" 2>/dev/null | head -1 || \
                  gsutil ls "${BUCKET}/wheels/${pkg}-*.whl" 2>/dev/null | head -1 || true)
            if [ -n "$whl" ]; then
                gcloud storage cp "$whl" "/tmp/${pkg}.whl" 2>/dev/null || \
                    gsutil cp "$whl" "/tmp/${pkg}.whl" 2>/dev/null || true
                [ -f "/tmp/${pkg}.whl" ] && pip install "/tmp/${pkg}.whl" --no-deps 2>/dev/null || true
            fi
        done
    }
fi

# ── PRE-FLIGHT ASSERTION: verify environment before proceeding ────────────────
report_phase "PRE_FLIGHT_CHECK"
echo "Running pre-flight assertions..."

ASSERT_ERRORS=""
python3 -c "import torch; import torch_xla" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS torch/torch_xla"
python3 -c "import transformers" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS transformers"
python3 -c "import hydra" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS hydra"

if [ -n "$ASSERT_ERRORS" ]; then
    echo "PRE_FLIGHT FAILED: missing$ASSERT_ERRORS"
    report_phase "FAILED_ENV:$ASSERT_ERRORS"
    # Write structured error to GCS for monitoring
    printf '{"tpu_name":"%s","phase":"FAILED_ENV","missing":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "$ASSERT_ERRORS" "$(date +%s)" > /tmp/boot_state.json
    gsutil cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json" 2>/dev/null || true
    exit 1
fi
echo "Pre-flight OK: torch=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# ── Download model ────────────────────────────────────────────────────────────
MODEL_PATH=/tmp/SmolLM2-135M
if [ ! -f "$MODEL_PATH/config.json" ]; then
    report_phase "DOWNLOADING_MODEL"
    echo "Downloading SmolLM2-135M..."
    mkdir -p "$MODEL_PATH"
    gcloud storage cp -r "${BUCKET}/models/SmolLM2-135M/*" \
        "$MODEL_PATH/" 2>/dev/null || \
    gsutil -m cp -r "${BUCKET}/models/SmolLM2-135M/*" \
        "$MODEL_PATH/" 2>/dev/null || true
fi

# ── Download experiment code ──────────────────────────────────────────────────
report_phase "DOWNLOADING_CODE"
mkdir -p ~/sf_bema/experiments

# Generic code bundle — ALWAYS refresh to pick up latest train_v2_tpu.py.
# Upload as sf_bema_code.tar.gz per LAUNCH.md. See ~/distributed_tpu_training/v2/LAUNCH.md.
echo "Refreshing experiment code bundle..."
gcloud storage cp "${BUCKET}/code/sf_bema_code.tar.gz" \
    /tmp/sf_bema_code.tar.gz 2>/dev/null || \
gsutil cp "${BUCKET}/code/sf_bema_code.tar.gz" \
    /tmp/sf_bema_code.tar.gz 2>/dev/null || true
[ -f /tmp/sf_bema_code.tar.gz ] && \
    tar xzf /tmp/sf_bema_code.tar.gz -C ~/sf_bema/experiments/ 2>/dev/null || true

# ── Download training data ────────────────────────────────────────────────────
DATA_DST=$HOME/sf_bema/experiments/exp10_smollm2_smoltalk/data
if [ ! -d "${DATA_DST}/train" ]; then
    report_phase "DOWNLOADING_DATA"
    echo "Downloading training data (~1.86GB)..."
    mkdir -p "$DATA_DST"
    # Use rsync-style for idempotency
    gcloud storage cp -r "${BUCKET}/data/smoltalk/data/train" \
        "$DATA_DST/" 2>/dev/null && \
    gcloud storage cp -r "${BUCKET}/data/smoltalk/data/val" \
        "$DATA_DST/" 2>/dev/null || \
    gsutil -m cp -r "${BUCKET}/data/smoltalk/data/train" \
        "$DATA_DST/" 2>/dev/null && \
    gsutil -m cp -r "${BUCKET}/data/smoltalk/data/val" \
        "$DATA_DST/" 2>/dev/null || true
    echo "Data: $(ls $DATA_DST 2>/dev/null | wc -l) dirs"
fi

# Symlink smoltalk data into every experiment work dir that needs it.
# Data lives at exp10_smollm2_smoltalk/data (shared source of truth).
DATA_SRC=~/sf_bema/experiments/exp10_smollm2_smoltalk/data
if [ -d "$DATA_SRC" ]; then
    for _wdir in ~/sf_bema/experiments/*/; do
        _name=$(basename "$_wdir")
        [ "$_name" = "exp10_smollm2_smoltalk" ] && continue  # skip source itself
        [ "$_name" = "shared" ] && continue
        [ -e "${_wdir}data" ] && continue  # already exists
        ln -sf "$DATA_SRC" "${_wdir}data"
    done
fi

# ── Download babysitter code ──────────────────────────────────────────────────
report_phase "DOWNLOADING_PULL_CODE"
mkdir -p ~/pull_code
gcloud storage cp "gs://gcp-researchcredits-blocklab-europe-west4/pull_code/*" \
    ~/pull_code/ 2>/dev/null || \
gcloud storage cp "gs://gcp-researchcredits-blocklab-us-east1/pull_code/*" \
    ~/pull_code/ 2>/dev/null || \
gcloud storage cp "gs://gcp-researchcredits-blocklab-1-us-central2/pull_code/*" \
    ~/pull_code/ 2>/dev/null || \
gsutil -m cp "gs://gcp-researchcredits-blocklab-1-us-central2/pull_code/*" \
    ~/pull_code/ 2>/dev/null || true

# ── Download XLA cache ────────────────────────────────────────────────────────
XLA_DIR=/tmp/xla_cache
mkdir -p $XLA_DIR
if [ "$(ls $XLA_DIR 2>/dev/null | wc -l)" -lt 5 ]; then
    report_phase "DOWNLOADING_XLA_CACHE"
    echo "Downloading XLA cache..."
    if $IS_V4; then
        XLA_GCS="${BUCKET}/xla_cache_v4"
    elif $IS_V5E; then
        XLA_GCS="gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e"
    else
        XLA_GCS="${BUCKET}/xla_cache_v6e"
    fi
    echo "XLA cache source: $XLA_GCS"
    gcloud storage cp -r "${XLA_GCS}/*" $XLA_DIR/ 2>/dev/null || \
    gsutil -m cp -r "${XLA_GCS}/*" $XLA_DIR/ 2>/dev/null || true
fi

# ── Detect chips and accelerator ─────────────────────────────────────────────
# v6e uses /dev/vfio/devices/vfio*, v5e uses /dev/vfio/[0-9]*, v4 uses /dev/accel[0-9]*
if ls /dev/vfio/devices/vfio* >/dev/null 2>&1; then
    CHIPS=$(ls /dev/vfio/devices/vfio* | wc -l)
elif ls /dev/vfio/[0-9]* >/dev/null 2>&1; then
    # v5e: /dev/vfio/0, /dev/vfio/1, etc. (4 per litepod-4)
    CHIPS=$(ls /dev/vfio/[0-9]* | wc -l)
elif ls /dev/accel[0-9]* >/dev/null 2>&1; then
    CHIPS=$(ls /dev/accel[0-9]* | wc -l)
else
    echo "ERROR: No TPU devices found!"
    report_phase "FAILED_NO_DEVICES"
    exit 1
fi

ACCEL=$(curl -s --connect-timeout 5 -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type \
    2>/dev/null || echo "")
if [ -z "$ACCEL" ]; then
    $IS_V4 && ACCEL="v4-8" || ACCEL="v6e-8"
fi

echo "Zone=$ZONE, Bucket=$BUCKET, Chips=$CHIPS, Accel=$ACCEL"

# Verify devices are free
for chip in $(seq 0 $((CHIPS-1))); do
    if ls /dev/vfio/${chip} >/dev/null 2>&1; then
        BUSY=$(fuser /dev/vfio/${chip} 2>/dev/null || true)
        if [ -n "$BUSY" ]; then
            echo "Device /dev/vfio/${chip} busy (PIDs: $BUSY) — force-killing"
            fuser -k -9 /dev/vfio/${chip} 2>/dev/null || true
            sleep 1
        fi
    fi
done

# ── Launch babysitter ─────────────────────────────────────────────────────────
report_phase "LAUNCHING_BABYSITTER"
echo "Launching babysitter (${CHIPS} chips)..."

export TPU_NAME=${TPU_NAME}
export ZONE=${ZONE:-}
export CONTROL_PLANE=${CTRL}
export BUCKET=${BUCKET}
export ACCELERATOR_TYPE=${ACCEL}
export MODEL_PATH=/tmp/SmolLM2-135M
export XLA_PERSISTENT_CACHE_PATH=${XLA_DIR}
export XLA_COMPILATION_CACHE_PATH=${XLA_DIR}
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CHIPS_PER_HOST=${CHIPS}
export GCS_CHECKPOINT_DIR=${BUCKET}/checkpoints  # babysitter.py appends /{exp_name}

# Use flock to prevent duplicate babysitters.
# --timeout=30: wait up to 30s if lock is still held (e.g., flock cleanup race).
# Lock file and flock wrapper are already killed above, so this should acquire immediately.
nohup flock --timeout=30 /tmp/tpu_babysitter.lock \
    python3 -u ~/pull_code/babysitter.py >> /tmp/babysitter.log 2>&1 &
BPID=$!

echo "Launched babysitter PID=$BPID (TPU=$TPU_NAME, CHIPS=$CHIPS)"
sleep 3

if kill -0 $BPID 2>/dev/null; then
    echo "Babysitter running OK (PID=$BPID)"
    report_phase "IDLE_AWAITING_WORK"
else
    echo "ERROR: Babysitter died! Last log:"
    tail -20 /tmp/babysitter.log
    report_phase "FAILED_BABYSITTER_DIED"
    exit 1
fi
