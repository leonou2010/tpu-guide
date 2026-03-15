#!/bin/bash
# deploy_ew4a.sh — v6e europe-west4-a setup
# Type: v6e-8, zone: europe-west4-a
# Torch: pre-installed (system), internet available
# Deps: pip install from PyPI
set -uo pipefail

# ── Single-instance lock ──────────────────────────────────────────────────────
exec 9>/tmp/deploy_babysitter_running.lock
flock -xn 9 || { echo "Another deploy running — exiting"; exit 0; }

sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log \
    /tmp/boot_state.json /tmp/tpu_babysitter.lock 2>/dev/null || true
exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_ew4a.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

export HOME=${HOME:-/root}
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
_PY=python3
ZONE=${ZONE:-europe-west4-a}
BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
if [ -z "${TPU_NAME:-}" ]; then
    TPU_NAME=$(curl -sf -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || hostname)
fi
CTRL=gs://gcp-researchcredits-blocklab-europe-west4/coord_v2

report_phase() {
    local ts; ts=$(date +%s)
    printf '{"tpu_name":"%s","zone":"%s","phase":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "$ZONE" "$1" "$ts" > /tmp/boot_state.json
    gsutil cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json" 2>/dev/null || true
    echo "[$(date -u '+%H:%M:%S')] PHASE: $1"
}

# ── Guard: skip if training active ───────────────────────────────────────────
if [ -z "${FORCE_REDEPLOY:-}" ]; then
    _now=$(date +%s)
    for _hb in $(gsutil ls "${CTRL}/heartbeats/${TPU_NAME}_chip*.json" 2>/dev/null || true); do
        _age=$(gsutil cat "$_hb" 2>/dev/null | $_PY -c "import sys,json,time; d=json.load(sys.stdin); print(int(time.time()-d.get('timestamp',0)))" 2>/dev/null || echo 9999)
        _st=$(gsutil cat "$_hb" 2>/dev/null | $_PY -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
        case "$_st" in training|xla_compile|uploading)
            [ "$_age" -lt 2700 ] && { echo "GUARD: active ($TPU_NAME $_st age=${_age}s)"; exit 0; };;
        esac
    done
fi

report_phase "BOOTING"

# ── Kill processes + release devices ─────────────────────────────────────────
tmux kill-server 2>/dev/null || true
sudo pkill -9 -f babysitter.py 2>/dev/null || pkill -9 -f babysitter.py 2>/dev/null || true
sudo pkill -9 -f train_v2_tpu.py 2>/dev/null || true
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio*; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 3
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio*; do
    [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
done
sleep 2
report_phase "RELEASING_DEVICES"

# ── Torch: pre-installed on ew4a — just verify ───────────────────────────────
report_phase "INSTALLING_TORCH"
# Purge any stale user-site torch that could shadow system install
_usp="$HOME/.local/lib/python3.10/site-packages"
rm -rf "${_usp}"/torch "${_usp}"/torch-*.dist-info "${_usp}"/torchgen \
       "${_usp}"/torch_xla "${_usp}"/torch_xla-*.dist-info \
       "${_usp}"/nvidia "${_usp}"/libtpu* 2>/dev/null || true

if ! $_PY -c "import torch; import torch_xla" 2>/dev/null; then
    echo "ew4a: system torch missing — installing from PyPI..."
    sudo $_PY -m pip install torch==2.9.0 torch_xla==2.9.0 \
        --extra-index-url https://download.pytorch.org/whl/cu128 -q 2>&1 | tail -5
    $_PY -c "import torch" 2>/dev/null || { report_phase "FAILED_ENV_TORCH_EW4A"; exit 1; }
    echo "ew4a: torch installed from PyPI"
else
    echo "ew4a: torch+torch_xla OK (system install)"
fi

# ── libtpu ───────────────────────────────────────────────────────────────────
_LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
if [ -z "$_LIBTPU_SO" ] || [ ! -f "$_LIBTPU_SO" ]; then
    echo "libtpu not found — installing from GCS..."
    mkdir -p /tmp/_libtpu && gsutil cp "${BUCKET}/wheels/tpu_core/libtpu-*.whl" /tmp/_libtpu/ 2>/dev/null || true
    ls /tmp/_libtpu/libtpu-*.whl 2>/dev/null && \
        $_PY -m pip install /tmp/_libtpu/libtpu-*.whl --no-deps -q 2>/dev/null || true
    rm -rf /tmp/_libtpu
    _LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
fi
[ -n "$_LIBTPU_SO" ] && [ -f "$_LIBTPU_SO" ] && export TPU_LIBRARY_PATH="$_LIBTPU_SO" && echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"

# ── TPU init test ─────────────────────────────────────────────────────────────
report_phase "TESTING_TPU_INIT"
TPU_OUT=$($_PY -c "
import torch_xla, os, sys
p = os.environ.get('PJRT_DEVICE','NOT_SET')
print('TPU_INIT_OK' if p=='TPU' else f'FAILED: PJRT_DEVICE={p}')
sys.exit(0 if p=='TPU' else 1)
" 2>&1)
echo "TPU init: $TPU_OUT"
echo "$TPU_OUT" | grep -q 'TPU_INIT_OK' || { report_phase "FAILED_ENV_TPU_INIT"; exit 1; }
echo "✓ TPU OK"

# ── Training deps ─────────────────────────────────────────────────────────────
MISSING=""
$_PY -c "import hydra" 2>/dev/null || MISSING="$MISSING hydra"
$_PY -c "import transformers" 2>/dev/null || MISSING="$MISSING transformers"
$_PY -c "import datasets" 2>/dev/null || MISSING="$MISSING datasets"
$_PY -c "import wandb" 2>/dev/null || MISSING="$MISSING wandb"
$_PY -c "import antlr4" 2>/dev/null || MISSING="$MISSING antlr4"

if [ -n "$MISSING" ]; then
    report_phase "INSTALLING_PACKAGES"
    echo "Missing:$MISSING"
    $_PY -m pip install 'setuptools<70' -q 2>/dev/null || true
    $_PY -m pip install antlr4-python3-runtime==4.9.3 -q 2>/dev/null || true
    $_PY -m pip install hydra-core omegaconf transformers datasets wandb -q 2>&1 | tail -3
fi

# ── Pre-flight check ──────────────────────────────────────────────────────────
report_phase "PRE_FLIGHT_CHECK"
ERR=""
$_PY -c "import torch, torch_xla" 2>/dev/null || ERR="$ERR torch"
$_PY -c "import transformers" 2>/dev/null || ERR="$ERR transformers"
$_PY -c "import hydra" 2>/dev/null || ERR="$ERR hydra"
[ -n "$ERR" ] && { echo "FATAL: missing$ERR"; report_phase "FAILED_ENV_PREFLIGHT"; exit 1; }
echo "✓ All deps OK"

# ── Download code + model ─────────────────────────────────────────────────────
mkdir -p ~/pull_code
gcloud storage cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
    gsutil -m cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || true
chmod +x ~/pull_code/babysitter.py ~/pull_code/gcs.py 2>/dev/null || true

# Model files
if [ ! -f /tmp/SmolLM2-135M/config.json ]; then
    mkdir -p /tmp/SmolLM2-135M
    gsutil -m cp "${BUCKET}/models/SmolLM2-135M/*" /tmp/SmolLM2-135M/ 2>/dev/null || true
fi

# Download experiment code (sf_bema) — required for train_tpu.py and smoke train
if [ ! -d "$HOME/sf_bema/experiments/exp13_smollm2_smoltalk" ]; then
    echo "Downloading sf_bema experiment code..."
    mkdir -p "$HOME/sf_bema"
    gsutil cp "${BUCKET}/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || \
        gsutil cp "gs://gcp-researchcredits-blocklab-us-east1/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || true
    [ -f /tmp/sf_bema_code.tar.gz ] && tar xzf /tmp/sf_bema_code.tar.gz -C "$HOME/sf_bema/" && \
        echo "sf_bema code extracted OK" || echo "WARNING: sf_bema code download failed"
fi

# Fix broken /scratch symlink and ensure data exists
_DATA_DIR="$HOME/sf_bema/experiments/exp13_smollm2_smoltalk/data"
if [ ! -d "${_DATA_DIR}/train" ]; then
    [ -L "${_DATA_DIR}" ] && rm -f "${_DATA_DIR}"
    mkdir -p "${_DATA_DIR}/train" "${_DATA_DIR}/val"
    gsutil -m cp "${BUCKET}/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || \
        gsutil -m cp "gs://gcp-researchcredits-blocklab-us-east1/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || true
    gsutil -m cp "${BUCKET}/data/smoltalk/data/val/"'*' "${_DATA_DIR}/val/" 2>/dev/null || \
        gsutil -m cp "gs://gcp-researchcredits-blocklab-us-east1/data/smoltalk/data/val/"'*' "${_DATA_DIR}/val/" 2>/dev/null || true
    echo "Data: $(ls ${_DATA_DIR}/train/ 2>/dev/null | wc -l) train files, $(ls ${_DATA_DIR}/val/ 2>/dev/null | wc -l) val files"
fi

# Training script
[ ! -f /tmp/train_v2_tpu.py ] && \
    gsutil cp "${BUCKET}/pull_code_v3/train_v2_tpu.py" /tmp/train_v2_tpu.py 2>/dev/null || true

# XLA cache — seed from GCS so chips don't recompile on every redeploy
_XLA_CACHE=/tmp/xla_cache
_XLA_FILES=$(ls "$_XLA_CACHE/" 2>/dev/null | wc -l)
if [ "$_XLA_FILES" -lt 10 ]; then
    echo "[$(date -u '+%H:%M:%S')] Seeding XLA cache from GCS..."
    mkdir -p "$_XLA_CACHE"
    # ew4a bucket only has 1 cache file — use us-east1 (41 files, same v6e hardware)
    gsutil -o GSUtil:check_hashes=never -m cp "${BUCKET}/xla_cache_v6e/*" "$_XLA_CACHE/" 2>/dev/null
    if [ "$(ls "$_XLA_CACHE/" 2>/dev/null | wc -l)" -lt 10 ]; then
        gsutil -o GSUtil:check_hashes=never -m cp "gs://gcp-researchcredits-blocklab-us-east1/xla_cache_v6e/*" "$_XLA_CACHE/" 2>/dev/null || true
    fi
    echo "[$(date -u '+%H:%M:%S')] XLA cache: $(ls "$_XLA_CACHE" 2>/dev/null | wc -l) files"
fi

# ── Launch babysitter ─────────────────────────────────────────────────────────
report_phase "LAUNCHING_BABYSITTER"
EXP=${EXP:-exp13_rerun3}
nohup $_PY -u ~/pull_code/babysitter.py \
    --tpu-name "$TPU_NAME" \
    --zone "$ZONE" \
    --bucket "$BUCKET" \
    --exp "$EXP" \
    >> /tmp/babysitter.log 2>&1 &
echo "Babysitter PID: $!"

sleep 5
pgrep -f babysitter.py > /dev/null || { echo "FATAL: babysitter not running"; report_phase "FAILED_BABYSITTER_DIED"; exit 1; }
report_phase "IDLE_AWAITING_WORK"
echo "✓ ew4a deploy complete — babysitter running"
