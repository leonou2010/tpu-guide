#!/bin/bash
# deploy_v5e.sh — v5e setup (europe-west4-b and us-central1-a)
# Type: v5litepod-8 (8 chips, /dev/accel[0-7]), zones: europe-west4-b, us-central1-a
# Torch: NOT pre-installed — install from GCS tpu_core wheels (libtpu, torch, torch_xla)
# gsutil: BROKEN on v5e — use gcloud storage for GCS downloads
# numpy: apt-get install python3-numpy (or manylinux wheel)
# CUDA stubs: required (torch is CUDA build but no CUDA hardware)
# Internet: NO — all installs from GCS or apt
# batch_size=4 only (OOM at bs=8, 7.78GB HBM used)
set -uo pipefail

# ── Single-instance lock ──────────────────────────────────────────────────────
exec 9>/tmp/deploy_babysitter_running.lock
flock -xn 9 || { echo "Another deploy running — exiting"; exit 0; }

sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log \
    /tmp/boot_state.json /tmp/tpu_babysitter.lock 2>/dev/null || true
exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_v5e.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

export HOME=${HOME:-/root}
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
_PY=python3

# Auto-detect ZONE
if [ -z "${ZONE:-}" ]; then
    ZONE=$(curl -sf --connect-timeout 5 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
        | awk -F/ '{print $NF}' || true)
fi

# Pick bucket based on zone
case "${ZONE:-}" in
    europe-west4-b)
        BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
        ;;
    us-central1-a)
        BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-1-us-central2}
        ;;
    *)
        BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}
        ;;
esac
_EW4A_BUCKET=gs://gcp-researchcredits-blocklab-europe-west4
_UE1D_BUCKET=gs://gcp-researchcredits-blocklab-us-east1

if [ -z "${TPU_NAME:-}" ]; then
    TPU_NAME=$(curl -sf -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || hostname)
fi
CTRL=gs://gcp-researchcredits-blocklab-europe-west4/coord_v2

# ── GCS helper: gcloud storage preferred (gsutil broken on v5e) ───────────────
gcs_cp() {
    # Usage: gcs_cp <src> <dst>
    gcloud storage cp "$1" "$2" 2>/dev/null || gsutil cp "$1" "$2" 2>/dev/null || true
}
gcs_cp_m() {
    # Usage: gcs_cp_m <src_glob> <dst_dir>
    gcloud storage cp "$1" "$2" 2>/dev/null || gsutil -m cp "$1" "$2" 2>/dev/null || true
}

report_phase() {
    local ts; ts=$(date +%s)
    printf '{"tpu_name":"%s","zone":"%s","phase":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "$ZONE" "$1" "$ts" > /tmp/boot_state.json
    gcs_cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json"
    echo "[$(date -u '+%H:%M:%S')] PHASE: $1"
}

# ── Guard: skip if training active ───────────────────────────────────────────
if [ -z "${FORCE_REDEPLOY:-}" ]; then
    for _hb in $(gcloud storage ls "${CTRL}/heartbeats/${TPU_NAME}_chip*.json" 2>/dev/null || \
                 gsutil ls "${CTRL}/heartbeats/${TPU_NAME}_chip*.json" 2>/dev/null || true); do
        _age=$(gcs_cp "$_hb" - 2>/dev/null | $_PY -c "import sys,json,time; d=json.load(sys.stdin); print(int(time.time()-d.get('timestamp',0)))" 2>/dev/null || echo 9999)
        _st=$(gcs_cp "$_hb" - 2>/dev/null | $_PY -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
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
for dev in /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 3
for dev in /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
done
sleep 2
report_phase "RELEASING_DEVICES"

# ── Torch: install from GCS tpu_core wheels ──────────────────────────────────
report_phase "INSTALLING_TORCH"

# Purge any wrong torch (CUDA or user-site)
_usp="$HOME/.local/lib/python3.10/site-packages"
rm -rf "${_usp}"/torch "${_usp}"/torch-*.dist-info "${_usp}"/torchgen \
       "${_usp}"/torch_xla "${_usp}"/torch_xla-*.dist-info \
       "${_usp}"/nvidia "${_usp}"/libtpu* 2>/dev/null || true

# Check if valid TPU torch already present
_torch_ok=false
if $_PY -c "import torch_xla; exit(0 if torch_xla._found_libtpu else 1)" 2>/dev/null; then
    _torch_ok=true
fi

if ! $_torch_ok; then
    echo "v5e: downloading tpu_core wheels from GCS..."
    rm -rf /tmp/tpu_core && mkdir -p /tmp/tpu_core
    gcs_cp_m "${BUCKET}/wheels/tpu_core/*.whl" /tmp/tpu_core/ || \
        gcs_cp_m "${_EW4A_BUCKET}/wheels/tpu_core/*.whl" /tmp/tpu_core/ || true
    ls /tmp/tpu_core/*.whl 2>/dev/null | grep -q . || {
        echo "ERROR: tpu_core wheels not found in GCS"
        report_phase "FAILED_ENV_TPU_CORE_WHEELS_MISSING"; exit 1
    }
    # Install system-wide (sudo). Order: libtpu → torch → torch_xla.
    sudo $_PY -m pip install /tmp/tpu_core/libtpu-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "libtpu OK" || echo "WARNING: libtpu failed"
    sudo $_PY -m pip install /tmp/tpu_core/torch-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch OK" || echo "WARNING: torch failed"
    sudo $_PY -m pip install /tmp/tpu_core/torch_xla-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch_xla OK" || echo "WARNING: torch_xla failed"
    rm -rf /tmp/tpu_core

    # Install torch runtime deps
    echo "v5e: installing numpy..."
    sudo apt-get install -y python3-numpy 2>/dev/null && echo "numpy OK via apt" || {
        echo "WARNING: apt numpy failed, trying manylinux wheel..."
        _npwhl=/tmp/numpy_whl
        mkdir -p "$_npwhl"
        gcs_cp "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/numpy-*.whl" "$_npwhl/" || true
        for _w in "$_npwhl"/numpy-*.whl; do
            [ -f "$_w" ] && sudo $_PY -m pip install "$_w" --no-deps -q 2>/dev/null && echo "numpy OK via wheel" && break || true
        done
        rm -rf "$_npwhl"
    }

    echo "v5e: installing filelock/sympy/jinja2/networkx from GCS wheels..."
    # networkx-3.4.2: supports Python>=3.10. networkx-3.6.1 requires Python>=3.11 (incompatible).
    _tdep=/tmp/torch_deps_v5e
    rm -rf "$_tdep" && mkdir -p "$_tdep"
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/filelock-*.whl" "$_tdep/" || true
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/typing_extensions-*.whl" "$_tdep/" || true
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/sympy-*.whl" "$_tdep/" || true
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/mpmath-*.whl" "$_tdep/" || true
    gcs_cp "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/networkx-3.4.2-py3-none-any.whl" "$_tdep/networkx-3.4.2-py3-none-any.whl" || true
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/jinja2-*.whl" "$_tdep/" || true
    gcs_cp_m "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/markupsafe-*.whl" "$_tdep/" || true
    for _w in "$_tdep"/*.whl; do
        [ -f "$_w" ] && $_PY -m pip install "$_w" --no-deps -q 2>&1 | grep -v "already installed" || true
    done
    rm -rf "$_tdep"
    echo "v5e: torch runtime deps installed"
else
    echo "v5e: torch+torch_xla already OK"
fi

# ── CUDA stubs ────────────────────────────────────────────────────────────────
echo "v5e: installing CUDA stubs..."
_nv=/usr/local/lib/python3.10/dist-packages/nvidia
sudo mkdir -p "$_nv" && sudo touch "$_nv/__init__.py" 2>/dev/null || true
_stgz=/tmp/nvidia_stubs_v6e.tar.gz
gcs_cp "${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" || \
    gcs_cp "${_EW4A_BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" || true
if [ -f "$_stgz" ]; then
    sudo tar xzf "$_stgz" -C /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true
    rm -f "$_stgz"
    echo "v5e: CUDA stubs installed"
else
    echo "WARNING: CUDA stubs tarball not found — building inline..."
    _mk() { local p=$1 l=$2 d="${_nv}/${p}/lib"; sudo mkdir -p "$d" && sudo touch "${_nv}/${p}/__init__.py" 2>/dev/null || true
        [ -f "$d/$l" ] || { printf 'void _stub(void){}' | sudo tee /tmp/stub.c >/dev/null && sudo gcc -shared -fPIC -Wl,-soname,"$l" -o "$d/$l" /tmp/stub.c 2>/dev/null && echo "stub: $l" || true; }; }
    _mk cublas libcublas.so.12; _mk cublas libcublasLt.so.12; _mk cuda_runtime libcudart.so.12
    _mk cudnn libcudnn.so.9; _mk cufft libcufft.so.11; _mk curand libcurand.so.10
    _mk cusolver libcusolver.so.11; _mk cusparse libcusparse.so.12; _mk nccl libnccl.so.2
    _mk nvjitlink libnvJitLink.so.12; _mk nvshmem libnvshmem_host.so.3; _mk nvtx libnvToolsExt.so.1
fi

# ── libtpu ────────────────────────────────────────────────────────────────────
_LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
if [ -n "$_LIBTPU_SO" ] && [ -f "$_LIBTPU_SO" ]; then
    export TPU_LIBRARY_PATH="$_LIBTPU_SO"
    echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
else
    echo "WARNING: libtpu.so path not found — may affect PJRT"
fi

# ── TPU init test ─────────────────────────────────────────────────────────────
report_phase "TESTING_TPU_INIT"
TPU_OUT=$($_PY -c "
import numpy as np
import torch_xla, os, sys
p = os.environ.get('PJRT_DEVICE','NOT_SET')
print('TPU_INIT_OK' if p=='TPU' else f'FAILED: PJRT_DEVICE={p}')
sys.exit(0 if p=='TPU' else 1)
" 2>&1)
echo "TPU init: $TPU_OUT"
echo "$TPU_OUT" | grep -q 'TPU_INIT_OK' || { report_phase "FAILED_ENV_TPU_INIT"; exit 1; }
echo "✓ TPU OK"

# ── Training deps (from GCS wheels — no internet) ────────────────────────────
MISSING=""
$_PY -c "import hydra" 2>/dev/null || MISSING="$MISSING hydra"
$_PY -c "import transformers" 2>/dev/null || MISSING="$MISSING transformers"
$_PY -c "import datasets" 2>/dev/null || MISSING="$MISSING datasets"
$_PY -c "import wandb" 2>/dev/null || MISSING="$MISSING wandb"
$_PY -c "import antlr4" 2>/dev/null || MISSING="$MISSING antlr4"

if [ -n "$MISSING" ]; then
    report_phase "INSTALLING_PACKAGES"
    echo "Missing:$MISSING — installing from GCS wheels..."
    cd /tmp
    gcs_cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz || \
        gcs_cp "${BUCKET}/wheels/tpu_wheels.tar.gz" /tmp/all_wheels.tar.gz || \
        gcs_cp "${_UE1D_BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz || \
        gcs_cp "${_EW4A_BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz || true
    if [ -f /tmp/all_wheels.tar.gz ]; then
        tar xzf /tmp/all_wheels.tar.gz -C /tmp/ 2>/dev/null || true
        [ -d /tmp/all_wheels ] || mv /tmp/tpu_wheels /tmp/all_wheels 2>/dev/null || true
        for whl in /tmp/all_wheels/*.whl; do
            [ -f "$whl" ] || continue
            name=$(basename "$whl" | tr '-' ' ' | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
            case "$name" in torch*|nvidia*|libtpu*|jax*|triton*) continue ;;
            *) $_PY -m pip install "$whl" --no-deps -q 2>/dev/null || true ;;
            esac
        done
        rm -f /tmp/all_wheels.tar.gz
    fi
fi

# ── Pre-flight check ──────────────────────────────────────────────────────────
report_phase "PRE_FLIGHT_CHECK"
ERR=""
$_PY -c "import torch, torch_xla" 2>/dev/null || ERR="$ERR torch"
$_PY -c "import numpy" 2>/dev/null || ERR="$ERR numpy"
$_PY -c "import transformers" 2>/dev/null || ERR="$ERR transformers"
$_PY -c "import hydra" 2>/dev/null || ERR="$ERR hydra"
[ -n "$ERR" ] && { echo "FATAL: missing$ERR"; report_phase "FAILED_ENV_PREFLIGHT"; exit 1; }
echo "✓ All deps OK"

# ── Download code + model ─────────────────────────────────────────────────────
mkdir -p ~/pull_code
gcs_cp_m "${BUCKET}/pull_code_v3/*" ~/pull_code/ || \
    gcs_cp_m "${_EW4A_BUCKET}/pull_code_v3/*" ~/pull_code/ || true
chmod +x ~/pull_code/babysitter.py ~/pull_code/gcs.py 2>/dev/null || true

if [ ! -f /tmp/SmolLM2-135M/config.json ]; then
    mkdir -p /tmp/SmolLM2-135M
    gcs_cp_m "${BUCKET}/models/SmolLM2-135M/*" /tmp/SmolLM2-135M/ || \
        gcs_cp_m "${_EW4A_BUCKET}/models/SmolLM2-135M/*" /tmp/SmolLM2-135M/ || true
fi
[ ! -f /tmp/train_v2_tpu.py ] && \
    gcs_cp "${BUCKET}/pull_code_v3/train_v2_tpu.py" /tmp/train_v2_tpu.py || true

# ── Launch babysitter ─────────────────────────────────────────────────────────
# v5e-8: CHIPS_PER_HOST=8, batch_size=4 (OOM at bs=8)
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
echo "✓ v5e deploy complete — babysitter running"
