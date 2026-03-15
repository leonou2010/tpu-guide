#!/bin/bash
# deploy_uc2b.sh — v4 us-central2-b setup
# Type: v4-8 (4 chips, /dev/accel[0-3]), zone: us-central2-b
# Torch: NOT pre-installed — install from GCS tpu_core wheels (libtpu, torch, torch_xla)
# numpy: apt-get install python3-numpy (pip wheel install silently fails on tpu-ubuntu2204-base)
# CUDA stubs: required (torch is CUDA build but no CUDA hardware — stubs satisfy dlopen)
# Internet: NO — all installs from GCS or apt
set -uo pipefail

# ── Single-instance lock ──────────────────────────────────────────────────────
exec 9>/tmp/deploy_babysitter_running.lock
flock -xn 9 || { echo "Another deploy running — exiting"; exit 0; }

sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log \
    /tmp/boot_state.json /tmp/tpu_babysitter.lock 2>/dev/null || true
exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_uc2b.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

export HOME=${HOME:-/root}
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
_PY=python3
ZONE=${ZONE:-us-central2-b}
BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-1-us-central2}
_EW4A_BUCKET=gs://gcp-researchcredits-blocklab-europe-west4
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
    echo "v4: downloading tpu_core wheels from GCS..."
    rm -rf /tmp/tpu_core && mkdir -p /tmp/tpu_core
    gsutil -m cp "${BUCKET}/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || true
    ls /tmp/tpu_core/*.whl 2>/dev/null | grep -q . || {
        echo "ERROR: tpu_core wheels not found in GCS"
        report_phase "FAILED_ENV_TPU_CORE_WHEELS_MISSING"; exit 1
    }
    # Install system-wide (sudo). Order: libtpu → torch → torch_xla.
    # --no-deps to avoid pulling in CUDA/nvidia packages from torch's deps.
    sudo $_PY -m pip install /tmp/tpu_core/libtpu-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "libtpu OK" || echo "WARNING: libtpu failed"
    sudo $_PY -m pip install /tmp/tpu_core/torch-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch OK" || echo "WARNING: torch failed"
    sudo $_PY -m pip install /tmp/tpu_core/torch_xla-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch_xla OK" || echo "WARNING: torch_xla failed"
    rm -rf /tmp/tpu_core

    # torch was installed --no-deps so its runtime deps are missing.
    # numpy: use apt-get (pip wheel install silently fails on tpu-ubuntu2204-base).
    # Other deps: pip install from GCS v6e wheels (filelock, sympy, etc.)
    echo "v4: installing numpy+filelock/sympy/jinja2/networkx from GCS wheels..."
    # apt-get install python3-numpy fails on no-internet VMs — use GCS wheel instead.
    # IMPORTANT: preserve full wheel filename (pip rejects short names like "networkx.whl")
    # networkx-3.4.2: supports Python>=3.10. networkx-3.6.1 requires Python>=3.11 (incompatible).
    _tdep=/tmp/torch_deps_v4
    rm -rf "$_tdep" && mkdir -p "$_tdep"
    gsutil -m cp \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/numpy-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/filelock-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/typing_extensions-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/sympy-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/mpmath-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/networkx-3.4.2-py3-none-any.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/jinja2-*.whl" \
        "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/markupsafe-*.whl" \
        "$_tdep/" 2>/dev/null || true
    for _w in "$_tdep"/*.whl; do
        [ -f "$_w" ] && $_PY -m pip install "$_w" --no-deps -q 2>&1 | grep -v "already installed" || true
    done
    rm -rf "$_tdep"
    echo "v4: torch runtime deps installed"
else
    echo "v4: torch+torch_xla already OK"
fi

# ── CUDA stubs ────────────────────────────────────────────────────────────────
# tpu_core torch is a CUDA build — dlopen(libcublas.so.12) at import time.
# v4 has no CUDA. Pre-built stubs from GCS satisfy the dlopen calls.
echo "v4: installing CUDA stubs..."
_nv=/usr/local/lib/python3.10/dist-packages/nvidia
sudo mkdir -p "$_nv" && sudo touch "$_nv/__init__.py" 2>/dev/null || true
_stgz=/tmp/nvidia_stubs_v6e.tar.gz
gsutil cp "${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" 2>/dev/null || \
    gsutil cp "${_EW4A_BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" 2>/dev/null || true
if [ -f "$_stgz" ]; then
    sudo tar xzf "$_stgz" -C /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true
    rm -f "$_stgz"
    echo "v4: CUDA stubs installed"
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
import numpy as np  # test numpy too
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
    rm -f /tmp/all_wheels.tar.gz  # clear any stale/corrupt file from prior run
    echo "Downloading all_wheels.tar.gz..."
    # IMPORTANT: all_wheels.tar.gz may be a composite GCS object (requires CRC32c).
    # crcmod C extension not installed on tpu-ubuntu2204-base Python 3.10.
    # Fix: -o GSUtil:check_hashes=never bypasses CRC32c requirement.
    gsutil -o GSUtil:check_hashes=never cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>&1 | tail -3 || \
        gsutil -o GSUtil:check_hashes=never cp "${BUCKET}/wheels/tpu_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>&1 | tail -3 || \
        gsutil -o GSUtil:check_hashes=never cp "gs://gcp-researchcredits-blocklab-us-east1/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>&1 | tail -3 || \
        gsutil -o GSUtil:check_hashes=never cp "${_EW4A_BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>&1 | tail -3 || true
    echo "Download result: $(ls -lh /tmp/all_wheels.tar.gz 2>/dev/null || echo 'FILE MISSING')"
    _WHEELS_SIZE=$(stat -c%s /tmp/all_wheels.tar.gz 2>/dev/null || echo 0)
    if [ "$_WHEELS_SIZE" -gt 1000000 ]; then
        tar xzf /tmp/all_wheels.tar.gz -C /tmp/ 2>/dev/null || true
        # Handle both all_wheels/ and tpu_wheels/ directory names in tarball
        [ -d /tmp/all_wheels ] || mv /tmp/tpu_wheels /tmp/all_wheels 2>/dev/null || true
        for whl in /tmp/all_wheels/*.whl; do
            [ -f "$whl" ] || continue
            name=$(basename "$whl" | tr '-' ' ' | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
            case "$name" in torch*|nvidia*|libtpu*|jax*|triton*) continue ;;
            *) $_PY -m pip install "$whl" --no-deps -q 2>/dev/null || true ;;
            esac
        done
        # antlr4-python3-runtime is a .tar.gz in the bundle (not .whl) — install separately
        if ! $_PY -c "import antlr4" 2>/dev/null; then
            if [ -f /tmp/all_wheels/antlr4-python3-runtime-4.9.3.tar.gz ]; then
                $_PY -m pip install /tmp/all_wheels/antlr4-python3-runtime-4.9.3.tar.gz --no-deps -q 2>/dev/null || true
            else
                # Fallback: download from uc2b or us-east1 bucket
                gsutil cp "${BUCKET}/wheels/antlr4-python3-runtime-4.9.3.tar.gz" /tmp/_antlr4.tar.gz 2>/dev/null || \
                    gsutil cp "gs://gcp-researchcredits-blocklab-us-east1/wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl" /tmp/_antlr4.tar.gz 2>/dev/null || true
                [ -f /tmp/_antlr4.tar.gz ] && $_PY -m pip install /tmp/_antlr4.tar.gz --no-deps -q 2>/dev/null || true
            fi
            echo "antlr4 install: $($_PY -c 'import antlr4; print("OK")' 2>/dev/null || echo 'FAIL')"
        fi
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
gsutil -m cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
    gsutil -m cp "${_EW4A_BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || true
chmod +x ~/pull_code/babysitter.py ~/pull_code/gcs.py 2>/dev/null || true

# Download experiment code (sf_bema) — required for train_tpu.py
if [ ! -d "$HOME/sf_bema/experiments/exp13_smollm2_smoltalk" ]; then
    echo "Downloading sf_bema experiment code..."
    mkdir -p "$HOME/sf_bema"
    gsutil cp "${BUCKET}/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || \
        gsutil cp "${_EW4A_BUCKET}/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || true
    [ -f /tmp/sf_bema_code.tar.gz ] && tar xzf /tmp/sf_bema_code.tar.gz -C "$HOME/sf_bema/" && \
        echo "sf_bema code extracted OK" || echo "WARNING: sf_bema code download failed"
fi

# Download training data — required at runtime
_DATA_DIR="$HOME/sf_bema/experiments/exp13_smollm2_smoltalk/data"
if [ ! -d "${_DATA_DIR}/train" ]; then
    echo "Downloading training data (~1.9GB)..."
    # Remove broken symlink if present (sf_bema tarball contains a /scratch symlink)
    [ -L "${_DATA_DIR}" ] && rm -f "${_DATA_DIR}"
    mkdir -p "${_DATA_DIR}/train" "${_DATA_DIR}/val"
    gsutil -m cp "${BUCKET}/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || true
    gsutil -m cp "${BUCKET}/data/smoltalk/data/val/"'*' "${_DATA_DIR}/val/" 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/data/smoltalk/data/val/"'*' "${_DATA_DIR}/val/" 2>/dev/null || true
    echo "Data: $(ls ${_DATA_DIR}/train/ 2>/dev/null | wc -l) train files, $(ls ${_DATA_DIR}/val/ 2>/dev/null | wc -l) val files"
fi
_train_count=$(ls ${_DATA_DIR}/train/ 2>/dev/null | wc -l)
[ "$_train_count" -eq 0 ] && { echo "FATAL: training data empty after download"; report_phase "FAILED_NO_DATA"; exit 1; }
echo "✓ Data verified: $_train_count train files"

if [ ! -f /tmp/SmolLM2-135M/config.json ]; then
    mkdir -p /tmp/SmolLM2-135M
    gsutil -m cp "${BUCKET}/models/SmolLM2-135M/*" /tmp/SmolLM2-135M/ 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/models/SmolLM2-135M/*" /tmp/SmolLM2-135M/ 2>/dev/null || true
fi
[ ! -f /tmp/train_v2_tpu.py ] && \
    gsutil cp "${BUCKET}/pull_code_v3/train_v2_tpu.py" /tmp/train_v2_tpu.py 2>/dev/null || true

# ── Download XLA cache ────────────────────────────────────────────────────────
_XLA_CACHE=/tmp/xla_cache
if [ ! -d "$_XLA_CACHE" ] || [ "$(ls "$_XLA_CACHE" 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "Downloading XLA cache from GCS..."
    mkdir -p "$_XLA_CACHE"
    gsutil -m cp "${BUCKET}/xla_cache_v4/*" "$_XLA_CACHE/" 2>/dev/null && \
        echo "XLA cache: $(ls $_XLA_CACHE | wc -l) files" || \
        echo "WARNING: XLA cache download failed — will recompile"
else
    echo "XLA cache already present: $(ls $_XLA_CACHE | wc -l) files"
fi

# ── Launch babysitter ─────────────────────────────────────────────────────────
report_phase "LAUNCHING_BABYSITTER"
EXP=${EXP:-exp13_rerun4}
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
echo "✓ uc2b deploy complete — babysitter running"
