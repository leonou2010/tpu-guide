#!/bin/bash
# deploy_ue1d.sh — v6e us-east1-d setup
# Type: v6e-8, zone: us-east1-d
# Torch: NOT pre-installed — install from GCS torch_v6e_cp310 wheels (sudo, system-wide)
# CUDA stubs: required (no CUDA hardware, torch needs libcublas etc.)
# Internet: NO — all installs from GCS
set -uo pipefail

# ── Single-instance lock ──────────────────────────────────────────────────────
exec 9>/tmp/deploy_babysitter_running.lock
flock -xn 9 || { echo "Another deploy running — exiting"; exit 0; }

sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log \
    /tmp/boot_state.json /tmp/tpu_babysitter.lock 2>/dev/null || true
exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_ue1d.sh started $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

export HOME=${HOME:-/root}
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
_PY=python3
ZONE=${ZONE:-us-east1-d}
BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-us-east1}
_UE1D_BUCKET=gs://gcp-researchcredits-blocklab-us-east1
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
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio*; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 3
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio*; do
    [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
done
sleep 2
report_phase "RELEASING_DEVICES"

# ── Torch: install from GCS wheels (system-wide via sudo) ───────────────────
report_phase "INSTALLING_TORCH"

# Purge any stale user-site torch first
_usp="$HOME/.local/lib/python3.10/site-packages"
rm -rf "${_usp}"/torch "${_usp}"/torch-*.dist-info "${_usp}"/torchgen \
       "${_usp}"/torch_xla "${_usp}"/torch_xla-*.dist-info \
       "${_usp}"/nvidia "${_usp}"/libtpu* 2>/dev/null || true

# Check if valid TPU torch already present (not CUDA torch)
_torch_ok=false
if PYTHONNOUSERSITE=1 $_PY -c "import torch, torch_xla" 2>/dev/null; then
    _ver=$(PYTHONNOUSERSITE=1 $_PY -c "import torch; print(torch.__version__)" 2>/dev/null || true)
    echo "$_ver" | grep -q '+cu' || _torch_ok=true
fi

if ! $_torch_ok; then
    echo "ue1d: installing torch+torch_xla from GCS torch_v6e_cp310 wheels..."
    _whl=/tmp/torch_v6e_wheels
    rm -rf "$_whl" && mkdir -p "$_whl"
    gsutil -m cp "${BUCKET}/wheels/torch_v6e_cp310/*.whl" "$_whl/" 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/wheels/torch_v6e_cp310/*.whl" "$_whl/" 2>/dev/null || true
    ls "$_whl"/torch-*.whl 2>/dev/null | head -1 | grep -q . || {
        echo "ERROR: torch_v6e_cp310 wheels not found in GCS"
        report_phase "FAILED_ENV_TORCH_WHEELS_MISSING"; exit 1
    }
    # Install deps one-by-one (skip failures gracefully)
    # networkx: pin to 3.4.2 (3.6.1 requires Python>=3.11; ue1d runs Python 3.10)
    # numpy: required by torch_xla xla_sharding.py (not pre-installed on v6e runtime image)
    for _w in filelock typing_extensions sympy mpmath jinja2 markupsafe fsspec; do
        _f=$(ls "$_whl"/${_w}-*.whl 2>/dev/null | head -1)
        [ -f "$_f" ] && sudo $_PY -m pip install "$_f" --no-deps -q 2>/dev/null || true
    done
    # networkx: explicitly use 3.4.2 (not 3.6.1)
    _nx=$(ls "$_whl"/networkx-3.4.2-py3-none-any.whl 2>/dev/null | head -1)
    [ -f "$_nx" ] && sudo $_PY -m pip install "$_nx" --no-deps -q 2>/dev/null || \
        echo "WARNING: networkx-3.4.2 not found in wheel bundle"
    # numpy: install system-wide so all users see it
    _np=$(ls "$_whl"/numpy-*.whl 2>/dev/null | head -1)
    [ -f "$_np" ] && sudo $_PY -m pip install "$_np" --no-deps -q 2>/dev/null || \
        echo "WARNING: numpy wheel not found in bundle"
    sudo $_PY -m pip install "$_whl"/torch-*.whl --no-deps -q 2>&1 | tail -2
    sudo $_PY -m pip install "$_whl"/torch_xla-*.whl --no-deps -q 2>&1 | tail -2
    $_PY -m pip install --user "$_whl"/libtpu-*.whl --no-deps -q 2>/dev/null || true
    rm -rf "$_whl"
    echo "ue1d: torch install complete"
else
    echo "ue1d: torch+torch_xla already OK"
fi

# ── CUDA stubs (ue1d has no CUDA hardware; torch needs libcublas.so.12 etc.) ─
echo "ue1d: installing CUDA stubs..."
_nv=/usr/local/lib/python3.10/dist-packages/nvidia
sudo mkdir -p "$_nv" && sudo touch "$_nv/__init__.py" 2>/dev/null || true
_stgz=/tmp/nvidia_stubs_v6e.tar.gz
gsutil cp "${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" 2>/dev/null || \
    gsutil cp "${_EW4A_BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stgz" 2>/dev/null || true
if [ -f "$_stgz" ]; then
    sudo tar xzf "$_stgz" -C /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true
    rm -f "$_stgz"
    echo "ue1d: CUDA stubs installed"
else
    echo "WARNING: CUDA stubs tarball not found — building inline..."
    _mk() { local p=$1 l=$2 d="${_nv}/${p}/lib"; sudo mkdir -p "$d" && sudo touch "${_nv}/${p}/__init__.py" 2>/dev/null || true
        [ -f "$d/$l" ] || { printf 'void _stub(void){}' | sudo tee /tmp/stub.c >/dev/null && sudo gcc -shared -fPIC -Wl,-soname,"$l" -o "$d/$l" /tmp/stub.c 2>/dev/null && echo "stub: $l" || true; }; }
    _mk cublas libcublas.so.12; _mk cublas libcublasLt.so.12; _mk cublas libnvblas.so.12
    _mk cuda_cupti libcupti.so.12; _mk cuda_nvrtc libnvrtc.so.12; _mk cuda_runtime libcudart.so.12
    _mk cudnn libcudnn.so.9; _mk cudnn libcudnn_adv.so.9; _mk cudnn libcudnn_cnn.so.9
    _mk cufft libcufft.so.11; _mk curand libcurand.so.10; _mk cusolver libcusolver.so.11
    _mk cusparse libcusparse.so.12; _mk nccl libnccl.so.2; _mk nvjitlink libnvJitLink.so.12
    _mk nvshmem libnvshmem_host.so.3; _mk nvtx libnvToolsExt.so.1
fi

# ── libtpu ────────────────────────────────────────────────────────────────────
_LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
if [ -z "$_LIBTPU_SO" ] || [ ! -f "$_LIBTPU_SO" ]; then
    echo "libtpu not found — installing from GCS..."
    mkdir -p /tmp/_libtpu
    gsutil cp "${BUCKET}/wheels/tpu_core/libtpu-*.whl" /tmp/_libtpu/ 2>/dev/null || \
        gsutil cp "${_EW4A_BUCKET}/wheels/tpu_core/libtpu-*.whl" /tmp/_libtpu/ 2>/dev/null || true
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
    echo "Downloading all_wheels.tar.gz from ${BUCKET}..."
    # IMPORTANT: all_wheels.tar.gz is a composite GCS object (requires CRC32c).
    # crcmod C extension not installed on ue1d (Python 3.10 tpu-ubuntu2204-base).
    # Fix: -o GSUtil:check_hashes=never bypasses CRC32c requirement for composite objects.
    gsutil -o GSUtil:check_hashes=never cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
        gsutil -o GSUtil:check_hashes=never cp "${_UE1D_BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
        echo "WARNING: all gsutil downloads failed"
    echo "Download result: $(ls -lh /tmp/all_wheels.tar.gz 2>/dev/null || echo 'FILE MISSING')"
    if [ -f /tmp/all_wheels.tar.gz ]; then
        echo "Extracting all_wheels.tar.gz..."
        tar xzf /tmp/all_wheels.tar.gz -C /tmp/ 2>/dev/null || true
        [ -d /tmp/all_wheels ] || mv /tmp/tpu_wheels /tmp/all_wheels 2>/dev/null || true
        echo "Wheels found: $(ls /tmp/all_wheels/*.whl 2>/dev/null | wc -l)"
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
                gsutil cp "${_UE1D_BUCKET}/wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl" /tmp/_antlr4.whl 2>/dev/null && \
                    $_PY -m pip install /tmp/_antlr4.whl --no-deps -q 2>/dev/null || true
            fi
            echo "antlr4 install: $($_PY -c 'import antlr4; print("OK")' 2>/dev/null || echo 'FAIL')"
        fi
        rm -f /tmp/all_wheels.tar.gz
    else
        echo "ERROR: all_wheels.tar.gz not downloaded — packages will be missing"
    fi
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
gsutil -m cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
    gsutil -m cp "${_EW4A_BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || true
chmod +x ~/pull_code/babysitter.py ~/pull_code/gcs.py 2>/dev/null || true

# Download experiment code (sf_bema) — required for train_tpu.py
if [ ! -d "$HOME/sf_bema/experiments/exp13_smollm2_smoltalk" ]; then
    echo "Downloading sf_bema experiment code..."
    mkdir -p "$HOME/sf_bema"
    gsutil cp "${_UE1D_BUCKET}/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || \
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
    gsutil -m cp "${_UE1D_BUCKET}/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || \
        gsutil -m cp "${_EW4A_BUCKET}/data/smoltalk/data/train/"'*' "${_DATA_DIR}/train/" 2>/dev/null || true
    gsutil -m cp "${_UE1D_BUCKET}/data/smoltalk/data/val/"'*' "${_DATA_DIR}/val/" 2>/dev/null || \
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
# Avoids 25-min recompile on every fresh VM. Cache is version-locked to torch_xla 2.9.0.
_XLA_CACHE=/tmp/xla_cache
if [ ! -d "$_XLA_CACHE" ] || [ "$(ls "$_XLA_CACHE" 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "Downloading XLA cache from GCS..."
    mkdir -p "$_XLA_CACHE"
    gsutil -m cp "${_UE1D_BUCKET}/xla_cache_v6e/*" "$_XLA_CACHE/" 2>/dev/null && \
        echo "XLA cache: $(ls $_XLA_CACHE | wc -l) files" || \
        echo "WARNING: XLA cache download failed — will recompile"
else
    echo "XLA cache already present: $(ls $_XLA_CACHE | wc -l) files"
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
echo "✓ ue1d deploy complete — babysitter running"
