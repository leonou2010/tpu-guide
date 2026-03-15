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
# Must remove stale/root-owned log BEFORE exec redirect (exec fails if file not writable)
sudo rm -f /tmp/deploy_babysitter.log 2>/dev/null || rm -f /tmp/deploy_babysitter.log 2>/dev/null || true
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
    # Remove stale root-owned file before writing (permission denied otherwise)
    sudo rm -f /tmp/boot_state.json 2>/dev/null || rm -f /tmp/boot_state.json 2>/dev/null || true
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
# Use sudo to kill root-owned processes (root babysitters launched by earlier sessions).
# If sudo is unavailable, fall through to user-level kill (handles non-root babysitters).
echo "Killing existing processes (sudo+user)..."
tmux kill-server 2>/dev/null || true
sudo pkill -9 -f babysitter.py 2>/dev/null || pkill -9 -f babysitter.py 2>/dev/null || true
sudo pkill -9 -f train_tpu.py 2>/dev/null || pkill -9 -f train_tpu.py 2>/dev/null || true
sudo pkill -9 -f run_tpu.py 2>/dev/null || pkill -9 -f run_tpu.py 2>/dev/null || true
# Kill training python (but not system python)
for pid in $(sudo pgrep -f 'python3.*run_tpu\|python3.*train_tpu\|python3.*babysitter' 2>/dev/null || pgrep -f 'python3.*run_tpu\|python3.*train_tpu\|python3.*babysitter' 2>/dev/null); do
    sudo kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
done
# Kill the flock wrapper (holds /tmp/tpu_babysitter.lock even after babysitter.py dies)
sudo pkill -9 -f 'flock.*tpu_babysitter' 2>/dev/null || pkill -9 -f 'flock.*tpu_babysitter' 2>/dev/null || true
# Remove stale lock file so flock --timeout=30 doesn't wait unnecessarily
sudo rm -f /tmp/tpu_babysitter.lock 2>/dev/null || rm -f /tmp/tpu_babysitter.lock 2>/dev/null || true
# Reset log file ownership so future deploys can write to them
sudo rm -f /tmp/babysitter.log /tmp/babysitter_chip*.log /tmp/boot_state.json 2>/dev/null || true
# Note: do NOT remove /tmp/deploy_babysitter.log here — it was opened at startup via exec >>
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

# ── Torch/TPU installation ────────────────────────────────────────────────────
# Strategy by VM type:
# - v6e ew4a: v2-alpha-tpuv6e runtime ships torch 2.9.0 + real nvidia CUDA libs in /usr/local/.
#             torch imports via DT_RPATH finding real libcudart.so.12 in nvidia/ → no stubs needed.
# - v6e ue1d: newer v2-alpha-tpuv6e image has NO torch. PyPI torch 2.9.0 requires Python 3.11+
#             for nvidia-* deps, so PyPI install fails. Install from pre-downloaded GCS wheels
#             (torch-2.9.0-cp310, torch_xla-2.9.0-cp310, libtpu-0.0.2, small deps).
#             Then create CUDA stubs in /usr/local/.../nvidia/ (matching torch's DT_RPATH).
# - v4/v5e: install torch+torch_xla+libtpu from GCS tpu_core wheels.

if [ "$IS_V6E" = "true" ]; then
    # Check without torch_xla.device() — libtpu not set yet, calling .device() would fail even on ew4a.
    # Only check importability: ew4a has torch+torch_xla in /usr/local/ → passes; ue1d has nothing → fails.
    if PYTHONNOUSERSITE=1 python3 -c "import torch; import torch_xla" 2>/dev/null; then
        echo "v6e: torch+torch_xla importable (system-wide) -- skipping install"
    else
        report_phase "INSTALLING_TPU_TORCH"
        echo "v6e: torch+torch_xla not found in /usr/local/. Installing from GCS wheels..."

        # Remove any stale CUDA torch from ~/.local/ that might shadow the correct system install
        _kw_sp="${HOME}/.local/lib/python3.10/site-packages"
        if ls "${_kw_sp}"/torch-*.dist-info 2>/dev/null | grep -q .; then
            echo "  Removing stale user torch from ~/.local/..."
            rm -rf "${_kw_sp}"/torch "${_kw_sp}"/torch-*.dist-info "${_kw_sp}"/torchgen 2>/dev/null || true
        fi

        # Install torch+torch_xla from pre-built GCS wheels.
        # PyPI torch 2.9.0 requires nvidia-*-cu12 which need Python 3.11+ (can't use pip install on 3.10).
        # Wheels at gs://.../wheels/torch_v6e_cp310/ are pre-downloaded cp310 builds from PyPI + libtpu-0.0.2.
        # storage.googleapis.com is reachable from all zones (Private Google Access); pypi.org is not on ue1d.
        if ! PYTHONNOUSERSITE=1 python3 -c "import torch_xla" 2>/dev/null; then
            echo "  Downloading torch wheels from GCS..."
            _whl_bucket="${BUCKET}/wheels/torch_v6e_cp310"
            _whl_dir="/tmp/torch_v6e_wheels"
            mkdir -p "$_whl_dir"
            # snap gsutil needs its user data dir; startup script runs as root which can create root-owned dirs.
            # Fix: chown snap dir to current user, or use sudo gsutil when non-root.
            _gsutil="gsutil"
            if [ "$(id -u)" != "0" ]; then
                sudo chown -R "$(id -un):$(id -gn)" "${HOME}/snap/google-cloud-cli" 2>/dev/null || true
                # If still failing, use sudo gsutil as fallback
                gsutil version 2>/dev/null || _gsutil="sudo gsutil"
            fi
            $_gsutil -m cp "${_whl_bucket}/*.whl" "$_whl_dir/" 2>&1 | grep -v 'crcmod' | tail -3 || echo "WARNING: GCS wheel download failed"

            if ls "$_whl_dir"/torch-*.whl 2>/dev/null | head -1 | grep -q .; then
                _pip="pip3"
                [ "$(id -u)" != "0" ] && _pip="sudo pip3"

                # Small deps first (safe to install with deps, they're tiny pure-python)
                $_pip install \
                    "$_whl_dir"/filelock*.whl \
                    "$_whl_dir"/typing_extensions*.whl \
                    "$_whl_dir"/sympy*.whl \
                    "$_whl_dir"/mpmath*.whl \
                    "$_whl_dir"/networkx*.whl \
                    "$_whl_dir"/jinja2*.whl \
                    "$_whl_dir"/markupsafe*.whl \
                    "$_whl_dir"/fsspec*.whl \
                    --no-deps -q 2>&1 | tail -2 || true

                # Install torch + torch_xla with --no-deps (avoids nvidia-*-cu12 deps, need Python 3.11+)
                $_pip install "$_whl_dir"/torch-*.whl --no-deps -q 2>&1 | tail -2 || true
                $_pip install "$_whl_dir"/torch_xla-*.whl --no-deps -q 2>&1 | tail -2 || true

                # Install libtpu to ~/.local/ (py3-none wheel, any Python 3; provides libtpu.so)
                pip3 install --user "$_whl_dir"/libtpu-0.0.2*.whl --no-deps -q 2>&1 | tail -2 || true

                # Install typing_extensions separately (required by torch at import; not a dep of torch_xla whl).
                # Use --force-reinstall to ensure it lands in /usr/local/ even if another version is present.
                PYTHONNOUSERSITE=1 python3 -c "import typing_extensions" 2>/dev/null || \
                    $_pip install "$_whl_dir"/typing_extensions*.whl --no-deps --force-reinstall -q 2>&1 | tail -2 || true
                # Verify torch importable now
                PYTHONNOUSERSITE=1 python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || \
                    echo "WARNING: torch still not importable after typing_extensions install"

                echo "  GCS wheel install complete"
                rm -rf "$_whl_dir"
            else
                echo "  ERROR: torch wheels not found after GCS download"
            fi
        fi

        # Create CUDA stub libs so torch imports succeed (TPU never uses CUDA).
        # Stubs in /usr/local/.../nvidia/ match torch's DT_RPATH — first ctypes.CDLL succeeds directly.
        _nv="/usr/local/lib/python3.10/dist-packages/nvidia"
        if [ "$(id -u)" = "0" ]; then
            mkdir -p "${_nv}"
            touch "${_nv}/__init__.py" 2>/dev/null || true
        else
            sudo mkdir -p "${_nv}" && sudo touch "${_nv}/__init__.py" 2>/dev/null || true
        fi

        _make_stub() {
            local pkg="$1" lib="$2"
            local dir="${_nv}/${pkg}/lib"
            if [ "$(id -u)" = "0" ]; then
                mkdir -p "$dir"
                touch "${_nv}/${pkg}/__init__.py" 2>/dev/null || true
                if [ ! -f "$dir/$lib" ]; then
                    printf 'void _stub(void) {}' | gcc -shared -fPIC -Wl,-soname,"$lib" -o "$dir/$lib" -x c - 2>/dev/null && echo "  stub: $lib" || echo "  WARN: gcc stub failed for $lib"
                fi
            else
                sudo mkdir -p "$dir" && sudo touch "${_nv}/${pkg}/__init__.py" 2>/dev/null || true
                if [ ! -f "$dir/$lib" ]; then
                    printf 'void _stub(void) {}' | sudo tee /tmp/stub_src.c > /dev/null && sudo gcc -shared -fPIC -Wl,-soname,"$lib" -o "$dir/$lib" /tmp/stub_src.c 2>/dev/null && echo "  stub: $lib" || { printf 'void _stub(void) {}' | sudo bash -c "gcc -shared -fPIC -Wl,-soname,\"$lib\" -o \"$dir/$lib\" -x c - 2>/dev/null" && echo "  stub: $lib" || echo "  WARN: gcc stub failed for $lib"; }
                fi
            fi
        }

        # Correct sonames verified from ew4a nvidia/ dir (torch 2.9.0+cu128)
        _make_stub cublas          libcublas.so.12
        _make_stub cublas          libcublasLt.so.12
        _make_stub cublas          libnvblas.so.12
        _make_stub cuda_cupti      libcupti.so.12
        _make_stub cuda_nvrtc      libnvrtc.so.12
        _make_stub cuda_runtime    libcudart.so.12
        _make_stub cudnn            libcudnn.so.9
        _make_stub cudnn            libcudnn_adv.so.9
        _make_stub cudnn            libcudnn_cnn.so.9
        _make_stub cudnn            libcudnn_engines_precompiled.so.9
        _make_stub cudnn            libcudnn_engines_runtime_compiled.so.9
        _make_stub cudnn            libcudnn_graph.so.9
        _make_stub cudnn            libcudnn_heuristic.so.9
        _make_stub cudnn            libcudnn_ops.so.9
        _make_stub cufft            libcufft.so.11
        _make_stub cufft            libcufftw.so.11
        _make_stub cufile            libcufile.so.0
        _make_stub curand            libcurand.so.10
        _make_stub cusolver          libcusolver.so.11
        _make_stub cusparse          libcusparse.so.12
        _make_stub cusparselt        libcusparseLt.so.0
        _make_stub nccl              libnccl.so.2
        _make_stub nvjitlink         libnvJitLink.so.12
        _make_stub nvshmem           libnvshmem_host.so.3
        _make_stub nvtx              libnvToolsExt.so.1

        # Verify with PYTHONNOUSERSITE=1 (same as babysitter)
        _import_test=$(PYTHONNOUSERSITE=1 python3 -c "import torch_xla; print('OK found_libtpu=' + str(torch_xla._found_libtpu))" 2>&1)
        echo "torch_xla (PYTHONNOUSERSITE=1): $_import_test"
        echo "$_import_test" | grep -q "^OK" || echo "WARNING: torch_xla still broken after setup"
    fi
else
    # v4/v5e: install tpu_core wheels if needed
    _NEEDS_TPU_TORCH=false
    python3 -c "import torch; import torch_xla" 2>/dev/null || _NEEDS_TPU_TORCH=true
    # Wrong: CUDA torch -- check via dist-info metadata (import fails, can't use torch.__version__)
    _usp="$HOME/.local/lib/python3.10/site-packages"
    for _meta in "${_usp}"/torch-*.dist-info/METADATA /usr/local/lib/python3.10/dist-packages/torch-*.dist-info/METADATA; do
        [ -f "$_meta" ] || continue
        grep -q '+cu\|nvidia\|cuda' "$_meta" 2>/dev/null && { echo "CUDA torch detected: $_meta"; _NEEDS_TPU_TORCH=true; break; }
    done
    python3 -c "import torch_xla; exit(0 if torch_xla._found_libtpu else 1)" 2>/dev/null || _NEEDS_TPU_TORCH=true
    echo "TPU_TORCH_NEEDED=$_NEEDS_TPU_TORCH"

    if [ "$_NEEDS_TPU_TORCH" = "true" ]; then
        report_phase "INSTALLING_TPU_TORCH"
        echo "Installing TPU torch+torch_xla+libtpu from GCS tpu_core wheels..."
        if ls "${_usp}"/torch-*.dist-info 2>/dev/null | grep -q .; then
            echo "Removing existing torch from ~/.local/..."
            rm -rf "${_usp}"/torch "${_usp}"/torch-*.dist-info "${_usp}"/torchgen \
                   "${_usp}"/torch_xla "${_usp}"/torch_xla-*.dist-info \
                   "${_usp}"/nvidia 2>/dev/null || true
        fi
        rm -rf /tmp/tpu_core && mkdir -p /tmp/tpu_core
        gsutil -m cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || \
            gsutil -m cp "gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || \
            gsutil -m cp "gs://gcp-researchcredits-blocklab-us-east1/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || true
        if ls /tmp/tpu_core/*.whl >/dev/null 2>&1; then
            pip install /tmp/tpu_core/libtpu-*.whl --force-reinstall --no-deps 2>/dev/null && echo "libtpu installed" || echo "WARNING: libtpu install failed"
            pip install /tmp/tpu_core/torch-*.whl --force-reinstall --no-deps 2>/dev/null && echo "torch installed" || echo "WARNING: torch install failed"
            pip install /tmp/tpu_core/torch_xla-*.whl --force-reinstall --no-deps 2>/dev/null && echo "torch_xla installed" || echo "WARNING: torch_xla install failed"
            pip install typing_extensions filelock jinja2 networkx sympy fsspec 2>/dev/null || true
            python3 -c "import typing_extensions" 2>/dev/null || {
                echo "typing_extensions missing -- installing from GCS wheels bundle..."
                cd /tmp
                [ -f /tmp/all_wheels.tar.gz ] || \
                    gcloud storage cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
                    gsutil cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || true
                if [ -f /tmp/all_wheels.tar.gz ]; then
                    tar xzf /tmp/all_wheels.tar.gz -C /tmp/ --wildcards '*/typing_extensions*.whl' 2>/dev/null || true
                    whl=$(ls /tmp/all_wheels/typing_extensions*.whl 2>/dev/null | head -1)
                    [ -n "$whl" ] && pip install "$whl" --no-deps 2>/dev/null && echo "typing_extensions installed from GCS" || true
                fi
            }
        else
            echo "ERROR: Could not download tpu_core wheels from any bucket"
        fi
        python3 -c "import torch_xla; print(f'torch_xla found_libtpu={torch_xla._found_libtpu}')" 2>/dev/null || echo "ERROR: torch_xla still broken after tpu_core install"
    fi
fi

# Set TPU_LIBRARY_PATH so _XLAC C extension can find libtpu.so at runtime.
# The Python libtpu package knows the .so location; _XLAC needs it via env var.
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
    _LIBTPU_SO=$(python3 -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
    if [ -n "$_LIBTPU_SO" ] && [ -f "$_LIBTPU_SO" ]; then
        export TPU_LIBRARY_PATH="$_LIBTPU_SO"
        echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
    else
        echo "WARNING: could not determine libtpu.so path — XLA device init may fail"
    fi
fi

# ── Hard TPU init gate ────────────────────────────────────────────────────────
# This is the ONLY test that matters: can we actually initialize the XLA device?
# Failures here (libtpu not found, PJRT errors, device busy) must stop deploy
# before babysitter claims tasks and fails at runtime.
report_phase "TESTING_TPU_INIT"
_TPU_INIT_OK=false
TPU_INIT_OUTPUT=$(python3 -c "
import torch_xla
d = torch_xla.device()
print(f'TPU_INIT_OK device={d}')
" 2>&1)
echo "TPU init test: $TPU_INIT_OUTPUT"
if echo "$TPU_INIT_OUTPUT" | grep -q "TPU_INIT_OK"; then
    _TPU_INIT_OK=true
    echo "✓ TPU device initialized successfully"
else
    echo "✗ TPU device init FAILED — cannot launch babysitter"
    echo "  Output: $TPU_INIT_OUTPUT"
    report_phase "FAILED_ENV_TPU_INIT"
    exit 1
fi

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
DATA_DST=$HOME/sf_bema/experiments/exp10_smollm2_smoltalk/data_full
if [ ! -d "${DATA_DST}/train" ]; then
    report_phase "DOWNLOADING_DATA"
    echo "Downloading training data (full ~7GB)..."
    mkdir -p "$DATA_DST"
    # Use rsync-style for idempotency
    gcloud storage cp -r "${BUCKET}/data/smoltalk_full/data/train" \
        "$DATA_DST/" 2>/dev/null && \
    gcloud storage cp -r "${BUCKET}/data/smoltalk_full/data/val" \
        "$DATA_DST/" 2>/dev/null || \
    gsutil -m cp -r "${BUCKET}/data/smoltalk_full/data/train" \
        "$DATA_DST/" 2>/dev/null && \
    gsutil -m cp -r "${BUCKET}/data/smoltalk_full/data/val" \
        "$DATA_DST/" 2>/dev/null || true
    echo "Data: $(ls $DATA_DST 2>/dev/null | wc -l) dirs"
fi

# Symlink smoltalk data into every experiment work dir that needs it.
# Data lives at exp10_smollm2_smoltalk/data_full (full 455k dataset).
DATA_SRC=~/sf_bema/experiments/exp10_smollm2_smoltalk/data_full
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
export TPU_LIBRARY_PATH=${TPU_LIBRARY_PATH:-}
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
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
