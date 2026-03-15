#!/bin/bash
# deploy_babysitter.sh (v3) — Strict phase-verified TPU worker setup.
#
# 6 phases, each with hard verification (exit 1 on failure + GCS telemetry):
#   BOOTING → DETECTING_TYPE → INSTALLING_TORCH → TESTING_TPU_INIT → INSTALLING_DEPS → LAUNCHING_BABYSITTER
#
# Key improvements vs v2:
#   - CUDA stubs from pre-built GCS tarball (falls back to inline GCC if tarball absent)
#   - Every phase: verify || { report_phase FAILED_*; exit 1; }
#   - USER variable: use $(id -un) — never assume $USER is set (startup SSH has minimal env)
#   - NEVER --user torch on v6e (installs CUDA torch from PyPI → shadows system torch)
#   - ue1d torch: GCS torch_v6e_cp310 wheels → sudo pip (system-wide)
#   - ew4a torch: system already has it → just verify, no install
#   - v4 torch: GCS tpu_core wheels → pip install
#
# Telemetry: gs://.../coord_v2/telemetry/{TPU_NAME}_boot.json at each phase.
# Logs: /tmp/deploy_babysitter.log (this script) + /tmp/babysitter.log (babysitter)
set -uo pipefail

# ── Single-instance deploy lock ───────────────────────────────────────────────
# Prevents concurrent deploys (e.g., vm_manager + manual SSH simultaneously).
# Use flock -xn: non-blocking, exclusive. If another deploy is running, exit 0.
_DEPLOY_LOCK=/tmp/deploy_babysitter_running.lock
exec 9>"$_DEPLOY_LOCK"
if ! flock -xn 9; then
    echo "[$(date -u '+%H:%M:%S')] Another deploy_babysitter.sh is running — exiting" >&2
    exit 0
fi
# Lock held for duration of this script (fd 9 stays open).

# ── Clean stale root-owned logs BEFORE exec redirect ─────────────────────────
# CRITICAL ORDER: must rm BEFORE exec >> (exec fails if file not writable)
sudo rm -f /tmp/deploy_babysitter.log /tmp/babysitter.log /tmp/babysitter_chip*.log \
    /tmp/boot_state.json /tmp/tpu_babysitter.lock 2>/dev/null || true

exec >> /tmp/deploy_babysitter.log 2>&1
echo "=== deploy_babysitter.sh (v3) started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# ── Environment setup ─────────────────────────────────────────────────────────
export HOME=${HOME:-/root}
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/sbin:$PATH
_CUR_USER=$(id -un 2>/dev/null || echo "kwokchunau")
_PY=python3

# Auto-detect ZONE
if [ -z "${ZONE:-}" ]; then
    ZONE=$(curl -s --connect-timeout 5 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
        | awk -F/ '{print $NF}' || true)
fi
# Auto-detect BUCKET from zone
case "${ZONE:-}" in
    europe-west4*) BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4} ;;
    us-east1*)     BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-us-east1} ;;
    us-central2*)  BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-1-us-central2} ;;
    *)             BUCKET=${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4} ;;
esac
# TPU_NAME: configured env var takes priority (vm_manager sets it via SSH)
if [ -z "${TPU_NAME:-}" ]; then
    TPU_NAME=$(curl -s --connect-timeout 5 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || hostname)
fi
CTRL=${CONTROL_PLANE:-gs://gcp-researchcredits-blocklab-europe-west4/coord_v2}

echo "TPU_NAME=$TPU_NAME ZONE=${ZONE:-unknown} BUCKET=$BUCKET USER=$_CUR_USER"

# ── GCS Telemetry ─────────────────────────────────────────────────────────────
report_phase() {
    local phase=$1
    local ts
    ts=$(date +%s)
    sudo rm -f /tmp/boot_state.json 2>/dev/null || rm -f /tmp/boot_state.json 2>/dev/null || true
    printf '{"tpu_name":"%s","zone":"%s","phase":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "${ZONE:-}" "$phase" "$ts" > /tmp/boot_state.json
    gsutil cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json" 2>/dev/null || true
    echo "[$(date -u '+%H:%M:%S')] PHASE: $phase"
}

report_phase "BOOTING"

# ── Guard: don't redeploy if training is active ───────────────────────────────
GUARD_TTL=2700
if [ -z "${FORCE_REDEPLOY:-}" ]; then
    _now_s=$(date +%s)
    for _hb_path in $(gsutil ls "${CTRL}/heartbeats/${TPU_NAME}_chip*.json" 2>/dev/null || true); do
        _hb_json=$(gsutil cat "$_hb_path" 2>/dev/null || true)
        [ -z "$_hb_json" ] && continue
        _ts=$(echo "$_hb_json" | $_PY -c "import sys,json; print(int(json.load(sys.stdin).get('timestamp',0)))" 2>/dev/null || echo 0)
        _status=$(echo "$_hb_json" | $_PY -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
        _age=$(( _now_s - _ts ))
        case "$_status" in
            training|xla_compile|uploading)
                if [ "$_age" -lt "$GUARD_TTL" ]; then
                    echo "GUARD: active chip at $_hb_path (age=${_age}s, status=$_status) — skipping deploy"
                    exit 0
                fi
                ;;
        esac
    done
fi

# ── Kill existing processes ───────────────────────────────────────────────────
echo "Killing existing processes..."
tmux kill-server 2>/dev/null || true
sudo pkill -9 -f babysitter.py 2>/dev/null || pkill -9 -f babysitter.py 2>/dev/null || true
sudo pkill -9 -f train_v2_tpu.py 2>/dev/null || pkill -9 -f train_v2_tpu.py 2>/dev/null || true
sudo pkill -9 -f 'flock.*tpu_babysitter' 2>/dev/null || pkill -9 -f 'flock.*tpu_babysitter' 2>/dev/null || true
sudo pkill -KILL -f 'TPU_VISIBLE_CHIPS' 2>/dev/null || pkill -KILL -f 'TPU_VISIBLE_CHIPS' 2>/dev/null || true
sudo rm -f /tmp/babysitter.log /tmp/babysitter_chip*.log /tmp/boot_state.json 2>/dev/null || true
sleep 3

report_phase "RELEASING_DEVICES"
# Release TPU device locks
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio* /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
done
sleep 2
for dev in /dev/vfio/[0-9]* /dev/vfio/devices/vfio* /dev/accel[0-9]*; do
    [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
done
sleep 2

report_phase "DETECTING_TYPE"
# ── VM type detection ─────────────────────────────────────────────────────────
# v4: /dev/accel[0-9]*
# v6e: /dev/vfio/devices/vfio* (subdirectory)
# v5e: /dev/vfio/[0-9]* (no devices/ subdir)
IS_V4=false
IS_V6E=false
IS_V5E=false
IS_UE1D=false  # us-east1-d: no torch pre-installed

if ls /dev/accel[0-9]* >/dev/null 2>&1; then
    IS_V4=true
    echo "Detected: v4 VM"
elif ls /dev/vfio/devices/vfio* >/dev/null 2>&1; then
    case "${ZONE:-}" in
        europe-west4-a*|us-east1*)
            IS_V6E=true
            echo "Detected: v6e VM (zone=${ZONE:-})"
            ;;
        europe-west4-b*|us-central1*)
            IS_V5E=true
            echo "Detected: v5e VM (zone=${ZONE:-})"
            ;;
        *)
            IS_V6E=true
            echo "Detected: v6e/v5e VM (zone unknown)"
            ;;
    esac
    # ue1d = us-east1-d: v6e with no pre-installed torch
    case "${ZONE:-}" in
        us-east1*) IS_UE1D=true; echo "  → ue1d zone: no pre-installed torch" ;;
    esac
elif ls /dev/vfio/[0-9]* >/dev/null 2>&1; then
    IS_V5E=true
    echo "Detected: v5e VM (/dev/vfio/[0-9]*)"
else
    echo "WARNING: no TPU devices found (may appear after boot)"
fi

report_phase "INSTALLING_TORCH"
# ── Torch installation ────────────────────────────────────────────────────────
#
# Rules (from TESTRUN_FINDINGS Session 3):
#   v6e ew4a: system has torch+torch_xla → just verify. No install.
#   v6e ue1d: no torch → install from GCS torch_v6e_cp310 wheels (sudo pip, system-wide)
#             + install CUDA stubs (tarball from GCS or inline GCC)
#   v4:       install from GCS tpu_core wheels (pip, user or system)
#   v5e:      install from GCS tpu_core wheels
#
# RULE: NEVER pip install --user torch on v6e. PyPI torch = CUDA, shadows/breaks libtpu.
# RULE: tpu_core GCS wheels = v4 ONLY. Never install on v6e.

_needs_torch() {
    # Returns 0 (true) if torch+torch_xla NOT importable
    PYTHONNOUSERSITE=1 $_PY -c "import torch; import torch_xla" 2>/dev/null
    local rc=$?
    [ $rc -ne 0 ] && return 0  # needs install
    # Also fail if CUDA torch is present (wrong for TPU)
    local ver
    ver=$(PYTHONNOUSERSITE=1 $_PY -c "import torch; print(torch.__version__)" 2>/dev/null || true)
    if echo "$ver" | grep -q '+cu'; then
        echo "CUDA torch detected ($ver) — will purge and reinstall"
        return 0
    fi
    return 1  # OK, TPU torch already there
}

_purge_user_torch() {
    local usp="$HOME/.local/lib/python3.10/site-packages"
    for d in torch torch_xla libtpu torchgen; do
        rm -rf "${usp}/${d}" "${usp}/${d}-"*.dist-info 2>/dev/null || true
    done
    echo "Purged user-site torch/torch_xla/libtpu"
}

if $IS_V6E; then
    if $IS_UE1D; then
        # ue1d: no torch at all. Install from GCS wheels (TPU build, system-wide).
        _purge_user_torch  # remove any stale user-site torch first
        if _needs_torch; then
            echo "ue1d: installing torch+torch_xla from GCS wheels..."
            _whl_dir=/tmp/torch_v6e_wheels
            rm -rf "$_whl_dir" && mkdir -p "$_whl_dir"
            gsutil -m cp "${BUCKET}/wheels/torch_v6e_cp310/*.whl" "$_whl_dir/" 2>&1 | tail -3 || \
                gsutil -m cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/*.whl" "$_whl_dir/" 2>&1 | tail -3 || \
                echo "WARNING: GCS torch_v6e_cp310 download failed"

            if ls "$_whl_dir"/torch-*.whl 2>/dev/null | head -1 | grep -q .; then
                # Install small deps one-by-one (system-wide via sudo).
                # One-at-a-time so a single bad wheel (e.g. networkx wrong Python metadata)
                # does NOT abort the whole batch and leave typing_extensions uninstalled.
                # Install deps one-by-one. Pin networkx-3.4.2 (3.6.1 requires Python>=3.11).
                # numpy: required by torch_xla xla_sharding.py (not pre-installed on ue1d).
                for _dep_whl in \
                        "$_whl_dir"/filelock*.whl \
                        "$_whl_dir"/typing_extensions*.whl \
                        "$_whl_dir"/sympy*.whl \
                        "$_whl_dir"/mpmath*.whl \
                        "$_whl_dir"/networkx-3.4.2-py3-none-any.whl \
                        "$_whl_dir"/jinja2*.whl \
                        "$_whl_dir"/markupsafe*.whl \
                        "$_whl_dir"/fsspec*.whl \
                        "$_whl_dir"/numpy-*.whl; do
                    [ -f "$_dep_whl" ] && sudo $_PY -m pip install "$_dep_whl" --no-deps -q 2>/dev/null || true
                done
                # Install torch + torch_xla system-wide (no --user, no --deps → avoids CUDA nvidia-* deps)
                sudo $_PY -m pip install "$_whl_dir"/torch-*.whl --no-deps -q 2>&1 | tail -2
                sudo $_PY -m pip install "$_whl_dir"/torch_xla-*.whl --no-deps -q 2>&1 | tail -2
                # libtpu to user site (py3-none wheel, fine as user install)
                $_PY -m pip install --user "$_whl_dir"/libtpu-0.0.2*.whl --no-deps -q 2>&1 | tail -2 || true
                rm -rf "$_whl_dir"
                echo "ue1d: torch install from GCS wheels complete"
            else
                echo "ERROR: torch_v6e_cp310 wheels not found in GCS"
                report_phase "FAILED_ENV_TORCH_WHEELS_MISSING"
                exit 1
            fi
        else
            echo "ue1d: torch+torch_xla already importable"
        fi

        # CUDA stubs — required for torch imports on ue1d (no CUDA libs installed)
        # Try pre-built tarball first; fall back to inline GCC if not available.
        _nv=/usr/local/lib/python3.10/dist-packages/nvidia
        sudo mkdir -p "$_nv" && sudo touch "$_nv/__init__.py" 2>/dev/null || true

        # Always re-install stubs (12KB tarball, fast; ensures nvshmem stub has correct symbols)
        echo "Installing CUDA stubs for ue1d..."
        # Try pre-built tarball (built by scripts/build_cuda_stubs.sh on ew4a)
        _stubs_tgz=/tmp/nvidia_stubs_v6e.tar.gz
        if gsutil cp "${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stubs_tgz" 2>/dev/null || \
           gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/nvidia_stubs_v6e.tar.gz" "$_stubs_tgz" 2>/dev/null; then
            echo "  Extracting pre-built CUDA stubs tarball..."
            # Tarball has nvidia/... structure; extract to dist-packages/ (not dist-packages/nvidia/)
            sudo tar xzf "$_stubs_tgz" -C /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true
            rm -f "$_stubs_tgz"
        else
                echo "  Pre-built tarball not found — building stubs inline with GCC..."
                _make_stub() {
                    local pkg="$1" lib="$2"
                    local dir="${_nv}/${pkg}/lib"
                    sudo mkdir -p "$dir" && sudo touch "${_nv}/${pkg}/__init__.py" 2>/dev/null || true
                    if [ ! -f "$dir/$lib" ]; then
                        printf 'void _stub(void) {}' | sudo tee /tmp/stub_src.c >/dev/null
                        sudo gcc -shared -fPIC -Wl,-soname,"$lib" -o "$dir/$lib" /tmp/stub_src.c 2>/dev/null && \
                            echo "  stub: $lib" || echo "  WARN: gcc failed for $lib"
                    fi
                }
                # Sonames confirmed from ew4a (torch 2.9.0+cu128)
                _make_stub cublas          libcublas.so.12
                _make_stub cublas          libcublasLt.so.12
                _make_stub cublas          libnvblas.so.12
                _make_stub cuda_cupti      libcupti.so.12
                _make_stub cuda_nvrtc      libnvrtc.so.12
                _make_stub cuda_runtime    libcudart.so.12
                _make_stub cudnn           libcudnn.so.9
                _make_stub cudnn           libcudnn_adv.so.9
                _make_stub cudnn           libcudnn_cnn.so.9
                _make_stub cudnn           libcudnn_engines_precompiled.so.9
                _make_stub cudnn           libcudnn_engines_runtime_compiled.so.9
                _make_stub cudnn           libcudnn_graph.so.9
                _make_stub cudnn           libcudnn_heuristic.so.9
                _make_stub cudnn           libcudnn_ops.so.9
                _make_stub cufft           libcufft.so.11
                _make_stub cufft           libcufftw.so.11
                _make_stub cufile          libcufile.so.0
                _make_stub curand          libcurand.so.10
                _make_stub cusolver        libcusolver.so.11
                _make_stub cusparse        libcusparse.so.12
                _make_stub cusparselt      libcusparseLt.so.0
                _make_stub nccl            libnccl.so.2
                _make_stub nvjitlink       libnvJitLink.so.12
                _make_stub nvshmem         libnvshmem_host.so.3
                _make_stub nvtx            libnvToolsExt.so.1
            fi
    else
        # ew4a: internet available — install torch+torch_xla from PyPI (sudo → system-wide).
        # PyPI auto-installs all nvidia-* packages (nccl, cublas, cudart, etc.) — no stubs needed.
        # V2 lesson: GCS wheels for ew4a is wrong — GCS lacks nvidia runtime packages.
        _purge_user_torch  # remove any stale user-site torch that would shadow system install
        if ! $_PY -c "import torch; import torch_xla" 2>/dev/null; then
            echo "v6e ew4a: torch not importable — installing from PyPI (system-wide)..."
            sudo $_PY -m pip install torch==2.9.0 torch_xla==2.9.0 \
                --extra-index-url https://download.pytorch.org/whl/cu128 \
                -q 2>&1 | tail -5
            if ! $_PY -c "import torch" 2>/dev/null; then
                echo "ERROR: torch PyPI install failed"
                report_phase "FAILED_ENV_TORCH_EW4A"
                exit 1
            fi
            echo "v6e ew4a: torch install from PyPI complete"
        else
            echo "v6e ew4a: torch+torch_xla OK"
        fi

    fi
elif $IS_V4 || $IS_V5E; then
    # v4/v5e: install from GCS tpu_core wheels if needed
    _NEEDS_TPU_TORCH=false
    $_PY -c "import torch; import torch_xla" 2>/dev/null || _NEEDS_TPU_TORCH=true
    # Detect CUDA torch (wrong for v4)
    _usp="$HOME/.local/lib/python3.10/site-packages"
    for _meta in "${_usp}"/torch-*.dist-info/METADATA /usr/local/lib/python3.10/dist-packages/torch-*.dist-info/METADATA; do
        [ -f "$_meta" ] || continue
        grep -q '+cu\|nvidia\|cuda' "$_meta" 2>/dev/null && { _NEEDS_TPU_TORCH=true; break; }
    done
    $_PY -c "import torch_xla; exit(0 if torch_xla._found_libtpu else 1)" 2>/dev/null || _NEEDS_TPU_TORCH=true

    if [ "$_NEEDS_TPU_TORCH" = "true" ]; then
        echo "v4/v5e: installing TPU torch from GCS tpu_core wheels..."
        # Purge any wrong user-site torch
        rm -rf "${_usp}"/torch "${_usp}"/torch-*.dist-info "${_usp}"/torchgen \
               "${_usp}"/torch_xla "${_usp}"/torch_xla-*.dist-info \
               "${_usp}"/nvidia 2>/dev/null || true
        rm -rf /tmp/tpu_core && mkdir -p /tmp/tpu_core
        gsutil -m cp "${BUCKET}/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || \
            gsutil -m cp "gs://gcp-researchcredits-blocklab-1-us-central2/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || \
            gsutil -m cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/tpu_core/*.whl" /tmp/tpu_core/ 2>/dev/null || true
        if ls /tmp/tpu_core/*.whl >/dev/null 2>&1; then
            # Use sudo so wheels install to system site-packages (not user site).
            # PYTHONNOUSERSITE=1 in the hard gate blocks user site — must install system-wide.
            # Install deps first (typing_extensions, filelock, etc.) — torch requires these even with --no-deps
            for _dep_whl in /tmp/tpu_core/*.whl; do
                _dname=$(basename "$_dep_whl" | sed 's/-[0-9].*//' | tr '[:upper:]' '[:lower:]')
                case "$_dname" in
                    torch*|torch_xla*|libtpu*) continue ;;
                    *) sudo $_PY -m pip install "$_dep_whl" --no-deps -q 2>/dev/null || true ;;
                esac
            done
            sudo $_PY -m pip install /tmp/tpu_core/libtpu-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "libtpu installed" || echo "WARNING: libtpu failed"
            sudo $_PY -m pip install /tmp/tpu_core/torch-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch installed" || echo "WARNING: torch failed"
            sudo $_PY -m pip install /tmp/tpu_core/torch_xla-*.whl --force-reinstall --no-deps -q 2>/dev/null && echo "torch_xla installed" || echo "WARNING: torch_xla failed"
            rm -rf /tmp/tpu_core
            # torch was installed --no-deps, so its runtime deps (typing_extensions, numpy,
            # filelock, sympy, etc.) are missing. Use apt-get (reliable on tpu-ubuntu2204-base).
            echo "v4/v5e: installing torch runtime deps via apt-get..."
            sudo apt-get install -y python3-numpy python3-typing-extensions 2>/dev/null || \
                sudo apt-get install -y python3-numpy 2>/dev/null || true
            # Install remaining deps from v6e wheel bundle (no apt package for filelock/sympy/etc)
            _tdep=/tmp/torch_deps_v4
            rm -rf "$_tdep" && mkdir -p "$_tdep"
            gsutil -m cp \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/filelock-*.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/typing_extensions-*.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/sympy-*.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/mpmath-*.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/networkx-3.4.2-py3-none-any.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/jinja2-*.whl" \
                "gs://gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/markupsafe-*.whl" \
                "$_tdep/" 2>/dev/null || true
            for _w in "$_tdep"/*.whl; do
                [ -f "$_w" ] && $_PY -m pip install "$_w" --no-deps -q 2>&1 | grep -v "already installed" || true
            done
            rm -rf "$_tdep"
            echo "v4/v5e: torch runtime deps installed"
        else
            echo "ERROR: tpu_core wheels not found in any GCS bucket"
            report_phase "FAILED_ENV_TPU_CORE_WHEELS_MISSING"
            exit 1
        fi
    else
        echo "v4/v5e: torch+torch_xla already OK"
    fi

    # CUDA stubs — tpu_core/torch-2.9.0 is a CUDA build that needs libcublas.so.12 etc.
    # v4 has no CUDA hardware, but torch only dlopen()'s these libs at import time.
    # torch_xla uses libtpu.so for actual computation, not CUDA. Stubs satisfy dlopen.
    _nv=/usr/local/lib/python3.10/dist-packages/nvidia
    sudo mkdir -p "$_nv" && sudo touch "$_nv/__init__.py" 2>/dev/null || true
    # Always re-install stubs (12KB; ensures nvshmem stub has correct symbols)
    echo "v4/v5e: installing CUDA stubs..."
    _stubs_tgz=/tmp/nvidia_stubs_v6e.tar.gz
    if gsutil cp "${BUCKET}/wheels/nvidia_stubs_v6e.tar.gz" "$_stubs_tgz" 2>/dev/null || \
       gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/nvidia_stubs_v6e.tar.gz" "$_stubs_tgz" 2>/dev/null; then
        # Tarball has nvidia/... structure; extract to dist-packages/ (not dist-packages/nvidia/)
        sudo tar xzf "$_stubs_tgz" -C /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true
        rm -f "$_stubs_tgz"
        echo "v4/v5e: CUDA stubs installed"
    else
        echo "WARNING: CUDA stubs tarball not found — torch import may fail"
    fi
fi

# ── Set TPU_LIBRARY_PATH ──────────────────────────────────────────────────────
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
    _LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
    if [ -z "$_LIBTPU_SO" ] || [ ! -f "$_LIBTPU_SO" ]; then
        # libtpu Python package not installed — download and install it.
        # All VM types need this: ew4a (system torch_xla 2.9.0 has no bundled libtpu.so),
        # ue1d (fresh install), v4 (tpu_core wheel may already have it from above).
        echo "libtpu not found — installing from GCS wheel..."
        mkdir -p /tmp/_libtpu_install
        gsutil cp "${BUCKET}/wheels/tpu_core/libtpu-*.whl" /tmp/_libtpu_install/ 2>/dev/null || \
            gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/tpu_core/libtpu-*.whl" /tmp/_libtpu_install/ 2>/dev/null || true
        if ls /tmp/_libtpu_install/libtpu-*.whl >/dev/null 2>&1; then
            $_PY -m pip install /tmp/_libtpu_install/libtpu-*.whl --no-deps -q 2>/dev/null || \
                sudo $_PY -m pip install /tmp/_libtpu_install/libtpu-*.whl --no-deps -q 2>/dev/null || true
            rm -rf /tmp/_libtpu_install
            _LIBTPU_SO=$($_PY -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
        fi
    fi
    if [ -n "$_LIBTPU_SO" ] && [ -f "$_LIBTPU_SO" ]; then
        export TPU_LIBRARY_PATH="$_LIBTPU_SO"
        echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
    else
        echo "WARNING: could not resolve libtpu.so path — PJRT_DEVICE may default to CPU"
    fi
fi

# ── HARD GATE: TPU device init ────────────────────────────────────────────────
# We verify TPU hardware is detected by checking PJRT_DEVICE after import.
# torch_xla sets PJRT_DEVICE=TPU at import time when it successfully loads libtpu.so.
# We do NOT call torch_xla.device() because torch_xla 2.9.0 has a known quirk:
# the library_path property raises EnvironmentError('libtpu not found') even when
# libtpu.so IS loaded (it can't find the .so path via the Python-level search).
# Checking PJRT_DEVICE is the correct proxy: if it's TPU, the hardware is ready.
report_phase "TESTING_TPU_INIT"
TPU_INIT_OUT=$($_PY -c "
import torch_xla, os, sys
pjrt = os.environ.get('PJRT_DEVICE', 'NOT_SET')
if pjrt == 'TPU':
    print(f'TPU_INIT_OK PJRT_DEVICE=TPU')
else:
    print(f'FAILED: PJRT_DEVICE={pjrt} (expected TPU)')
    sys.exit(1)
" 2>&1)
echo "TPU init: $TPU_INIT_OUT"
if ! echo "$TPU_INIT_OUT" | grep -q 'TPU_INIT_OK'; then
    echo "FATAL: TPU device init failed — PJRT_DEVICE not set to TPU"
    echo "       (libtpu.so not found or TPU devices not present)"
    report_phase "FAILED_ENV_TPU_INIT"
    exit 1
fi
echo "✓ TPU device initialized (PJRT_DEVICE=TPU)"

# ── Install training dependencies ─────────────────────────────────────────────
MISSING=""
$_PY -c "import hydra" 2>/dev/null || MISSING="$MISSING hydra"
$_PY -c "import transformers" 2>/dev/null || MISSING="$MISSING transformers"
$_PY -c "import sympy" 2>/dev/null || MISSING="$MISSING sympy"
$_PY -c "import antlr4" 2>/dev/null || MISSING="$MISSING antlr4"
$_PY -c "import datasets" 2>/dev/null || MISSING="$MISSING datasets"
$_PY -c "import wandb" 2>/dev/null || MISSING="$MISSING wandb"

if [ -n "$MISSING" ]; then
    report_phase "INSTALLING_PACKAGES"
    echo "Missing:$MISSING — installing..."
    # Check internet first (no-internet VMs: ue1d, uc2b — skip PyPI to avoid timeout)
    _HAS_INTERNET=false
    curl -sf --connect-timeout 5 https://pypi.org > /dev/null 2>&1 && _HAS_INTERNET=true
    if $_HAS_INTERNET; then
        # Internet available: try PyPI with short timeout
        $_PY -m pip install 'setuptools<70' --timeout 15 -q 2>/dev/null || true
        $_PY -m pip install antlr4-python3-runtime==4.9.3 --timeout 15 -q 2>/dev/null || true
        $_PY -m pip install hydra-core omegaconf transformers sympy datasets wandb --timeout 15 -q 2>/dev/null || true
    fi
    # Fallback: GCS wheels bundle (no-internet VMs: ue1d, uc2b)
    $_PY -c "import transformers" 2>/dev/null || {
        echo "pip failed — installing from GCS wheels bundle..."
        cd /tmp
        gsutil cp "${BUCKET}/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
            gsutil cp "${BUCKET}/wheels/tpu_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
            gsutil cp "gs://gcp-researchcredits-blocklab-us-east1/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || \
            gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/wheels/all_wheels.tar.gz" /tmp/all_wheels.tar.gz 2>/dev/null || true
        if [ -f /tmp/all_wheels.tar.gz ]; then
            tar xzf all_wheels.tar.gz -C /tmp/ 2>/dev/null || true
            # Handle both all_wheels/ and tpu_wheels/ directory names
            [ -d /tmp/all_wheels ] || mv /tmp/tpu_wheels /tmp/all_wheels 2>/dev/null || true
            for whl in /tmp/all_wheels/*.whl; do
                [ -f "$whl" ] || continue
                name=$(basename "$whl" | tr '-' ' ' | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
                case "$name" in
                    torch*|nvidia*|libtpu*|jax*|triton*) continue ;;
                    *) $_PY -m pip install "$whl" --no-deps -q 2>/dev/null || true ;;
                esac
            done
        fi
    }
fi

# ── PRE-FLIGHT: verify all deps before proceeding ────────────────────────────
report_phase "PRE_FLIGHT_CHECK"
ASSERT_ERRORS=""
$_PY -c "import torch; import torch_xla" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS torch/torch_xla"
$_PY -c "import transformers" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS transformers"
$_PY -c "import hydra" 2>/dev/null || ASSERT_ERRORS="$ASSERT_ERRORS hydra"

if [ -n "$ASSERT_ERRORS" ]; then
    echo "PRE_FLIGHT FAILED: missing$ASSERT_ERRORS"
    report_phase "FAILED_ENV_PACKAGES:${ASSERT_ERRORS}"
    printf '{"tpu_name":"%s","phase":"FAILED_ENV","missing":"%s","timestamp":%s}\n' \
        "$TPU_NAME" "$ASSERT_ERRORS" "$(date +%s)" > /tmp/boot_state.json
    gsutil cp /tmp/boot_state.json "$CTRL/telemetry/${TPU_NAME}_boot.json" 2>/dev/null || true
    exit 1
fi
echo "Pre-flight OK: torch=$(PYTHONNOUSERSITE=1 $_PY -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# ── Download model ────────────────────────────────────────────────────────────
MODEL_PATH=/tmp/SmolLM2-135M
if [ ! -f "$MODEL_PATH/config.json" ]; then
    report_phase "DOWNLOADING_MODEL"
    mkdir -p "$MODEL_PATH"
    gcloud storage cp -r "${BUCKET}/models/SmolLM2-135M/*" "$MODEL_PATH/" 2>/dev/null || \
    gsutil -m cp -r "${BUCKET}/models/SmolLM2-135M/*" "$MODEL_PATH/" 2>/dev/null || true
fi

# ── Download experiment code ──────────────────────────────────────────────────
report_phase "DOWNLOADING_CODE"
mkdir -p ~/sf_bema/experiments
gcloud storage cp "${BUCKET}/code/sf_bema_code.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || \
    gsutil cp "${BUCKET}/code/sf_bema_code.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || true
[ -f /tmp/sf_bema_code.tar.gz ] && tar xzf /tmp/sf_bema_code.tar.gz -C ~/sf_bema/experiments/ 2>/dev/null || true

# ── Download training data ────────────────────────────────────────────────────
DATA_DST=$HOME/sf_bema/experiments/exp10_smollm2_smoltalk/data_full
if [ ! -d "${DATA_DST}/train" ]; then
    report_phase "DOWNLOADING_DATA"
    mkdir -p "$DATA_DST"
    gcloud storage cp -r "${BUCKET}/data/smoltalk_full/data/train" "$DATA_DST/" 2>/dev/null && \
    gcloud storage cp -r "${BUCKET}/data/smoltalk_full/data/val" "$DATA_DST/" 2>/dev/null || \
    gsutil -m cp -r "${BUCKET}/data/smoltalk_full/data/train" "$DATA_DST/" 2>/dev/null && \
    gsutil -m cp -r "${BUCKET}/data/smoltalk_full/data/val" "$DATA_DST/" 2>/dev/null || true
fi
# Symlink data into each experiment dir
DATA_SRC=~/sf_bema/experiments/exp10_smollm2_smoltalk/data_full
if [ -d "$DATA_SRC" ]; then
    for _wdir in ~/sf_bema/experiments/*/; do
        _name=$(basename "$_wdir")
        [ "$_name" = "exp10_smollm2_smoltalk" ] && continue
        [ "$_name" = "shared" ] && continue
        [ -e "${_wdir}data" ] && continue
        ln -sf "$DATA_SRC" "${_wdir}data"
    done
fi

# ── Download babysitter (v3) code ────────────────────────────────────────────
report_phase "DOWNLOADING_PULL_CODE"
mkdir -p ~/pull_code
gcloud storage cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
gcloud storage cp "gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
gsutil -m cp "${BUCKET}/pull_code_v3/*" ~/pull_code/ 2>/dev/null || \
gsutil -m cp "gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/*" ~/pull_code/ 2>/dev/null || true

# ── Download XLA cache ────────────────────────────────────────────────────────
XLA_DIR=/tmp/xla_cache
mkdir -p $XLA_DIR
if [ "$(ls $XLA_DIR 2>/dev/null | wc -l)" -lt 5 ]; then
    report_phase "DOWNLOADING_XLA_CACHE"
    if $IS_V4; then
        XLA_GCS="${BUCKET}/xla_cache_v4"
    elif $IS_V5E; then
        XLA_GCS="gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_v5e"
    else
        XLA_GCS="${BUCKET}/xla_cache_v6e"
    fi
    echo "XLA cache: $XLA_GCS"
    gcloud storage cp -r "${XLA_GCS}/*" $XLA_DIR/ 2>/dev/null || \
    gsutil -m cp -r "${XLA_GCS}/*" $XLA_DIR/ 2>/dev/null || true
fi

# ── Detect chips ──────────────────────────────────────────────────────────────
if ls /dev/vfio/devices/vfio* >/dev/null 2>&1; then
    CHIPS=$(ls /dev/vfio/devices/vfio* | wc -l)
elif ls /dev/vfio/[0-9]* >/dev/null 2>&1; then
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
[ -z "$ACCEL" ] && { $IS_V4 && ACCEL="v4-8" || ACCEL="v6e-8"; }
echo "Zone=$ZONE Chips=$CHIPS Accel=$ACCEL"

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
export CHIPS_PER_HOST=${CHIPS}
export GCS_CHECKPOINT_DIR=${BUCKET}/checkpoints

nohup flock --timeout=30 /tmp/tpu_babysitter.lock \
    $_PY -u ~/pull_code/babysitter.py >> /tmp/babysitter.log 2>&1 &
BPID=$!

echo "Launched babysitter PID=$BPID"
sleep 3

if kill -0 $BPID 2>/dev/null; then
    echo "Babysitter running OK (PID=$BPID)"
    report_phase "IDLE_AWAITING_WORK"
else
    echo "ERROR: Babysitter died. Last log:"
    tail -20 /tmp/babysitter.log
    report_phase "FAILED_BABYSITTER_DIED"
    exit 1
fi
