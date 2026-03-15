#!/bin/bash
# build_cuda_stubs.sh — One-time script to build CUDA stub .so files for ue1d v6e VMs.
#
# Run on blocklab. Connects to ew4a-1 to read exact sonames, then builds stubs locally.
# Output: ~/distributed_tpu_training/v3/stubs/nvidia_stubs_v6e.tar.gz
# Upload: gsutil cp ... gs://.../wheels/nvidia_stubs_v6e.tar.gz (all 3 buckets)
#
# WHY: ue1d has no CUDA libs. torch 2.9.0+cu128 dlopen's 25+ .so files at import.
# Stubs satisfy the dlopen without providing real CUDA functionality (TPU never uses CUDA).
# The sonames must exactly match what torch expects (verified from ew4a system libs).
#
# USAGE:
#   bash ~/distributed_tpu_training/v3/scripts/build_cuda_stubs.sh
#   # Then upload to GCS:
#   for bucket in \
#       gs://gcp-researchcredits-blocklab-europe-west4 \
#       gs://gcp-researchcredits-blocklab-us-east1 \
#       gs://gcp-researchcredits-blocklab-1-us-central2; do
#       gsutil cp ~/distributed_tpu_training/v3/stubs/nvidia_stubs_v6e.tar.gz \
#           ${bucket}/wheels/nvidia_stubs_v6e.tar.gz
#   done

set -euo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
EW4A_VM=${1:-v6e-ew4a-1}
EW4A_ZONE=europe-west4-a
OUTPUT_DIR=~/distributed_tpu_training/v3/stubs
mkdir -p "$OUTPUT_DIR"

echo "=== Building CUDA stubs for ue1d v6e VMs ==="
echo "Reference VM: $EW4A_VM ($EW4A_ZONE)"

# ── Step 1: Get exact sonames from ew4a nvidia/ directory ────────────────────
echo ""
echo "Step 1: Querying exact sonames from $EW4A_VM..."
SONAME_LIST=$($GCLOUD alpha compute tpus tpu-vm ssh "$EW4A_VM" \
    --zone="$EW4A_ZONE" --project=$PROJECT --tunnel-through-iap \
    --command="
python3 -c \"
import os, re
nv = '/usr/local/lib/python3.10/dist-packages/nvidia'
libs = {}
for pkg in sorted(os.listdir(nv)):
    lib_dir = os.path.join(nv, pkg, 'lib')
    if not os.path.isdir(lib_dir):
        continue
    for f in os.listdir(lib_dir):
        m = re.match(r'(.+\.so\.\d+)', f)
        if m and not os.path.islink(os.path.join(lib_dir, f)):
            libs[pkg] = libs.get(pkg, []) + [m.group(1)]
for pkg, ls in sorted(libs.items()):
    for l in sorted(ls):
        print(f'{pkg} {l}')
\"
" 2>/dev/null)

echo "Found sonames:"
echo "$SONAME_LIST"

# ── Step 2: Build stubs locally ───────────────────────────────────────────────
echo ""
echo "Step 2: Building stubs..."

BUILD_DIR=$(mktemp -d)
NV_DIR="$BUILD_DIR/nvidia"
mkdir -p "$NV_DIR"
touch "$NV_DIR/__init__.py"

_make_stub() {
    local pkg="$1" lib="$2"
    local dir="$NV_DIR/$pkg/lib"
    mkdir -p "$dir"
    touch "$NV_DIR/$pkg/__init__.py"
    if [ ! -f "$dir/$lib" ]; then
        # Create stub with empty functions + versioned symbol for linker compatibility
        cat > /tmp/stub_src.c << 'CSRC'
/* CUDA stub for TPU VMs — satisfies dlopen without providing real CUDA */
void __attribute__((constructor)) stub_init(void) {}
CSRC
        gcc -shared -fPIC -Wl,-soname,"$lib" -o "$dir/$lib" /tmp/stub_src.c 2>/dev/null && \
            echo "  + $pkg/$lib" || echo "  WARN: gcc failed for $lib"
    fi
}

# Parse soname list and build stubs
while IFS=' ' read -r pkg lib; do
    [ -z "$pkg" ] || [ -z "$lib" ] && continue
    _make_stub "$pkg" "$lib"
done <<< "$SONAME_LIST"

# Ensure all known-required stubs exist (fallback if ew4a SSH failed)
declare -A REQUIRED_STUBS=(
    ["cublas"]="libcublas.so.12 libcublasLt.so.12 libnvblas.so.12"
    ["cuda_cupti"]="libcupti.so.12"
    ["cuda_nvrtc"]="libnvrtc.so.12"
    ["cuda_runtime"]="libcudart.so.12"
    ["cudnn"]="libcudnn.so.9 libcudnn_adv.so.9 libcudnn_cnn.so.9 libcudnn_engines_precompiled.so.9 libcudnn_engines_runtime_compiled.so.9 libcudnn_graph.so.9 libcudnn_heuristic.so.9 libcudnn_ops.so.9"
    ["cufft"]="libcufft.so.11 libcufftw.so.11"
    ["cufile"]="libcufile.so.0"
    ["curand"]="libcurand.so.10"
    ["cusolver"]="libcusolver.so.11"
    ["cusparse"]="libcusparse.so.12"
    ["cusparselt"]="libcusparseLt.so.0"
    ["nccl"]="libnccl.so.2"
    ["nvjitlink"]="libnvJitLink.so.12"
    ["nvshmem"]="libnvshmem_host.so.3"
    ["nvtx"]="libnvToolsExt.so.1"
)
for pkg in "${!REQUIRED_STUBS[@]}"; do
    for lib in ${REQUIRED_STUBS[$pkg]}; do
        dir="$NV_DIR/$pkg/lib"
        [ -f "$dir/$lib" ] || _make_stub "$pkg" "$lib"
    done
done

# ── Step 3: Create tarball ────────────────────────────────────────────────────
echo ""
echo "Step 3: Creating tarball..."
OUTPUT_TGZ="$OUTPUT_DIR/nvidia_stubs_v6e.tar.gz"
tar czf "$OUTPUT_TGZ" -C "$BUILD_DIR" nvidia/
rm -rf "$BUILD_DIR"

SIZE=$(du -sh "$OUTPUT_TGZ" | cut -f1)
echo "Created: $OUTPUT_TGZ ($SIZE)"

# ── Step 4: Upload to GCS ─────────────────────────────────────────────────────
echo ""
echo "Step 4: Upload to GCS..."
for bucket in \
    gs://gcp-researchcredits-blocklab-europe-west4 \
    gs://gcp-researchcredits-blocklab-us-east1 \
    gs://gcp-researchcredits-blocklab-1-us-central2; do
    echo "  Uploading to $bucket/wheels/nvidia_stubs_v6e.tar.gz..."
    gsutil cp "$OUTPUT_TGZ" "${bucket}/wheels/nvidia_stubs_v6e.tar.gz" && \
        echo "  ✓ $bucket" || echo "  ✗ $bucket FAILED"
done

echo ""
echo "=== Done. Test with: ==="
echo "  gcloud alpha compute tpus tpu-vm ssh v6e-ue1d-1 \\"
echo "    --zone=us-east1-d --project=$PROJECT --tunnel-through-iap \\"
echo "    --command='sudo tar xzf /tmp/nvidia_stubs_v6e.tar.gz -C /usr/local/lib/python3.10/dist-packages/ && PYTHONNOUSERSITE=1 python3 -c \"import torch; import torch_xla; print(torch_xla.device())\"'"
