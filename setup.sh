#!/bin/bash
# Full TPU VM setup script — run once on each new VM
# Usage: bash <(gsutil cat gs://gcp-researchcredits-blocklab-europe-west4/config/setup.sh)
set -e

BUCKET="${BUCKET:-gs://gcp-researchcredits-blocklab-europe-west4}"

# Fix crcmod issue on v4 VMs (composite objects need CRC32c)
echo -e "[GSUtil]\ncheck_hashes = never" > /tmp/boto_crc && export BOTO_CONFIG=/tmp/boto_crc

echo "[1/7] Loading secrets..."
gsutil cp $BUCKET/config/secrets.env /tmp/secrets.env
source /tmp/secrets.env
rm /tmp/secrets.env

echo "[2/7] Configuring git..."
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

echo "[3/7] Ensuring Python 3.10+..."
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_MINOR" -lt 10 ]; then
  echo "  Python $PY_VER detected — installing Miniconda (Python 3.10)..."
  gsutil cp $BUCKET/tools/Miniconda3-py310.sh /tmp/miniconda.sh 2>/dev/null || \
    gsutil cp gs://gcp-researchcredits-blocklab-1-us-central2/tools/Miniconda3-py310.sh /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $HOME/miniconda3 -u
  rm /tmp/miniconda.sh
  export PATH=$HOME/miniconda3/bin:$PATH
  echo "export PATH=\$HOME/miniconda3/bin:\$PATH" >> ~/.bashrc
  echo "  Now using: $(python3 --version)"
else
  echo "  Python $PY_VER OK"
fi
export PATH=$HOME/miniconda3/bin:$HOME/.local/bin:$PATH

echo "[4/7] Installing Python packages..."
# Install from pre-downloaded wheels on GCS (VMs may not have public internet)
# Force fresh download: remove file AND gsutil tracker/resume state
rm -rf /tmp/tpu_wheels /tmp/tpu_wheels.tar.gz
rm -rf ~/.gsutil/tracker-files 2>/dev/null || true
for attempt in 1 2 3; do
  rm -f /tmp/tpu_wheels.tar.gz
  gsutil cp $BUCKET/wheels/tpu_wheels.tar.gz /tmp/tpu_wheels.tar.gz
  if gzip -t /tmp/tpu_wheels.tar.gz 2>/dev/null; then
    echo "  Download verified OK (attempt $attempt)"
    break
  else
    echo "  WARNING: corrupt download (attempt $attempt), retrying..."
    rm -rf ~/.gsutil/tracker-files 2>/dev/null || true
    if [ "$attempt" = "3" ]; then echo "FATAL: tarball corrupt after 3 attempts"; exit 1; fi
  fi
done
mkdir -p /tmp/tpu_wheels && tar xzf /tmp/tpu_wheels.tar.gz -C /tmp/tpu_wheels/ --strip-components=1 2>/dev/null || tar xzf /tmp/tpu_wheels.tar.gz -C /tmp/tpu_wheels/
python3 -m pip install --upgrade pip setuptools wheel -f /tmp/tpu_wheels --no-index -q
pip3 install /tmp/tpu_wheels/*.whl --no-deps --force-reinstall -q
rm -rf /tmp/tpu_wheels /tmp/tpu_wheels.tar.gz

echo "[5/8] Logging into W&B..."
mkdir -p ~/.config/wandb
echo -e "machine api.wandb.ai\n  login user\n  password $WANDB_API_KEY" > ~/.netrc
chmod 600 ~/.netrc

echo "[6/8] Deploying code from GCS..."
mkdir -p ~/sf_bema/experiments
gsutil cp $BUCKET/code/sf_bema_code_12c.tar.gz /tmp/sf_bema_code_12c.tar.gz
tar -xz -C ~/sf_bema/experiments/ -f /tmp/sf_bema_code_12c.tar.gz

echo "[7/8] Downloading dataset..."
mkdir -p ~/sf_bema/experiments/exp10_smollm2_smoltalk/data
gsutil -m cp -r "$BUCKET/data/smoltalk/data/train" "$BUCKET/data/smoltalk/data/val" ~/sf_bema/experiments/exp10_smollm2_smoltalk/data/

echo "[8/8] Setting XLA compilation cache + mmap limit..."
echo "export XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache" >> ~/.bashrc
echo "export PJRT_DEVICE=TPU" >> ~/.bashrc
mkdir -p /tmp/xla_cache
export XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache
export PJRT_DEVICE=TPU
sudo sysctl -w vm.max_map_count=2147483647

echo ""
echo "✓ Setup done. VM is ready for training."
