#!/usr/bin/env bash
set -euo pipefail

TPU_NAME="${TPU_NAME:?TPU_NAME required}"
ZONE="${ZONE:?ZONE required}"
BUCKET="${BUCKET:?BUCKET required}"
CONTROL_PLANE="${CONTROL_PLANE:-gs://gcp-researchcredits-blocklab-europe-west4/coord_v2}"

EXP_DIR="${EXP_DIR:-$HOME/sf_bema/experiments/exp13_smollm2_smoltalk}"
MODEL_DIR="${MODEL_DIR:-/tmp/SmolLM2-135M}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"

RUNSTAMP="${RUNSTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
TEST_ID="${TEST_ID:-unit_${RUNSTAMP}_${TPU_NAME}_chip0}"
GCS_OUT="${GCS_OUT:-${BUCKET}/checkpoints/${TEST_ID}}"

echo "[setup_v6e_us] TPU_NAME=${TPU_NAME} ZONE=${ZONE} BUCKET=${BUCKET}"
echo "[setup_v6e_us] EXP_DIR=${EXP_DIR}"
echo "[setup_v6e_us] GCS_OUT=${GCS_OUT}"

export TPU_NAME ZONE BUCKET CONTROL_PLANE RUNSTAMP TEST_ID GCS_OUT

# Ensure experiment code exists on a fresh VM.
# NOTE: Don't pre-create EXP_DIR before this, or deploy_ue1d.sh will skip downloading code.
if [ ! -f "${EXP_DIR}/exp13_tpu_rerun/train_tpu.py" ]; then
  echo "[setup_v6e_us] Experiment code missing — downloading sf_bema_exp13_rerun3.tar.gz..."
  mkdir -p "$HOME/sf_bema"
  # Preserve any existing data payload (might be large) before we replace the directory.
  if [ -e "${EXP_DIR}/data" ]; then
    rm -rf /tmp/exp13_data_backup 2>/dev/null || true
    mv "${EXP_DIR}/data" /tmp/exp13_data_backup 2>/dev/null || true
  fi
  rm -rf "${EXP_DIR}" 2>/dev/null || true
  (gsutil cp "${BUCKET}/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz 2>/dev/null || \
   gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/code/sf_bema_exp13_rerun3.tar.gz" /tmp/sf_bema_code.tar.gz)
  tar xzf /tmp/sf_bema_code.tar.gz -C "$HOME/sf_bema/"
  if [ -e /tmp/exp13_data_backup ]; then
    rm -rf "${EXP_DIR}/data" 2>/dev/null || true
    mv /tmp/exp13_data_backup "${EXP_DIR}/data"
  fi
fi

mkdir -p "${EXP_DIR}"

if [ -L "${DATA_DIR}" ]; then
  echo "[setup_v6e_us] Removing data symlink: ${DATA_DIR} -> $(readlink "${DATA_DIR}")"
  rm -f "${DATA_DIR}"
fi
mkdir -p "${DATA_DIR}"

echo "[setup_v6e_us] Ensuring dataset present (smoltalk)..."
if [ ! -d "${DATA_DIR}/train" ]; then
  gsutil -m cp -r "${BUCKET}/data/smoltalk/data/train" "${DATA_DIR}/" 2>/dev/null || true
fi
if [ ! -d "${DATA_DIR}/val" ]; then
  gsutil -m cp -r "${BUCKET}/data/smoltalk/data/val" "${DATA_DIR}/" 2>/dev/null || true
fi
[ -d "${DATA_DIR}/train" ] || { echo "[setup_v6e_us] ERROR: missing ${DATA_DIR}/train after download"; exit 1; }
[ -d "${DATA_DIR}/val" ] || { echo "[setup_v6e_us] ERROR: missing ${DATA_DIR}/val after download"; exit 1; }

echo "[setup_v6e_us] Ensuring model present..."
mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  gsutil -m cp "${BUCKET}/models/SmolLM2-135M/*" "${MODEL_DIR}/" 2>/dev/null || true
fi
[ -f "${MODEL_DIR}/config.json" ] || { echo "[setup_v6e_us] ERROR: missing ${MODEL_DIR}/config.json after model download"; exit 1; }

echo "[setup_v6e_us] Running environment deploy (deploy_ue1d.sh)..."
TMP_DEPLOY="/tmp/deploy_ue1d.sh"
gsutil cp "${BUCKET}/pull_code_v3/deploy_ue1d.sh" "${TMP_DEPLOY}" 2>/dev/null || \
  gsutil cp "gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/deploy_ue1d.sh" "${TMP_DEPLOY}" 2>/dev/null
chmod +x "${TMP_DEPLOY}"
TPU_NAME="${TPU_NAME}" ZONE="${ZONE}" BUCKET="${BUCKET}" CONTROL_PLANE="${CONTROL_PLANE}" FORCE_REDEPLOY=1 \
  bash "${TMP_DEPLOY}" || true

echo "[setup_v6e_us] Stopping babysitter to run isolated smoke test..."
pkill -9 -f babysitter.py 2>/dev/null || true
pkill -9 -f train_tpu.py 2>/dev/null || true
pkill -9 -f train_v2_tpu.py 2>/dev/null || true

echo "[setup_v6e_us] Running smoke training..."
cd "${EXP_DIR}"
export TPU_NUM_WORKERS=1
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=0
export LAUNCH_MODE=single
export CHIPS_PER_HOST=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache
export XLA_COMPILATION_CACHE_PATH=/tmp/xla_cache
export TRAIN_LOSS_JSONL="/tmp/${TEST_ID}.jsonl"
export GCS_CHECKPOINT_DIR="${GCS_OUT}"

python3 -u exp13_tpu_rerun/train_tpu.py \
  optimizer=adamw_ema_pullback_v4_tpu optimizer.lr=0.001 optimizer.lambda_pullback=0.01 optimizer.clamp_pullback=true \
  optimizer.ema_kappa=0.3 optimizer.use_ema_eval=false optimizer.ema_update_freq=5 +warmup_steps=50 \
  model.name="${MODEL_DIR}" model.max_seq_len=128 \
  training.batch_size=1 training.gradient_accumulation_steps=1 training.eval_interval=1 \
  +training.max_steps=3 +training.max_train_seconds=900 \
  >"/tmp/${TEST_ID}_stdout.log" 2>&1

gsutil cp "/tmp/${TEST_ID}_stdout.log" "${GCS_OUT}/${TEST_ID}_stdout.log" 2>/dev/null || true

RUN_NAME="$(grep '^DONE:' "/tmp/${TEST_ID}_stdout.log" | tail -1 | sed 's/^DONE: //')"
export RUN_NAME
python3 - <<PY >"/tmp/${TEST_ID}_loss_result.json"
import json, os, time
test_id=os.environ['TEST_ID']
run_name=os.environ.get('RUN_NAME','')
out=os.environ['GCS_OUT']
result={
  "test_id": test_id,
  "vm_id": os.environ.get("TPU_NAME",""),
  "zone": os.environ.get("ZONE",""),
  "vm_type": "v6e_us",
  "bucket": os.environ.get("BUCKET",""),
  "control_plane": os.environ.get("CONTROL_PLANE",""),
  "run_name": run_name,
  "gcs_train_loss_jsonl": f"{out}/{run_name}_train_loss.jsonl" if run_name else "",
  "gcs_stdout_log": f"{out}/{test_id}_stdout.log",
  "timestamp": time.time(),
}
print(json.dumps(result, sort_keys=True))
PY

gsutil cp "/tmp/${TEST_ID}_loss_result.json" "${GCS_OUT}/${TEST_ID}_loss_result.json" 2>/dev/null || true
echo "LOSS_RESULT: $(cat "/tmp/${TEST_ID}_loss_result.json")"
