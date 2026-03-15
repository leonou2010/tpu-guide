# VM Reliability Proposal — v4 Design

## Observed Failure Modes (v3)

1. **Env contract not enforced** — deploy scripts don't consistently export BUCKET/CONTROL_PLANE/GCS_CHECKPOINT_DIR. Babysitter starts but uploads go to wrong place or nowhere. Symptom: `BUCKET: NOT SET` in babysitter log.

2. **Broken data symlinks** — sf_bema tarball built with `data → /scratch/blocklab/...`. ue1d/uc2b have no `/scratch`. Deploy "succeeds" (deps install OK) but training crashes with `FileNotFoundError: data/train not found`. Fixed in v3 per-zone scripts but not enforced.

3. **Multi-host barrier misconfiguration** — `train_tpu.py` defaults `TPU_NUM_WORKERS=16`. Babysitter sets `TPU_NUM_WORKERS=1` in subprocess env, but any out-of-band training invocation hangs forever in rendezvous.

4. **GCS download reliability on no-internet VMs** — composite objects (large tarballs) fail without `crcmod` C extension (`GSUtil:check_hashes=never` workaround required). Reoccurs for any new large artifact.

5. **v5e provisioning instability** — repeated `VM creation failed` in vm_manager log. Not a setup issue — VM never reliably exists. HEALTH_CHECKS quota exhaustion blocks ALL new VM creation across all types.

6. **xla_compile label overloaded** — `status=xla_compile` means `step==0`, which covers: startup + dep install check + XLA compilation + first step. Can mask setup failures that look like "slow compile."

7. **Per-zone script drift** — `deploy_ew4a.sh`, `deploy_ue1d.sh`, `deploy_uc2b.sh` have diverged. Invariants enforced in one script may be missing in another. No shared validation layer.

---

## P0 Fixes (Highest Impact)

### 1. Canary Training Gate Before Task Claiming

Before babysitter enters the claim loop, run a 1-step smoke train:
```bash
timeout 300 python3 -u train_tpu.py ... +training.max_steps=1 TRAIN_LOSS_JSONL=/tmp/canary.jsonl
```
- Require that `/tmp/canary.jsonl` exists and contains valid JSON
- Only after canary passes: start babysitter claim loop
- If canary fails: write `telemetry/{TPU_NAME}_boot.json` with phase=FAILED_CANARY + error
- vm_manager sees FAILED_CANARY → force redeploy

This would have caught: broken symlink (FileNotFoundError), missing model, TPU init fail, missing deps.

### 2. Hard Env Contract

Define required env before babysitter starts:
```bash
# Required — fail fast if missing
: "${TPU_NAME:?TPU_NAME not set}"
: "${ZONE:?ZONE not set}"
: "${BUCKET:?BUCKET not set}"

# Required paths
test -f /tmp/SmolLM2-135M/config.json || { echo "FATAL: model missing"; exit 1; }
test -d "${_DATA_DIR}/train" || { echo "FATAL: data/train missing"; exit 1; }
python3 -c "import torch_xla; import omegaconf; import hydra" || { echo "FATAL: deps missing"; exit 1; }
```

### 3. Data Path — No /scratch Assumptions

All deploy scripts must:
```bash
# Remove broken symlink before mkdir
[ -L "${_DATA_DIR}" ] && rm -f "${_DATA_DIR}"
mkdir -p "${_DATA_DIR}/train" "${_DATA_DIR}/val"
```
Verify after download:
```bash
python3 -c "from datasets import load_from_disk; load_from_disk('${_DATA_DIR}/train')" || \
    { echo "FATAL: data/train not loadable"; exit 1; }
```

### 4. Richer Status Labels

Babysitter heartbeat status should distinguish:
```
booting           → VM just started, downloading deps
downloading_data  → gsutil cp in progress
canary            → running 1-step smoke test
idle              → no task, polling
xla_compile       → claimed task, step==0 (true XLA compile)
training          → step > 0
done              → no more tasks
```

Replace binary `xla_compile`/`idle` split with the above. Lets dashboard show real state.

---

## P1 Improvements

### 5. READY Registry

Only VMs that passed canary can claim tasks:
```
coord_v2/ready/{TPU_NAME}.json   → written after canary pass
coord_v2/quarantine/{TPU_NAME}.json → written after N canary failures
```
`gcs.py claim_task()` checks `ready/{worker_prefix}.json` before claiming.

### 6. Versioned XLA Cache

Key cache by `(torch_xla_version, libtpu_hash, model_config_hash)`:
```
xla_cache_v6e_tx290_ltp_<hash>/
xla_cache_v4_tx260_ltp_<hash>/
```
Prevents cache-busting when versions change. Old cache silently produces UNIMPLEMENTED warnings (self-healing but wastes 5 min).

### 7. v5e Via QueuedResource Only

v5e VMs never reliably provisioned via direct `tpu-vm create`. Always use QueuedResource:
- GCP handles capacity queuing
- FAILED/SUSPENDED → delete + requeue automatically
- Reduces "creation loop" log noise

---

## P2 (Future)

### 8. Baked Artifact

Instead of on-VM pip install + GCS wheel bundle:
- Pre-built GCS tarball per (arch, torch version): `setup_v6e_eu.tar.gz`, `setup_v4.tar.gz`
- VM boot: download + unpack (no pip), run canary
- Eliminates dep drift, reduces setup time from 10-15 min to 2-3 min
- Already partially done in v3 (`/tmp/setup_packages/setup_v4.tar.gz` etc.)

### 9. HEALTH_CHECKS Auto-Cleanup

vm_manager.py: when quota near-full AND pending==0, delete idle VMs (>10 min all-chips-idle).
Already implemented in v3 (`IDLE_SCALE_DOWN_S = 600`).

---

## SRE Runbook — Triage Checklist

### VM appears alive but not training
```bash
# 1. Check boot telemetry
gsutil cat gs://.../coord_v2/telemetry/{TPU_NAME}_boot.json

# 2. Check babysitter log
gcloud alpha compute tpus tpu-vm ssh {VM} --zone={ZONE} \
    --project=gcp-research-credits-489020 --tunnel-through-iap \
    --command='tail -50 /tmp/babysitter.log'

# 3. Check heartbeat
gsutil cat gs://.../coord_v2/heartbeats/{TPU_NAME}_chip0.json

# 4. Check running tasks assigned to this VM
gsutil ls gs://.../coord_v2/running/ | while read f; do
  gsutil cat "$f" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['worker_id'], d['task_id'])" 2>/dev/null
done | grep {TPU_NAME}
```

### Task in failed/ permanently
```bash
# Check last_error
gsutil cat gs://.../coord_v2/failed/{task_id}.json | python3 -m json.tool

# Check failure log
gsutil cat gs://.../coord_v2/logs/task_fail_{task_id}.log | tail -30

# Requeue (reset retries)
task_id="exp13_rerun3__v4_c0.001_lr0.001_k0.5"
gsutil cat gs://.../failed/${task_id}.json | python3 -c "
import json,sys; d=json.load(sys.stdin)
d['retries']=0; d.pop('last_error',None); d.pop('failed_at',None)
print(json.dumps(d))" | gsutil cp - gs://.../pending/${task_id}.json
gsutil rm gs://.../failed/${task_id}.json
```

### HEALTH_CHECKS quota exhausted
```bash
# Check quota
gcloud compute project-info describe --project=gcp-research-credits-489020 \
    --format='table(quotas.metric,quotas.usage,quotas.limit)' | grep HEALTH

# Delete idle VMs manually to free quota
gcloud alpha compute tpus tpu-vm delete {VM} --zone={ZONE} \
    --project=gcp-research-credits-489020 --quiet
```

### KPIs to track
- VMs in each state: provisioning / booting / canary / training / xla_compile / failed
- Time-to-ready: VM create → canary pass (p50 target: <15 min)
- Task success rate: validated / (validated + permanently_failed)
- Chip utilization: chips_training / total_chips_in_fleet
