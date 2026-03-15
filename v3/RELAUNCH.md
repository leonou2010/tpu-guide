# exp13_rerun3 Relaunch Guide
# Generated: 2026-03-15 02:41 UTC

## Current State (saved before shutdown)
- **112/120 validated** — run was stopped deliberately to clean up health checks
- **8 tasks remaining** (some in running/, some failed) — must requeue before next launch
- VMs: all 15 deleted (health check cleanup)

## Pre-Relaunch Checklist

### 1. Verify HEALTH_CHECKS quota is free
```bash
gcloud compute project-info describe --project=gcp-research-credits-489020 \
    --format='table(quotas.metric,quotas.usage,quotas.limit)' | grep HEALTH
# Need: usage < 10 (should be near 0 after VM deletions)
```

### 2. Move stale running tasks back to pending
```bash
# Tasks in running/ with no active workers need to be reclaimed
python3 ~/distributed_tpu_training/v3/monitor.py --once --stale-ttl 60 \
    --exp exp13_rerun3:120 2>/dev/null
# Or manual requeue:
gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/running/ | while read f; do
  task_id=$(basename "$f" .json)
  gsutil cat "$f" | python3 -c "
import json,sys; d=json.load(sys.stdin)
d.pop('worker_id',None); d.pop('claimed_at',None); d['status']='pending'
print(json.dumps(d))" | gsutil cp - "gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/pending/${task_id}.json"
  gsutil rm "$f"
done
```

### 3. Check what's still needed
```bash
# See what's not yet validated
comm -23 \
  <(python3 ~/distributed_tpu_training/v3/populate.py --dry-run 2>/dev/null | grep task_id | sort) \
  <(ls ~/sf_bema/results/exp13_rerun3/validated/ | sed 's/.json//' | sort)
# Quick count:
echo "Validated: $(ls ~/sf_bema/results/exp13_rerun3/validated/ | wc -l)/120"
echo "Pending: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/pending/ | wc -l)"
echo "Running: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/running/ | wc -l)"
echo "Failed: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/failed/ | wc -l)"
```

### 4. Requeue any failed tasks
```bash
# Requeue all failed (reset retries to 0)
for f in $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/failed/ 2>/dev/null); do
  task_id=$(basename "$f" .json)
  gsutil cat "$f" | python3 -c "
import json,sys; d=json.load(sys.stdin)
d['retries']=0; d.pop('last_error',None); d.pop('failed_at',None); d['status']='pending'
print(json.dumps(d))" | gsutil cp - "gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/pending/${task_id}.json"
  gsutil rm "$f"
  echo "Requeued: ${task_id##*__}"
done
```

### 5. Clear ALL health checks (reset quota to 0 before launch)
```bash
~/google-cloud-sdk/bin/gcloud compute health-checks list \
    --project=gcp-research-credits-489020 --format='value(name)' | \
  xargs -I{} ~/google-cloud-sdk/bin/gcloud compute health-checks delete {} \
    --project=gcp-research-credits-489020 --quiet 2>/dev/null
# Confirm 0/75:
~/google-cloud-sdk/bin/gcloud compute project-info describe \
    --project=gcp-research-credits-489020 \
    --format='table(quotas.metric,quotas.usage,quotas.limit)' | grep HEALTH
```
# Safe to delete all — vm_manager recreates what it needs. Always do this before launch.

## Launch Commands

### Start vm_manager (creates all VMs automatically)
```bash
nohup python3 -u ~/distributed_tpu_training/v3/vm_manager.py \
    >> /tmp/vm_manager_v3.log 2>&1 &
echo "PID: $!"
```

### Start monitor (validates completions, reclaims stale)
```bash
nohup python3 -u ~/distributed_tpu_training/v3/monitor.py \
    --exp exp13_rerun3:120 --interval 60 --stale-ttl 1800 \
    >> /tmp/monitor_v3.log 2>&1 &
echo "PID: $!"
```

### Watch progress
```bash
watch -n60 'echo "Validated: $(ls ~/sf_bema/results/exp13_rerun3/validated/ | wc -l)/120"; \
  echo "Running: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/running/ 2>/dev/null | wc -l)"; \
  echo "Pending: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/pending/ 2>/dev/null | wc -l)"; \
  echo "Failed: $(gsutil ls gs://gcp-researchcredits-blocklab-europe-west4/coord_v2/failed/ 2>/dev/null | wc -l)"'
```

## Known Bugs Fixed in v3 (do NOT regress)

### 1. Broken /scratch symlink → data/train not found
- **File**: deploy_ue1d.sh, deploy_uc2b.sh (line ~248)
- **Fix**: `[ -L "${_DATA_DIR}" ] && rm -f "${_DATA_DIR}"` before mkdir
- **Verify**: `grep -n "scratch\|symlink" ~/distributed_tpu_training/v3/deploy_uc2b.sh`

### 2. TPU infra errors misclassified as code errors → permanent fail
- **File**: babysitter.py line ~587
- **Fix**: Added `'TPU initialization failed'`, `'Deadline exceeded'`, `'PRE_START_SESSION_BARRIER'` to infra_fail patterns
- **Verify**: `grep -A5 "infra_fail" ~/distributed_tpu_training/v3/babysitter.py`

### 3. experiment=None in upload_result()
- **File**: babysitter.py
- **Fix**: `_resolve_exp(task)` helper falls back to task_id prefix
- **Verify**: `grep "_resolve_exp" ~/distributed_tpu_training/v3/babysitter.py`

### 4. XLA_STUCK_S too short (killed babysitters mid-compile)
- **File**: vm_manager.py line ~53
- **Fix**: XLA_STUCK_S = 7200 (was 2100)
- **Verify**: `grep XLA_STUCK_S ~/distributed_tpu_training/v3/vm_manager.py`

### 5. ew4a VMs had no XLA cache download → cold compile every redeploy
- **File**: deploy_ew4a.sh
- **Fix**: Added XLA cache download from gs://gcp-researchcredits-blocklab-us-east1/xla_cache_v6e/
- **Verify**: `grep xla_cache ~/distributed_tpu_training/v3/deploy_ew4a.sh`

### 6. Empty pending task files block claim_task()
- **Symptom**: Chips idle despite tasks in pending/, claim_task() returns None
- **Detection**: `gsutil stat <file> | grep Content-Length` → 0 bytes
- **Fix**: Delete empty files + re-run populate.py

### 7. vm_manager crashes → 60 chips go dead silently
- **Symptom**: All ue1d + uc2b heartbeats disappear, vm_manager log stops
- **Detection**: `pgrep -f vm_manager.py` returns nothing
- **Fix**: Restart vm_manager, it auto-redeploys unhealthy babysitters

### 8. gsutil cp -r directory collision
- **Fix**: Use explicit glob `bucket/subdir/*` with pre-created destination dirs

### 9. TPU_NAME overwritten by GCP metadata
- **Symptom**: Heartbeats go to `t1v-n-*` instead of named VM keys
- **Fix**: babysitter uses TPU_NAME from env, not metadata; vm_manager sets it in deploy command

### 10. HEALTH_CHECKS quota exhaustion blocks ALL VM creation
- **Quota**: 75/75 = blocks all types (v4, v5e, v6e)
- **Empirical rate**: ~5 health checks per VM. 16 v6e VMs = ~80 checks → exceeds 75 limit.
- **Fix**: Just delete all health checks — safe to do at any time, vm_manager recreates what it needs:
```bash
~/google-cloud-sdk/bin/gcloud compute health-checks list \
    --project=gcp-research-credits-489020 --format='value(name)' | \
  xargs -I{} ~/google-cloud-sdk/bin/gcloud compute health-checks delete {} \
    --project=gcp-research-credits-489020 --quiet 2>/dev/null
# Verify clean:
~/google-cloud-sdk/bin/gcloud compute project-info describe \
    --project=gcp-research-credits-489020 \
    --format='table(quotas.metric,quotas.usage,quotas.limit)' | grep HEALTH
```
- **When to run**: if vm_manager log shows "HEALTH_CHECKS near limit" or new VMs fail to create

## Setup Packages (local + GCS)
| Arch | Local | GCS |
|------|-------|-----|
| v4 (uc2b) | /tmp/setup_packages/setup_v4.tar.gz | gs://gcp-researchcredits-blocklab-1-us-central2/setup_packages/ |
| v6e EU (ew4a) | /tmp/setup_packages/setup_v6e_eu.tar.gz | gs://gcp-researchcredits-blocklab-europe-west4/setup_packages/ |
| v6e US (ue1d) | /tmp/setup_packages/setup_v6e_us.tar.gz | gs://gcp-researchcredits-blocklab-us-east1/setup_packages/ |
| v5e (ew4b/uc1a) | /tmp/setup_packages/setup_v5e.tar.gz | both buckets /setup_packages/ |

## Fleet Definition
```
v6e-ew4a-1..8  europe-west4-a   v6e-8  (internet, 8 chips each) — 64 chip quota, fully tested ✅
v6e-ue1d-1..8  us-east1-d       v6e-8  (NO internet, 8 chips each) — 64 chip quota, fully tested ✅
v4-uc2b-1..5   us-central2-b    v4-8   (NO internet, 4 chips each) — zone unreliable (code 13 GCP internal errors)
v5e-ew4b-1..3  europe-west4-b   v5e-8  (NO internet, QUOTA UNAVAILABLE — serving quota 4, need GCP support ticket)
v5e-uc1a-1..3  us-central1-a    v5e-8  (NO internet, QUOTA UNAVAILABLE — serving quota 4, need GCP support ticket)
```

## Unit Test Results (2026-03-15) — v6e CONFIRMED WORKING
- v6e_eu (europe-west4-a): 5/5 PASS, loss=1.632, monotonic ✅
- v6e_us (us-east1-d): 5/5 PASS, loss=1.632, monotonic ✅
- v4 (us-central2-b): PENDING — zone down (GCP internal error since 01:00 UTC)
- Raw: /tmp/unit_test_results/results.tsv | Report: /tmp/unit_test_results/stress_test_report.md

## Design Principles (v3)

### 1. GCS is the single source of truth for everything
Before every launch, all artifacts must be on GCS — not assumed to be present on the VM:
- **Setup packages** (`setup_<arch>.tar.gz`) — wheels, deploy scripts, code, data
- **Model files** (`SmolLM2-135M/`) — downloaded by deploy script on VM creation
- **Training data** (`data/smoltalk/`) — downloaded by deploy script, verified non-empty
- **XLA cache** (`xla_cache_v6e/`) — seeded on deploy to avoid cold recompile (~15-45 min)
- **Training code** (`sf_bema_exp13_rerun3.tar.gz`) — downloaded by deploy script

If anything is missing from GCS, VMs silently fail or produce garbage. Upload everything before launch.

### 2. Fleet is actively maintained, not fire-and-forget
vm_manager continuously:
- Creates VMs when slots are empty
- Detects unhealthy babysitters and redeploys
- Cleans orphan health checks when count mismatches active VMs
- Scales down idle VMs when no pending work
- Watchdog cron (every 5 min) restarts vm_manager if it crashes

### 3. XLA compile is NOT a success indicator
A VM that reaches `xla_compile` status is NOT runnable yet. XLA compile:
- Takes 15-45 min on cold cache, 5 min with cache
- Can hang indefinitely (XLA_STUCK_S=7200 timeout)
- Does NOT mean the VM will produce valid results

**The only indicator of a truly runnable VM is actual result logs** — JSONL loss output written to GCS. Until that appears, the VM is unproven.

### 4. Monitor validates, not just counts
monitor.py:
- Downloads JSONL and verifies loss values are real numbers
- Reclaims tasks whose heartbeats go stale (worker died mid-train)
- Only increments validated/ count on confirmed good results
- Stall detection tracks validated count (not completed count)

## Efficiency Lessons
- v4 tasks failed repeatedly because data download used `|| true` — ALWAYS verify data after download
- babysitter permanent-fail after 3 retries was too aggressive — TPU infra errors look like code errors
- vm_manager has no watchdog — it can die silently, killing all redeploy logic
- Consider: `systemd` unit or cron to restart vm_manager if it dies
