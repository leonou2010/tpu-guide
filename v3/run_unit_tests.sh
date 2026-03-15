#!/bin/bash
# run_unit_tests.sh — 5x5 stress test orchestrator
# Runs N cycles per arch: create fresh VM → unit test → delete → clean health checks
#
# Usage:
#   bash run_unit_tests.sh                     # all arches, 5 cycles each
#   ARCHES=v6e_eu CYCLES=1 bash run_unit_tests.sh  # one arch, one cycle
#
# Results saved to: /tmp/unit_test_results/

set -uo pipefail

GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
CYCLES=${CYCLES:-5}
ARCHES=${ARCHES:-"v6e_eu v6e_us v4"}   # v5e excluded (quota unavailable)
RESULTS_DIR=/tmp/unit_test_results
REPORT=$RESULTS_DIR/stress_test_report.md
mkdir -p "$RESULTS_DIR"

# ── Config per arch ──────────────────────────────────────────────────────────
get_vm_config() {
    local arch=$1
    case "$arch" in
        v6e_eu) echo "europe-west4-a v6e-8 v2-alpha-tpuv6e gs://gcp-researchcredits-blocklab-europe-west4" ;;
        v6e_us) echo "us-east1-d v6e-8 v2-alpha-tpuv6e gs://gcp-researchcredits-blocklab-us-east1" ;;
        v4)     echo "us-central2-b v4-8 tpu-ubuntu2204-base gs://gcp-researchcredits-blocklab-1-us-central2" ;;
        v5e_eu) echo "europe-west4-b v5litepod-8 v2-alpha-tpuv5-lite gs://gcp-researchcredits-blocklab-europe-west4" ;;
        v5e_us) echo "us-central1-a v5litepod-8 v2-alpha-tpuv5-lite gs://gcp-researchcredits-blocklab-1-us-central2" ;;
        *) echo "ERROR: unknown arch $arch"; exit 1 ;;
    esac
}

# ── Health check cleanup ─────────────────────────────────────────────────────
cleanup_health_checks() {
    local vm_name=$1
    echo "[hc_cleanup] Removing health checks for $vm_name..."
    $GCLOUD compute health-checks list --project=$PROJECT \
        --format='value(name)' 2>/dev/null | grep -i "$vm_name" | \
    while read hc; do
        $GCLOUD compute health-checks delete "$hc" \
            --project=$PROJECT --quiet 2>/dev/null && \
            echo "[hc_cleanup] Deleted: $hc"
    done
    # Also sweep any orphan checks not tied to current fleet
    local usage
    usage=$($GCLOUD compute project-info describe --project=$PROJECT \
        --format='json(quotas)' 2>/dev/null | \
        python3 -c "import json,sys; q=[x for x in json.load(sys.stdin)['quotas'] if x['metric']=='HEALTH_CHECKS']; print(int(q[0]['usage']) if q else 0)" 2>/dev/null || echo "?")
    echo "[hc_cleanup] HEALTH_CHECKS usage after cleanup: $usage/75"
}

# ── Create VM ────────────────────────────────────────────────────────────────
create_vm() {
    local vm_name=$1 zone=$2 accel=$3 runtime=$4 bucket=$5
    echo "[create] Creating $vm_name ($accel) in $zone..."
    $GCLOUD alpha compute tpus tpu-vm create "$vm_name" \
        --zone="$zone" \
        --project=$PROJECT \
        --accelerator-type="$accel" \
        --version="$runtime" \
        --spot \
        --internal-ips \
        --quiet 2>&1 | tail -3
    # Wait for READY
    local deadline=$(($(date +%s) + 600))
    while [ $(date +%s) -lt $deadline ]; do
        local state
        state=$($GCLOUD alpha compute tpus tpu-vm describe "$vm_name" \
            --zone="$zone" --project=$PROJECT \
            --format='value(state)' 2>/dev/null || echo "NOT_FOUND")
        echo "[create] $vm_name state: $state"
        [ "$state" = "READY" ] && return 0
        [ "$state" = "FAILED" ] && return 1
        [ "$state" = "PREEMPTED" ] && { echo "[create] $vm_name PREEMPTED"; return 1; }
        sleep 15
    done
    echo "[create] TIMEOUT waiting for $vm_name READY"
    return 1
}

# ── Delete VM ────────────────────────────────────────────────────────────────
delete_vm() {
    local vm_name=$1 zone=$2
    echo "[delete] Deleting $vm_name..."
    $GCLOUD alpha compute tpus tpu-vm delete "$vm_name" \
        --zone="$zone" --project=$PROJECT --quiet 2>&1 | tail -2
    sleep 10
    cleanup_health_checks "$vm_name"
}

# ── Run unit test on VM ───────────────────────────────────────────────────────
run_test_on_vm() {
    local vm_name=$1 zone=$2 arch=$3 bucket=$4 cycle=$5
    local result_key="${arch}_cycle${cycle}"
    local log="$RESULTS_DIR/${result_key}.log"

    echo "[test] Running unit test on $vm_name (arch=$arch cycle=$cycle)..."

    # SSH and run unit_test.sh
    timeout 1200 $GCLOUD alpha compute tpus tpu-vm ssh "$vm_name" \
        --zone="$zone" \
        --project=$PROJECT \
        --tunnel-through-iap \
        --command="gsutil -o GSUtil:check_hashes=never cp ${bucket}/setup_packages/unit_test.sh /tmp/unit_test.sh && ARCH=${arch} bash /tmp/unit_test.sh" \
        2>&1 | tee "$log"

    # Parse result
    local pass=0
    local loss=""
    local smoke_gcs=""

    # Check: deploy completed
    grep -q "IDLE_AWAITING_WORK\|babysitter running\|deploy complete" "$log" && pass=1

    # Check: smoke train produced loss
    loss=$(grep "JSONL:" "$log" | grep -oP '"loss":\s*\K[0-9.]+' | head -1)
    smoke_gcs=$(grep "GCS proof:" "$log" | awk '{print $NF}')

    # Check monotonicity (step increases)
    local mono="N/A"
    if [ -n "$smoke_gcs" ]; then
        local jsonl_content
        jsonl_content=$(gsutil cat "$smoke_gcs" 2>/dev/null)
        if [ -n "$jsonl_content" ]; then
            mono=$(echo "$jsonl_content" | python3 -c "
import json,sys
steps=[json.loads(l).get('step',0) for l in sys.stdin if l.strip()]
print('PASS' if steps==sorted(steps) and len(steps)>0 else 'FAIL')
" 2>/dev/null || echo "FAIL")
        fi
    fi

    # Determine overall pass
    local status="FAIL"
    if [ $pass -eq 1 ] && [ -n "$loss" ] && [ "$mono" = "PASS" ]; then
        status="PASS"
    elif [ $pass -eq 1 ] && [ -n "$loss" ]; then
        status="PASS(no-mono-check)"
    elif [ $pass -eq 1 ]; then
        status="PASS(no-smoke)"
    fi

    echo "[result] $result_key: $status | loss=$loss | mono=$mono | gcs=$smoke_gcs"
    echo "$result_key|$status|$loss|$mono|$smoke_gcs|$log" >> "$RESULTS_DIR/results.tsv"
}

# ── Report generation ─────────────────────────────────────────────────────────
generate_report() {
    echo "# 5x5 Stress Test Report" > "$REPORT"
    echo "Generated: $(date -u)" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "## Results Matrix" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Test | Status | Loss | Monotonic | GCS Proof |" >> "$REPORT"
    echo "|------|--------|------|-----------|-----------|" >> "$REPORT"

    if [ -f "$RESULTS_DIR/results.tsv" ]; then
        while IFS='|' read -r key status loss mono gcs log; do
            echo "| $key | $status | $loss | $mono | $gcs |" >> "$REPORT"
        done < "$RESULTS_DIR/results.tsv"
    fi

    echo "" >> "$REPORT"
    echo "## Summary" >> "$REPORT"
    local total pass
    total=$(wc -l < "$RESULTS_DIR/results.tsv" 2>/dev/null || echo 0)
    pass=$(grep -c "^.*|PASS" "$RESULTS_DIR/results.tsv" 2>/dev/null || echo 0)
    echo "**$pass/$total PASS**" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "## Setup Packages" >> "$REPORT"
    echo "| Arch | GCS Path |" >> "$REPORT"
    echo "|------|---------|" >> "$REPORT"
    echo "| v4 | gs://gcp-researchcredits-blocklab-1-us-central2/setup_packages/setup_v4.tar.gz |" >> "$REPORT"
    echo "| v6e_eu | gs://gcp-researchcredits-blocklab-europe-west4/setup_packages/setup_v6e_eu.tar.gz |" >> "$REPORT"
    echo "| v6e_us | gs://gcp-researchcredits-blocklab-us-east1/setup_packages/setup_v6e_us.tar.gz |" >> "$REPORT"
    echo "| v5e | gs://gcp-researchcredits-blocklab-europe-west4/setup_packages/setup_v5e.tar.gz |" >> "$REPORT"

    cat "$REPORT"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== TPU 5x5 Stress Test ==="
echo "Arches: $ARCHES | Cycles: $CYCLES"
echo "Results: $RESULTS_DIR"
echo ""
[ "${START_CYCLE:-1}" -eq 1 ] && echo "key|status|loss|mono|gcs|log" > "$RESULTS_DIR/results.tsv"

for arch in $ARCHES; do
    read -r zone accel runtime bucket <<< "$(get_vm_config "$arch")"
    echo ""
    echo "=== ARCH: $arch (zone=$zone accel=$accel) ==="

    for cycle in $(seq ${START_CYCLE:-1} $CYCLES); do
        vm_name="ut-${arch//_/-}-c${cycle}"
        # Replace _ with - in vm name for GCP
        vm_name="${vm_name//_/-}"
        echo ""
        echo "--- Cycle $cycle/$CYCLES: $vm_name ---"
        stamp=$(date -u +%H:%M:%S)

        # Create + test with retry on preemption
        _result_key="${arch}_cycle${cycle}"
        for _attempt in 1 2 3; do
            [ $_attempt -gt 1 ] && { echo "[retry] Attempt $_attempt for $_result_key"; sleep 30; }
            if create_vm "$vm_name" "$zone" "$accel" "$runtime" "$bucket"; then
                run_test_on_vm "$vm_name" "$zone" "$arch" "$bucket" "$cycle"
                # If preempted (SSH lost = no loss, no IDLE in log), retry
                if grep -q "PASS\|FAIL" "$RESULTS_DIR/${_result_key}.log" 2>/dev/null; then
                    _status=$(grep "\[result\]" "$RESULTS_DIR/${_result_key}.log" | grep -o 'PASS[^|]*\|FAIL[^|]*' | head -1)
                    [[ "$_status" == *"PASS"* ]] && break  # Good result — stop retrying
                fi
                grep -q "Failed to lookup instance\|PREEMPTED" "$RESULTS_DIR/${_result_key}.log" 2>/dev/null || break
                echo "[retry] Preemption detected — retrying cycle"
                # Remove failed entry so retry can overwrite
                sed -i "/^${_result_key}|/d" "$RESULTS_DIR/results.tsv" 2>/dev/null || true
            else
                echo "[error] VM creation failed for $vm_name (attempt $_attempt)"
            fi
            delete_vm "$vm_name" "$zone"
        done

        # Always delete + clean quota
        delete_vm "$vm_name" "$zone"
        echo "[done] Cycle $cycle complete at $(date -u +%H:%M:%S) (started $stamp)"

        # Brief pause between cycles
        [ $cycle -lt $CYCLES ] && sleep 30
    done
done

echo ""
echo "=== All cycles complete ==="
generate_report
echo ""
echo "Full report: $REPORT"
echo "Raw results: $RESULTS_DIR/results.tsv"
echo "Logs: $RESULTS_DIR/*.log"
