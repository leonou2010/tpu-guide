# Setup Packages (v3)

This folder contains per-VM-type setup packages plus a small unit-test runner.

Goal: a *dedicated* setup package for each VM class (v4, v6e US, v6e EU) that can:
- sanitize the VM to a known baseline
- ensure model + dataset are present locally
- run a short TPU smoke training workload
- upload a loss JSONL artifact to GCS (proof of correctness)
- emit a machine-readable result JSON for reporting

The packages are intentionally small and fetch large artifacts (data/model/wheels)
from GCS at runtime.

