# Robust TPU Orchestrator Checklist (Pull-Based System)

This checklist is for the **pull-based system** under `distributed_tpu_training/pull/` (VM babysitter + GCS blackboard + blocklab monitor + VM requester).

Primary goal: **tasks start, make progress, and finish** (validated), despite preemption, VM churn, quiet stdout during XLA compile, and partial failures.

## 0) Constraints (Non-Negotiable)
- [ ] Blocklab cannot accept inbound connections from TPU VMs (GCS blackboard only).
- [ ] Do not rely on GCS “locking”; avoid global mutable queue files.
- [ ] Some VMs have no public internet; installs must work from GCS wheel bundles.
- [ ] v6e commonly runs 8 procs/chips; v4 runs 4 in your configs; v5e is often OOM/low-capacity.

## 1) Definition Of Done (Make This Explicit)
- [ ] A task is considered **finished** only when:
- [ ] result summary exists in the expected location and parses as JSON
- [ ] validation passes (non-NaN, sufficient steps, any experiment-specific checks)
- [ ] a validated artifact is written (local `~/sf_bema/results/<exp>/validated/<label>.json` and/or `validated/<task_id>.json` on GCS)
- [ ] A task is not “done” just because it reached `completed/` (that is only a worker report).

## 2) Task Attempt Model (So Mid-Run Restarts Don’t Stall)
- [ ] Every task has a monotonic `attempt` counter stored in GCS state (`running/` record at minimum).
- [ ] Every attempt has a globally unique `attempt_id` (include `worker_id`, timestamps, and attempt number).
- [ ] Every state transition writes an append-only receipt (`receipts/<task_id>/<attempt_id>.json`) including:
- [ ] start/end timestamps, rc, failure reason, last observed step, and code version (`git_sha`).
- [ ] Retries are policy-driven:
- [ ] infra failure (preempt, lost heartbeat, SIGKILL/rc=-9, TPU lock contention) => retry (high limit)
- [ ] validation failure => retry (low limit) and mark `invalidated/`
- [ ] deterministic config error => limited retries then `failed/`

## 3) Checkpoint/Resume Contract (Optional But Required If You Care About Wasted Hours)
- [ ] Decide one of:
- [ ] `restart-from-scratch`: acceptable only for short tasks
- [ ] `resume-on-retry`: requires periodic checkpointing to durable storage (GCS)
- [ ] If resume-on-retry:
- [ ] checkpoint path is per-task per-attempt: `checkpoints/<task_id>/<attempt_id>/...` (never shared across tasks)
- [ ] on retry, worker checks for latest checkpoint and resumes deterministically
- [ ] checkpoint cleanup is label/attempt-specific (never `ckpt_*.pt` across the whole VM)

## 4) Control Plane Layout (Per Experiment)
- [ ] Define a single control plane prefix: `CONTROL_PLANE=gs://.../coord_v2` (or similar).
- [ ] Per-experiment task IDs are immutable and unique: `task_id = <exp>__<label>`.
- [ ] Task-state directories (exactly one at a time, per task_id):
- [ ] `pending/<task_id>.json`
- [ ] `running/<task_id>.json`
- [ ] `completed/<task_id>.json`
- [ ] `failed/<task_id>.json`
- [ ] `invalidated/<task_id>.json` (validation failed; separate from infra failure)
- [ ] Heartbeats:
- [ ] `heartbeats/<worker_id>.json` (must exist for every active worker proc)

## 5) Global Invariants (Monitor/Watchdog Must Enforce These)
- [ ] Uniqueness: each `task_id` is present in at most one of `pending/`, `running/`, `completed/`, `failed/`, `invalidated/`.
- [ ] Completeness: each task is present in at least one state directory (no “missing tasks”).
- [ ] Progress definition: the metric that matters is **validated results**, not “completed receipts”.
- [ ] Recovery: if a VM disappears, tasks in `running/` must return to `pending/` after TTL.
- [ ] Idempotency: re-running `populate.py` and `reclaim_stale()` is always safe.

## 6) Worker/VM Invariants (Babysitter Must Guarantee These)
- [ ] Exactly one babysitter instance per VM (not per chip):
- [ ] Use `flock` + `nohup` (or `systemd` if available) to guarantee single-instance.
- [ ] Never silently exit due to lock contention:
- [ ] Use `flock --timeout=<N>` and log clearly when it fails.
- [ ] Orphan-proof execution:
- [ ] Spawn training with its own process group (`start_new_session=True` / `setsid`).
- [ ] Always `killpg()` in `finally` on babysitter exit, and do startup cleanup of stale TPU holders.
- [ ] Heartbeats are independent of stdout:
- [ ] A background timer/thread updates heartbeat every 60–120s even during quiet phases (XLA compile).
- [ ] Chip isolation is explicit:
- [ ] One proc per chip via `TPU_VISIBLE_CHIPS` and `LAUNCH_MODE=single` (no hidden child spawns).

## 7) Validation Semantics (Stop “Phantom Complete”)
- [ ] `completed/` means “worker reported completion”, not “result is valid”.
- [ ] Monitor must validate results:
- [ ] If valid: create local `validated/<label>.json` and keep/record `completed/`.
- [ ] If invalid: move `completed/<task_id>.json` -> `invalidated/<task_id>.json` and requeue to `pending/` with attempt increment.
- [ ] Retries:
- [ ] Attempt number must be monotonic and recorded in `running/` and receipts.
- [ ] Infra failures (preempt, SIGKILL, stale heartbeat) and validation failures must be distinguishable.

## 8) Atomicity & Race Safety (GCS Is An Object Store)
- [ ] Claims must be “good enough” race-safe:
- [ ] If staying CLI-only: verify post-claim that `running/<task_id>` matches your worker_id before deleting `pending/<task_id>`.
- [ ] If races become real: migrate claim to GCS preconditions (generation-match) via Python client (blocklab only).
- [ ] Writes must not corrupt JSON:
- [ ] Never use string `repr()` tricks; write exact bytes via stdin/heredoc or temp file.

## 9) VM Acquisition Robustness (Stop Churn)
- [ ] `CREATING` handling (vm_requester):
- [ ] Track per-VM create start times; apply a 20–30 min grace per name before declaring stuck.
- [ ] Treat `ALREADY_EXISTS` as “re-check state”, not “success” or “no capacity”.
- [ ] Treat `RESOURCE_EXHAUSTED` / internal errors as “cooldown” for that zone (backoff + jitter).
- [ ] Do not redeploy to every READY VM every cycle:
- [ ] Deploy only when VM is missing a healthy babysitter (based on heartbeats / remote check).
- [ ] Keep a durable “deployed” marker in `$HOME` (not only `/tmp`) or use heartbeat presence as truth.

## 9A) Hidden Global Quota: Compute `HEALTH_CHECKS` (Caps TPU VM Count)
This is easy to miss because TPU discussions focus on **regional TPU quota** (chips), but TPU VMs also consume **Compute Engine global project quotas**.

- [ ] **What it is**
- [ ] `HEALTH_CHECKS` is a **Compute Engine project quota** for the number of health check resources you can have in the project.
- [ ] When you create a TPU VM, Google auto-creates multiple health checks for that TPU node (names look like `tpu-<projectNumber>-<nodeId>-tensorflow-service`, `...-preemption-event`, etc.).

- [ ] **Why it blocks “more VMs” even when regional TPU quota exists**
- [ ] Each TPU VM consumes about **5** health checks.
- [ ] Approx max TPU VM count from this quota is: `floor(HEALTH_CHECKS_limit / 5)`.
- [ ] If you hit this quota, TPU VM creates can fail with: `You have reached HEALTH_CHECKS limit`.

- [ ] **How to check (source of truth)**
- [ ] Check the quota itself:
```bash
gcloud compute project-info describe --project gcp-research-credits-489020 \
  --format='table(quotas.metric,quotas.limit,quotas.usage)' | rg HEALTH_CHECKS
```
- [ ] Count health check resources:
```bash
gcloud compute health-checks list --project gcp-research-credits-489020 --format='value(name)' | wc -l
```

- [ ] **Observed in this project**
- [ ] As of `2026-03-12T00:22Z`, `HEALTH_CHECKS limit=75` and `usage=70` (and there were `70` health-check resources).
- [ ] That leaves room for only about **one** more TPU VM before hard-failing (needs ~5 checks).

- [ ] **What to do**
- [ ] Best fix: request a higher **Compute Engine → Health checks** quota in Cloud Console Quotas (target `5 * desired_TPU_VMs`, add headroom).
- [ ] Short-term triage: delete TPU VMs you don't need (v4/v5e/v6e all “spend” health checks), then recreate the mix you want.
- [ ] Do not rely on “regional TPU quota” alone; always check `HEALTH_CHECKS` when VM count caps at ~15.

## 10) Watchdog Auto-Repair (Overnight Survival)
- [ ] Stall definition uses validated count (and/or queue movement), not just “VM READY”.
- [ ] On stall, automatically run repair steps:
- [ ] Repopulate missing tasks: `populate.py --skip-completed` (idempotent).
- [ ] Force reclaim stale tasks (tight TTL for emergency): `reclaim_stale(stale_ttl_s=...)`.
- [ ] Requeue invalidated tasks according to retry policy.
- [ ] Ensure monitor and vm_requester are running; restart if not.
- [ ] Fix `pgrep` patterns to avoid false matches (match full path).

## 11) Startup/Deploy Consistency (Prevent Self-Inflicted Conflicts)
- [ ] `startup.sh` and deploy script must launch babysitter the same way:
- [ ] single babysitter per VM, same lockfile, same env contract.
- [ ] Do not run per-chip tmux sessions if babysitter already multiplexes chips.
- [ ] After any babysitter or startup change:
- [ ] Upload to all buckets
- [ ] Redeploy only broken VMs (avoid killing healthy training)

## 12) Versioning & Rollout
- [ ] Put `git_sha` (or build ID) in heartbeats and receipts.
- [ ] Canary first:
- [ ] One VM completes one task end-to-end (pending -> running -> completed -> validated) before broad rollout.
- [ ] Mixed-version policy:
- [ ] Either block mixed versions, or allow but record `git_sha` in every artifact for forensic debugging.

## 13) Acceptance Tests (Run These Before Trusting Overnight)
- [ ] Lock contention: two deploys overlap; babysitter still starts within timeout and does not silently exit.
- [ ] Preempt mid-task: task returns to pending and later validates.
- [ ] Quiet stdout during XLA: heartbeats continue; no false reclaim.
- [ ] Orphan test: kill babysitter; no TPU lock remains; new babysitter makes progress.
- [ ] Invalid result: corrupt/NaN result is moved to `invalidated/` and retried.
- [ ] Missing task repair: delete one task object; watchdog repopulates it.

## 14) Minimal Patch Order (High ROI To Finish Tasks)
1. [ ] Ensure babysitter never silently exits (lock timeout + clear logs).
2. [ ] Ensure exactly one babysitter per VM (startup and deploy consistency).
3. [ ] Ensure monitor requeues invalid results (`invalidated/` + retry).
4. [ ] Ensure watchdog auto-repairs missing/stuck tasks (populate + reclaim + requeue).
5. [ ] Ensure training processes cannot orphan TPU locks (process groups + killpg + startup cleanup).
6. [ ] Ensure acquisition doesn’t thrash (CREATING grace + cooldown + no redeploy spam).

## Hidden Global Quota: Compute `HEALTH_CHECKS`

- **Each TPU VM auto-creates ~5 Compute Engine health-check resources.**
- Default project quota: `HEALTH_CHECKS` limit=75 → max ~15 TPU VMs total.
- When you hit the cap: `gcloud tpu-vm create` returns `RESOURCE_EXHAUSTED: You have reached HEALTH_CHECKS limit`.
- **Check current usage**: `gcloud compute project-info describe --format='json' | python3 -c "import sys,json; [print(q) for q in json.load(sys.stdin)['quotas'] if 'HEALTH' in q['metric']]"`
- **Fix**: Request increase to 300–500 at `console.cloud.google.com → IAM & Admin → Quotas → HEALTH_CHECKS`. Free, approval minutes–days.
- Larger VMs (v6e-8) give more chips per VM per health check slot — prefer fewer bigger VMs when quota-constrained.

## W&B / Per-Step Metrics

- **W&B is disabled on all TPU VMs** (`WANDB_MODE=disabled` in babysitter.py). Validated JSON only has ~8 eval-loss points (one per eval_interval=100 steps).
- **Per-step training loss is NOT recorded** — to get it, either:
  - **Option A** (W&B offline): Set `WANDB_MODE=offline` on internet VMs, sync after task. Requires WANDB_API_KEY in deploy env.
  - **Option B** (local JSONL): Append `{"step": N, "loss": X}` to `train_loss.jsonl` every step in `train_v2_tpu.py`. Upload to GCS at task end.
- Without per-step loss, sweeps can only compare final eval performance — can't detect divergence or slow convergence mid-run.
- **TODO**: Implement before next experiment. See `NEXT_ITERATION_TODO.md` item #1.
