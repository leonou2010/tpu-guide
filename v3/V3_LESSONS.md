# V3 Lessons Learned

**Last updated**: 2026-03-14

---

## From ~/levanter/, ~/xpk/, ~/tpunanny/

### tpunanny
- QueuedResource API: `tpu_v2.QueuedResource` with `spot=tpu_v2.QueuedResource.Spot()`. State: PROVISIONING→WAITING_FOR_RESOURCES→ACTIVE/SUSPENDED/FAILED.
- Delete strategy: check state first, only delete FAILED/SUSPENDED. Wait for absence before recreating (`_wait_for_absence()`, 5s poll, 300s timeout).
- Rich TUI: parallel GCS queries, pre-fetch all nodes once (avoids N+1 queries).

### levanter
- Health check: `actor.healthy.remote()` with 60s timeout. Parallel via `ray.wait()`.
- Preemption detection: `get_current_tpu_is_preempted()` — queries kernel state.
- Process group cleanup: `os.killpg()` after subprocess wait. `start_new_session=True` in Popen.
- libtpu lock: `_hacky_remove_tpu_lockfile()` on new task start. Same issue we hit with `/dev/vfio/`.
- TPU config registry: hardcoded chip counts per type (v4-8=4chips, v6e-8=8chips).

### xpk
- Batch parallel execution: `subprocess.Popen()` list, poll all until complete.
- Exponential backoff: `DELAY = BASE_DELAY * (2 ** (attempt - 1))` — **we should adopt this**.
- Cluster/workload separation: provisioning (persistent) vs jobs (ephemeral). GKE-based.
- Kueue: admission control, resource reservation, priority queues.

---

## Bug Fix Log — deploy_babysitter.sh

### 2026-03-13: Fix 1 — ew4a torch check
- **Bug**: `_needs_torch()` rejected `2.9.0+cu128` as CUDA torch
- **Fix**: Direct importability test in ew4a branch, no version string check

### 2026-03-13: Fix 2 — v4 sudo pip
- **Bug**: `pip install` without sudo → user-site. `PYTHONNOUSERSITE=1` in hard gate blocked it.
- **Fix**: All v4 tpu_core installs use `sudo pip install` (system-wide)

### 2026-03-13: Fix 3 — v4 deps before torch
- **Bug**: `torch --no-deps` fails because typing_extensions not in system python
- **Fix**: Pre-install loop for all non-torch deps from tpu_core/*.whl with sudo

### 2026-03-13: Fix 4 — PJRT_DEVICE gate
- **Bug**: `torch_xla.device()` → `OSError: libtpu not found` even when libtpu IS loaded
- **Root cause**: torch_xla 2.9.0 `library_path` property triggers fresh `import libtpu` call; fails even if libtpu.so is already dlopen'd
- **Fix**: Check `os.environ.get('PJRT_DEVICE')` after `import torch_xla`. Set to 'TPU' by torch_xla at import time if libtpu found.

### 2026-03-14: Fix 5 — libtpu not installed on ew4a
- **Bug**: `PJRT_DEVICE=CPU` on ew4a-1. `_found_libtpu=False`. Deploy → FAILED_ENV_TPU_INIT.
- **Root cause**: torch_xla 2.9.0 `_setup_tpu_vm_library_path()` checks: (1) TPU_LIBRARY_PATH env, (2) `torch_xla/lib/libtpu.so` (doesn't exist in system install), (3) `import libtpu` (package not installed). All fail → `PJRT_DEVICE=CPU`.
- **Fix**: Download `libtpu-0.0.2-py3-none-any.whl` from `$BUCKET/wheels/tpu_core/` and install (pip user or sudo) when `import libtpu` fails. Wheel contains `libtpu/libtpu.so`.
- **Confirmed**: ew4a-1 reaches IDLE_AWAITING_WORK after this fix.

---

## Architecture Decisions

### Pull-based coordinator (keep)
GCS-based pending/running/completed/failed dirs. Workers pull tasks. No central bottleneck.
Better than push-based for bursty workloads and preemption-heavy environments.

### QueuedResource per VM (keep from tpunanny)
GCP handles queuing for spot capacity. vm_manager requests QR per VM, GCP provisions when available.
Eliminates "internal error" retry loops from v2 vm_requester.

### Thread-per-VM in vm_manager (keep)
Each VM managed by dedicated thread. Detected stuck/dead VMs in 60s (vs 45min in v2 single loop).

### Stale-ttl 1800s (keep from exp13 lessons)
900s caused 7 false reclaims in exp13 run. 1800s safer. Also: heartbeat step count as secondary check.

### LAUNCH_MODE=single (mandatory)
`torch_xla.launch()` with >1 process → device conflicts. `debug_single_process=True` + `TPU_VISIBLE_CHIPS=N` per worker.

---

## V2 Reference — What Worked

### v2 working setup per VM type (use as ground truth for v3):
- **ew4a (v6e, internet)**: `sudo pip install torch==2.9.0 torch_xla==2.9.0` from PyPI. This auto-installs all nvidia runtime packages (real libs, no stubs needed). libtpu from pip: `pip install --user libtpu-nightly` or from wheel.
- **ue1d (v6e, no internet)**: GCS wheels `torch_v6e_cp310/*.whl` + `nvidia_stubs_v6e.tar.gz`. The stubs must export ALL symbols required by `libtorch_nvshmem.so` (fixed in Fix 10).
- **uc2b (v4, no internet)**: GCS wheels `tpu_core/*.whl`. Same CUDA stubs requirement.
- **Key lesson**: Never use GCS wheel approach for ew4a when PyPI is available — GCS wheels lack nvidia runtime packages and require manual stubs.

---

## TODO / Open Issues

### 2026-03-14: Fix 6 — v4 needs CUDA stubs too
- **Bug**: `ValueError: libcublas.so.*[0-9] not found` on uc2b-1 during TESTING_TPU_INIT
- **Root cause**: `tpu_core/torch-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl` is a CUDA-enabled build. `import torch` calls `_preload_cuda_deps()` which dlopen's libcublas.so.12. v4 has no CUDA hardware and no CUDA libs installed.
- **Fix**: Added CUDA stubs installation (same `nvidia_stubs_v6e.tar.gz` as ue1d) to the v4/v5e section of deploy_babysitter.sh. Stubs satisfy torch's dlopen calls; actual training uses libtpu.so via torch_xla.
- **Key insight**: The `tpu_core/` wheels are CUDA+TPU builds (same as v6e wheels). Any VM using these needs CUDA stubs, even if the hardware has no GPU.

---

## TODO / Open Issues

### 2026-03-14: QueuedResource API — NOT usable with this project
- **Bug**: `tpu_v2.TpuClient()` → `DefaultCredentialsError` (needs ADC, not set up)
- **Bug**: `gcloud alpha compute tpus queued-resources create --spot` → "STANDARD provisioning model incompatible with spot requests"
- **Fix**: Revert to direct `gcloud alpha compute tpus tpu-vm create --spot --internal-ips` (v2-style, confirmed working).
- vm_manager.py: replaced all QR API methods with direct `tpu-vm create/describe/delete`.

### 2026-03-14: v5e added to fleet
- Added 3x v5e-ew4b + 3x v5e-uc1a (v5litepod-8, runtime=v2-alpha-tpuv5-lite).
- deploy_babysitter.sh handles v5e same as v4 (CUDA stubs + tpu_core wheels).
- v5e is slow (~100s/step). Useful when v6e quota exhausted.

---

### 2026-03-14: Fix 7 — ue1d typing_extensions batch install killed by networkx

- **Bug**: `ModuleNotFoundError: No module named 'typing_extensions'` at TESTING_TPU_INIT on ue1d
- **Root cause**: `torch_v6e_cp310/networkx-3.6.1` wheel has wrong Python version metadata (`>=3.11`). When all deps installed in one `pip install` batch, pip exits with error (networkx incompatible), silently skipping typing_extensions too.
- **Fix**: Install each dep wheel individually with `|| true` per wheel so one failure doesn't block others.

---

### 2026-03-14: Fix 10 — nvshmem stub missing required symbols (version NVSHMEM)

- **Bug**: `ImportError: libtorch_nvshmem.so: undefined symbol: nvshmem_malloc, version NVSHMEM`
- **Root cause**: `libtorch_nvshmem.so` (bundled with torch 2.9.0) needs 16 symbols from `libnvshmem_host.so.3` with GNU version `NVSHMEM`. The stub built by `build_cuda_stubs.sh` only had empty functions, no symbol exports.
- **Fix**: Rebuilt `libnvshmem_host.so.3` stub using gcc with a version script that exports all 16 symbols under `NVSHMEM` version: `nvshmem_malloc`, `nvshmem_free`, `nvshmem_ptr`, `nvshmem_info_get_version`, `nvshmem_team_*`, `nvshmemid_*`, `nvshmemx_*`. Also changed stubs install in deploy script to always re-extract (tarball is only 12KB; removes need to check if stubs are "already present" since old stub may have wrong symbols).
- **Key insight**: Empty stubs are not enough when the dependent lib needs specific symbols with GNU versioning. Must export all needed symbols even if they return dummy values. Use `readelf`/`nm -D` on the dependent `.so` to find required undefined symbols.

### 2026-03-14: Fix 9 — CUDA stubs extracted to wrong path (double nvidia/nvidia/)

- **Bug**: `ValueError: libcublas.so.*[0-9] not found` at TESTING_TPU_INIT on ew4a (and all zones)
- **Root cause**: CUDA stubs tarball has structure `nvidia/cublas/lib/libcublas.so.12`. Deploy script extracted with `tar xzf ... -C "$_nv/"` where `_nv=dist-packages/nvidia`. This created `dist-packages/nvidia/nvidia/cublas/lib/libcublas.so.12`. torch's `_preload_cuda_deps` searches `<sys.path>/nvidia/<subdir>/lib/` → wrong path. The fallback `-C dist-packages/` never ran because the first `tar` "succeeded" (just wrong dest). `_stubs_ok()` check for `dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12` returns false (file not there), causing re-install attempts that also go to wrong path.
- **Fix**: Changed all 3 stubs sections to extract directly to `/usr/local/lib/python3.10/dist-packages/` (not to `$_nv/`). Tarball's `nvidia/...` entries land at the correct `dist-packages/nvidia/...`.
- **Affects**: ALL VM types (ew4a, ue1d, v4, v5e) — same bug in all 3 sections.

### 2026-03-14: Fix 8 — ew4a torch NOT pre-installed (v2-alpha-tpuv6e has no system torch)

- **Bug**: Force-redeploy on ew4a VMs → `FAILED_ENV_TORCH_EW4A`. `python3 -c "import torch"` → `No module named 'torch'`
- **Root cause**: `v2-alpha-tpuv6e` runtime does NOT pre-install torch in `/usr/local/lib/python3.10/dist-packages/`. The old deploy scripts (v2) installed torch to user-site (`~/.local/lib/python3.10/site-packages/`). Subsequent deploys with `PYTHONNOUSERSITE=1` check didn't find it. After `_purge_user_torch` removed user-site, torch gone completely.
- **Fix**: ew4a section now installs from GCS `torch_v6e_cp310` wheels (same as ue1d) + CUDA stubs, if torch not importable. Removed `PYTHONNOUSERSITE=1` from the check (user-site is fine for ew4a).
- **Key insight**: Never assume runtime pre-installs torch. Always install from GCS wheels. Only difference from ue1d: ew4a has internet (could pip install from PyPI) but we use GCS for consistency.
- **SUPERSEDED by Fix 11**: GCS wheels for ew4a are wrong — they lack nvidia-nccl and other runtime packages. Use PyPI instead.

### 2026-03-14: Fix 11 — ew4a MUST use PyPI, not GCS wheels
- **Bug**: FAILED_ENV_TPU_INIT on ew4a: `libtorch_cuda.so: undefined symbol: ncclGetUniqueId`
- **Root cause**: GCS torch_v6e_cp310 wheels only contain torch+torch_xla, no nvidia-nccl-cu12 or other runtime packages. The comprehensive stubs (Fixes 9/10) covered cublas/cudart/cudnn etc. but not NCCL. NCCL is a full shared lib, not a single symbol stub. Adding NCCL symbols to stubs is playing whack-a-mole.
- **Fix**: ew4a section changed to `sudo pip install torch==2.9.0 torch_xla==2.9.0 --extra-index-url https://download.pytorch.org/whl/cu128`. PyPI auto-installs ALL nvidia-* packages (nccl, cublas, cudart, etc.) — no stubs needed on ew4a.
- **Key insight**: Stubs are only for no-internet VMs (ue1d, v4, v5e). ew4a has internet — always use PyPI for torch install. Saves ~1GB GCS download and eliminates all stub complexity on ew4a.
- **V2 reference confirms**: v2 working setup used `sudo pip install torch==2.9.0 torch_xla==2.9.0` from PyPI for ew4a.

### 2026-03-14: Fix 12 — CUDA stubs missing cusparselt, curand, cufile, libnvblas, libcufftw
- **Bug**: After nccl was added (Fix ~10+), ue1d still failed: `libcusparseLt.so.0: cannot open shared object file: No such file or directory`
- **Root cause**: gen_stubs.py only had the "big 9" libs (cublas, cudart, cudnn, cufft, cupti, nvrtc, cusolver, cusparse, nvshmem). Missing: nccl, cusparselt, curand, cufile, libnvblas, libcufftw — all present in original `build_cuda_stubs.sh` REQUIRED_STUBS but never ported to gen_stubs.py.
- **Fix**: Added all missing libs to gen_stubs.py. Final stub set: 19 .so files covering 100% of build_cuda_stubs.sh REQUIRED_STUBS.
- **Complete stubs list** (42KB tarball): libcudart.so.12, libcublas.so.12, libcublasLt.so.12, libnvblas.so.12, libcudnn.so.9, libcufft.so.11, libcufftw.so.11, libcupti.so.12, libnvrtc.so.12, libcusolver.so.11, libcusparse.so.12, libcusparseLt.so.0, libnccl.so.2, libcurand.so.10, libcufile.so.0, libnvshmem_host.so.3, libnvToolsExt.so.1, libnvJitLink.so.12
- **cusparseLt required symbols** (from readelf -W): cusparseLtInit, cusparseLtDenseDescriptorInit, cusparseLtStructuredDescriptorInit, cusparseLtMatDescriptorDestroy, cusparseLtMatmulDescriptorInit, cusparseLtMatmulDescSetAttribute, cusparseLtMatmulAlgSelectionInit, cusparseLtMatmulAlgGetAttribute, cusparseLtMatmulAlgSetAttribute, cusparseLtMatmulPlanInit, cusparseLtMatmulPlanDestroy, cusparseLtMatmulGetWorkspace, cusparseLtMatmul, cusparseLtMatmulSearch, cusparseLtSpMMACompressedSize2, cusparseLtSpMMACompress2
- **NCCL required symbols**: see readelf output — includes ncclBcast, ncclGetLastError, ncclGroupSimulateEnd, ncclRedOpCreatePreMulSum, ncclRedOpDestroy, ncclCommWindowRegister, ncclCommWindowDeregister, ncclCommInitRankScalable
- **How to get full symbol list**: `readelf -W --dyn-syms libtorch_cuda.so | grep " UND " | awk '{print $NF}' | grep <libname_prefix> | sort -u`
- **Key insight**: Always compare against build_cuda_stubs.sh REQUIRED_STUBS as ground truth. Any lib in that list that's missing from gen_stubs.py will cause a "file not found" error at torch import.

### 2026-03-14: Fix 13 (v2) — PYTHONNOUSERSITE=1 + sudo pip = wrong Python → packages not found

- **Bug**: babysitter.py fails `verify_environment()` with `FATAL: ['hydra', 'omegaconf', 'transformers']` even though deploy_babysitter.sh reported IDLE_AWAITING_WORK.
- **Root cause v1**: deploy installs to user-site but babysitter launched with PYTHONNOUSERSITE=1.
- **Fix v1 (wrong)**: Changed to `sudo $_PY -m pip install`. But `sudo python3` on Ubuntu resolves to `/usr/bin/python3` (sudo strips PATH, ignores miniconda). Packages installed to `/usr/local/lib/python3.10/dist-packages/`. But babysitter uses system python3 too, which SHOULD see `/usr/local/lib/...` — except on Ubuntu 22.04, sudo pip fails silently with "externally managed environment" restrictions.
- **Diagnosed**: `ls ~/.local/lib/python3.10/site-packages/hydra_core*` → EXISTS. `ls /usr/local/lib/python3.10/dist-packages/hydra_core*` → NOT FOUND. `sudo python3 -c "import hydra"` → fails. Confirmed: sudo pip install silently failed.
- **Fix v2 (correct)**: (1) Remove `export PYTHONNOUSERSITE=1` from babysitter launch block — packages in user-site ARE accessible. (2) Remove PYTHONNOUSERSITE=1 from MISSING/PRE_FLIGHT checks. (3) Use `$_PY -m pip install` (no sudo) — goes to user-site which babysitter can see.
- **Key insight**: `sudo python3 -m pip install` is dangerous on Ubuntu 22.04 — installs to a different path than regular `python3` if miniconda is in PATH, AND may fail silently with "externally managed environment". Safest approach: use `$_PY -m pip install --user` and don't export PYTHONNOUSERSITE=1.

### 2026-03-14: Fix 17 — `nohup VAR=val bash` fails ("No such file or directory")

- **Bug**: vm_manager's fire-and-forget SSH command `nohup FORCE_REDEPLOY=1 bash ~/pull_code/deploy_babysitter.sh` fails on all VMs with "nohup: failed to run command 'FORCE_REDEPLOY=1': No such file or directory". Manual redeployment also failed for same reason.
- **Root cause**: `nohup` treats its first argument as the command to run, not as shell env var assignment. `nohup FORCE_REDEPLOY=1 bash` → nohup tries to exec "FORCE_REDEPLOY=1" as a binary.
- **Fix**: Move env vars BEFORE nohup: `FORCE_REDEPLOY=1 TPU_NAME=... nohup bash ~/pull_code/deploy_babysitter.sh`. Shell handles `VAR=val cmd` prefix as env for `cmd` (which is `nohup`), which then inherits it for `bash`.
- **Key insight**: `nohup VAR=value cmd` is wrong. Always: `VAR=value nohup cmd` or `env VAR=value nohup cmd`.

### 2026-03-14: Fix 16 — PRE_START_SESSION_BARRIER: 8 chips init PJRT simultaneously

- **Bug**: All chips on ew4a-2/3/4/5 fail training with `RuntimeError: TPU initialization failed: Session barrier id PRE_START_SESSION_BARRIER is not ready`. Chips 0-3 get the barrier error, chip1 gets "Deadline Exceeded", chip2 gets "Timeout waiting for barrier master".
- **Root cause**: Babysitter's 8 chip_worker threads run indefinitely in idle mode. When 120 tasks appear in the queue, all 8 simultaneously claim tasks and call `subprocess.Popen` for training. All 8 training subprocesses call `initialize_singleprocess()` at the same time → PJRT's PRE_START_SESSION_BARRIER requires sequential initialization. Chip0's barrier master starts but doesn't see the other 7 chips as partners (each chip only sees 1 chip via TPU_VISIBLE_CHIPS), causing timeout/failure.
- **Why ew4a-1 worked**: Babysitter had just started (IDLE_AWAITING_WORK age=34s). Only chip0 thread was running (chip1 hadn't started yet — 45s stagger between THREAD starts). chip0 initialized PJRT alone → success.
- **Fix**: Added `_TPU_INIT_LOCK = threading.Lock()` global in babysitter.py. In `run_training()`, acquire lock before `subprocess.Popen`, hold for `_TPU_INIT_STAGGER_S = 90s`, then release. This serializes PJRT initialization: each chip has 90s to initialize before the next chip starts. 8 chips × 90s = 12 min overhead per VM startup — acceptable vs 2.4h training time.
- **Key insight**: v6e PJRT session initialization requires exclusive access to the TPU machine-level session setup. The existing 45s stagger (thread startup) prevented conflicts when training starts immediately, but didn't help when all threads were already in idle mode and simultaneously spotted tasks. Must serialize at the PJRT init call site, not at thread creation.

### 2026-03-14: Fix 15 — SSH deploy timeout on v4 (240s too short for 15-20 min deploy)

- **Bug**: v4 deploy always times out — `Deploy attempt 1 timed out (240s)`. v4 needs to download 1GB tpu_core wheels + install + download 10GB+ XLA cache. Total: 15-20 min. vm_manager had 240s SSH timeout.
- **Fix**: Changed deploy command to fire-and-forget: `nohup ... bash ~/pull_code/deploy_babysitter.sh > /tmp/deploy_babysitter.log 2>&1 &`. SSH exits immediately (rc=0), deploy runs in background on VM. SSH timeout reduced to 60s (just for connection + launch). vm_manager tracks success via GCS telemetry + heartbeats within DEPLOY_GRACE_S (600s).
- **Key insight**: For slow operations (long downloads, installs), never wait synchronously via SSH. Launch in background, monitor via side channel (GCS telemetry). This is the fire-and-forget pattern from xpk's async workload submission.

### 2026-03-14: Fix 14 — vm_manager continuous redeploy loop (no grace period)

- **Bug**: vm_manager repeatedly redeploys all VMs every 60s, killing freshly started babysitters before they can write heartbeats.
- **Root cause**: `_babysitter_healthy()` returns False when no fresh heartbeat (stale = old session, or no heartbeat = first deploy). After deploy succeeds, vm_manager waits only POLL_INTERVAL (60s) then checks health again. Babysitter needs up to 150s to write first heartbeat (verify_environment=120s + first idle heartbeat=30s). So 60s re-check sees unhealthy → immediate redeploy → babysitter killed at 60s, never writes heartbeat → infinite loop.
- **Fix**: Added `DEPLOY_GRACE_S = 600s` in vm_manager. After successful deploy, `_last_deploy_time = time.time()`. Main loop skips health check for 600s, logging "Deploy grace period (Xs remaining)". Confirmed working: `[ew4a-2] Deploy grace period (538s remaining)`.
- **Key insight**: Matches levanter's `_HEALTH_CHECK_TIMEOUT = 60s` concept — don't declare a VM dead immediately after deployment. Always have a startup grace period.

### 2026-03-14: Fix 21 — numpy not in us-east1 / us-central2 GCS wheel bundles

- **Bug**: `import torch_xla` → `ModuleNotFoundError: No module named 'numpy'` on ue1d and v4
- **Root cause**: `torch_xla/distributed/spmd/xla_sharding.py` imports numpy. numpy was only uploaded to `gcp-researchcredits-blocklab-europe-west4/wheels/torch_v6e_cp310/`. ue1d (us-east1) and v4 (us-central2) use different buckets with no numpy wheel.
- **Fix**: Copied numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.whl to all 3 GCS buckets. Added numpy to install list in deploy_ue1d.sh and deploy_babysitter.sh ue1d/v4 sections. On v4: also try `apt-get install python3-numpy` first (but `libopenblas-base` deps may be missing on tpu-ubuntu2204-base without apt-get update).
- **Key insight**: Always copy ALL dependency wheels to ALL 3 GCS buckets. ue1d, v4, v5e all need the same torch_v6e_cp310 deps (minus torch/torch_xla themselves which come from type-specific bundles).

### 2026-03-14: Fix 20 — networkx-3.6.1 incompatible with Python 3.10 (v4/v5e)

- **Bug**: `import torch_xla` → `ModuleNotFoundError: No module named 'networkx'` on v4 uc2b
- **Root cause 1**: torch_xla 2.9.0 chains: `_patched_functions → scan.py → functorch → partitioners.py → import networkx`. networkx is a runtime dep of torch_xla.
- **Root cause 2**: GCS bundle `torch_v6e_cp310/networkx-3.6.1-py3-none-any.whl` requires Python `!=3.14.1,>=3.11`. v4 VMs run Python 3.10.6 → pip refuses to install → `|| true` hides the error.
- **Root cause 3**: When install loop downloads to `/tmp/deps_fix/` then saves as `/tmp/networkx.whl`, pip rejects "networkx.whl is not a valid wheel filename" (wheel spec requires `name-ver-py-abi-plat.whl` format).
- **Fix**: Download networkx-3.4.2 (supports Python >=3.10) and upload to GCS, replacing 3.6.1. Remove old 3.6.1 from GCS. Deploy scripts MUST preserve full wheel filename (use `gsutil cp .../networkx-3.4.2-py3-none-any.whl /tmp/deps/` not `/tmp/networkx.whl`).
- **Key insight**: Always check wheel metadata for Python version requirements when creating GCS bundles. `pip show networkx` on a Python 3.11+ machine installs 3.6.x, incompatible with 3.10 VMs.

---

### 2026-03-14: Fix 22 — all_wheels.tar.gz missing from us-central2 (uc2b) bucket
**Root cause**: `all_wheels.tar.gz` (4.12 GiB) only existed in us-east1 bucket. eu-w4a fallback also missing it. uc2b VMs had no way to install training packages (transformers, hydra, etc.).
**Fix**: Copied `all_wheels.tar.gz` from us-east1 to us-central2 bucket. Also added `tpu_wheels.tar.gz` as alias fallback in all deploy scripts (uc2b has `tpu_wheels.tar.gz` which has same content). Added directory rename fallback `mv /tmp/tpu_wheels /tmp/all_wheels` in case tarball uses different top-level dir name.

### 2026-03-14: Fix 23 — PyPI pip install hangs on no-internet VMs (ue1d, uc2b)
**Root cause**: deploy_babysitter.sh ran `pip install hydra-core omegaconf transformers sympy datasets wandb` without `--timeout` or internet check. On no-internet VMs, pip DNS queries hang 10-15 min before timing out. With 6 packages, total wait = 10-15 min before failing.
**Fix**: Added internet check (`curl -sf --connect-timeout 5 https://pypi.org`) before PyPI attempts. Skip pip entirely on no-internet VMs, go straight to GCS wheel fallback. Per-type scripts (deploy_ue1d.sh, deploy_uc2b.sh) never try PyPI.

### 2026-03-14: Fix 24 — typing_extensions missing from v4/v5e torch runtime deps
**Root cause**: deploy_uc2b.sh and deploy_babysitter.sh v4 section installed filelock/sympy/jinja2/networkx/markupsafe but not `typing_extensions`. torch/__init__.py line 35: `from typing_extensions import ParamSpec` → ImportError → FAILED_ENV_TPU_INIT.
**Fix**: Added `typing_extensions-*.whl` to gsutil -m cp list in deploy_uc2b.sh, deploy_v5e.sh, and deploy_babysitter.sh v4 section.

### 2026-03-14: Fix 25 — vm_manager used monolithic deploy_babysitter.sh for all VM types
**Root cause**: vm_manager._deploy_babysitter() always ran `deploy_babysitter.sh` regardless of VM type. Per-type scripts (deploy_ew4a.sh, deploy_ue1d.sh, deploy_uc2b.sh, deploy_v5e.sh) existed but were never called.
**Fix**: vm_manager now selects script based on VM name: ew4a→deploy_ew4a.sh, ue1d→deploy_ue1d.sh, uc2b→deploy_uc2b.sh, v5e→deploy_v5e.sh. Same logic in _startup_script(). vm_manager restarted to pick up change.

### 2026-03-14: Fix 26 — deploy_v5e.sh created (missing entirely)
**Root cause**: v5e VMs had no dedicated deploy script. vm_manager would fall through to deploy_babysitter.sh which has limited v5e support.
**Fix**: Created deploy_v5e.sh based on deploy_uc2b.sh with v5e-specific changes: (1) gcloud storage preferred over gsutil (gsutil broken on v5e), (2) gcs_cp() helper for single files, gcs_cp_m() for globs, (3) zone-aware bucket selection (ew4b→eu-w4a, uc1a→us-central2), (4) 8 chips per host (v5e-8), (5) all_wheels.tar.gz fallback chain includes us-east1 bucket.

### 2026-03-14: Fix 30 — antlr4-python3-runtime in all_wheels.tar.gz is .tar.gz not .whl (loop skips it)
**Root cause**: `all_wheels.tar.gz` bundle contains `antlr4-python3-runtime-4.9.3.tar.gz` as a SOURCE distribution, not a binary wheel. The install loop only iterates `*.whl` files → antlr4 never installed → `omegaconf` can't import (`from antlr4 import ParserRuleContext`) → `import hydra` fails → FAILED_ENV_PREFLIGHT.
**Symptom**: `python3 -c "import hydra"` fails with `ModuleNotFoundError: No module named 'antlr4'`. But `pip show hydra-core` shows it's installed. The problem is antlr4, not hydra itself.
**Fix**: After the `*.whl` loop, check if antlr4 is importable. If not, try `/tmp/all_wheels/antlr4-python3-runtime-4.9.3.tar.gz` (pip can install source dists). Fallback: download the proper wheel `antlr4_python3_runtime-4.9.3-py3-none-any.whl` from GCS bucket.
**GCS wheel**: `gs://gcp-researchcredits-blocklab-us-east1/wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl`
**Deployed**: 2026-03-14 ~13:05 UTC to deploy_ue1d.sh + deploy_uc2b.sh.

### 2026-03-14: Fix 29 — numpy missing at TESTING_TPU_INIT on uc2b (apt-get fails, no internet)
**Root cause**: deploy_uc2b.sh installed numpy via `apt-get install python3-numpy`. No internet on us-central2-b → apt-get fails silently (2>/dev/null). numpy not installed before TESTING_TPU_INIT → `No module named 'numpy'` → FAILED_ENV_TPU_INIT.
**Fix**: Removed apt-get approach. Added `numpy-*.whl` to the gsutil -m cp list that downloads filelock/sympy/etc from GCS (line ~108). numpy-2.2.6 wheel is in `${_EW4A_BUCKET}/wheels/torch_v6e_cp310/`. Installed with pip --no-deps before TPU init.
**Deployed**: 2026-03-14 ~12:53 UTC.

### 2026-03-14: Fix 28 — same composite GCS object CRC32c issue in deploy_uc2b.sh
**Root cause**: Identical to Fix 27. deploy_uc2b.sh had `2>/dev/null` on all `gsutil cp all_wheels.tar.gz` calls. CRC32c error swallowed silently → 0-byte file → 17s INSTALLING_PACKAGES → FAILED_ENV_PREFLIGHT.
**Fix**: Applied same `gsutil -o GSUtil:check_hashes=never cp` fix to deploy_uc2b.sh. Added `rm -f` to clear stale file, size check (`_WHEELS_SIZE > 1000000`) before extracting, error output visible in log.
**Deployed**: 2026-03-14 ~12:50 UTC.

### 2026-03-14: Fix 27 — all_wheels.tar.gz download fails on ue1d (composite GCS object, CRC32c)
**Root cause**: `all_wheels.tar.gz` in us-east1 bucket was uploaded as a **composite GCS object** (composed from multiple parts). Downloading composite objects requires CRC32c integrity checking. The `crcmod` Python package's C extension is NOT installed on ue1d VMs (Python 3.10, tpu-ubuntu2204-base). gsutil cp exits with:
  `CommandException: Downloading this composite object requires integrity checking with CRC32c, but your crcmod installation isn't using the module's C extension`
The old script had `2>/dev/null` on the gsutil cp so the error was swallowed silently — appeared as if download succeeded but file was 0 bytes (or missing). Install phase then showed "INSTALLING_PACKAGES" for ~8 seconds (no real work), then PRE_FLIGHT_CHECK failed: missing transformers/hydra.
**Detection**: Manually ran `gsutil cp gs://.../wheels/all_wheels.tar.gz /tmp/test.tar.gz` on ue1d-1 → got the CRC32c error.
**Fix**: Added `-o GSUtil:check_hashes=never` flag to ALL gsutil cp calls for all_wheels.tar.gz in deploy_ue1d.sh. Also removed `2>/dev/null` from download to surface errors in deploy log. Removed `2>/dev/null` to log errors. Uploaded 2026-03-14 ~12:36 UTC.
**Symptom pattern**: INSTALLING_PACKAGES takes only ~8s (should be 60+ min for 4.12 GiB), then FAILED_ENV_PREFLIGHT with missing transformers/hydra. Key: if INSTALLING_PACKAGES < 30s and pre_flight fails on imports → download silently failed.

## TODO / Open Issues

- [x] Verify ue1d deploy — Fix 7 deployed (typing_extensions one-by-one)
- [x] Verify ew4a force-redeploy — Fix 8 deployed (install from GCS wheels) — SUPERSEDED by Fix 11 (PyPI)
- [x] ew4a CONFIRMED WORKING (2026-03-14 00:10 UTC): ALL 5 VMs, 40/40 chips, ew4a-2/3 at step=1 training. Fixes 11/13/14/15/16/17/18/19 all active.
- [x] Fix 17 (nohup VAR=val bug), Fix 18 (v4 typing_extensions+numpy deps), Fix 19 (FAILED grace period) deployed
- [x] Fix 22-26 deployed 2026-03-14 ~16:15 UTC: all_wheels in uc2b, PyPI timeout bypass, typing_extensions for v4/v5e, per-type vm_manager routing, deploy_v5e.sh created
- [ ] v4 uc2b: typing_extensions fix deployed — expect TESTING_TPU_INIT to pass in next cycle
- [x] Fix 27 deployed 2026-03-14 ~12:36 UTC: composite GCS CRC32c fix — gsutil -o GSUtil:check_hashes=never for all_wheels.tar.gz in deploy_ue1d.sh
- [x] ue1d: ALL 5 VMs IDLE_AWAITING_WORK as of ~13:15 UTC 2026-03-14 (fixes 27+30 deployed)
- [x] uc2b-1: IDLE_AWAITING_WORK. uc2b-2/3/4/5: completing package install, antlr4 pre-installed
- [x] antlr4 pre-installed manually on uc2b-2/3/4/5 to skip fail cycle
- [x] ue1d-1 chip0: xla_compile step=0 — first ue1d training in v3!
- [ ] v5e: no VMs provisioned yet (quota/capacity unavailable in ew4b/uc1a)
- [ ] Exponential backoff in vm_manager retry (adopt from xpk)
- [ ] Alert in dashboard if no step progress across fleet for >30 min
- [ ] Add git_sha to heartbeats (detect stale code versions mid-run)
- [x] monitor.py running (PID 208339) — validates, reclaims stale tasks

## Session 2026-03-14 — TPU_NAME env var root cause

### Bug: Internal hostname used instead of assigned VM name

**Symptom**: ue1d/uc2b babysitters wrote heartbeats to `coord_v2/heartbeats/t1v-n-XXXXXX_chip*.json` instead of `v6e-ue1d-1_chip*.json`. Dashboard showed 0 training chips for ue1d/uc2b. vm_manager saw stale named heartbeats and kept redeploying.

**Root cause**: babysitter.py reads `TPU_NAME` from environment variable. When babysitters were started manually via SSH (`nohup python3 babysitter.py --tpu-name v6e-ue1d-1 ...`), the `--tpu-name` flag is NOT an argparse argument — babysitter.py ignores it. If `TPU_NAME` is not in the shell environment, babysitter falls back to `socket.gethostname()` = internal GCP name `t1v-n-XXXXXXXX-w-0`.

**Fix**: Always start babysitter with `TPU_NAME=v6e-ue1d-1` in environment:
```bash
# WRONG (flag is ignored):
python3 babysitter.py --tpu-name v6e-ue1d-1 ...

# CORRECT (env var):
TPU_NAME=v6e-ue1d-1 python3 babysitter.py ...
# OR:
export TPU_NAME=v6e-ue1d-1
python3 babysitter.py ...
```

vm_manager correctly sets `TPU_NAME={self.name}` before running deploy scripts. Manual SSH must do the same.

**Detection**: `gsutil ls coord_v2/heartbeats/ | grep t1v-n` — any t1v-n entries = wrong TPU_NAME.

**Cleanup**: 
1. Delete t1v-n heartbeats: `gsutil ls .../heartbeats/ | grep t1v-n | xargs gsutil rm`
2. Reclaim t1v-n owned tasks: scan running/ for worker_id containing t1v-n, move back to pending/
3. Trigger vm_manager redeploy for VMs with old babysitters by setting heartbeat timestamp to NOW-3601

### Bug: HEALTHY_TTL too short for XLA compile on fresh VMs

**Symptom**: vm_manager redeployed VMs that were still in XLA compile phase (first compile takes 25-30 min, HEALTHY_TTL was 2700s = 45 min but babysitters stopped writing heartbeats when using wrong worker_id).

**Fix**: Already fixed (XLA_STUCK_S=2100, HEALTHY_TTL=2700). The real fix is ensuring correct TPU_NAME so heartbeats go to right keys.

### Bug: stale running/ tasks from orphaned internal-name babysitters

**Symptom**: After killing internal-name babysitters, their running/ tasks remained claimed. New named babysitters found nothing to claim (0 pending).

**Fix**: Scan running/ for t1v-n worker_ids and immediately reclaim to pending/ instead of waiting 1800s stale-TTL.


## Prevention: add argparse to babysitter.py

To prevent the TPU_NAME env var confusion, add proper CLI argparse:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tpu-name', default='')
parser.add_argument('--zone', default='')
parser.add_argument('--bucket', default='')
parser.add_argument('--exp', default='exp13_rerun3')
args, _ = parser.parse_known_args()

# Override env vars with CLI args if provided
if args.tpu_name: os.environ['TPU_NAME'] = args.tpu_name
if args.zone: os.environ['ZONE'] = args.zone
if args.bucket: os.environ['BUCKET'] = args.bucket
if args.exp: os.environ['EXP'] = args.exp
```

This makes `--tpu-name v6e-ue1d-1` actually work from SSH.

---

## 2026-03-14 Afternoon — XLA_STUCK kills and broken data symlink

### Fix 31: vm_manager XLA_STUCK_S too short → killed babysitters mid-compile

**Root cause**: `XLA_STUCK_S = 2100` (35 min). On ue1d/uc2b, `xla_compile` status = step==0,
which covers ALL of: startup + XLA compile + first step. Without matched XLA cache (~5-10 min
per config), this easily exceeds 35 min. vm_manager killed babysitters → heartbeats stale →
monitor reclaimed tasks → retry+1 → after 4 kills → failed/. 45 tasks failed this way.

**Symptoms**: All ue1d chips stuck in `xla_compile`, never training. Tasks accumulating in
failed/ with `retries=4, exit_code=None, error=""`.

**Fix**: Increased `XLA_STUCK_S` from 2100 → 7200 (120 min). Only kill truly stuck VMs.
Also requeued 45 failed tasks.

**Rule**: Never set XLA_STUCK_S < 2× the worst-case XLA compile time (no cache = 45-60 min).
Use 120 min as safe floor.

---

### Fix 32: sf_bema code tarball has broken /scratch symlink for data/

**Root cause**: `sf_bema_exp13_rerun3.tar.gz` was created on a GPU cluster where
`data → /scratch/blocklab/ka3094/sf_bema/experiments/exp10_smollm2_smoltalk/data`.
On TPU VMs, that symlink is extracted but the target doesn't exist.

**Symptoms**: Training fails with `FileNotFoundError: Directory .../data/train not found`.
Only affects newly deployed VMs or VMs that got a fresh code extract (ue1d-4, uc2b-3).
Other VMs had data already downloaded in a prior session.

**Fix**: 
1. SSH to affected VMs: `rm ~/sf_bema/.../data` (remove symlink), then download from GCS.
2. Updated deploy_ue1d.sh + deploy_uc2b.sh to detect and remove broken symlink before mkdir:
   ```bash
   [ -L "${_DATA_DIR}" ] && rm -f "${_DATA_DIR}"
   mkdir -p "${_DATA_DIR}/train" "${_DATA_DIR}/val"
   ```
3. Changed `gsutil cp -r` to `gsutil cp 'path/*'` (direct file copy, avoids dir collision).

**Rule**: When packaging code tarballs, always dereference symlinks (`tar -h`) or exclude
data/ directories entirely. Data lives in GCS, not in code archives.

---

### VM Delivery Counter (2026-03-14 ~17:30 UTC)

Current state: 36/120 validated, all 5 ew4a working, 3/5 ue1d working (ue1d-4 data downloading),
4/5 uc2b working (uc2b-3 data downloading). ETA: first ue1d completions ~20:00 UTC.
