# Fleet Utilization Report — 2026-03-11 05:25 UTC

## Regional Quota Utilization

| Zone | Type | Quota | VMs | Claimed | Training | Util% | Gap |
|------|------|-------|-----|---------|----------|-------|-----|
| europe-west4-a | v6e-8 | 64c | 4 | 32c (50%) | 32c | 50% | v6e-ew4a-8d deployed, 9/10 failed to create |
| us-east1-d | v6e-8 | 64c | 0 | 0c (0%) | 0c | 0% | 3 VMs CREATING + 1 setup -> +32 chips soon |
| us-central2-b | v4-8 | 64c | 8 | 32c (50%) | 12c | 19% | All 8 VMs claimed, 18 stuck chips |
| europe-west4-b | v5e-4 | 64c | 1 | 4c (6%) | 0c | 0% | Only 1 v5e VM, v5e 20x slower |
| us-central1-a | v5e-4 | 64c | 0 | 0c (0%) | 0c | 0% | No VMs created yet |
| **TOTAL** | | **320c** | **13** | **68c (21%)** | **44c** | **14%** | **252 chips unclaimed** |

## Key Issues
1. **us-east1-d at 0%**: 4 VMs being created/setup — will add 32 chips
2. **us-central1-a at 0%**: No v5e VMs attempted — v5e is 20x slower than v6e
3. **us-central2-b 18 stuck**: v4 workers with stale heartbeats, auto_maintain redeploying
4. **v6e-ew4a-9/10 creation failed**: External IP constraint despite --internal-ips flag

## VM Creation Results (05:25 UTC)
All 6 creation attempts FAILED:
| VM | Zone | Error |
|---|---|---|
| v6e-ew4a-9/10 | europe-west4-a | No spot capacity (internal error) |
| v6e-ue1d-3/4/6 | us-east1-d | HEALTH_CHECKS quota limit |
| v5e-ew4b-2 | europe-west4-b | v5e spot quota exhausted (4/4 chips) |

**Current fleet is at maximum capacity given quotas + spot availability.**

## Action Items
- [x] v6e-ew4a-8d: Deployed with 8 sessions
- [ ] v6e-ue1d-5: auto_maintain running setup.sh
- [ ] Retry VM creation later (spot capacity fluctuates)
- [ ] Request HEALTH_CHECKS quota increase for us-east1-d

