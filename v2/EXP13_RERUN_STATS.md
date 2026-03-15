# exp13_rerun Run Statistics

## Summary
| Metric | Value |
|--------|-------|
| Run start | 2026-03-12 ~23:00 UTC |
| First completion | 2026-03-13 04:07 UTC |
| Last completion | 2026-03-13 13:56 UTC |
| Total wall time | ~15 hours |
| Tasks completed | 120/120 |
| Avg completion rate | 8.0 tasks/hr |
| Peak rate | ~42 tasks/hr (07:05-08:05 UTC burst) |
| Theoretical max rate | ~20 tasks/hr (at full 60-chip fleet) |
| Efficiency | ~40% (60 chip-hours of waste from bugs) |

## Fleet
| Period | VMs | Chips | Notes |
|--------|-----|-------|-------|
| 23:00–05:30 UTC | 10 (5 v6e + 5 v4) | 60 | Stable |
| 05:30–06:22 UTC | 2 v6e + 5 v4 = 7 | 36 | Preemption wave + libtpu bug VMs deleted |
| 06:22–14:00 UTC | 10 (5 v6e + 5 v4) | 60 | Rebuilt with fixed deploy script |

**Average VMs: ~9.8 VMs (60 chips 93% of the time)**

### VM List
| VM | Type | Zone | Chips | Notes |
|----|------|------|-------|-------|
| v6e-ew4a-1 | v6e-8 | europe-west4-a | 8 | Stable all night |
| v6e-ew4a-2 | v6e-8 | europe-west4-a | 8 | Stable; chip1/chip7 had user-site torch_xla issue |
| v6e-ew4a-3 | v6e-8 | europe-west4-a | 8 | Preempted 05:30 UTC, redeployed 06:22 with libtpu fix |
| v6e-ew4a-4 | v6e-8 | europe-west4-a | 8 | Deleted 05:25 (libtpu bug), redeployed 06:22 with fix |
| v6e-ew4a-5 | v6e-8 | europe-west4-a | 8 | Deployed with libtpu fix 06:22 |
| v4-uc2b-1 | v4-8 | us-central2-b | 4 | Stable all night |
| v4-uc2b-2 | v4-8 | us-central2-b | 4 | Stable; chip2 had user-site torch_xla issue |
| v4-uc2b-3 | v4-8 | us-central2-b | 4 | Stable all night |
| v4-uc2b-4 | v4-8 | us-central2-b | 4 | Stable all night |
| v4-uc2b-5 | v4-8 | us-central2-b | 4 | Stable; chip2 had heartbeat_stale false-reclaim |

## Hourly Completions (UTC)
```
04:xx  ████████████████████  20
05:xx  (0 — monitor stuck / preemption wave)
06:xx  ██████████████  14
07:xx  ████████████████████  20
08:xx  █████████  9
09:xx  ███████████████████████████████  31  ← peak burst
10:xx  ██████  6
11:xx  ████████  8
12:xx  ████████  8
13:xx  ████  4
```

## Chip Performance
| Type | VMs | Chips | Steps/sec | Min/task (1778 steps) | Tasks/chip/day |
|------|-----|-------|-----------|----------------------|----------------|
| v6e | 5 | 40 | ~0.18/s (5.5s/step) | ~163 min | ~8.8 |
| v4  | 5 | 20 | ~0.12/s (8.5s/step) | ~252 min | ~5.7 |

## Bugs Found (8 total → v3 backlog)
| Bug | Impact | Status |
|-----|--------|--------|
| O1: monitor.py stuck 4h (01:04-05:11 UTC) | No validation for 4h | Fixed at 05:11 UTC |
| O2: libtpu not found on new v6e VMs | 50+ tasks burned on ew4a-4/7 | Fixed in deploy_babysitter.sh |
| O3: Spot preemption wave at 05:30 UTC | 4 VMs lost simultaneously | Auto-recovered by vm_requester |
| O4: pgrep -af regex alternation bug | False "no process" alarm | Identified, use ps aux instead |
| O5: heartbeat_stale false-reclaim 11× | 1 task re-run 11× unnecessarily | Requeued manually |
| O6: User-site torch_xla on v4 (deleted buffer) | 8 task failures on uc2b-2 | Requeued; same root as O2 |
| O7: 5 zombie tasks in running/ for 7-9h | 5 tasks never completed | Requeued manually at 11:20 UTC |
| O8: 4 tasks had 0-byte JSON (never claimed 11h) | 4 tasks unclaimed entire run | Rebuilt from module at 11:20 UTC |

## Wasted Chip-Hours Estimate
| Cause | Waste |
|-------|-------|
| O2: libtpu (50 tasks × instant fail × 5 retries) | ~0 (immediate fail, no training) |
| O7: zombie tasks (5 tasks × full duration per chip) | ~0 (chips moved to other tasks, zombies just sat in GCS) |
| O8: empty task files (4 tasks × 11h unclaimed) | 4 tasks × 11h on idle chips = overhead only |
| O1: monitor outage (4h × 60 chips × validation delay) | ~4h delay, not wasted compute |
| O5/O6: false-reclaims and bad-chip retries | ~12 task re-runs × 0-163min = up to ~1h |
| Preemption recovery | ~1h of re-compilation |
| **Total meaningful waste** | **~2-3 chip-hours** (surprisingly low — most bugs were coordination-layer, not compute) |
