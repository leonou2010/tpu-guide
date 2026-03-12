#!/usr/bin/env python3
"""
dashboard.py — Rich TUI dashboard for pull-based coordinator.

Reads from GCS coord_v2/ to show real-time sweep progress.

Usage:
    python3 ~/distributed_tpu_training/pull/dashboard.py --once              # single snapshot
    watch -c -n30 'python3 ~/distributed_tpu_training/pull/dashboard.py --once'  # live refresh
    python3 ~/distributed_tpu_training/pull/dashboard.py --interval 30       # auto-refresh
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcs import gcs_list, gcs_read, gcs_read_batch, CONTROL_PLANE

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── Data fetching ────────────────────────────────────────────────────────

def fetch_state():
    """Fetch full queue state from GCS. Returns dict."""
    # List all prefixes in parallel
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(gcs_list, f"{CONTROL_PLANE}/pending"): 'pending',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/running"): 'running',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/completed"): 'completed',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/failed"): 'failed',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/heartbeats"): 'heartbeats',
        }
        lists = {}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                lists[key] = fut.result()
            except Exception:
                lists[key] = []

    # Read running tasks and heartbeats in parallel
    paths_to_read = lists.get('running', []) + lists.get('heartbeats', [])
    raw_data = {}
    if paths_to_read:
        raw_data = gcs_read_batch(paths_to_read, max_workers=20)

    # Parse running tasks
    running_tasks = {}
    for path in lists.get('running', []):
        raw = raw_data.get(path)
        if raw:
            try:
                running_tasks[os.path.basename(path).replace('.json', '')] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    # Parse heartbeats
    heartbeats = {}
    for path in lists.get('heartbeats', []):
        raw = raw_data.get(path)
        if raw:
            try:
                heartbeats[os.path.basename(path).replace('.json', '')] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    # Count per experiment
    exp_counts = defaultdict(lambda: {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0})
    for path in lists.get('pending', []):
        tid = os.path.basename(path).replace('.json', '')
        exp = tid.split('__')[0] if '__' in tid else 'unknown'
        exp_counts[exp]['pending'] += 1
    for tid in running_tasks:
        exp = tid.split('__')[0] if '__' in tid else 'unknown'
        exp_counts[exp]['running'] += 1
    for path in lists.get('completed', []):
        tid = os.path.basename(path).replace('.json', '')
        exp = tid.split('__')[0] if '__' in tid else 'unknown'
        exp_counts[exp]['completed'] += 1
    for path in lists.get('failed', []):
        tid = os.path.basename(path).replace('.json', '')
        exp = tid.split('__')[0] if '__' in tid else 'unknown'
        exp_counts[exp]['failed'] += 1

    # Local validated counts
    for exp in list(exp_counts.keys()):
        vdir = os.path.expanduser(f"~/sf_bema/results/{exp}/validated")
        if os.path.isdir(vdir):
            exp_counts[exp]['validated'] = len([f for f in os.listdir(vdir) if f.endswith('.json')])
        else:
            exp_counts[exp]['validated'] = 0

    return {
        'pending_count': len(lists.get('pending', [])),
        'running_count': len(lists.get('running', [])),
        'completed_count': len(lists.get('completed', [])),
        'failed_count': len(lists.get('failed', [])),
        'running_tasks': running_tasks,
        'heartbeats': heartbeats,
        'exp_counts': dict(exp_counts),
        'timestamp': time.time(),
    }


# ── Rendering ────────────────────────────────────────────────────────────

EXP_TARGETS = {'exp13': 120, 'exp12_1': 185}


def format_age(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{seconds/3600:.1f}h"


def _build_panels(state, panels):
    """Build all dashboard panels into the panels list."""
    now = time.time()

    # ── Header ───────────────────────────────────────────────────────
    total = state['pending_count'] + state['running_count'] + state['completed_count'] + state['failed_count']
    header = Text()
    header.append("PULL-BASED COORDINATOR", style="bold cyan")
    header.append(f"  {time.strftime('%H:%M:%S')}")
    header.append(f"  total={total}")
    header.append(f"  pending=", style="dim")
    header.append(f"{state['pending_count']}", style="yellow bold")
    header.append(f"  running=", style="dim")
    header.append(f"{state['running_count']}", style="green bold")
    header.append(f"  completed=", style="dim")
    header.append(f"{state['completed_count']}", style="blue bold")
    if state['failed_count'] > 0:
        header.append(f"  failed=", style="dim")
        header.append(f"{state['failed_count']}", style="red bold")
    panels.append(Panel(header, border_style="cyan"))

    # ── Per-experiment progress ───────────────────────────────────────
    exp_table = Table(show_header=True, header_style="bold", expand=True)
    exp_table.add_column("Experiment", style="cyan", min_width=10)
    exp_table.add_column("Target", justify="right", min_width=6)
    exp_table.add_column("Pend", justify="right", style="yellow", min_width=5)
    exp_table.add_column("Run", justify="right", style="green", min_width=5)
    exp_table.add_column("Done", justify="right", style="blue", min_width=5)
    exp_table.add_column("Fail", justify="right", style="red", min_width=5)
    exp_table.add_column("Valid", justify="right", style="bold green", min_width=5)
    exp_table.add_column("Progress", justify="right", min_width=30)
    exp_table.add_column("ETA", justify="right", style="cyan", min_width=10)

    # ETA: track validated rate across dashboard invocations via state file
    _eta_path = "/tmp/dashboard_eta_state.json"
    _now_ts = time.time()
    try:
        with open(_eta_path) as _f:
            _eta_state = json.load(_f)
    except Exception:
        _eta_state = {}

    for exp, counts in sorted(state['exp_counts'].items()):
        target = EXP_TARGETS.get(exp, '?')
        validated = counts.get('validated', 0)
        bar_len = 40
        if isinstance(target, int) and target > 0:
            filled = int(validated / target * bar_len)
            bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_len - filled)}[/dim] {validated*100//target}%"
        else:
            bar = "?"

        # ETA from recent validation rate
        eta_str = "—"
        if isinstance(target, int) and validated >= target:
            eta_str = "[green]done[/green]"
        elif isinstance(target, int):
            prev = _eta_state.get(exp, {})
            elapsed = _now_ts - prev.get('ts', _now_ts)
            delta = validated - prev.get('validated', validated)
            if elapsed > 120 and delta > 0:
                secs_left = (target - validated) * elapsed / delta
                if secs_left < 3600:
                    eta_str = f"~{int(secs_left/60)}m"
                else:
                    h, m = divmod(int(secs_left / 60), 60)
                    eta_str = f"~{h}h{m:02d}m"
        _eta_state[exp] = {'validated': validated, 'ts': _now_ts}

        exp_table.add_row(
            exp, str(target),
            str(counts['pending']), str(counts['running']),
            str(counts['completed']), str(counts['failed']),
            str(validated), bar, eta_str
        )

    try:
        with open(_eta_path, 'w') as _f:
            json.dump(_eta_state, _f)
    except Exception:
        pass

    panels.append(Panel(exp_table, title="Experiments", border_style="blue"))

    # ── Regional Quota Utilization ────────────────────────────────────
    QUOTAS = {
        'eu-w4a': {'label': 'eu-w4a (v6e)', 'type': 'v6e-8', 'quota': 64, 'per_vm': 8},
        'us-e1d': {'label': 'us-e1d (v6e)', 'type': 'v6e-8', 'quota': 64, 'per_vm': 8},
        'us-c2b': {'label': 'us-c2b (v4)',  'type': 'v4-8',  'quota': 64, 'per_vm': 4},
        'eu-w4b': {'label': 'eu-w4b (v5e)', 'type': 'v5e-4', 'quota': 64, 'per_vm': 4},
        'us-c1a': {'label': 'us-c1a (v5e)', 'type': 'v5e-4', 'quota': 64, 'per_vm': 4},
    }

    # Map internal VM names (t1v-n-*) to zones by chip count or heartbeat zone field
    _vm_chip_counts = defaultdict(int)
    _vm_zone_from_hb = {}  # vm_name -> zone string from heartbeat JSON
    for wid, hb in state['heartbeats'].items():
        if '_chip' in wid:
            vm = wid.rsplit('_chip', 1)[0]
            _vm_chip_counts[vm] += 1
        else:
            vm = wid
        # Extract zone from heartbeat if present (new babysitters include this)
        hb_zone = hb.get('zone', '') if isinstance(hb, dict) else ''
        hb_tpu = hb.get('tpu_name', '') if isinstance(hb, dict) else ''
        if hb_zone and vm not in _vm_zone_from_hb:
            _vm_zone_from_hb[vm] = hb_zone
        if hb_tpu and vm not in _vm_zone_from_hb:
            _vm_zone_from_hb[vm] = hb_tpu  # use tpu_name for zone lookup below

    def _zone_from_name(name):
        if 'ew4a' in name or 'europe-west4-a' in name: return 'eu-w4a'
        if 'ew4b' in name or 'europe-west4-b' in name: return 'eu-w4b'
        if 'ue1d' in name or 'us-east1' in name: return 'us-e1d'
        if 'uc2b' in name or 'us-central2' in name: return 'us-c2b'
        if 'uc1a' in name or 'us-central1' in name: return 'us-c1a'
        return ''

    def _zone_key(vm_name):
        # 1. Check name directly (our configured VM names)
        zk = _zone_from_name(vm_name)
        if zk: return zk
        # 2. Use zone from heartbeat JSON (new babysitters send this)
        hb_info = _vm_zone_from_hb.get(vm_name, '')
        if hb_info:
            zk = _zone_from_name(hb_info)
            if zk: return zk
        # 3. Fallback: infer from chip count (imprecise for multi-zone v6e)
        chips = _vm_chip_counts.get(vm_name, 0)
        if chips == 4: return 'us-c2b'   # v4-8 has 4 chips
        if chips >= 8: return 'eu-w4a'   # v6e-8 (best guess, may be ue1d)
        if chips > 0: return 'eu-w4a'
        return '?'

    # Aggregate heartbeats by zone
    zone_hb = defaultdict(lambda: {'vms': set(), 'training': 0, 'compiling': 0, 'stuck': 0})
    for wid, hb in state['heartbeats'].items():
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        zk = _zone_key(vm)
        zone_hb[zk]['vms'].add(vm)
        status = hb.get('status', '?')
        age = now - hb.get('timestamp', 0)
        if status == 'training' and age < 900:
            zone_hb[zk]['training'] += 1
        elif status in ('xla_compile', 'starting') and age < 2700:
            zone_hb[zk]['compiling'] += 1
        elif age > 900:
            zone_hb[zk]['stuck'] += 1
        else:
            zone_hb[zk]['compiling'] += 1

    reg_table = Table(show_header=True, header_style="bold", expand=True)
    reg_table.add_column("Zone", style="cyan", min_width=16)
    reg_table.add_column("Type", min_width=6)
    reg_table.add_column("Quota", justify="right", min_width=5)
    reg_table.add_column("VMs", justify="right", min_width=4)
    reg_table.add_column("Claimed", justify="right", min_width=10)
    reg_table.add_column("Training", justify="right", min_width=10)
    reg_table.add_column("Utilization", min_width=28)

    tot_q = tot_c = tot_t = 0
    for zk in ['eu-w4a', 'us-e1d', 'us-c2b', 'eu-w4b', 'us-c1a']:
        q = QUOTAS[zk]
        zh = zone_hb.get(zk, {'vms': set(), 'training': 0, 'compiling': 0, 'stuck': 0})
        n_vms = len(zh['vms'])
        claimed = n_vms * q['per_vm']
        training = zh['training']
        quota = q['quota']
        tot_q += quota; tot_c += claimed; tot_t += training

        cpct = claimed * 100 // quota if quota else 0
        tpct = training * 100 // quota if quota else 0

        # Visual bar: green=training, yellow=claimed-but-not-training, dim=unclaimed
        bar_w = 40
        t_fill = int(training / quota * bar_w) if quota else 0
        c_fill = int(claimed / quota * bar_w) if quota else 0
        bar = (f"[green]{'█' * t_fill}[/green]"
               f"[yellow]{'▒' * (c_fill - t_fill)}[/yellow]"
               f"[dim]{'░' * (bar_w - c_fill)}[/dim]"
               f" {tpct}%")

        claimed_str = f"{claimed}c ({cpct}%)"
        train_str = f"[green]{training}c[/green]" if training > 0 else f"[dim]0c[/dim]"

        reg_table.add_row(q['label'], q['type'], f"{quota}c", str(n_vms),
                         claimed_str, train_str, bar)

    # Totals row
    tot_cpct = tot_c * 100 // tot_q if tot_q else 0
    tot_tpct = tot_t * 100 // tot_q if tot_q else 0
    reg_table.add_row(
        "[bold]TOTAL[/bold]", "", f"[bold]{tot_q}c[/bold]", "",
        f"[bold]{tot_c}c ({tot_cpct}%)[/bold]",
        f"[bold green]{tot_t}c[/bold green]",
        f"[bold]{tot_tpct}% overall[/bold]",
        style="dim"
    )

    reg_summary = Text()
    reg_summary.append(f"  Unclaimed: {tot_q - tot_c} chips", style="yellow")
    reg_summary.append(f"  |  Efficiency: {tot_t}/{tot_c} claimed = ", style="dim")
    eff = tot_t * 100 // tot_c if tot_c else 0
    eff_style = "green bold" if eff > 70 else "yellow bold" if eff > 40 else "red bold"
    reg_summary.append(f"{eff}%", style=eff_style)

    panels.append(Panel(Group(reg_summary, reg_table), title="Regional Quota Utilization", border_style="magenta"))

    # ── Active workers ───────────────────────────────────────────────
    worker_table = Table(show_header=True, header_style="bold", expand=True)
    worker_table.add_column("Worker", style="cyan", min_width=18)
    worker_table.add_column("Task", min_width=25)
    worker_table.add_column("Step", justify="right", min_width=10)
    worker_table.add_column("Status", min_width=12)
    worker_table.add_column("HB Age", justify="right", min_width=8)
    worker_table.add_column("Claim Age", justify="right", min_width=10)

    # Group by VM
    vm_workers = defaultdict(list)
    for task_id, task in state['running_tasks'].items():
        worker_id = task.get('worker_id', '?')
        # Extract VM name from worker_id (format: vmname_chipN)
        vm = worker_id.rsplit('_chip', 1)[0] if '_chip' in worker_id else worker_id
        vm_workers[vm].append((task_id, task, worker_id))

    total_running = 0
    total_stuck = 0
    for vm in sorted(vm_workers.keys()):
        for task_id, task, worker_id in sorted(vm_workers[vm]):
            label = task.get('label', '?')
            claim_age = now - task.get('claimed_at', now)

            hb = state['heartbeats'].get(worker_id, {})
            hb_age = now - hb.get('timestamp', 0) if hb else -1
            step = hb.get('step', 0) if hb else 0
            status = hb.get('status', 'unknown') if hb else 'no_hb'

            # Color status
            if status == 'training':
                status_str = f"[green]{status}[/green]"
                total_running += 1
            elif status == 'xla_compile':
                status_str = f"[yellow]{status}[/yellow]"
                total_running += 1
            elif status in ('idle', 'uploading'):
                status_str = f"[blue]{status}[/blue]"
            elif 'failed' in status:
                status_str = f"[red]{status}[/red]"
                total_stuck += 1
            else:
                status_str = f"[dim]{status}[/dim]"

            hb_str = format_age(hb_age) if hb_age >= 0 else "[red]none[/red]"
            step_str = str(step) if step > 0 else "—"

            worker_table.add_row(
                worker_id, label, step_str, status_str,
                hb_str, format_age(claim_age)
            )

    # Summary line
    summary = Text()
    summary.append(f"  {len(vm_workers)} VMs", style="bold")
    summary.append(f" | ")
    summary.append(f"{total_running} active", style="green")
    if total_stuck > 0:
        summary.append(f" | ")
        summary.append(f"{total_stuck} stuck", style="red")

    if vm_workers:
        panels.append(Panel(Group(summary, worker_table), title="Active Workers", border_style="green"))
    else:
        panels.append(Panel(Text("  No active workers", style="dim"), title="Active Workers", border_style="green"))

    # ── Per-VM health (from heartbeats, grouped by zone) ────────────
    # Build VM stats from heartbeats (source of truth, not running tasks)
    vm_health = defaultdict(lambda: {'training': 0, 'compile': 0, 'stuck': 0, 'idle': 0,
                                      'max_step': 0, 'min_age': 99999, 'type': '?', 'zone': '?'})
    for wid, hb in state['heartbeats'].items():
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        status = hb.get('status', 'unknown')
        hb_age = now - hb.get('timestamp', 0)
        step = hb.get('step', 0)

        vm_health[vm]['max_step'] = max(vm_health[vm]['max_step'], step)
        vm_health[vm]['min_age'] = min(vm_health[vm]['min_age'], hb_age)

        if status == 'training' and hb_age < 900:
            vm_health[vm]['training'] += 1
        elif status in ('xla_compile', 'starting') and hb_age < 2700:
            vm_health[vm]['compile'] += 1
        elif status == 'idle':
            vm_health[vm]['idle'] += 1
        elif hb_age > 900:
            vm_health[vm]['stuck'] += 1
        else:
            vm_health[vm]['compile'] += 1

        # Determine type and zone from VM name
        if 'v4' in vm:
            vm_health[vm]['type'] = 'v4-8'
        elif 'v5e' in vm:
            vm_health[vm]['type'] = 'v5e-4'
        elif 'v6e' in vm:
            vm_health[vm]['type'] = 'v6e-8'
        else:
            # Internal name: infer from chip count
            chips = _vm_chip_counts.get(vm, 0)
            if chips >= 8:
                vm_health[vm]['type'] = 'v6e-8'
            elif chips == 4:
                vm_health[vm]['type'] = 'v4-8'

        zk = _zone_key(vm)
        if zk != '?':
            vm_health[vm]['zone'] = zk
        elif 'ew4a' in vm:
            vm_health[vm]['zone'] = 'eu-w4a'
        elif 'ew4b' in vm:
            vm_health[vm]['zone'] = 'eu-w4b'
        elif 'ue1d' in vm:
            vm_health[vm]['zone'] = 'us-e1d'
        elif 'uc2b' in vm:
            vm_health[vm]['zone'] = 'us-c2b'

    # Build table grouped by zone
    vm_table = Table(show_header=True, header_style="bold", expand=True)
    vm_table.add_column("VM", style="cyan", min_width=18)
    vm_table.add_column("Type", min_width=6)
    vm_table.add_column("Zone", min_width=7)
    vm_table.add_column("Train", justify="right", style="green", min_width=5)
    vm_table.add_column("Comp", justify="right", style="yellow", min_width=5)
    vm_table.add_column("Stuck", justify="right", style="red", min_width=5)
    vm_table.add_column("MaxStep", justify="right", min_width=7)
    vm_table.add_column("FreshHB", justify="right", min_width=7)
    vm_table.add_column("Health", min_width=10)

    # Group by zone
    zone_vms = defaultdict(list)
    for vm in sorted(vm_health.keys()):
        zone_vms[vm_health[vm]['zone']].append(vm)

    # Aggregate stats
    agg = {'vms': 0, 'training': 0, 'compiling': 0, 'stuck': 0, 'idle': 0}
    zone_agg = defaultdict(lambda: {'vms': 0, 'training': 0, 'compiling': 0, 'stuck': 0})

    for zone in sorted(zone_vms.keys()):
        for vm in sorted(zone_vms[zone]):
            h = vm_health[vm]
            total_active = h['training'] + h['compile'] + h['stuck'] + h['idle']
            agg['vms'] += 1
            agg['training'] += h['training']
            agg['compiling'] += h['compile']
            agg['stuck'] += h['stuck']
            agg['idle'] += h['idle']
            zone_agg[zone]['vms'] += 1
            zone_agg[zone]['training'] += h['training']
            zone_agg[zone]['compiling'] += h['compile']
            zone_agg[zone]['stuck'] += h['stuck']

            if h['stuck'] > 0 and h['training'] == 0:
                health = "[red bold]DEGRADED[/red bold]"
            elif h['training'] > 0:
                health = "[green bold]HEALTHY[/green bold]"
            elif h['compile'] > 0:
                health = "[yellow]WARMING[/yellow]"
            elif h['idle'] > 0:
                health = "[blue]IDLE[/blue]"
            else:
                health = "[dim]UNKNOWN[/dim]"

            fresh = format_age(h['min_age']) if h['min_age'] < 99999 else "—"
            max_s = str(h['max_step']) if h['max_step'] > 0 else "—"

            vm_table.add_row(vm, h['type'], h['zone'],
                           str(h['training']), str(h['compile']), str(h['stuck']),
                           max_s, fresh, health)

        # Zone subtotal
        za = zone_agg[zone]
        vm_table.add_row(
            f"[bold]{zone} total[/bold]", "", "",
            f"[bold]{za['training']}[/bold]", f"[bold]{za['compiling']}[/bold]",
            f"[bold]{za['stuck']}[/bold]", "", "", "",
            style="dim"
        )

    # Aggregate summary
    agg_text = Text()
    agg_text.append(f"  {agg['vms']} VMs", style="bold")
    agg_text.append(f" | ")
    agg_text.append(f"{agg['training']} training", style="green bold")
    agg_text.append(f" | ")
    agg_text.append(f"{agg['compiling']} compiling", style="yellow")
    if agg['stuck'] > 0:
        agg_text.append(f" | ")
        agg_text.append(f"{agg['stuck']} stuck", style="red bold")
    if agg['idle'] > 0:
        agg_text.append(f" | ")
        agg_text.append(f"{agg['idle']} idle", style="blue")
    total_chips = agg['training'] + agg['compiling'] + agg['stuck'] + agg['idle']
    agg_text.append(f" | {total_chips} total chips")

    panels.append(Panel(Group(agg_text, vm_table), title="Fleet Health", border_style="cyan"))

    # ── Idle workers (heartbeat but no running task) ──────────────────
    idle_workers = []
    for wid, hb in state['heartbeats'].items():
        if hb.get('status') == 'idle':
            idle_workers.append(wid)
    if idle_workers:
        idle_text = Text()
        idle_text.append(f"  {len(idle_workers)} idle workers: ", style="dim")
        idle_text.append(", ".join(sorted(idle_workers)[:10]), style="yellow")
        if len(idle_workers) > 10:
            idle_text.append(f" ... +{len(idle_workers)-10} more", style="dim")
        panels.append(Panel(idle_text, title="Idle", border_style="yellow"))

    # Done building panels


def render_rich(state, force_color=False, use_pager=False):
    """Render dashboard using rich library."""
    console = Console(force_terminal=force_color)
    panels = []
    _build_panels(state, panels)

    if use_pager:
        with console.pager(styles=True):
            for p in panels:
                console.print(p)
    else:
        for p in panels:
            console.print(p)


def render_plain(state):
    """Fallback plain text rendering."""
    total = state['pending_count'] + state['running_count'] + state['completed_count'] + state['failed_count']
    print(f"\n=== PULL COORDINATOR | {time.strftime('%H:%M:%S')} | total={total} ===")
    print(f"pending={state['pending_count']} running={state['running_count']} "
          f"completed={state['completed_count']} failed={state['failed_count']}")

    for exp, counts in sorted(state['exp_counts'].items()):
        target = EXP_TARGETS.get(exp, '?')
        print(f"  {exp}: validated={counts.get('validated',0)}/{target} "
              f"pending={counts['pending']} running={counts['running']} "
              f"completed={counts['completed']} failed={counts['failed']}")

    now = time.time()
    for task_id, task in sorted(state['running_tasks'].items()):
        wid = task.get('worker_id', '?')
        label = task.get('label', '?')
        hb = state['heartbeats'].get(wid, {})
        step = hb.get('step', 0)
        status = hb.get('status', '?')
        age = format_age(now - hb.get('timestamp', 0)) if hb else 'no_hb'
        print(f"  {wid}: {label} step={step} status={status} hb_age={age}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pull-based coordinator dashboard')
    parser.add_argument('--once', action='store_true', help='Single snapshot')
    parser.add_argument('--pager', action='store_true', help='Scrollable colored output (less -R)')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval')
    parser.add_argument('--plain', action='store_true', help='Plain text (no rich)')
    args = parser.parse_args()

    if args.plain or not HAS_RICH:
        render = render_plain
    else:
        render = lambda state: render_rich(state, force_color=args.pager, use_pager=args.pager)

    if args.once:
        state = fetch_state()
        render(state)
        return

    if args.pager:
        state = fetch_state()
        render_rich(state, force_color=True, use_pager=True)
        return

    # Live auto-refresh — clear + reprint so terminal scroll works
    if HAS_RICH and not args.plain:
        console = Console()
        try:
            while True:
                console.clear()
                state = fetch_state()
                panels_list = []
                _build_panels(state, panels_list)
                for p in panels_list:
                    console.print(p)
                console.print(f"\n[dim]Refreshing in {args.interval}s... (Ctrl+C to quit, scroll up to see all)[/dim]")

                # Check if all done
                all_done = True
                for exp, target in EXP_TARGETS.items():
                    counts = state['exp_counts'].get(exp, {})
                    if counts.get('validated', 0) < target:
                        all_done = False
                if all_done:
                    console.print("\n[bold green]ALL EXPERIMENTS COMPLETE![/bold green]")
                    break

                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        # Fallback plain loop
        while True:
            os.system('clear')
            state = fetch_state()
            render(state)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
