#!/usr/bin/env python3
"""
dashboard.py — Rich TUI dashboard for pull-based coordinator (v3).

Adds VM Boot State panel showing deploy phase telemetry per VM.
Supports --exp name:N to set per-experiment target counts at runtime.

Usage:
    python3 ~/distributed_tpu_training/v3/dashboard.py --once
    python3 ~/distributed_tpu_training/v3/dashboard.py --exp exp13_rerun3:120 --once
    watch -c -n30 'python3 ~/distributed_tpu_training/v3/dashboard.py --exp exp13_rerun3:120 --once'
    python3 ~/distributed_tpu_training/v3/dashboard.py --exp exp13_rerun3:120 --interval 30
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
    from google.cloud import tpu_v2
    _TPU_CLIENT = tpu_v2.TpuClient()
    _HAS_TPU_API = True
except Exception:
    _TPU_CLIENT = None
    _HAS_TPU_API = False

_PROJECT = os.environ.get('PROJECT', 'gcp-research-credits-489020')

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
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(gcs_list, f"{CONTROL_PLANE}/pending"): 'pending',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/running"): 'running',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/completed"): 'completed',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/failed"): 'failed',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/heartbeats"): 'heartbeats',
            pool.submit(gcs_list, f"{CONTROL_PLANE}/telemetry"): 'telemetry',
            pool.submit(fetch_qr_states): 'qr_states',
        }
        lists = {}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                lists[key] = fut.result()
            except Exception:
                lists[key] = [] if key != 'qr_states' else {}

    qr_states = lists.pop('qr_states', {})

    # Read running tasks, heartbeats, and telemetry in parallel
    paths_to_read = (lists.get('running', []) + lists.get('heartbeats', [])
                     + lists.get('telemetry', []))
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

    # Parse telemetry (boot state)
    vm_boot_states = {}
    now = time.time()
    for path in lists.get('telemetry', []):
        fname = os.path.basename(path)
        if not fname.endswith('_boot.json'):
            continue
        raw = raw_data.get(path)
        if raw:
            try:
                data = json.loads(raw)
                vm_name = data.get('tpu_name', fname.replace('_boot.json', ''))
                vm_boot_states[vm_name] = {
                    'phase': data.get('phase', '?'),
                    'zone': data.get('zone', '?'),
                    'age_s': now - data.get('timestamp', now),
                }
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
        'vm_boot_states': vm_boot_states,
        'qr_states': qr_states,
        'exp_counts': dict(exp_counts),
        'timestamp': time.time(),
    }


# ── QueuedResource state fetch ──────────────────────────────────────────────

def fetch_qr_states():
    """Fetch QueuedResource states for all FLEET VMs from GCP API.
    Returns dict: {vm_name: {state, zone, accel, age_s}} or empty if API unavailable.
    """
    if not _HAS_TPU_API or _TPU_CLIENT is None:
        return {}
    try:
        qrs = _TPU_CLIENT.list_queued_resources(
            parent=f'projects/{_PROJECT}/locations/-',
            timeout=10,
        )
        result = {}
        now = time.time()
        for qr in qrs:
            qr_id = qr.name.split('/')[-1]
            zone = qr.name.split('/')[3]
            state = qr.state.state.name if qr.state else 'UNKNOWN'
            accel = ''
            try:
                accel = qr.tpu.node_spec[0].node.accelerator_type
            except Exception:
                pass
            age_s = 0
            try:
                age_s = now - qr.create_time.timestamp()
            except Exception:
                pass
            result[qr_id] = {'state': state, 'zone': zone, 'accel': accel, 'age_s': age_s}
        return result
    except Exception:
        return {}


# ── Rendering ────────────────────────────────────────────────────────────

# Default targets — overridden by --exp name:N args
EXP_TARGETS = {
    'exp13': 120,
    'exp12_1': 185,
    'exp13_rerun': 120,
    'exp13_rerun2': 120,
    'exp13_rerun3': 120,
}


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
    header.append("PULL-BASED COORDINATOR v3", style="bold cyan")
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
    exp_table.add_column("Progress", justify="right", min_width=20)
    exp_table.add_column("ETA", justify="right", style="cyan", min_width=10)

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
        bar_len = 20
        if isinstance(target, int) and target > 0:
            filled = int(validated / target * bar_len)
            bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_len - filled)}[/dim] {validated*100//target}%"
        else:
            bar = "?"

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
            else:
                TOTAL_STEPS = 1778
                V6E_SECS = 5.5
                V4_SECS = 8.5
                total_remaining_chip_secs = 0.0
                active_chips = 0
                for wid, hb in state['heartbeats'].items():
                    if (hb.get('task_id') or '').startswith(exp + '__') or \
                       any(t.get('experiment') == exp for t in state.get('running_tasks', {}).values()
                           if t.get('worker_id', '').rsplit('_chip', 1)[0] == hb.get('tpu_name', wid.rsplit('_chip', 1)[0])):
                        step = hb.get('step', 0)
                        tpu = hb.get('tpu_name', wid)
                        secs_per_step = V6E_SECS if 'v6e' in tpu or 'ew4' in tpu else V4_SECS
                        remaining = max(0, TOTAL_STEPS - step) * secs_per_step
                        total_remaining_chip_secs += remaining
                        active_chips += 1
                if active_chips > 0:
                    v6e_chips = sum(1 for wid in state['heartbeats']
                                    if 'v6e' in wid or 'ew4' in wid)
                    v4_chips = active_chips - v6e_chips
                    tasks_per_sec = max(0.001,
                        (v6e_chips / (TOTAL_STEPS * V6E_SECS)) +
                        (v4_chips / (TOTAL_STEPS * V4_SECS)))
                    remaining_tasks = target - validated
                    secs_left = remaining_tasks / tasks_per_sec
                    if secs_left < 3600:
                        eta_str = f"~{int(secs_left/60)}m*"
                    else:
                        h, m = divmod(int(secs_left / 60), 60)
                        eta_str = f"~{h}h{m:02d}m*"
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

    _vm_chip_counts = defaultdict(int)
    _vm_zone_from_hb = {}
    for wid, hb in state['heartbeats'].items():
        if '_chip' in wid:
            vm = wid.rsplit('_chip', 1)[0]
            _vm_chip_counts[vm] += 1
        else:
            vm = wid
        hb_zone = hb.get('zone', '') if isinstance(hb, dict) else ''
        hb_tpu = hb.get('tpu_name', '') if isinstance(hb, dict) else ''
        if hb_zone and vm not in _vm_zone_from_hb:
            _vm_zone_from_hb[vm] = hb_zone
        if hb_tpu and vm not in _vm_zone_from_hb:
            _vm_zone_from_hb[vm] = hb_tpu

    def _zone_from_name(name):
        if 'ew4a' in name or 'europe-west4-a' in name: return 'eu-w4a'
        if 'ew4b' in name or 'europe-west4-b' in name: return 'eu-w4b'
        if 'ue1d' in name or 'us-east1' in name: return 'us-e1d'
        if 'uc2b' in name or 'us-central2' in name: return 'us-c2b'
        if 'uc1a' in name or 'us-central1' in name: return 'us-c1a'
        return ''

    def _zone_key(vm_name):
        zk = _zone_from_name(vm_name)
        if zk: return zk
        hb_info = _vm_zone_from_hb.get(vm_name, '')
        if hb_info:
            zk = _zone_from_name(hb_info)
            if zk: return zk
        chips = _vm_chip_counts.get(vm_name, 0)
        if chips == 4: return 'us-c2b'
        if chips >= 8: return 'eu-w4a'
        if chips > 0: return 'eu-w4a'
        return '?'

    zone_hb = defaultdict(lambda: {'vms': set(), 'training': 0, 'compiling': 0, 'stuck': 0})
    for wid, hb in state['heartbeats'].items():
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        zk = _zone_key(vm)
        zone_hb[zk]['vms'].add(vm)
        status = hb.get('status', '?')
        age = now - hb.get('timestamp', 0)
        if status == 'training' and age < 1800:
            zone_hb[zk]['training'] += 1
        elif status in ('xla_compile', 'starting') and age < 2700:
            zone_hb[zk]['compiling'] += 1
        elif age > 1800:
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

    vm_workers = defaultdict(list)
    for task_id, task in state['running_tasks'].items():
        worker_id = task.get('worker_id', '?')
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

    # ── Fleet Health ─────────────────────────────────────────────────
    vm_health = defaultdict(lambda: {'training': 0, 'compile': 0, 'stuck': 0, 'idle': 0,
                                      'max_step': 0, 'min_age': 99999, 'type': '?', 'zone': '?'})
    for wid, hb in state['heartbeats'].items():
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        status = hb.get('status', 'unknown')
        hb_age = now - hb.get('timestamp', 0)
        step = hb.get('step', 0)

        vm_health[vm]['max_step'] = max(vm_health[vm]['max_step'], step)
        vm_health[vm]['min_age'] = min(vm_health[vm]['min_age'], hb_age)

        if status == 'training' and hb_age < 1800:
            vm_health[vm]['training'] += 1
        elif status in ('xla_compile', 'starting') and hb_age < 2700:
            vm_health[vm]['compile'] += 1
        elif status == 'idle':
            vm_health[vm]['idle'] += 1
        elif hb_age > 1800:
            vm_health[vm]['stuck'] += 1
        else:
            vm_health[vm]['compile'] += 1

        if 'v4' in vm:
            vm_health[vm]['type'] = 'v4-8'
        elif 'v5e' in vm:
            vm_health[vm]['type'] = 'v5e-4'
        elif 'v6e' in vm:
            vm_health[vm]['type'] = 'v6e-8'
        else:
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

    zone_vms = defaultdict(list)
    for vm in sorted(vm_health.keys()):
        zone_vms[vm_health[vm]['zone']].append(vm)

    agg = {'vms': 0, 'training': 0, 'compiling': 0, 'stuck': 0, 'idle': 0}
    zone_agg = defaultdict(lambda: {'vms': 0, 'training': 0, 'compiling': 0, 'stuck': 0})

    for zone in sorted(zone_vms.keys()):
        for vm in sorted(zone_vms[zone]):
            h = vm_health[vm]
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

        za = zone_agg[zone]
        vm_table.add_row(
            f"[bold]{zone} total[/bold]", "", "",
            f"[bold]{za['training']}[/bold]", f"[bold]{za['compiling']}[/bold]",
            f"[bold]{za['stuck']}[/bold]", "", "", "",
            style="dim"
        )

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

    # ── VM Boot State ─────────────────────────────────────────────────
    # Green: IDLE_AWAITING_WORK | Yellow: INSTALLING_*/DOWNLOADING_*/TESTING_*/intermediate
    # Red: FAILED_* | Dim: no telemetry yet
    boot_table = Table(show_header=True, header_style="bold", expand=True)
    boot_table.add_column("VM", style="cyan", min_width=18)
    boot_table.add_column("Zone", min_width=13)
    boot_table.add_column("Phase", min_width=32)
    boot_table.add_column("Age", justify="right", min_width=6)

    vm_boot_states = state.get('vm_boot_states', {})
    # Also show VMs from fleet that have no telemetry yet
    all_vm_names = set(vm_boot_states.keys())
    # Add VMs from heartbeats that might not have boot telemetry
    for wid in state['heartbeats']:
        vm = wid.rsplit('_chip', 1)[0] if '_chip' in wid else wid
        all_vm_names.add(vm)

    if all_vm_names:
        for vm in sorted(all_vm_names):
            boot = vm_boot_states.get(vm)
            if boot:
                phase = boot['phase']
                zone_str = boot['zone'] if boot['zone'] != '?' else _zone_key(vm)
                age_str = format_age(boot['age_s'])
                if phase == 'IDLE_AWAITING_WORK':
                    phase_str = f"[green]{phase} ✓[/green]"
                elif phase.startswith('FAILED_'):
                    phase_str = f"[red]{phase} ✗[/red]"
                else:
                    phase_str = f"[yellow]{phase}[/yellow]"
            else:
                phase_str = "[dim]no telemetry[/dim]"
                zone_str = _zone_key(vm)
                age_str = "—"
            boot_table.add_row(vm, zone_str, phase_str, age_str)

        n_ready = sum(1 for b in vm_boot_states.values() if b['phase'] == 'IDLE_AWAITING_WORK')
        n_failed = sum(1 for b in vm_boot_states.values() if b['phase'].startswith('FAILED_'))
        n_in_progress = len(vm_boot_states) - n_ready - n_failed

        boot_summary = Text()
        boot_summary.append(f"  {n_ready} ready", style="green bold")
        if n_in_progress > 0:
            boot_summary.append(f" | {n_in_progress} deploying", style="yellow")
        if n_failed > 0:
            boot_summary.append(f" | {n_failed} failed", style="red bold")
        no_telemetry = len(all_vm_names) - len(vm_boot_states)
        if no_telemetry > 0:
            boot_summary.append(f" | {no_telemetry} no telemetry", style="dim")

        panels.append(Panel(Group(boot_summary, boot_table), title="VM Boot State", border_style="yellow"))
    else:
        panels.append(Panel(Text("  No VMs in telemetry yet", style="dim"), title="VM Boot State", border_style="yellow"))

    # ── QueuedResource state (metric #5: queuing visibility) ────────────────
    qr_states = state.get('qr_states', {})
    if qr_states or _HAS_TPU_API:
        qr_table = Table(show_header=True, header_style="bold", expand=True)
        qr_table.add_column("VM", style="cyan", min_width=18)
        qr_table.add_column("Zone", min_width=13)
        qr_table.add_column("Type", min_width=6)
        qr_table.add_column("QR State", min_width=22)
        qr_table.add_column("Age", justify="right", min_width=8)
        qr_table.add_column("Deploy", min_width=22)

        QR_COLORS = {
            'ACTIVE': 'green',
            'WAITING_FOR_RESOURCES': 'yellow',
            'PROVISIONING': 'yellow',
            'DELETING': 'dim',
            'FAILED': 'red',
            'SUSPENDED': 'red',
            'UNKNOWN': 'dim',
        }

        n_active = n_waiting = n_failed_qr = 0
        for vm_name in sorted(qr_states.keys()):
            qr = qr_states[vm_name]
            st = qr.get('state', 'UNKNOWN')
            color = QR_COLORS.get(st, 'dim')
            st_str = f"[{color}]{st}[/{color}]"
            if st == 'ACTIVE': n_active += 1
            elif st in ('WAITING_FOR_RESOURCES', 'PROVISIONING'): n_waiting += 1
            elif st in ('FAILED', 'SUSPENDED'): n_failed_qr += 1

            age_s = qr.get('age_s', 0)
            age_str = format_age(age_s) if age_s > 0 else '?'

            # Cross-reference with boot telemetry
            boot = state.get('vm_boot_states', {}).get(vm_name)
            if boot:
                phase = boot['phase']
                if phase == 'IDLE_AWAITING_WORK':
                    deploy_str = f"[green]{phase} ✓[/green]"
                elif phase.startswith('FAILED_'):
                    deploy_str = f"[red]{phase}[/red]"
                else:
                    deploy_str = f"[yellow]{phase}[/yellow]"
            else:
                deploy_str = "[dim]no telemetry[/dim]"

            qr_table.add_row(vm_name, qr.get('zone', '?'), qr.get('accel', '?'),
                             st_str, age_str, deploy_str)

        qr_summary = Text()
        if not qr_states:
            qr_summary.append("  No QueuedResources yet (vm_manager not started)", style="dim")
        else:
            qr_summary.append(f"  {n_active} ACTIVE", style="green bold")
            if n_waiting:
                qr_summary.append(f" | {n_waiting} WAITING", style="yellow bold")
            if n_failed_qr:
                qr_summary.append(f" | {n_failed_qr} FAILED/SUSPENDED", style="red bold")
            untracked = len(qr_states) - n_active - n_waiting - n_failed_qr
            if untracked:
                qr_summary.append(f" | {untracked} other", style="dim")

        panels.append(Panel(Group(qr_summary, qr_table) if qr_states else qr_summary,
                           title="QueuedResource State", border_style="blue"))

    # ── Idle workers ─────────────────────────────────────────────────
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


def render_rich(state, force_color=False, use_pager=False):
    """Render dashboard using rich library."""
    console = Console(force_terminal=force_color, width=160)
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
    print(f"\n=== PULL COORDINATOR v3 | {time.strftime('%H:%M:%S')} | total={total} ===")
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

    print("\n-- VM Boot States --")
    for vm, boot in sorted(state.get('vm_boot_states', {}).items()):
        print(f"  {vm}: {boot['phase']} (zone={boot['zone']} age={format_age(boot['age_s'])})")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pull-based coordinator dashboard v3')
    parser.add_argument('--once', action='store_true', help='Single snapshot')
    parser.add_argument('--pager', action='store_true', help='Scrollable colored output (less -R)')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval')
    parser.add_argument('--plain', action='store_true', help='Plain text (no rich)')
    parser.add_argument('--exp', nargs='+', metavar='NAME[:N]',
                        help='Experiments to track. Use name:N to set target count (e.g. exp13_rerun3:120)')
    args = parser.parse_args()

    # Update EXP_TARGETS from --exp args
    if args.exp:
        for spec in args.exp:
            if ':' in spec:
                name, n = spec.rsplit(':', 1)
                try:
                    EXP_TARGETS[name] = int(n)
                except ValueError:
                    EXP_TARGETS[name] = spec
            else:
                # Just register the experiment (use existing target if known)
                if spec not in EXP_TARGETS:
                    EXP_TARGETS[spec] = '?'

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

                all_done = True
                for exp, target in EXP_TARGETS.items():
                    if not isinstance(target, int):
                        continue
                    counts = state['exp_counts'].get(exp, {})
                    if counts.get('validated', 0) < target:
                        all_done = False
                if all_done and EXP_TARGETS:
                    console.print("\n[bold green]ALL EXPERIMENTS COMPLETE![/bold green]")
                    break

                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        while True:
            os.system('clear')
            state = fetch_state()
            render(state)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
