#!/usr/bin/env python3
"""
TPU Fleet Dashboard — rich-based terminal UI.

Single command: python3 ~/tpu_guide/dashboard.py [--exp EXP_NAME]
Shows: fleet status, zone capacity, experiment progress, per-VM workers, phase tracking.
Auto-refreshes every 60s.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import BarColumn, Progress, TextColumn, TaskProgressColumn
    from rich import box
except ImportError:
    print("ERROR: rich not found. Install: pip3 install rich")
    sys.exit(1)

GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
PROJECT = "gcp-research-credits-489020"
TPU_GUIDE = os.path.expanduser("~/tpu_guide")
RESULTS_BASE = os.path.expanduser("~/sf_bema/results")

# Per-zone quota grants
QUOTA_GRANTS = [
    {"zone": "europe-west4-a", "type": "v6e", "chips": 64, "mode": "spot", "usable": True, "internet": True},
    {"zone": "us-east1-d", "type": "v6e", "chips": 64, "mode": "spot", "usable": True, "internet": False},
    {"zone": "us-central2-b", "type": "v4", "chips": 64, "mode": "spot+ondemand", "usable": True, "internet": False},
    {"zone": "us-central1-a", "type": "v5e", "chips": 64, "mode": "spot", "usable": True, "internet": False},
    {"zone": "europe-west4-b", "type": "v5e", "chips": 64, "mode": "spot", "usable": True, "internet": True},
]

# Granted zones to scan (usable only)
GRANTED_ZONES = ["europe-west4-a", "us-east1-d", "us-central2-b", "us-central1-a", "europe-west4-b"]

# Chip specs
CHIP_SPECS = {
    "v6e": {"hbm": 32, "step_s": 4.9, "runtime": "v2-alpha-tpuv6e"},
    "v4": {"hbm": 32, "step_s": 8.4, "runtime": "v2-alpha-tpuv4"},
    "v5e": {"hbm": 16, "step_s": 0, "runtime": ""},  # OOM
}


def run_cmd(cmd, timeout=30):
    """Run shell command, return stdout or empty string on error."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def scan_vms():
    """Scan all granted zones for TPU VMs."""
    vms = []
    for zone in GRANTED_ZONES:
        raw = run_cmd(
            f"{GCLOUD} alpha compute tpus tpu-vm list --zone={zone} --project={PROJECT} "
            f"--format='table[no-heading](name,acceleratorType,state)' 2>/dev/null",
            timeout=20
        )
        if not raw:
            continue
        for line in raw.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                name, accel, state = parts[0], parts[1], parts[2]
                chip_match = re.search(r"(\d+)$", accel)
                chips = int(chip_match.group(1)) if chip_match else 0
                chip_type = re.match(r"(v\w+)-", accel)
                chip_type = chip_type.group(1) if chip_type else "?"
                vms.append({
                    "name": name, "accel": accel, "state": state,
                    "zone": zone, "chips": chips, "type": chip_type,
                })
    return vms


def load_vm_configs():
    """Read all vm_configs/*.env files."""
    vm_dir = os.path.join(TPU_GUIDE, "vm_configs")
    configs = []
    if not os.path.isdir(vm_dir):
        return configs
    for f in sorted(os.listdir(vm_dir)):
        if not f.endswith(".env"):
            continue
        cfg = {}
        with open(os.path.join(vm_dir, f)) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
        configs.append(cfg)
    return configs


def load_exp_config(exp_name):
    """Load experiment .env config."""
    path = os.path.join(TPU_GUIDE, "experiments", f"{exp_name}.env")
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def get_total_configs(exp_cfg):
    """Get total config count from experiment module."""
    work_dir = exp_cfg.get("WORK_DIR", "")
    module = exp_cfg.get("EXP_MODULE", "")
    if not module:
        return 0
    try:
        raw = run_cmd(
            f"cd ~/sf_bema/experiments/{work_dir} && python3 -c \""
            f"import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,os.path.expanduser('~/sf_bema/experiments')); "
            f"m=__import__('{module}',fromlist=['build_configs']); print(len(m.build_configs()))\"",
            timeout=10
        )
        return int(raw) if raw.isdigit() else 0
    except Exception:
        return 0


def get_wandb_info(exp_cfg):
    """Extract WANDB_PROJECT and WANDB_ENTITY from experiment module."""
    work_dir = exp_cfg.get("WORK_DIR", "")
    module = exp_cfg.get("EXP_MODULE", "")
    if not module:
        return None, None
    try:
        raw = run_cmd(
            f"cd ~/sf_bema/experiments/{work_dir} && python3 -c \""
            f"import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,os.path.expanduser('~/sf_bema/experiments')); "
            f"m=__import__('{module}',fromlist=['WANDB_PROJECT','WANDB_ENTITY']); "
            f"print(getattr(m,'WANDB_ENTITY',''),getattr(m,'WANDB_PROJECT',''))\"",
            timeout=10
        )
        parts = raw.strip().split(" ", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
    except Exception:
        pass
    return None, None


def get_validated_count(exp_name):
    """Count validated result files."""
    vdir = os.path.join(RESULTS_BASE, exp_name, "validated")
    if not os.path.isdir(vdir):
        return 0
    return len([f for f in os.listdir(vdir) if f.endswith(".json")])


def get_state(exp_name):
    """Read coordinator state.json."""
    path = os.path.join(RESULTS_BASE, exp_name, "state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def query_vm_live(vm_cfg, exp_name):
    """Query VM status via GCS heartbeats (fast, no SSH needed).

    Heartbeat files contain: {worker_id, vm, timestamp, step, label}
    Status is derived from heartbeat freshness and step progress.
    """
    tpu = vm_cfg.get("TPU_NAME", "")
    bucket = vm_cfg.get("BUCKET", "")
    if not tpu or not bucket:
        return {"status": "NO CONFIG", "color": "dim"}

    prefix = f"{bucket}/coord/{exp_name}"

    # Read all heartbeat files for this VM (fast GCS list + cat)
    hb_paths = run_cmd(f"{GCLOUD} storage ls {prefix}/heartbeat/ 2>/dev/null", timeout=15)
    heartbeats = []
    if hb_paths:
        for p in hb_paths.strip().split("\n"):
            p = p.strip()
            if not p or not tpu in p:
                continue
            content = run_cmd(f"{GCLOUD} storage cat {p} 2>/dev/null", timeout=10)
            if content:
                try:
                    hb = json.loads(content)
                    heartbeats.append(hb)
                except Exception:
                    pass

    if not heartbeats:
        # No heartbeats — worker starting up (loading model, compiling, etc.)
        # Check if assignment exists
        assign_raw = run_cmd(f"{GCLOUD} storage cat {prefix}/assignments/{tpu}.json 2>/dev/null", timeout=10)
        if assign_raw:
            return {
                "status": "STARTING", "color": "yellow",
                "procs": 0, "step": 0, "total_steps": 0,
                "train_loss": "—", "best_val": "—", "time_left": "waiting",
                "done": 0, "config": "—",
            }
        return {"status": "NO ASSIGNMENT", "color": "dim"}

    # Find best heartbeat: prefer highest step (active worker), break ties by freshest
    now = time.time()
    best_hb = max(heartbeats, key=lambda h: (h.get("step", 0), h.get("timestamp", 0)))
    age_s = now - best_hb.get("timestamp", 0)
    cur_step = best_hb.get("step", 0)
    cur_label = best_hb.get("label", "—")

    # Count active workers (heartbeat < 10 min old)
    active_workers = sum(1 for h in heartbeats if now - h.get("timestamp", 0) < 600)
    # Count unique labels being worked on
    active_labels = {h.get("label") for h in heartbeats if now - h.get("timestamp", 0) < 600}

    # Determine status from heartbeat age + step
    # Heartbeats update every 100 steps (~500s at 5s/step), so age < 900s is normal
    if cur_step == 0 and age_s < 1500:
        status, color = "XLA COMPILE", "yellow"
    elif cur_step == 0 and age_s >= 1500:
        status, color = "STUCK (step=0)", "red"
    elif cur_step > 0 and age_s > 3600:
        status, color = "DEAD", "red"
    elif cur_step > 0 and age_s > 900:
        status, color = "STALE", "yellow"
    elif cur_step > 0:
        status, color = "RUNNING", "green"
    else:
        status, color = "UNKNOWN", "dim"

    return {
        "status": status, "color": color,
        "procs": active_workers, "step": cur_step, "total_steps": 0,
        "train_loss": "—", "best_val": "—",
        "time_left": f"{age_s/60:.0f}m ago" if cur_step > 0 else "~10min",
        "done": 0, "config": cur_label,
        "active_labels": len(active_labels),
    }


def query_vm_gcs(vm_cfg, exp_name):
    """Query GCS for assignment count and active heartbeat count for a VM."""
    tpu = vm_cfg.get("TPU_NAME", "")
    bucket = vm_cfg.get("BUCKET", "")
    if not bucket:
        return {"gcs_assigned": 0, "gcs_heartbeats": 0}

    prefix = f"{bucket}/coord/{exp_name}"

    # Read current assignment count (= remaining work for this VM)
    assign_raw = run_cmd(f"{GCLOUD} storage cat {prefix}/assignments/{tpu}.json 2>/dev/null", timeout=15)
    gcs_assigned = 0
    assigned_labels = []
    if assign_raw:
        try:
            assigned_labels = json.loads(assign_raw)
            gcs_assigned = len(assigned_labels)
        except Exception:
            pass

    # Count done receipts that match assigned labels (= completed from current assignment)
    done_raw = run_cmd(f"{GCLOUD} storage ls {prefix}/done/ 2>/dev/null", timeout=15)
    gcs_done = 0
    if done_raw:
        done_files = {os.path.basename(f).replace('.done', '') for f in done_raw.strip().split('\n') if f.strip()}
        assigned_label_set = {item.get('label', '') for item in assigned_labels} if assigned_labels else set()
        if assigned_label_set:
            gcs_done = len(done_files & assigned_label_set)
        else:
            gcs_done = len(done_files)

    return {"gcs_done": gcs_done, "gcs_assigned": gcs_assigned}


def build_vm_detail_table(vm_configs, exp_name, steps_per_config):
    """Build per-VM detail table with live SSH + GCS data (parallel queries)."""
    import concurrent.futures

    if not vm_configs or not exp_name:
        return Panel("[dim]No VM configs found. Create vm_configs/*.env for READY VMs.[/dim]",
                     title="Per-VM Workers", border_style="magenta")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta",
                  show_lines=False, padding=(0, 1))
    table.add_column("VM", style="bold", no_wrap=True)
    table.add_column("Chips", justify="right")
    table.add_column("Status", no_wrap=True)
    table.add_column("Step", justify="right", no_wrap=True)
    table.add_column("Train", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("ETA", justify="right")
    table.add_column("D/A", justify="right", no_wrap=True)
    table.add_column("cfg/h", justify="right")

    # Parallel SSH + GCS queries
    ssh_results = {}
    gcs_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        ssh_futures = {}
        gcs_futures = {}
        for cfg in vm_configs:
            tpu = cfg.get("TPU_NAME", "")
            pph = int(cfg.get("PROCS_PER_HOST", "0"))
            if pph == 0:
                continue
            ssh_futures[tpu] = pool.submit(query_vm_live, cfg, exp_name)
            gcs_futures[tpu] = pool.submit(query_vm_gcs, cfg, exp_name)

        for tpu, fut in ssh_futures.items():
            try:
                ssh_results[tpu] = fut.result(timeout=50)
            except Exception:
                ssh_results[tpu] = {"status": "TIMEOUT", "color": "red"}

        for tpu, fut in gcs_futures.items():
            try:
                gcs_results[tpu] = fut.result(timeout=20)
            except Exception:
                gcs_results[tpu] = {"gcs_done": 0, "gcs_assigned": 0}

    for cfg in vm_configs:
        tpu = cfg.get("TPU_NAME", "")
        accel = cfg.get("ACCELERATOR_TYPE", "")
        pph = int(cfg.get("PROCS_PER_HOST", "0"))
        hosts = int(cfg.get("TPU_NUM_WORKERS", "1"))
        chips = int(re.search(r"(\d+)$", accel).group(1)) if re.search(r"(\d+)$", accel) else 0

        if pph == 0:
            table.add_row(tpu, str(chips), "[red]OOM[/red]", "—", "—", "—", "—", "—", "—")
            continue

        procs = hosts * pph
        chip_type = re.match(r"(v\w+)-", accel)
        chip_type = chip_type.group(1) if chip_type else "v6e"
        step_s = CHIP_SPECS.get(chip_type, {}).get("step_s", 5.0)
        cfg_h = steps_per_config * step_s / 3600
        tput = f"{procs / cfg_h:.1f} cfg/h"

        live = ssh_results.get(tpu, {})
        gcs = gcs_results.get(tpu, {})

        status_str = f"[{live.get('color', 'dim')}]{live.get('status', '?')}[/{live.get('color', 'dim')}]"
        step_str = f"{live.get('step', '—')}/{steps_per_config}" if live.get('step', 0) > 0 else "—"

        gcs_done = gcs.get('gcs_done', 0)
        gcs_assigned = gcs.get('gcs_assigned', 0)
        done_asgn = f"{gcs_done}/{gcs_assigned}"

        table.add_row(
            tpu, str(chips),
            status_str,
            step_str,
            str(live.get("train_loss", "—")),
            str(live.get("best_val", "—")),
            str(live.get("time_left", "—")),
            done_asgn,
            tput,
        )

    return Panel(table, title="Per-VM Workers", border_style="magenta")


def detect_phase(vms, vm_configs, exp_name, validated, total):
    """Detect current workflow phase."""
    ready_vms = [v for v in vms if v["state"] == "READY"]
    creating_vms = [v for v in vms if v["state"] == "CREATING"]

    if total > 0 and validated >= total:
        return "DONE", "All configs validated!", "green"

    state = get_state(exp_name)
    if state.get("assignments"):
        if validated > 0:
            return "MONITORING", f"Sweep running — {validated}/{total} validated", "green"
        return "SWEEPING", "Workers executing configs", "cyan"

    if not vms:
        return "ACQUIRING", "No VMs — creating Spot instances", "yellow"
    if creating_vms and not ready_vms:
        return "ACQUIRING", f"{len(creating_vms)} VMs CREATING, waiting for READY", "yellow"
    if ready_vms and not vm_configs:
        return "SETUP NEEDED", f"{len(ready_vms)} READY VMs, need vm_configs + setup", "yellow"
    if vm_configs and not state.get("assignments"):
        return "INIT NEEDED", "VMs configured, run coordinator --init", "yellow"

    return "ACQUIRING", f"{len(ready_vms)} READY, {len(creating_vms)} CREATING", "yellow"


def build_fleet_table(vms):
    """Build fleet status table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("VM Name", style="bold")
    table.add_column("Type")
    table.add_column("Chips", justify="right")
    table.add_column("Zone")
    table.add_column("Status")

    for vm in sorted(vms, key=lambda v: (v["zone"], v["name"])):
        state = vm["state"]
        if state == "READY":
            style = "[green]READY[/green]"
        elif state == "CREATING":
            style = "[yellow]CREATING[/yellow]"
        elif state == "DELETING":
            style = "[red]DELETING[/red]"
        else:
            style = f"[red]{state}[/red]"
        table.add_row(vm["name"], vm["accel"], str(vm["chips"]), vm["zone"], style)

    if not vms:
        table.add_row("[dim]no VMs found[/dim]", "", "", "", "")

    return table


def build_quota_table(vms):
    """Build quota usage table."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Zone")
    table.add_column("Type")
    table.add_column("Quota")
    table.add_column("Used", justify="right")
    table.add_column("Status")

    for grant in QUOTA_GRANTS:
        zone = grant["zone"]
        used = sum(v["chips"] for v in vms if v["zone"] == zone and v["state"] in ("READY", "CREATING"))
        limit = grant["chips"]
        pct = used * 100 // limit if limit else 0

        if not grant["usable"]:
            status = "[dim]OOM — skip[/dim]"
        elif used == 0:
            status = "[red]empty[/red]"
        elif used >= limit:
            status = "[green]FULL[/green]"
        else:
            status = f"[yellow]{pct}%[/yellow]"

        internet = " [cyan](internet)[/cyan]" if grant["internet"] else ""
        table.add_row(
            f"{zone}{internet}",
            grant["type"],
            f"{limit} {grant['mode']}",
            str(used) if grant["usable"] else "—",
            status,
        )

    return table


def compute_fleet_eta(vm_configs, exp_cfg, total, validated):
    """Compute total fleet ETA based on all VM throughputs."""
    steps = int(exp_cfg.get("STEPS_PER_CONFIG", 1778))
    remaining = total - validated
    if remaining <= 0:
        return "done", 0

    total_tput = 0  # configs/hour
    for cfg in vm_configs:
        pph = int(cfg.get("PROCS_PER_HOST", "0"))
        hosts = int(cfg.get("TPU_NUM_WORKERS", "1"))
        if pph == 0:
            continue
        procs = hosts * pph
        accel = cfg.get("ACCELERATOR_TYPE", "")
        chip_type = re.match(r"(v\w+)-", accel)
        chip_type = chip_type.group(1) if chip_type else "v6e"
        step_s = CHIP_SPECS.get(chip_type, {}).get("step_s", 5.0)
        if step_s == 0:
            continue
        cfg_h = steps * step_s / 3600
        total_tput += procs / cfg_h

    if total_tput == 0:
        return "unknown", 0

    eta_h = remaining / total_tput
    return f"{eta_h:.1f}h (~{eta_h/24:.1f}d)", total_tput


def build_experiment_panel(exp_name, exp_cfg, validated, total, state, vm_configs=None, wandb_info=None):
    """Build experiment progress panel."""
    if not exp_name:
        return Panel("[dim]No experiment specified. Use --exp EXP_NAME[/dim]", title="Experiment")

    steps = int(exp_cfg.get("STEPS_PER_CONFIG", 1778))
    assignments = state.get("assignments", {})
    failed = state.get("failed", [])

    lines = []
    lines.append(f"[bold]{exp_name}[/bold] — {total} configs × {steps} steps")

    # W&B link
    if wandb_info and wandb_info[0] and wandb_info[1]:
        entity, project = wandb_info
        url = f"https://wandb.ai/{entity}/{project}"
        lines.append(f"  W&B: [link={url}]{url}[/link]")

    lines.append("")

    # Progress bar
    if total > 0:
        pct = validated * 100 // total
        bar_width = 40
        filled = validated * bar_width // total
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_width - filled)
        lines.append(f"  Progress: {bar} {validated}/{total} ({pct}%)")
    else:
        lines.append("  Progress: [dim]unknown total[/dim]")

    lines.append(f"  Failed:   {len(failed)}")

    # Fleet ETA
    if vm_configs and total > 0:
        eta_str, total_tput = compute_fleet_eta(vm_configs, exp_cfg, total, validated)
        lines.append(f"  Fleet:    {total_tput:.1f} cfg/h  │  ETA: [bold]{eta_str}[/bold]")

    lines.append("")

    # Per-VM assignments (only show VMs with >0 configs)
    if assignments:
        active_assignments = {vm: labels for vm, labels in sorted(assignments.items())
                              if (len(labels) if isinstance(labels, list) else labels) > 0}
        if active_assignments:
            lines.append("  [bold]Assignments:[/bold]")
            for vm, labels in sorted(active_assignments.items()):
                count = len(labels) if isinstance(labels, list) else labels
                lines.append(f"    {vm}: {count} configs")
        else:
            lines.append("  [dim]All assignments completed[/dim]")
    else:
        lines.append("  [dim]No assignments yet (run coordinator --init)[/dim]")

    return Panel("\n".join(lines), title="Experiment", border_style="blue")


def build_phase_panel(phase, detail, color):
    """Build phase indicator panel."""
    phases = ["ACQUIRING", "SETUP NEEDED", "INIT NEEDED", "SWEEPING", "MONITORING", "DONE"]
    parts = []
    for p in phases:
        if p == phase:
            parts.append(f"[bold {color}]▶ {p}[/bold {color}]")
        elif phases.index(p) < phases.index(phase) if phase in phases else False:
            parts.append(f"[green]✓ {p}[/green]")
        else:
            parts.append(f"[dim]○ {p}[/dim]")

    phase_line = "  →  ".join(parts)
    return Panel(f"{phase_line}\n\n[{color}]{detail}[/{color}]", title="Phase", border_style=color)


def build_actions_panel(phase, exp_name, vm_configs, vms):
    """Suggest next actions based on current phase."""
    ready_vms = [v for v in vms if v["state"] == "READY"]
    lines = []

    if phase == "ACQUIRING":
        lines.append("[yellow]Waiting for VMs to become READY.[/yellow]")
        lines.append("If stuck, try different sizes or zones.")
        lines.append("")
        lines.append("[bold]Retry creates:[/bold]")
        for zone in GRANTED_ZONES:
            zone_vms = [v for v in vms if v["zone"] == zone]
            if not zone_vms:
                lines.append(f"  {GCLOUD} alpha compute tpus tpu-vm create <name> \\")
                lines.append(f"    --zone={zone} --project={PROJECT} \\")
                lines.append(f"    --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot --async")

    elif phase == "SETUP NEEDED":
        lines.append("[yellow]VMs are READY. Create vm_configs and run setup:[/yellow]")
        for vm in ready_vms:
            lines.append(f"  EXP={exp_name} TPU_NAME={vm['name']} bash ~/tpu_guide/submit.sh --setup")

    elif phase == "INIT NEEDED":
        lines.append("[yellow]Run coordinator init to distribute configs:[/yellow]")
        lines.append(f"  EXP={exp_name} python3 ~/tpu_guide/coordinator.py --init")

    elif phase in ("SWEEPING", "MONITORING"):
        lines.append("[green]Sweep is running. Monitor with:[/green]")
        lines.append(f"  EXP={exp_name} bash ~/tpu_guide/watch.sh")
        lines.append(f"  EXP={exp_name} python3 ~/tpu_guide/coordinator.py --status")

    elif phase == "DONE":
        lines.append("[green bold]All configs validated! Pull results:[/green bold]")
        lines.append(f"  ls ~/sf_bema/results/{exp_name}/validated/")

    if not lines:
        lines.append("[dim]No suggested actions.[/dim]")

    return Panel("\n".join(lines), title="Next Actions", border_style="dim")


def build_automation_panel():
    """Show babysitter/monitor process status and next scheduled actions."""
    lines = []

    # Check babysitter processes
    try:
        r = subprocess.run("ps aux | grep 'babysit_exp' | grep -v grep", shell=True,
                          capture_output=True, text=True, timeout=5)
        babysit_lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        if babysit_lines:
            for bl in babysit_lines:
                parts = bl.split()
                pid = parts[1]
                # Extract exp name from command
                exp_match = re.search(r'babysit_(\w+)', bl)
                exp_id = exp_match.group(1) if exp_match else "?"
                lines.append(f"  [green]babysit_{exp_id}[/green] PID={pid}")
        else:
            lines.append("  [red]No babysitter running[/red]")
    except Exception:
        lines.append("  [dim]Could not check babysitter[/dim]")

    # Check monitor processes
    try:
        r = subprocess.run("ps aux | grep 'coordinator.py --monitor' | grep -v grep", shell=True,
                          capture_output=True, text=True, timeout=5)
        mon_lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        if mon_lines:
            lines.append(f"  [green]monitor[/green] × {len(mon_lines)} process(es)")
        else:
            lines.append("  [red]No monitor running[/red]")
    except Exception:
        lines.append("  [dim]Could not check monitor[/dim]")

    # Babysitter cycle info
    lines.append("")
    lines.append("  [dim]Babysitter: checks every 5min, VM health every ~30min[/dim]")
    lines.append("  [dim]Monitor: pulls GCS results every ~60s[/dim]")

    # Check for VM scanner
    try:
        r = subprocess.run("ps aux | grep 'vm_scanner' | grep -v grep", shell=True,
                          capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            lines.append("  [green]VM scanner running[/green]")
    except Exception:
        pass

    return Panel("\n".join(lines), title="Automation", border_style="green")


def build_dashboard(exp_name=None):
    """Build the full dashboard layout."""
    console = Console()

    # Scan fleet
    vms = scan_vms()
    vm_configs = load_vm_configs()

    # Experiment data
    exp_cfg = load_exp_config(exp_name) if exp_name else {}
    total = get_total_configs(exp_cfg) if exp_cfg else 0
    validated = get_validated_count(exp_name) if exp_name else 0
    state = get_state(exp_name) if exp_name else {}
    wandb_info = get_wandb_info(exp_cfg) if exp_cfg else (None, None)

    # Detect phase
    phase, detail, color = detect_phase(vms, vm_configs, exp_name or "", validated, total)

    # Fleet stats
    ready_chips = sum(v["chips"] for v in vms if v["state"] == "READY")
    creating_chips = sum(v["chips"] for v in vms if v["state"] == "CREATING")

    # Build components
    output_parts = []

    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = Text()
    header.append("  TPU Fleet Dashboard", style="bold white")
    header.append(f"  │  {now}  │  ", style="dim")
    header.append(f"{ready_chips} READY", style="bold green")
    header.append(" + ", style="dim")
    header.append(f"{creating_chips} CREATING", style="bold yellow")
    header.append(" chips", style="dim")
    output_parts.append(Panel(header, box=box.DOUBLE))

    # Phase
    output_parts.append(build_phase_panel(phase, detail, color))

    # Fleet + Quota side by side
    fleet_table = build_fleet_table(vms)
    quota_table = build_quota_table(vms)
    output_parts.append(Panel(fleet_table, title="Fleet Status", border_style="cyan"))
    output_parts.append(Panel(quota_table, title="Quota Usage", border_style="dim"))

    # Experiment
    if exp_name:
        output_parts.append(build_experiment_panel(exp_name, exp_cfg, validated, total, state, vm_configs, wandb_info))

    # Per-VM workers (only if vm_configs exist and experiment is set)
    if exp_name and vm_configs:
        steps = int(exp_cfg.get("STEPS_PER_CONFIG", 1778))
        output_parts.append(build_vm_detail_table(vm_configs, exp_name, steps))

    # Next actions
    output_parts.append(build_actions_panel(phase, exp_name or "EXP", vm_configs, vms))

    # Refresh note
    output_parts.append(Text("  Auto-refreshes every 60s. Ctrl+C to exit.", style="dim"))

    return output_parts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TPU Fleet Dashboard")
    parser.add_argument("--exp", default=None, nargs='+', help="Experiment name(s) (e.g., exp12_1 exp13)")
    parser.add_argument("--once", action="store_true", help="Print once and exit (no live refresh)")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds")
    args = parser.parse_args()

    # Auto-detect experiment from EXP env var
    exp_names = args.exp or [os.environ.get("EXP")] if (args.exp or os.environ.get("EXP")) else [None]
    if isinstance(exp_names, str):
        exp_names = [exp_names]
    # Use the first experiment for the main display (backwards compatible)
    exp_name = exp_names[0] if exp_names else None

    console = Console(force_terminal=True)

    def render_all(target_console):
        """Render fleet once, then each experiment's panels."""
        # --- Shared fleet section (once) ---
        vms = scan_vms()
        vm_configs = load_vm_configs()

        ready_chips = sum(v["chips"] for v in vms if v["state"] == "READY")
        creating_chips = sum(v["chips"] for v in vms if v["state"] == "CREATING")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Text()
        header.append("  TPU Fleet Dashboard", style="bold white")
        header.append(f"  │  {now}  │  ", style="dim")
        header.append(f"{ready_chips} READY", style="bold green")
        header.append(" + ", style="dim")
        header.append(f"{creating_chips} CREATING", style="bold yellow")
        header.append(" chips", style="dim")
        exp_list = ", ".join(n for n in exp_names if n)
        if exp_list:
            header.append(f"  │  ", style="dim")
            header.append(exp_list, style="bold cyan")
        target_console.print(Panel(header, box=box.DOUBLE))

        # Fleet + Quota
        target_console.print(Panel(build_fleet_table(vms), title="Fleet Status", border_style="cyan"))
        target_console.print(Panel(build_quota_table(vms), title="Quota Usage", border_style="dim"))

        # Automation status
        target_console.print(build_automation_panel())

        # --- Per-experiment sections ---
        for en in exp_names:
            if not en:
                continue
            exp_cfg = load_exp_config(en)
            total = get_total_configs(exp_cfg) if exp_cfg else 0
            validated = get_validated_count(en)
            state = get_state(en)
            wandb_info = get_wandb_info(exp_cfg) if exp_cfg else (None, None)

            # Phase (per-experiment)
            phase, detail, color = detect_phase(vms, vm_configs, en, validated, total)

            target_console.print(build_experiment_panel(en, exp_cfg, validated, total, state, vm_configs, wandb_info))

            # Per-VM workers — only show VMs assigned to this experiment
            if vm_configs:
                steps = int(exp_cfg.get("STEPS_PER_CONFIG", 1778))
                assigned_vms = set(state.get("assignments", {}).keys())
                if assigned_vms:
                    filtered_configs = [c for c in vm_configs if c.get("TPU_NAME", "") in assigned_vms]
                else:
                    filtered_configs = vm_configs
                if filtered_configs:
                    target_console.print(build_vm_detail_table(filtered_configs, en, steps))

    if args.once:
        render_all(console)
        return

    # Refresh mode
    import shutil
    outfile = f"/tmp/dashboard_{'_'.join(n for n in exp_names if n)}.txt"
    while True:
        try:
            term_w = shutil.get_terminal_size().columns
            from io import StringIO
            buf = StringIO()
            file_console = Console(file=buf, width=term_w, force_terminal=True)

            render_all(file_console)
            file_console.print(f"\n[dim]Updated: {datetime.now().strftime('%H:%M:%S')} | Next in {args.interval}s | Ctrl+C to quit[/dim]\n")

            # Write to file (atomic via rename)
            tmp = outfile + ".tmp"
            with open(tmp, "w") as f:
                f.write(buf.getvalue())
            os.replace(tmp, outfile)

            # Also print to terminal with clear
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(buf.getvalue())
            sys.stdout.flush()

            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
