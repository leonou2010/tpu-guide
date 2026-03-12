#!/usr/bin/env python3
"""
upload_wandb.py — Upload experiment results to W&B from local validated files.

Run from blocklab after experiment completes:
    python3 ~/distributed_tpu_training/pull/upload_wandb.py --exp exp13

Reads from: ~/sf_bema/results/{exp}/validated/
Per-step train loss: {label}_train_loss.jsonl (if present)
Eval logs: from {label}.json summary/eval_logs
"""

import argparse
import json
import os
import sys


def load_jsonl(path):
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return entries


def upload_run(label, result_path, jsonl_path, project, entity, group, dry_run):
    with open(result_path) as f:
        data = json.load(f)

    config = data.get('config', {})
    summary = data.get('summary', {})
    eval_logs = data.get('eval_logs', [])

    # Flatten nested config for W&B
    run_config = {}
    for section, vals in config.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                run_config[f"{section}.{k}"] = v
        else:
            run_config[section] = vals

    train_entries = load_jsonl(jsonl_path) if jsonl_path and os.path.exists(jsonl_path) else []

    if dry_run:
        best_val = summary.get('best_val_loss', float('nan'))
        print(f"  [dry-run] {label}: best_val={best_val:.4f}, "
              f"steps={summary.get('total_steps', 0)}, train_jsonl={len(train_entries)}")
        return

    import wandb

    print(f"  Uploading {label}...", end='', flush=True)
    run = wandb.init(
        project=project,
        entity=entity,
        name=label,
        config=run_config,
        group=group,
        reinit=True,
        resume="allow",
        id=f"{group}__{label}".replace('/', '_').replace(' ', '_'),
    )

    # Log per-step train loss from JSONL
    if train_entries:
        for entry in train_entries:
            wandb.log({'train/loss': entry['loss']}, step=entry['step'], commit=False)
        print(f" {len(train_entries)} train steps", end='', flush=True)

    # Log eval checkpoints
    for log in eval_logs:
        step = log['step']
        log_dict = {
            'val/loss': log.get('val_loss'),
            'val/best_loss': log.get('best_val_loss'),
        }
        if log.get('vanilla_val_loss') is not None:
            log_dict['val/vanilla_loss'] = log['vanilla_val_loss']
        if log.get('ema_val_loss') is not None:
            log_dict['val/ema_loss'] = log['ema_val_loss']
        if log.get('train_loss') is not None:
            log_dict['train/loss_at_eval'] = log['train_loss']
        wandb.log(log_dict, step=step, commit=True)

    # Summary metrics
    for k in ['best_val_loss', 'total_steps', 'total_hours', 'peak_hbm_gb']:
        if summary.get(k) is not None:
            wandb.run.summary[k] = summary[k]

    wandb.finish()
    print(f", {len(eval_logs)} eval points — done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help='Experiment name (e.g. exp13)')
    parser.add_argument('--project', default=None, help='W&B project name (default: {exp}-tpu)')
    parser.add_argument('--entity', default='ka3094-columbia-university', help='W&B entity')
    parser.add_argument('--group', default=None, help='W&B group (default: exp name)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--label', default=None, help='Upload only this label')
    args = parser.parse_args()

    project = args.project or f"{args.exp}-tpu"
    group = args.group or args.exp
    validated_dir = os.path.expanduser(f'~/sf_bema/results/{args.exp}/validated')

    if not os.path.isdir(validated_dir):
        print(f"ERROR: {validated_dir} not found", file=sys.stderr)
        sys.exit(1)

    result_files = sorted(f for f in os.listdir(validated_dir) if f.endswith('.json'))
    if args.label:
        result_files = [f for f in result_files if f.replace('.json', '') == args.label]

    print(f"[upload_wandb] exp={args.exp}, project={project}, entity={args.entity}")
    print(f"[upload_wandb] {len(result_files)} validated results in {validated_dir}")

    if not args.dry_run:
        import wandb
        print(f"[upload_wandb] W&B version: {wandb.__version__}")

    for fname in result_files:
        label = fname.replace('.json', '')
        result_path = os.path.join(validated_dir, fname)
        jsonl_path = os.path.join(validated_dir, f'{label}_train_loss.jsonl')
        upload_run(label, result_path, jsonl_path, project, args.entity, group, args.dry_run)

    print(f"\n[upload_wandb] Done. {len(result_files)} runs.")
    if not args.dry_run:
        print(f"View at: https://wandb.ai/{args.entity}/{project}")


if __name__ == '__main__':
    main()
