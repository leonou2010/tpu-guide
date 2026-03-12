#!/bin/bash
# watch.sh — Auto-refresh monitor dashboard
# Usage: EXP=exp12c bash ~/distributed_tpu_training/watch.sh
EXP=${EXP:?'EXP env var required (e.g. EXP=exp12c)'}
while true; do clear; EXP=$EXP bash ~/distributed_tpu_training/monitor.sh; echo "Next refresh in 5 min... (Ctrl+C to stop)"; sleep 300; done
