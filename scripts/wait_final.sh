#!/usr/bin/env bash
# Block until the final evaluation runs finish, then print every arm's numbers.
cd "$HOME/githubs/mimir-weight-space"
while systemctl --user is-active --quiet ws-final-0 || systemctl --user is-active --quiet ws-final-1; do sleep 30; done
for d in results/base_dev results/final_base_v1 results/final_q3_v0 results/final_q3_v1; do
  [ -d "$d" ] && .venv/bin/python src/analyze_run.py "$d" 2>/dev/null
done
