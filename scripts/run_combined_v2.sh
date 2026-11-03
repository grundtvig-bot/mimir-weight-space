#!/usr/bin/env bash
# Full Danish + English EuroEval suite for the refit combined checkpoint.
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
mkdir -p results/euroeval_combined_v2
cd results/euroeval_combined_v2
CUDA_VISIBLE_DEVICES=0 "$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval" \
  -m "$HOME/githubs/mimir-weight-space/checkpoints/combined_v2" \
  -l da -l en --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force \
  --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" > out.log 2>&1
grep -c "benchmarks)" out.log
