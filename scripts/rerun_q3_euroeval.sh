#!/usr/bin/env bash
# Re-run the q3 EuroEval arm with a checkpoint saved by the fixed saver (the earlier one had
# num_hidden_layers=128 in its config, which costs a few points on its own).
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/save_edit.py --patch specs/q3.json --out checkpoints/q3_fixed
mkdir -p results/euroeval_q3_fixed
cd results/euroeval_q3_fixed
CUDA_VISIBLE_DEVICES=0 "$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval" \
  -m "$HOME/githubs/mimir-weight-space/checkpoints/q3_fixed" \
  -l da -l en --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force \
  --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" > out.log 2>&1
grep -c "benchmarks)" out.log
