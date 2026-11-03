#!/usr/bin/env bash
# Both edits at once: save the checkpoint, then run the full Danish + English EuroEval suite.
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/save_edit.py --patch specs/combined_q3_acc.json --out checkpoints/combined_q3_acc
mkdir -p results/euroeval_combined
cd results/euroeval_combined
CUDA_VISIBLE_DEVICES=0 "$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval" \
  -m "$HOME/githubs/mimir-weight-space/checkpoints/combined_q3_acc" \
  -l da -l en --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force \
  --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" > out.log 2>&1
grep -c "benchmarks)" out.log
