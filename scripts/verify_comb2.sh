#!/usr/bin/env bash
# Save the refit combined edit and verify it on the two acceptability datasets.
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/save_edit.py --patch specs/combined_q3_acc_refit.json --out checkpoints/combined_v2
mkdir -p results/verify_comb2
cd results/verify_comb2
CUDA_VISIBLE_DEVICES=0 "$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval" \
  -m "$HOME/githubs/mimir-weight-space/checkpoints/combined_v2" \
  --dataset scala-en --dataset dala --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" \
  > out.log 2>&1
grep -E "Matthew's" out.log
