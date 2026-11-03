#!/usr/bin/env bash
# Save the chosen acceptability edit and check that the saved checkpoint reproduces the patched run.
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/save_edit.py --patch specs/acc_en_a05.json --out checkpoints/acc_en_a05
mkdir -p results/verify_a05
cd results/verify_a05
CUDA_VISIBLE_DEVICES=0 "$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval" \
  -m "$HOME/githubs/mimir-weight-space/checkpoints/acc_en_a05" \
  --dataset scala-en --dataset dala --num-iterations 1 --zero-shot \
  --generative-type instruction_tuned --device cuda --no-progress-bar --force \
  --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" > out.log 2>&1
grep -E "Matthew's" out.log
