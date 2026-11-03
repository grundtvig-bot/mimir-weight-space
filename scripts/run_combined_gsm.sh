#!/usr/bin/env bash
# GSM-Symbolic held-out arm for the combined checkpoint (variant 1, 100 problems per language).
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
.venv/bin/python src/eval_gsm.py --model "$HOME/githubs/mimir-weight-space/checkpoints/combined_q3_acc" \
  --langs dan eng --variant 1 --device cuda:1 --batch-size 8 --max-new-tokens 768 --dtype fp16 \
  --out results/final_combined_v1
