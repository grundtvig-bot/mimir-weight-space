#!/usr/bin/env bash
# Screen several edit specs on a fixed Danish dev subset, one arm after another on one GPU.
#   scripts/screen.sh <gpu> <templates> <arm> [<arm> ...]
# where an arm is either "base" or "specs/<name>.json".
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
GPU="$1"; N="$2"; shift 2
for arm in "$@"; do
  name="$(basename "$arm" .json)"
  out="results/screen_$name"
  mkdir -p "$out"
  if [ -s "$out/dan.jsonl" ]; then echo "skip $name (done)"; continue; fi
  patch=()
  [ "$arm" != "base" ] && patch=(--patch "$arm")
  echo "=== $name on cuda:$GPU ($N templates) ==="
  .venv/bin/python src/eval_gsm.py --langs dan --variant 0 --max-templates "$N" \
    --device "cuda:$GPU" --batch-size 12 --max-new-tokens 768 --dtype fp16 \
    --out "$out" "${patch[@]}" || echo "FAILED $name"
done
