#!/usr/bin/env bash
# Final evaluation of one edit spec: 100 dev (variant 0) + 100 held-out (variant 1) problems,
# Danish and English, against the already-measured baseline (results/base_dev, variant 0).
#   scripts/final.sh <gpu> <spec|base> <langs> <variants>
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
GPU="$1"; SPEC="$2"; LANGS="$3"; VARIANTS="$4"
name="$(basename "$SPEC" .json)"
for lang in $LANGS; do
  for v in $VARIANTS; do
    out="results/final_${name}_v${v}"
    mkdir -p "$out"
    if [ -s "$out/$lang.jsonl" ]; then echo "skip $name v$v $lang"; continue; fi
    patch=()
    [ "$SPEC" != "base" ] && patch=(--patch "$SPEC")
    echo "=== $name variant $v $lang on cuda:$GPU ==="
    .venv/bin/python src/eval_gsm.py --langs "$lang" --variant "$v" \
      --device "cuda:$GPU" --batch-size 8 --max-new-tokens 768 --dtype fp16 \
      --out "$out" "${patch[@]}" || echo "FAILED $name v$v $lang"
  done
done
