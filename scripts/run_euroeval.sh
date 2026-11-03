#!/usr/bin/env bash
# EuroEval (Danish + English) on a Mimir checkpoint — baseline or edited.
#   scripts/run_euroeval.sh <model-path-or-id> <outdir> [extra euroeval args...]
# The vLLM backend is used (the same one the earlier Mimir runs used); note that vLLM's hrm_text
# implementation ignores the PrefixLM mask, so levels are not comparable to the model card — the
# baseline/edit contrast is, because both arms run through this exact command.
set -euo pipefail
MODEL="${1:?model}"; OUT="${2:?outdir}"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_BIN="$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval"
mkdir -p "$OUT"
cd "$OUT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
"$EVAL_BIN" \
  -m "$MODEL" \
  -l da -l en \
  --num-iterations 1 --zero-shot --generative-type instruction_tuned \
  --device cuda --no-progress-bar --force \
  --cache-dir "$REPO/.euroeval_cache" "$@"
