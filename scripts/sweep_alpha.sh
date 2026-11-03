#!/usr/bin/env bash
# Sweep the acceptability edit strength on the scala-en test split (disjoint from the reported val subset).
set -uo pipefail
cd "$HOME/githubs/mimir-weight-space"
EVAL="$HOME/githubs/mimir-weight-space/.venv-eval/bin/euroeval"
for a in "$@"; do
  out="results/sweep/${TAG:-a}$a"
  mkdir -p "$out"
  [ -s "$out/euroeval_benchmark_results.jsonl" ] && { echo "skip a$a"; continue; }
  ( cd "$out" && CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH="$HOME/githubs/mimir-weight-space/src/patch_accept" \
      MIMIR_ACC_ALPHA="$a" \
      MIMIR_ACC_DIR="$HOME/githubs/mimir-weight-space/results/accept_probe_scala/${DIR:-u_gold}.pt" \
      "$EVAL" -m danish-foundation-models/DFM-Mimir-v1.5 --dataset scala-en --evaluate-test-split \
        --num-iterations 1 --zero-shot --generative-type instruction_tuned --device cuda \
        --no-progress-bar --force --cache-dir "$HOME/githubs/mimir-weight-space/.euroeval_cache" \
        > out.log 2>&1 )
  python3 - <<EOF
import json
r=json.loads(open("$out/euroeval_benchmark_results.jsonl").readline())["evaluation_results"][0]
print("alpha", "$a", round(r["score_details"]["score"], 2), flush=True)
EOF
done
