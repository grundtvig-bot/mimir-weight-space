#!/usr/bin/env bash
# Block until both EuroEval arms have written results, then print the comparison.
cd "$HOME/githubs/mimir-weight-space"
while true; do
  base=0; q3=0
  [ -s results/euroeval_base/euroeval_benchmark_results.jsonl ] && base=1
  [ -s results/euroeval_q3/euroeval_benchmark_results.jsonl ] && q3=1
  running=$(pgrep -fc "euroeval -m" || true)
  if [ "$base" = 1 ] && [ "$q3" = 1 ] && [ "${running:-0}" -eq 0 ]; then break; fi
  sleep 60
done
.venv/bin/python src/analyze_euroeval.py results/euroeval_base results/euroeval_q3
