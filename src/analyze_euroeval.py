"""Compare EuroEval runs of Mimir with and without the prior-steering intervention.

Usage: python src/analyze_euroeval.py results/euroeval/base results/euroeval/m025 [results/euroeval/m05 ...]

Reads `euroeval_benchmark_results.jsonl` from each run directory (one line per dataset) and uses the
dataset's primary metric (the first metric EuroEval reports for it).
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(run_dir: str) -> dict[str, dict]:
    path = Path(run_dir) / "euroeval_benchmark_results.jsonl"
    out = {}
    for line in open(path):
        rec = json.loads(line)
        res = rec["evaluation_results"][0]  # primary metric
        details = rec["eval_library"]["additional_details"]
        ds = res["source_data"]["dataset_name"]
        out[ds] = {
            "metric": res["evaluation_name"],
            "score": res["score_details"]["score"],
            "task": details.get("task", "?"),
            "lang": details.get("languages", "?").strip('["]'),
            "failed": float(res["score_details"]["details"].get("num_failed_instances", 0) or 0),
        }
    return out


def main(base_dir: str, *other_dirs: str) -> None:
    base = load(base_dir)
    runs = {Path(d).name: load(d) for d in other_dirs}
    header = f"| dataset | task | base | " + " | ".join(runs) + " | " + " | ".join(f"Δ {r}" for r in runs) + " |"
    print(header)
    print("|---" * (3 + 2 * len(runs)) + "|")
    for ds in sorted(base, key=lambda d: (base[d]["task"], d)):
        cells = [f"{base[ds]['score']:.2f}"]
        deltas = []
        for name, run in runs.items():
            if ds in run:
                cells.append(f"{run[ds]['score']:.2f}")
                deltas.append(f"{run[ds]['score'] - base[ds]['score']:+.2f}")
            else:
                cells.append("–")
                deltas.append("–")
        print(f"| {ds} | {base[ds]['task']} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")

    print()
    for name, run in runs.items():
        common = [d for d in base if d in run]
        delta = np.array([run[d]["score"] - base[d]["score"] for d in common])
        tasks: dict[str, list[float]] = {}
        for d in common:
            tasks.setdefault(base[d]["task"], []).append(run[d]["score"] - base[d]["score"])
        print(f"{name}: n={len(common)} datasets, mean Δ={delta.mean():+.2f}, "
              f"wins/ties/losses={int((delta > 0.05).sum())}/{int((abs(delta) <= 0.05).sum())}/{int((delta < -0.05).sum())}, "
              f"mean base={np.mean([base[d]['score'] for d in common]):.2f}, "
              f"mean steered={np.mean([run[d]['score'] for d in common]):.2f}")
        print("   by task: " + ", ".join(f"{t} {np.mean(v):+.2f} (n={len(v)})" for t, v in sorted(tasks.items())))


if __name__ == "__main__":
    main(*sys.argv[1:])
