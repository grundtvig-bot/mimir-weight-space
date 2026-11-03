"""Summarise a GSM-Symbolic run: accuracy, language drift, and where the errors are.

Usage: python src/analyze_run.py results/base_dev [results/e1_dev ...]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

DA = set(
    "og er det som på ikke til af for med den der har kan skal hvor mange meget hvis hun han vi jeg så "
    "også fra ved bliver være trin beregn samlet først derefter total antal hvor".split()
)
EN = set(
    "the and is of to in that it for with are this be by as at from we you they not have has can will how "
    "many much step calculate total first then answer each number".split()
)


def lang_frac(text: str) -> float:
    toks = re.findall(r"[a-zæøåA-ZÆØÅ]+", text.lower())
    d = sum(t in DA for t in toks)
    e = sum(t in EN for t in toks)
    return d / (d + e) if d + e else 0.5


def main() -> None:
    for run in map(Path, sys.argv[1:]):
        for f in sorted(run.glob("*.jsonl")):
            recs = [json.loads(line) for line in open(f)]
            lang = recs[0]["lang"]
            frac = np.array([lang_frac(r["response"]) for r in recs])
            prompt_is_da = lang == "dan"
            if prompt_is_da:
                drift = frac < 0.75  # drifted away from Danish
            else:
                drift = frac > 0.25  # drifted away from English
            acc = np.array([r["correct"] for r in recs])
            toks = np.array([r["n_resp_tokens"] for r in recs])
            print(
                f"{run.name}/{f.stem:4s} n={len(recs):3d} acc={acc.mean():.3f} "
                f"drift={drift.mean():.2f} acc|stay={acc[~drift].mean() if (~drift).any() else float('nan'):.3f} "
                f"acc|drift={acc[drift].mean() if drift.any() else float('nan'):.3f} "
                f"toks={toks.mean():.0f}"
            )


if __name__ == "__main__":
    main()
