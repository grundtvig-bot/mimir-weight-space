"""Paired comparison of two GSM-Symbolic runs on the same problems (McNemar, exact-ish binomial).

    python src/pair_test.py results/base_dev results/screen_e1

Reports accuracy for both arms, wins/losses, the two-sided binomial p-value on discordant pairs,
and the drift-rate change (language of the chain of thought).
"""

import json
import math
import sys
from pathlib import Path

from analyze_run import lang_frac


def binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial test against p=0.5."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    a_dir, b_dir = map(Path, sys.argv[1:3])
    for f in sorted(a_dir.glob("*.jsonl")):
        if not (b_dir / f.name).exists():
            continue
        a = {r["template"]: r for r in map(json.loads, open(f))}
        b = {r["template"]: r for r in map(json.loads, open(b_dir / f.name))}
        keys = sorted(set(a) & set(b))
        ca = [a[k]["correct"] for k in keys]
        cb = [b[k]["correct"] for k in keys]
        wins = sum(1 for x, y in zip(ca, cb) if y and not x)
        losses = sum(1 for x, y in zip(ca, cb) if x and not y)
        acc_a, acc_b = sum(ca) / len(ca), sum(cb) / len(cb)
        is_drift = (lambda t: lang_frac(t) < 0.75) if f.stem == "dan" else (lambda t: lang_frac(t) > 0.25)
        da = sum(is_drift(a[k]["response"]) for k in keys) / len(keys)
        db = sum(is_drift(b[k]["response"]) for k in keys) / len(keys)
        print(
            f"{f.stem}: n={len(keys)}  {a_dir.name} {acc_a:.3f} -> {b_dir.name} {acc_b:.3f} "
            f"({acc_b - acc_a:+.3f})  wins {wins} losses {losses}  p={binom_p(min(wins, losses), wins + losses):.4f}  "
            f"drift {da:.2f} -> {db:.2f}"
        )


if __name__ == "__main__":
    main()
