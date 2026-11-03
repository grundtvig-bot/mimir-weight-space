"""Failure taxonomy for Mimir's GSM-symbolic responses (no GPU).

Reads results/<run>/<lang>.jsonl and reports, per language:
  acc, direct-answer rate, truncation, missing \\boxed{}, and — where the response shows its work —
  whether the arithmetic in the chain of thought is self-consistent.

The arithmetic check parses expressions like `3 * 20 = 60`, `$12 \\times 4 = 48$`, `\frac{1}{2}`-free
simple forms, and verifies each one. A wrong intermediate value is a *calculation* error; a chain that
is arithmetically clean but ends at the wrong number is a *setup / final-step* error.

Usage: python src/diagnose.py results/base_dev
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EXPR = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×x·÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def check_arithmetic(text: str) -> tuple[int, int]:
    """(n_expressions, n_wrong) over `a op b = c` patterns in the response."""
    n = wrong = 0
    for a, op, b, c in EXPR.findall(text):
        a, b, c = float(a), float(b), float(c)
        op = {"×": "*", "x": "*", "·": "*", "÷": "/"}.get(op, op)
        try:
            got = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("nan")}[op]
        except (KeyError, ZeroDivisionError):
            continue
        n += 1
        if abs(got - c) > 1e-2 * max(1.0, abs(c)):
            wrong += 1
    return n, wrong


def main() -> None:
    run = Path(sys.argv[1])
    for f in sorted(run.glob("*.jsonl")):
        recs = [json.loads(line) for line in open(f)]
        acc = np.mean([r["correct"] for r in recs])
        direct = np.mean([r["direct"] for r in recs])
        trunc = np.mean([not r["finished"] for r in recs])
        nobox = np.mean([not r["boxed"] for r in recs])
        nopred = np.mean([r["pred"] is None for r in recs])
        toks = np.array([r["n_resp_tokens"] for r in recs])
        wrong_calc = []
        for r in recs:
            n, w = check_arithmetic(r["response"])
            if n >= 2:
                wrong_calc.append(w > 0)
        print(f"\n=== {f.stem}: n={len(recs)} ===")
        print(f"acc {acc:.3f} | direct {direct:.3f} | truncated {trunc:.3f} | no-boxed {nobox:.3f} | no-pred {nopred:.3f}")
        print(f"response tokens: mean {toks.mean():.0f} p50 {np.percentile(toks,50):.0f} p90 {np.percentile(toks,90):.0f} max {toks.max()}")
        if wrong_calc:
            print(f"responses with >=2 checkable equations: {len(wrong_calc)}, of which >=1 wrong: "
                  f"{sum(wrong_calc)} ({np.mean(wrong_calc):.2f})")
        buckets = Counter()
        selection_err = 0
        for r in recs:
            n, w = check_arithmetic(r["response"])
            gold_str = f"{r['gold']:g}"
            gold_seen = gold_str in r["response"].replace(",", "")
            if not r["finished"]:
                buckets["truncated"] += 1
            elif r["direct"]:
                buckets["direct answer"] += 1
            elif r["correct"]:
                buckets["correct"] += 1
            elif gold_seen:
                buckets["gold in CoT, not boxed"] += 1
                selection_err += 1
            elif w > 0:
                buckets["wrong arithmetic in CoT"] += 1
            else:
                buckets["clean CoT, wrong number"] += 1
        for k, v in buckets.most_common():
            print(f"  {k:28s} {v:4d}  {v/len(recs):.2f}")
        if selection_err:
            print(f"  -> of the errors, {selection_err} had the right number in the response already "
                  f"({selection_err/max(1,len(recs)-sum(1 for r in recs if r['correct'])):.2f} of all errors)")
        if len(sys.argv) > 2 and sys.argv[2] == "--examples":
            for r in recs:
                if not r["correct"] and not r["direct"] and r["finished"]:
                    n, w = check_arithmetic(r["response"])
                    print(f"\n--- template {r['template']} gold={r['gold']} pred={r['pred']} eqs={n} wrong={w}")
                    print(r["response"][-600:])


if __name__ == "__main__":
    main()
