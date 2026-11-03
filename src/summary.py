"""Aggregate the final GSM-Symbolic arms into one table (results/final_summary.json + markdown).

    python src/summary.py
"""

import json
import math
from pathlib import Path

from analyze_run import lang_frac

# held-out arms only: variant 1 of every template, never used to fit u or choose the dose
ARMS = {
    "base": ["results/final_base_v1"],
    "q3": ["results/final_q3_v1"],
}


def binom_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(runs: list[str], lang: str) -> dict[str, dict]:
    out = {}
    for run in runs:
        p = Path(run) / f"{lang}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            out.setdefault(r["template"], r)  # first run wins (variant 0 before variant 1)
    return out


def stats(recs: dict[str, dict], lang: str) -> dict:
    rs = list(recs.values())
    frac_drift = 0.75 if lang == "dan" else 0.25
    drift = [lang_frac(r["response"]) < 0.75 if lang == "dan" else lang_frac(r["response"]) > 0.25 for r in rs]
    acc = sum(r["correct"] for r in rs) / len(rs)
    return {
        "n": len(rs),
        "acc": acc,
        "drift": sum(drift) / len(rs),
        "acc_stay": sum(r["correct"] for r, d in zip(rs, drift) if not d) / max(1, sum(not d for d in drift)),
        "n_drift": sum(drift),
        "acc_drift": (sum(r["correct"] for r, d in zip(rs, drift) if d) / sum(drift)) if any(drift) else None,
        "direct": sum(r["direct"] for r in rs) / len(rs),
        "trunc": sum(not r["finished"] for r in rs) / len(rs),
        "toks": sum(r["n_resp_tokens"] for r in rs) / len(rs),
        "_recs": recs,
        "_drift": drift,
        "_frac": frac_drift,
    }


def main() -> None:
    table = {}
    for lang in ("dan", "eng"):
        base = load(ARMS["base"], lang)
        edited = load(ARMS["q3"], lang)
        keys = sorted(set(base) & set(edited), key=int)
        b = stats({k: base[k] for k in keys}, lang)
        e = stats({k: edited[k] for k in keys}, lang)
        wins = sum(1 for k in keys if edited[k]["correct"] and not base[k]["correct"])
        losses = sum(1 for k in keys if base[k]["correct"] and not edited[k]["correct"])
        res = {
            "n": len(keys),
            "base": {k: v for k, v in b.items() if not k.startswith("_")},
            "edited": {k: v for k, v in e.items() if not k.startswith("_")},
            "delta_acc": e["acc"] - b["acc"],
            "wins": wins,
            "losses": losses,
            "mcnemar_p": binom_p(min(wins, losses), wins + losses),
            "delta_drift": e["drift"] - b["drift"],
            "rescued_drift_cases": sum(
                1 for k, db, de in zip(keys, b["_drift"], e["_drift"]) if db and not de
            ),
            "new_drift_cases": sum(1 for k, db, de in zip(keys, b["_drift"], e["_drift"]) if de and not db),
        }
        table[lang] = res
        print(
            f"{lang}: n={res['n']} acc {b['acc']:.3f} -> {e['acc']:.3f} ({res['delta_acc']:+.3f}) "
            f"wins {wins} losses {losses} p={res['mcnemar_p']:.3f} | drift {b['drift']:.2f} -> {e['drift']:.2f} "
            f"(rescued {res['rescued_drift_cases']}, new {res['new_drift_cases']}) | "
            f"direct {b['direct']:.2f} -> {e['direct']:.2f}"
        )
    Path("results/final_summary.json").write_text(json.dumps(table, indent=1))

    md = ["| lang | n | base acc | edited acc | Δ | wins/losses | McNemar p | drift base→edited |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for lang, r in table.items():
        md.append(
            f"| {lang} | {r['n']} | {r['base']['acc']:.3f} | {r['edited']['acc']:.3f} | {r['delta_acc']:+.3f} | "
            f"{r['wins']}/{r['losses']} | {r['mcnemar_p']:.3f} | {r['base']['drift']:.2f} → {r['edited']['drift']:.2f} |"
        )
    Path("results/final_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
