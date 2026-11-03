"""One figure: every arm plotted as (language drift, accuracy).

Moving left = less drift. The control and the magnitude-selected arms sit on top of the baseline; the
signed ablations move left; none of them moves up.

    python src/figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_run import lang_frac

SCREEN = {  # 40 Danish problems, variant 0
    "base": ("results/screen_base", "baseline"),
    "e1 (mag, α=1)": ("results/screen_e1", "magnitude-selected"),
    "e3 (layer 15)": ("results/screen_e3", "magnitude-selected"),
    "e5 (layer 15, k=32)": ("results/screen_e5", "magnitude-selected"),
    "q4 (k=256, α=−1)": ("results/screen_q4", "signed, α=−1"),
    "q6 (α=−2)": ("results/screen_q6", "signed, α=−1"),
    "q3 (k=64, α=−1)": ("results/screen_q3", "signed, α=−1"),
    "q5 (random dir)": ("results/screen_q5ctrl", "control"),
}
HELDOUT = {"base": "results/final_base_v1", "q3": "results/final_q3_v1"}


def arm(run: str, lang: str = "dan") -> tuple[float, float]:
    recs = [json.loads(line) for line in open(Path(run) / f"{lang}.jsonl")]
    drift = np.mean([lang_frac(r["response"]) < 0.75 for r in recs])
    acc = np.mean([r["correct"] for r in recs])
    return float(drift), float(acc)


def main() -> None:
    colours = {"baseline": "#444444", "magnitude-selected": "#c9a227", "signed, α=−1": "#1f77b4", "control": "#d62728"}
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (run, kind) in SCREEN.items():
        if not (Path(run) / "dan.jsonl").exists():
            continue
        d, a = arm(run)
        ax.scatter(d, a, s=90 if kind == "baseline" else 70, color=colours[kind], zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(label, (d, a), textcoords="offset points", xytext=(7, -3), fontsize=7.5, color="#333")
    for label, run in HELDOUT.items():
        if not (Path(run) / "dan.jsonl").exists():
            continue
        d, a = arm(run)
        ax.scatter(d, a, marker="*", s=320, color="#2ca02c", zorder=4, edgecolor="white", linewidth=0.6)
        ax.annotate(f"held-out {label}", (d, a), textcoords="offset points", xytext=(9, 4), fontsize=8,
                    color="#2ca02c")
    for kind, colour in colours.items():
        ax.scatter([], [], color=colour, label=kind)
    ax.scatter([], [], marker="*", s=200, color="#2ca02c", label="held-out (100 problems)")
    ax.set_xlabel("share of Danish answers that drift out of Danish")
    ax.set_ylabel("GSM-Symbolic accuracy (Danish)")
    ax.set_title("Weight-space programming on DFM Mimir v1.5: the edit moves the behaviour, not the score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = Path("results/figures/arms.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
