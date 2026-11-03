"""Offline sweep of the head-row edit: how much of the gold direction should the decision use?

The edit adds α·u to the `no` row of the output head and α·u to the `yes` row with the opposite sign, so
the decision margin becomes

    margin_α = (logp_no + α⟨y, u⟩) − (logp_yes − α⟨y, u⟩) = margin_0 + 2α⟨y, u⟩

which is exactly linear in α, so the whole sweep costs nothing once the states are recorded
(src/accept_probe.py --split train|val). Fit α on the train split, report the held-out val curve.

    python src/bias_sweep.py --dataset scala-en --probe results/accept_probe_scala
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from accept import mcc


def load(path: str) -> dict:
    d = np.load(path)
    return {
        "Y": torch.from_numpy(d["Y"].astype(np.float32)),
        "logp_yes": d["logp_yes"],
        "logp_no": d["logp_no"],
        "gold_no": d["gold_no"],
    }


def score(margin: np.ndarray, gold_no: np.ndarray) -> tuple[float, float, tuple]:
    pred = margin > 0
    tp = int((pred & gold_no).sum())
    fp = int((pred & ~gold_no).sum())
    tn = int((~pred & ~gold_no).sum())
    fn = int((~pred & gold_no).sum())
    return float((pred == gold_no).mean()), mcc(tp, tn, fp, fn), (tp, tn, fp, fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scala-en")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--alphas", default="0,2,5,10,20,40,80")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    u = torch.load(Path(args.probe) / "u_gold.pt", weights_only=False)["u"]
    rows = []
    for split in ("train", "val"):
        d = load(Path(args.probe) / f"{args.dataset}_{split}_states.npz")
        proj = (d["Y"] @ u).numpy()  # ⟨y, u⟩
        base = d["logp_no"] - d["logp_yes"]
        for alpha in [float(a) for a in args.alphas.split(",")]:
            acc, m, cells = score(base + 2 * alpha * proj, d["gold_no"])
            rows.append({"split": split, "alpha": alpha, "acc": acc, "mcc": m, "cells": cells})
            print(f"{split:5s} alpha={alpha:5.1f}  acc={acc:.4f}  MCC={m:+.4f}  tp/tn/fp/fn={cells}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
