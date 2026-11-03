"""Build a weight-edit spec from the localisation in results/langdir/columns.json.

    python src/make_edit.py --name e1 --layers 15 10 12 --k 8 --alpha 1.0 --out specs/e1.json
    python src/make_edit.py --name ctrl --layers 15 10 12 --k 8 --alpha 1.0 --random --out specs/ctrl.json

`--random` replaces the language direction with a random unit vector of the same norm (control).
"""

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--langdir", default="results/langdir")
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--k", type=int, default=8, help="columns per layer")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sign", default="abs", choices=["abs", "positive", "negative"])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--random-columns", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = json.load(open(Path(args.langdir) / "columns.json"))
    by_layer = {r["layer"]: r for r in cols["layers"]}
    scores_npz = Path(args.langdir) / "column_scores.npz"
    if scores_npz.exists():  # full per-column scores, so any k is available
        import numpy as np

        scores = np.load(scores_npz)
        for layer in by_layer:
            s = scores[f"layer{layer}"]
            key = {"abs": np.abs(s), "positive": s, "negative": -s}[args.sign]
            order = np.argsort(-key)
            by_layer[layer] = {"top_columns": order.tolist(), "score": s}
    u = torch.load(Path(args.langdir) / "u.pt", weights_only=False)["u"]
    if args.random:
        direction_file = Path(args.langdir) / "u_random.pt"
        torch.save({"u": torch.randn(len(u)) / torch.randn(len(u)).norm() * u.norm()}, direction_file)
        note = "control: random unit direction with the same norm as u"
    else:
        direction_file = Path(args.langdir) / "u.pt"
        note = "language direction u (mean English - mean Danish final state)"

    gen = torch.Generator().manual_seed(0)
    edits = []
    for layer in args.layers:
        top = by_layer[layer]["top_columns"][: args.k]
        if args.random_columns:
            top = torch.randperm(1536, generator=gen)[: args.k].tolist()
        edits.append(
            {
                "param": f"model.H_module.layers.{layer}.mlp.down_proj.weight",
                "kind": "project_out",
                "columns": top,
                "direction": f"{direction_file}:u",
                "alpha": args.alpha,
            }
        )
    spec = {
        "name": args.name,
        "note": args.note or note,
        "lang": cols["lang"],
        "auroc_heldout": cols["auroc_heldout"],
        "random_direction": args.random,
        "random_columns": args.random_columns,
        "k": args.k,
        "alpha": args.alpha,
        "edits": edits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1))
    n = sum(len(e["columns"]) for e in edits) * 1536
    print(f"{out}: {len(edits)} layers x {args.k} columns = {n:,} weights "
          f"({100 * n / 1_786_775_040:.4f}% of 1.787B)")


if __name__ == "__main__":
    main()
