"""Weight-space editing: declarative edit specs applied to a loaded Mimir checkpoint.

An edit spec is a JSON file:

    {"name": "...", "note": "...",
     "edits": [{"param": "model.H_module.layers.7.mlp.down_proj.weight",
                "kind": "project_out" | "scale" | "add",
                "columns": null | [ints],
                "direction": "results/prior_mix.pt:d_prior",   # for project_out / add
                "alpha": 0.5}]}

`project_out` removes the component of each selected column along `direction`:
    w_j <- w_j - alpha * <w_j, d_hat> * d_hat
i.e. those neurons stop writing the direction into the residual stream. `scale` multiplies the
columns by (1 + alpha). `add` adds alpha * beta * d_hat to the columns.

Everything is applied in float32 and cast back to the parameter's dtype.
"""

import json
from pathlib import Path

import torch


def load_direction(spec: str) -> torch.Tensor:
    """`path/to/file.pt:key` -> the vector stored under that key (flat, float32)."""
    path, _, key = spec.partition(":")
    obj = torch.load(path, weights_only=False)
    vec = obj[key] if key else obj
    return vec.detach().float().flatten()


def apply_edits(model, edits: list[dict]) -> dict:
    """Apply edits in place; returns {param: n_changed_weights}."""
    changed: dict[str, int] = {}
    for e in edits:
        param = e["param"]
        mod = model.get_parameter(param)
        kind = e["kind"]
        cols = e.get("columns")
        alpha = float(e.get("alpha", 1.0))
        with torch.no_grad():
            w = mod.detach().float()
            sel = slice(None) if cols is None else torch.tensor(cols, dtype=torch.long, device=w.device)
            n_cols = w.shape[1] if cols is None else len(cols)
            if kind == "project_out":
                d = load_direction(e["direction"]).to(w.device)
                d = d / d.norm()
                sub = w[:, sel]  # [out, n_sel]
                coef = sub.T @ d  # [n_sel]
                sub = sub - alpha * coef.unsqueeze(0) * d.unsqueeze(1)
                w[:, sel] = sub
                changed[param] = n_cols * w.shape[0]
            elif kind == "scale":
                w[:, sel] = w[:, sel] * (1.0 + alpha)
                changed[param] = n_cols * w.shape[0]
            elif kind == "add":
                d = load_direction(e["direction"]).to(w.device)
                d = d / d.norm()
                w[:, sel] = w[:, sel] + alpha * d.unsqueeze(1)
                changed[param] = n_cols * w.shape[0]
            else:
                raise ValueError(f"unknown edit kind {kind!r}")
            mod.copy_(w.to(mod.dtype))
    return changed


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
