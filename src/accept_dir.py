"""Locate the "say yes" vs "say no" decision direction for linguistic acceptability.

Same machinery as src/langdir.py, applied to a different behaviour: for every item we record the final
prediction state y at the position that produces the answer, label it by the label the model actually
emitted, and take

    u = mean(y | answered "no") − mean(y | answered "yes")

then rank the H-module MLP output columns by their signed contribution along u.

    python src/accept_dir.py --dataset scala-en --device cuda:0 --out results/accept_dir_scala
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from accept import LABELS, PREFIX, PROMPT, load_items  # noqa: E402
from langdir import LastHRecorder, auroc  # noqa: E402

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"


@torch.no_grad()
def record(tok, model, rec, prompts, device, batch_size=8):
    """y at the last prompt position, for each prompt."""
    ys, acts = [], {}
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        rec.start()
        model(input_ids=ids, attention_mask=att, token_type_ids=att.clone(), use_cache=False)
        # last non-pad position per row
        idx = att.sum(1) - 1
        ys.append(rec.y[torch.arange(len(chunk)), idx].cpu())
        for layer_idx, a in rec.a.items():
            acts.setdefault(layer_idx, []).append(a[torch.arange(len(chunk)), idx].cpu())
        print(f"  {min(b+batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(ys), {k: torch.cat(v) for k, v in acts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scala-en")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--said", default="results/accept_plm", help="run whose answers label the states")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang = args.lang or ("en" if args.dataset.endswith("-en") else "da")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    model.requires_grad_(False)

    said = [json.loads(l) for l in open(Path(args.said) / f"{args.dataset}.jsonl")]
    items = load_items(args.dataset, args.split)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": PREFIX[lang] + PROMPT[lang].format(text=it["text"])}],
                                tokenize=False, add_generation_prompt=True)
        for it in items
    ]
    rec = LastHRecorder(model)
    Y, A = record(tok, model, rec, prompts, args.device)
    rec.remove()

    yes, no = LABELS[lang]
    assert len(said) == len(items), (len(said), len(items))
    is_no = torch.tensor([no in (r["raw"] or "") and yes not in (r["raw"] or "") for r in said])
    print(f"answered no: {int(is_no.sum())} / {len(is_no)}", flush=True)

    half = len(Y) // 2
    def pool(idx):
        return Y[idx], is_no[idx]
    fit_Y, fit_l = pool(slice(0, half))
    val_Y, val_l = pool(slice(half, None))
    delta = fit_Y[fit_l].mean(0) - fit_Y[~fit_l].mean(0)
    u = delta / delta.norm()
    auc_fit = auroc(fit_Y[fit_l] @ u, fit_Y[~fit_l] @ u)
    auc_val = auroc(val_Y[val_l] @ u, val_Y[~val_l] @ u)
    print(f"AUROC(answered) fit {auc_fit:.3f} held-out {auc_val:.3f} | |delta| {delta.norm():.2f}", flush=True)

    # does the state encode the *gold* judgement (rather than the model's own answer)?
    gold_no = torch.tensor([it["label"] != "correct" for it in items])
    g_fit = fit_Y[gold_no[0:half]].mean(0) - fit_Y[~gold_no[0:half]].mean(0)
    g = g_fit / g_fit.norm()
    g_auc_fit = auroc(fit_Y[gold_no[0:half]] @ g, fit_Y[~gold_no[0:half]] @ g)
    g_auc_val = auroc(val_Y[gold_no[half:]] @ g, val_Y[~gold_no[half:]] @ g)
    agree = (is_no == gold_no).float().mean().item()
    print(f"AUROC(gold) fit {g_auc_fit:.3f} held-out {g_auc_val:.3f} | model agrees with gold on {agree:.3f}",
          flush=True)
    torch.save({"u": u, "auroc_fit": auc_fit, "auroc_heldout": auc_val}, out / "u_no.pt")

    rows, scores = [], {}
    for layer_idx in sorted(A):
        W = model.model.H_module.layers[layer_idx].mlp.down_proj.weight.detach().float().cpu()
        proj = W.T @ u.cpu()                      # [inter] how much each neuron writes along u
        c = A[layer_idx] * proj.unsqueeze(0)      # [n, inter] contribution to +/- u
        signed = (c[is_no].mean(0) - c[~is_no].mean(0))
        scores[layer_idx] = signed.numpy()
        top = torch.topk(signed.abs(), 64)
        rows.append({"layer": layer_idx, "mean_abs": float(signed.abs().mean()),
                     "top_columns": top.indices.tolist(), "top_scores": [float(x) for x in top.values]})
        print(f"layer {layer_idx:2d}: max|dc| {top.values[0]:.4f} mean {signed.abs().mean():.5f}", flush=True)
    np.savez_compressed(out / "column_scores.npz", **{f"layer{k}": v for k, v in scores.items()})
    rows.sort(key=lambda r: -r["mean_abs"])
    json.dump({"dataset": args.dataset, "auroc_fit": auc_fit, "auroc_heldout": auc_val, "layers": rows},
              open(out / "columns.json", "w"), indent=1)
    print("top layers by mean |contribution|:", [r["layer"] for r in rows[:6]], flush=True)


if __name__ == "__main__":
    main()
