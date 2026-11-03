"""Locate the direction that writes "answer in English" inside a Danish chain of thought.

Method (no gradients):
  1. Teacher-force the *baseline* Danish answers and record, for every response token, the final
     prediction state y (output of the last H-module application) and, for every H-layer, the MLP
     intermediate activations a (the input of `down_proj`).
  2. Label each response token English or Danish from its own frequency in an English vs a Danish
     corpus (Pile vs Dynaword counts over the 262,144-token Gemma vocabulary). Numbers, maths and
     other language-neutral tokens stay unlabelled.
  3. u = mean(y | English token) - mean(y | Danish token), normalised. Report how well it separates
     the two classes (AUROC), fitted on even-indexed problems and validated on odd-indexed ones.
  4. Score every MLP output column by how much it writes along u:  c_j = a_j * <w_j, u_hat>, and rank
     the columns of every H-layer by that score. Those are the candidate edit sites.

Usage: python src/langdir.py --lang dan --device cuda:0 --variant 0 --out results/langdir
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from eval_gsm import build_prompt, load_questions  # noqa: E402
from multilingual_gsm_symbolic import load_instruction  # noqa: E402

EN_COUNTS = Path(__file__).parent.parent / "results/unigram_pile.npy"
DA_COUNTS = Path(__file__).parent.parent / "results/unigram_dynaword.npy"


def token_labels(tok) -> torch.Tensor:
    """+1 = English-leaning token, -1 = Danish-leaning, 0 = language-neutral."""
    en = torch.from_numpy(np.load(EN_COUNTS)).double()
    da = torch.from_numpy(np.load(DA_COUNTS)).double()
    p_en = (en + 1) / (en.sum() + len(en))
    p_da = (da + 1) / (da.sum() + len(da))
    alpha = torch.tensor(
        [s.replace("▁", "").isalpha() and len(s.strip()) > 1 for s in tok.convert_ids_to_tokens(range(len(en)))],
        dtype=torch.bool,
    )
    lab = torch.zeros(len(en), dtype=torch.int8)
    strong = alpha & (en + da >= 50)
    lab[strong & (p_en > 10 * p_da)] = 1
    lab[strong & (p_da > 10 * p_en)] = -1
    return lab


class LastHRecorder:
    """y = output of the last H-module application; a[i] = down_proj input of H layer i, same position."""

    def __init__(self, model):
        self.n_h = model.config.H_cycles
        self.handles = [model.model.H_module.register_forward_hook(self._on_stack)]
        for i, layer in enumerate(model.model.H_module.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._down_hook(i)))
        self.start()

    def start(self):
        self.stack_calls = 0
        self.layer_calls = {i: 0 for i in range(len(self.handles) - 1)}
        self.y = None
        self.a = {}

    def _on_stack(self, _mod, _inp, out):
        self.stack_calls += 1
        if self.stack_calls == self.n_h:
            self.y = out.detach().float()

    def _down_hook(self, i):
        def hook(_mod, inp, _out):
            self.layer_calls[i] += 1
            if self.layer_calls[i] == self.n_h:
                self.a[i] = inp[0].detach().float()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def record(model, rec, prompts, responses, device, max_len=1024):
    out = []
    for prompt, resp in zip(prompts, responses):
        resp = resp[: max_len - len(prompt)]
        ids = torch.tensor([prompt + resp], device=device)
        tt = torch.zeros_like(ids)
        tt[0, : len(prompt)] = 1
        rec.start()
        model(input_ids=ids, token_type_ids=tt, use_cache=False)
        sl = slice(len(prompt) - 1, len(prompt) + len(resp) - 1)  # state at t predicts resp[t]
        out.append({"y": rec.y[0, sl].cpu(), "a": {i: v[0, sl].cpu() for i, v in rec.a.items()}, "resp": resp})
        print(f"  recorded {len(out)}/{len(prompts)}", flush=True)
    return out


def auroc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """P(score_pos > score_neg), ties counted 0.5, computed from ranks with tie averaging."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    s = torch.cat([scores_pos, scores_neg])
    uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    order = s.argsort()
    ranks = torch.empty(len(s), dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    sums = torch.zeros(len(uniq)).index_add_(0, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(scores_pos)].sum()
    return float((r_pos - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danish-foundation-models/DFM-Mimir-v1.5")
    ap.add_argument("--lang", default="dan")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--max-templates", type=int, default=None)
    ap.add_argument("--baseline", default="results/base_dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype={"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        )
        .to(args.device)
        .eval()
    )
    model.requires_grad_(False)

    base = {json.loads(l)["template"]: json.loads(l) for l in open(f"{args.baseline}/{args.lang}.jsonl")}
    qs = [q for q in load_questions(args.lang, 1, args.max_templates, "v0.5.0", args.variant) if q["template"] in base]
    prompts = [build_prompt(tok, load_instruction(args.lang), q["question"]) for q in qs]
    responses = [tok(base[q["template"]]["response"], add_special_tokens=False)["input_ids"] for q in qs]
    print(f"{len(qs)} problems, {sum(map(len, responses))} response tokens", flush=True)

    rec = LastHRecorder(model)
    data = record(model, rec, prompts, responses, args.device)
    rec.remove()

    lab = token_labels(tok)
    labels = [lab[torch.tensor(d["resp"])] for d in data]
    print(f"labelled tokens: EN {int(sum((l == 1).sum() for l in labels))} "
          f"DA {int(sum((l == -1).sum() for l in labels))} of {sum(len(l) for l in labels)}", flush=True)

    # ---- 1. the direction: fit on even problems, validate on odd ones
    def pool(ids: list[int]):
        ys = torch.cat([data[i]["y"][labels[i] != 0] for i in ids])
        ls = torch.cat([labels[i][labels[i] != 0] for i in ids])
        return ys, ls

    ev, lv = pool(list(range(0, len(data), 2)))
    tv, lt = pool(list(range(1, len(data), 2)))
    delta = ev[lv == 1].mean(0) - ev[lv == -1].mean(0)
    u = delta / delta.norm()
    auc_train = auroc(ev[lv == 1] @ u, ev[lv == -1] @ u)
    auc_val = auroc(tv[lt == 1] @ u, tv[lt == -1] @ u)
    print(f"|mean_en - mean_da| = {delta.norm():.2f}   AUROC train {auc_train:.3f} held-out {auc_val:.3f}", flush=True)
    torch.save({"u": u, "auroc_train": auc_train, "auroc_heldout": auc_val, "lang": args.lang}, out / "u.pt")

    # ---- 2. rank MLP output columns by how much they write along u
    rows = []
    scores = {}
    for layer_idx in sorted(data[0]["a"]):
        W = model.model.H_module.layers[layer_idx].mlp.down_proj.weight.detach().float().cpu()  # [d, inter]
        proj = W.T @ u.cpu()  # [inter]
        pos = torch.cat([data[i]["a"][layer_idx][labels[i] == 1] for i in range(len(data))])
        neg = torch.cat([data[i]["a"][layer_idx][labels[i] == -1] for i in range(len(data))])
        signed = (pos * proj).mean(0) - (neg * proj).mean(0)
        scores[layer_idx] = signed.numpy()
        d = signed.abs()
        top = torch.topk(d, 256)
        rows.append(
            {
                "layer": layer_idx,
                "mean_abs_contrib": float(d.mean()),
                "top_columns": top.indices.tolist(),
                "top_scores": [float(x) for x in top.values],
                "peak_ratio": float(top.values[0] / (d.mean() + 1e-12)),
            }
        )
        print(f"layer {layer_idx:2d}: max|dc| {top.values[0]:.4f} mean {d.mean():.5f} "
              f"ratio {rows[-1]['peak_ratio']:.1f}", flush=True)
    np.savez_compressed(out / "column_scores.npz", **{f"layer{k}": v for k, v in scores.items()})  # signed
    rows.sort(key=lambda r: -r["mean_abs_contrib"])
    json.dump({"lang": args.lang, "auroc_train": auc_train, "auroc_heldout": auc_val, "layers": rows},
              open(out / "columns.json", "w"), indent=1)
    print("H layers ranked by mean |contribution along u|:", [r["layer"] for r in rows], flush=True)


if __name__ == "__main__":
    main()
