"""Mechanism check: does the edit reduce the incentive to write English inside a Danish answer?

Teacher-forces the *baseline* answers again under the base model and under an edited checkpoint and
compares, at the positions where the baseline answer wrote an English word:
  * the probability the model gives to that same token,
  * the rank of that token,
  * the share of the top-50 probability mass sitting on English-leaning vs Danish-leaning tokens.

Same measurement on English prompts (where the answer is supposed to be English) is the control: an
edit that simply damages English shows up there.

    python src/mechanism.py --patch specs/e1.json --lang dan --device cuda:0
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
from langdir import token_labels  # noqa: E402
from multilingual_gsm_symbolic import load_instruction  # noqa: E402
from weights import apply_edits, load_spec  # noqa: E402


@torch.no_grad()
def measure(tok, model, prompts, responses, labels, device, max_len=1024):
    """At every response position, keep the stats needed for the comparison."""
    out = []
    for prompt, resp in zip(prompts, responses):
        resp = resp[: max_len - len(prompt)]
        ids = torch.tensor([prompt + resp], device=device)
        tt = torch.zeros_like(ids)
        tt[0, : len(prompt)] = 1
        logits = model(input_ids=ids, token_type_ids=tt, use_cache=False).logits[0].float()
        # row t predicts resp[t+1]; align to the token being predicted
        logits = logits[len(prompt) - 1 : len(prompt) - 1 + len(resp) - 1]
        targets = torch.tensor(resp[1:], device=device)
        logp = torch.log_softmax(logits, -1)
        p = logp.gather(1, targets[:, None]).squeeze(1).exp()
        rank = (logits > logits.gather(1, targets[:, None])).sum(1) + 1
        top = torch.topk(logits, 50).indices
        lab = labels[top.cpu()]
        mass_en = (logp.gather(1, top)[lab == 1].sum(1) if (lab == 1).any() else torch.zeros(len(top), device=device)).cpu()
        mass_da = (logp.gather(1, top)[lab == -1].sum(1) if (lab == -1).any() else torch.zeros(len(top), device=device)).cpu()
        out.append(
            {
                "lab": labels[torch.tensor(resp[1:])].numpy(),
                "p": p.cpu().numpy(),
                "rank": rank.cpu().numpy(),
                "mass_en": mass_en.numpy(),
                "mass_da": mass_da.numpy(),
            }
        )
    return out


def summarise(data, name: str) -> dict:
    lab = np.concatenate([d["lab"] for d in data])
    p = np.concatenate([d["p"] for d in data])
    rank = np.concatenate([d["rank"] for d in data])
    men = np.concatenate([d["mass_en"] for d in data])
    mda = np.concatenate([d["mass_da"] for d in data])
    res = {}
    for k, sel in (("en", lab == 1), ("da", lab == -1)):
        if sel.sum() == 0:
            continue
        res[k] = {
            "n": int(sel.sum()),
            "p_target": float(p[sel].mean()),
            "rank_median": float(np.median(rank[sel])),
            "top50_rank1": float((rank[sel] == 1).mean()),
            "mass_en_top50": float(men[sel].mean()),
            "mass_da_top50": float(mda[sel].mean()),
        }
    print(f"{name}: " + " | ".join(
        f"{k}: p {v['p_target']:.3f} rank#{v['rank_median']:.0f} top1 {v['top50_rank1']:.2f} "
        f"m_en {v['mass_en_top50']:.2f} m_da {v['mass_da_top50']:.2f}" for k, v in res.items()
    ), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danish-foundation-models/DFM-Mimir-v1.5")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--langs", nargs="+", default=["dan", "eng"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--baseline", default="results/base_dev")
    ap.add_argument("--out", default="results/mechanism")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    labels = token_labels(tok)
    result = {}

    for lang in args.langs:
        base = {json.loads(l)["template"]: json.loads(l) for l in open(f"{args.baseline}/{lang}.jsonl")}
        qs = [q for q in load_questions(lang, 1, None, "v0.5.0", args.variant) if q["template"] in base]
        prompts = [build_prompt(tok, load_instruction(lang), q["question"]) for q in qs]
        responses = [tok(base[q["template"]]["response"], add_special_tokens=False)["input_ids"] for q in qs]

        arms = {}
        for name, patch in (("base", None), ("edited", args.patch)):
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()
            model.requires_grad_(False)
            if patch:
                apply_edits(model, load_spec(patch)["edits"])
            arms[name] = summarise(measure(tok, model, prompts, responses, labels, args.device), f"[{lang}] {name}")
            del model
            torch.cuda.empty_cache()
        result[lang] = arms

    name = Path(args.patch).stem if args.patch else "base"
    json.dump(result, open(out / f"{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
