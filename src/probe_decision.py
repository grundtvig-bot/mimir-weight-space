"""Fast decision probe: how much probability does the model put on starting the answer in English?

For every problem we run one forward pass over the prompt and split the probability mass of the first
response token into English-leaning, Danish-leaning and language-neutral tokens (labels from the Pile
vs Dynaword counts). The first token is where the language of the whole answer is decided: half of the
drifting Danish answers start in English outright, the rest switch later.

One forward pass per problem, so this screens an edit in ~1 minute instead of a 25-minute generation.

    python src/probe_decision.py --lang dan --device cuda:0 --patch specs/e7.json --out results/dec_e7
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


@torch.no_grad()
def first_logits(model, prompts, device, batch_size=8):
    outs = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), L), 0)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, L - len(p) :] = torch.tensor(p)
            att[i, L - len(p) :] = 1
        out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                    token_type_ids=att.clone().to(device), use_cache=False)
        outs.append(out.logits[:, -1, :].float().cpu())
    return torch.cat(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danish-foundation-models/DFM-Mimir-v1.5")
    ap.add_argument("--lang", default="dan")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--max-templates", type=int, default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--baseline", default="results/base_dev", help="run whose responses define the drift split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    labels = token_labels(tok)
    lab = labels.float()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype={"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ).to(args.device).eval()
    model.requires_grad_(False)
    if args.patch:
        from weights import apply_edits, load_spec

        apply_edits(model, load_spec(args.patch)["edits"])
        print(f"[patch] {args.patch}", flush=True)

    base = {json.loads(l)["template"]: json.loads(l) for l in open(f"{args.baseline}/{args.lang}.jsonl")}
    qs = [q for q in load_questions(args.lang, 1, args.max_templates, "v0.5.0", args.variant) if q["template"] in base]
    prompts = [build_prompt(tok, load_instruction(args.lang), q["question"]) for q in qs]
    logits = first_logits(model, prompts, args.device)
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    mass_en = (p * (lab == 1)).sum(-1)
    mass_da = (p * (lab == -1)).sum(-1)
    ratio = mass_en / (mass_en + mass_da + 1e-9)

    # the baseline's own first token: did that answer start in English?
    starts_en = []
    for q in qs:
        resp = tok(base[q["template"]]["response"], add_special_tokens=False)["input_ids"]
        starts_en.append(bool(resp) and bool(labels[resp[0]] == 1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, q in enumerate(qs):
        recs.append(
            {
                "template": q["template"],
                "mass_en": float(mass_en[i]),
                "mass_da": float(mass_da[i]),
                "ratio": float(ratio[i]),
                "base_start_en": starts_en[i],
                "base_correct": base[q["template"]]["correct"],
            }
        )
    with open(out / f"{args.lang}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    r = np.array([x["ratio"] for x in recs])
    se = np.array(starts_en)
    print(f"{args.lang} {Path(args.patch).stem if args.patch else 'base'}: n={len(r)} "
          f"mean EN-ratio {r.mean():.3f} | started-EN {se.mean():.2f} "
          f"| ratio|startedEN {r[se].mean() if se.any() else float('nan'):.3f} "
          f"| ratio|startedDA {r[~se].mean() if (~se).any() else float('nan'):.3f}", flush=True)


if __name__ == "__main__":
    main()
