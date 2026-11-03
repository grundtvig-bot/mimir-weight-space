"""Fast behavioural probe of Mimir at the answer position (few forward passes, no generation).

For every problem we run a single forward pass over the prompt and look at the distribution the model
would sample the *first response token* from. That token decides the shape of the whole answer: either
the model opens its chain of thought ("Trin", "Step", "##", "\n"), or it jumps straight to the final
answer ("\\boxed", "Svaret", "The answer").

Reports per language: mean entropy, the mass on each family of openers, and per-problem top tokens.
Optionally records the final-state projection onto a direction (the prior / "ignorance" direction).

Usage: python src/probe_state.py --langs dan eng --device cuda:0 --variant 0 --patch spec.json
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

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"

COT_OPENERS = ["Trin", "Step", "##", "\n", "First", "Først", "For", "Lad", "We", "Vi", "**", "\u2022", "1"]
ANSWER_OPENERS = ["\\boxed", "\\(", "Svar", "Answer", "The final", "Det", "Result", "$"]


@torch.no_grad()
def first_token_logits(tok, model, prompts: list[list[int]], device, batch_size=24):
    outs = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, L - len(p) :] = torch.tensor(p)
            att[i, L - len(p) :] = 1
        out = model(
            input_ids=ids.to(device),
            attention_mask=att.to(device),
            token_type_ids=att.clone().to(device),
            use_cache=False,
        )
        logits = out.logits[:, -1, :].float()
        outs.append(logits.cpu())
        print(f"  forwarded {min(b + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(outs)


def opener_mass(logp: torch.Tensor, tok, openers: list[str]) -> tuple[float, str]:
    """Total probability of the token ids that start each opener string, and the most likely opener."""
    ids, names = [], []
    for s in openers:
        e = tok.encode(s, add_special_tokens=False)
        if e:
            ids.append(e[0])
            names.append(s)
    ids = torch.tensor(sorted(set(ids)))
    mass = logp[ids].exp().sum().item()
    best = names[int(logp[ids].argmax())]
    return mass, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--langs", nargs="+", default=["dan", "eng"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--max-templates", type=int, default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--revision", default="v0.5.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    model.requires_grad_(False)
    if args.patch:
        from weights import apply_edits, load_spec

        apply_edits(model, load_spec(args.patch)["edits"])
        print(f"[patch] {args.patch}", flush=True)

    summary = {}
    for lang in args.langs:
        qs = load_questions(lang, 1, args.max_templates, args.revision, args.variant)
        prompts = [build_prompt(tok, load_instruction(lang), q["question"]) for q in qs]
        logits = first_token_logits(tok, model, prompts, args.device)
        logp = torch.log_softmax(logits, -1)
        ent = -(logp.exp() * logp).sum(-1)
        recs = []
        for i, q in enumerate(qs):
            cot, cot_best = opener_mass(logp[i], tok, COT_OPENERS)
            ans, ans_best = opener_mass(logp[i], tok, ANSWER_OPENERS)
            top = torch.topk(logits[i], 5)
            recs.append(
                {
                    "lang": lang,
                    "template": q["template"],
                    "entropy": float(ent[i]),
                    "cot_mass": cot,
                    "answer_mass": ans,
                    "cot_best": cot_best,
                    "answer_best": ans_best,
                    "top": [tok.decode([int(t)]) for t in top.indices],
                    "top_p": [float(p) for p in top.values.softmax(-1)],
                }
            )
        with open(out / f"{lang}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[lang] = {
            "entropy": float(ent.mean()),
            "cot_mass": float(np.mean([r["cot_mass"] for r in recs])),
            "answer_mass": float(np.mean([r["answer_mass"] for r in recs])),
            "answer_dominant": float(np.mean([r["answer_mass"] > r["cot_mass"] for r in recs])),
            "top1": [r["top"][0] for r in recs],
        }
        print(f"[{lang}] entropy {summary[lang]['entropy']:.2f} cot_mass {summary[lang]['cot_mass']:.3f} "
              f"answer_mass {summary[lang]['answer_mass']:.3f} answer_dominant {summary[lang]['answer_dominant']:.2f}",
              flush=True)
    json.dump(summary, open(out / "summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
