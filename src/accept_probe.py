"""Read the acceptability decision at the label position, and test whether a bias edit can move it.

Instead of generating the whole JSON answer, we teacher-force the prompt plus `{"label": "` and look at the
distribution over the two label tokens (`ja`/`nej`, `yes`/`no`). That position is where the judgement is
actually made, so a direction fitted there is a far better handle than the end of the prompt.

Reports per item: margin log p(no) − log p(yes), the emitted answer, the gold label.
Also fits a direction between "gold correct" and "gold incorrect" states and reports its held-out AUROC.

    python src/accept_probe.py --dataset scala-en --device cuda:0 --out results/accept_probe_scala
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from accept import LABELS, PREFIX, PROMPT, load_items, mcc  # noqa: E402
from langdir import LastHRecorder, auroc  # noqa: E402

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"
OPEN = '{"label": "'


@torch.no_grad()
def probe(tok, model, rec, prompts, lang, device, batch_size=8):
    yes, no = LABELS[lang]
    id_yes = tok.encode(yes, add_special_tokens=False)[0]
    id_no = tok.encode(no, add_special_tokens=False)[0]
    out = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        suffix = tok(OPEN, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids = torch.cat([ids, suffix.expand(len(chunk), -1)], 1)
        att = torch.cat([att, torch.ones((len(chunk), suffix.shape[1]), dtype=att.dtype, device=device)], 1)
        rec.start()
        logits = model(input_ids=ids, attention_mask=att, token_type_ids=att.clone(), use_cache=False).logits[:, -1].float()
        y = rec.y[:, -1].cpu()
        lp = torch.log_softmax(logits, -1)
        for i in range(len(chunk)):
            out.append(
                {
                    "logp_yes": float(lp[i, id_yes]),
                    "logp_no": float(lp[i, id_no]),
                    "y": y[i],
                }
            )
        print(f"  {min(b+batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dataset", default="scala-en")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang = args.lang or ("en" if args.dataset.endswith("-en") else "da")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()
    model.requires_grad_(False)
    items = load_items(args.dataset, args.split, args.limit)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": PREFIX[lang] + PROMPT[lang].format(text=it["text"])}],
                                tokenize=False, add_generation_prompt=True)
        for it in items
    ]
    rec = LastHRecorder(model)
    res = probe(tok, model, rec, prompts, lang, args.device)
    rec.remove()

    margin = np.array([r["logp_no"] - r["logp_yes"] for r in res])
    gold_no = np.array([it["label"] != "correct" for it in items])
    pred = margin > 0
    tp = int((pred & gold_no).sum())     # ungrammatical, answered "no"
    fp = int((pred & ~gold_no).sum())    # grammatical, answered "no"
    tn = int((~pred & ~gold_no).sum())   # grammatical, answered "yes"
    fn = int((~pred & gold_no).sum())    # ungrammatical, answered "yes"
    print(f"{args.dataset} [{args.split}] n={len(margin)} acc={((pred == gold_no).mean()):.4f} "
          f"MCC={mcc(tp, tn, fp, fn):.4f} | tp {tp} tn {tn} fp {fp} fn {fn}")
    print(f"margin: mean {margin.mean():+.3f} median {np.median(margin):+.3f} "
          f"(p(no) wins on {(margin > 0).mean():.2f} of items)")

    Y = torch.stack([r["y"] for r in res])
    half = len(Y) // 2
    g = Y[:half][gold_no[:half]].mean(0) - Y[:half][~gold_no[:half]].mean(0)
    g = g / g.norm()
    print(f"AUROC(gold) fit {auroc(Y[:half][gold_no[:half]] @ g, Y[:half][~gold_no[:half]] @ g):.3f} "
          f"held-out {auroc(Y[half:][gold_no[half:]] @ g, Y[half:][~gold_no[half:]] @ g):.3f}")
    torch.save({"u": g}, out / "u_gold.pt")
    np.savez_compressed(out / f"{args.dataset}_{args.split}_states.npz",
                        Y=Y.numpy().astype(np.float16),
                        logp_yes=np.array([r["logp_yes"] for r in res], dtype=np.float32),
                        logp_no=np.array([r["logp_no"] for r in res], dtype=np.float32),
                        gold_no=gold_no)
    with open(out / f"{args.dataset}_{args.split}.jsonl", "w") as f:
        for it, r, m in zip(items, res, margin):
            f.write(json.dumps({"text": it["text"], "corruption_type": it.get("corruption_type"),
                                "gold": it["label"], "margin_no_minus_yes": float(m)}) + "\n")


if __name__ == "__main__":
    main()
