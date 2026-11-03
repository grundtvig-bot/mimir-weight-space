"""Would the drifting Danish answers be right if the model stayed in Danish?

No weights are touched here: we re-run the problems whose baseline answer drifted into English, but
this time the response is *prefilled* with a Danish reasoning opener ("Trin 1:"). If accuracy jumps,
the language switch is what broke those answers; if it does not, the switch was a symptom of the model
not being able to solve the problem.

    python src/prefill_test.py --lang dan --variant 0 --device cuda:3 --out results/prefill
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from analyze_run import lang_frac  # noqa: E402
from eval_gsm import build_prompt, final_answer, load_questions  # noqa: E402
from multilingual_gsm_symbolic import load_instruction  # noqa: E402

PREFILL = {"dan": "Trin 1:", "eng": "Step 1:"}


@torch.no_grad()
def generate(tok, model, prompts, device, max_new_tokens=768, batch_size=6):
    outs = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, L - len(p) :] = torch.tensor(p)
            att[i, L - len(p) :] = 1
        gen = model.generate(
            input_ids=ids.to(device),
            attention_mask=att.to(device),
            token_type_ids=att.clone().to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        for row in gen[:, L:].tolist():
            if tok.eos_token_id in row:
                row = row[: row.index(tok.eos_token_id) + 1]
            outs.append([t for t in row if t != tok.pad_token_id])
        print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danish-foundation-models/DFM-Mimir-v1.5")
    ap.add_argument("--lang", default="dan")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--baseline", default="results/base_dev")
    ap.add_argument("--which", default="drift", choices=["drift", "stay", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype={"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ).to(args.device).eval()
    model.requires_grad_(False)

    base = {json.loads(l)["template"]: json.loads(l) for l in open(f"{args.baseline}/{args.lang}.jsonl")}
    qs = [q for q in load_questions(args.lang, 1, None, "v0.5.0", args.variant) if q["template"] in base]
    drifted = {k: (lang_frac(v["response"]) < 0.75) for k, v in base.items()}
    if args.which == "drift":
        qs = [q for q in qs if drifted[q["template"]]]
    elif args.which == "stay":
        qs = [q for q in qs if not drifted[q["template"]]]
    print(f"{len(qs)} problems ({args.which})", flush=True)

    instruction = load_instruction(args.lang)
    prefill = tok(PREFILL.get(args.lang, "Step 1:"), add_special_tokens=False)["input_ids"]
    prompts = [build_prompt(tok, instruction, q["question"]) + prefill for q in qs]
    resps = generate(tok, model, prompts, args.device)

    recs = []
    for q, r in zip(qs, resps):
        text = PREFILL.get(args.lang, "Step 1:") + tok.decode(r, skip_special_tokens=True)
        pred, boxed = final_answer(text)
        correct = pred is not None and abs(pred - q["gold"]) < 1e-4 * max(1, abs(q["gold"]))
        recs.append(
            {
                "template": q["template"],
                **q,
                "response": text,
                "pred": pred,
                "correct": bool(correct),
                "lang_frac": lang_frac(text),
                "base_correct": base[q["template"]]["correct"],
            }
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.lang}_{args.which}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = np.mean([r["correct"] for r in recs])
    base_acc = np.mean([r["base_correct"] for r in recs])
    still_drift = np.mean([r["lang_frac"] < 0.75 for r in recs])
    wins = sum(r["correct"] and not r["base_correct"] for r in recs)
    losses = sum(r["base_correct"] and not r["correct"] for r in recs)
    print(
        f"[{args.lang}/{args.which}] n={len(recs)} base {base_acc:.3f} -> prefilled {acc:.3f} "
        f"({wins} wins / {losses} losses)  still-drifts {still_drift:.2f}"
    )


if __name__ == "__main__":
    main()
