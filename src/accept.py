"""Linguistic acceptability (ScaLA-en / DaLA) with the EuroEval prompt, per item.

EuroEval's own numbers are reproduced here so that every item carries a prediction, a corruption type and
a confidence: scala-en 64.90 MCC, dala 97.67 MCC (vLLM, causal mask only).

    python src/accept.py --dataset scala-en --device cuda:0 --out results/accept_base

`--patch` applies an edit spec first. `--limit` truncates for quick screens.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from weights import apply_edits, load_spec  # noqa: E402

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"
PROMPT = {
    "en": 'Sentence: {text}\n\nDetermine whether the sentence is grammatically correct or not. '
          'Answer with {{"label": "yes"}} or {{"label": "no"}}, and nothing else.',
    "da": 'Sætning: {text}\n\nBestem om sætningen er grammatisk korrekt eller ej. '
          'Svar kun med {{"label": "ja"}} eller {{"label": "nej"}}, og intet andet.',
}
LABELS = {"en": ("yes", "no"), "da": ("ja", "nej")}
# EuroEval renders `prompt_prefix` before the instruction (this line is what makes the task work at all:
# without it Mimir answers "ja" to everything, MCC 0.00 on DaLA)
PREFIX = {
    "en": "The following are sentences and whether they are grammatically correct.\n\n",
    "da": "Følgende er sætninger og om de er grammatisk korrekte.\n\n",
}


def load_items(dataset: str, split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset(f"EuroEval/{dataset}", split=split, cache_dir=".euroeval_cache")
    rows = list(ds)
    if limit:
        rows = rows[:limit]
    return rows


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / den if den else 0.0


@torch.no_grad()
def run(tok, model, items, lang, device, batch_size=8, max_new_tokens=16, prefix_lm=False):
    ids_yes = tok.encode(" " + LABELS[lang][0], add_special_tokens=False)[-1]
    ids_no = tok.encode(" " + LABELS[lang][1], add_special_tokens=False)[-1]
    prompts = []
    for it in items:
        text = PREFIX[lang] + PROMPT[lang].format(text=it["text"])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True))
    out = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        kw = {k: v.to(device) for k, v in enc.items()}
        if prefix_lm:
            kw["token_type_ids"] = kw["attention_mask"].clone()
        gen = model.generate(**kw, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1] :]
        for row in new.tolist():
            if tok.eos_token_id in row:
                row = row[: row.index(tok.eos_token_id)]
            out.append(tok.decode(row, skip_special_tokens=True))
        print(f"  {min(b+batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out, (ids_yes, ids_no)


def parse_label(text: str, lang: str) -> str | None:
    m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
    cand = m.group(1).lower() if m else text.strip().lower()
    for lab in LABELS[lang]:
        if lab in cand:
            return lab
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scala-en", "dala"])
    ap.add_argument("--lang", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--prefix-lm", action="store_true", help="bidirectional prompt block, as the model card specifies")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang = args.lang or ("en" if args.dataset.endswith("-en") else "da")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype={"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ).to(args.device).eval()
    model.requires_grad_(False)
    if args.patch:
        apply_edits(model, load_spec(args.patch)["edits"])
        print(f"[patch] {args.patch}", flush=True)

    items = load_items(args.dataset, args.split, args.limit)
    print(f"{args.dataset}: {len(items)} items", flush=True)
    texts, _ = run(tok, model, items, lang, args.device, prefix_lm=args.prefix_lm)

    yes, no = LABELS[lang]
    recs, cm = [], {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for it, raw in zip(items, texts):
        lab = parse_label(raw, lang)
        gold = "correct" if it["label"] == "correct" else "incorrect"
        pred = None
        if lab == yes:
            pred = "correct"
        elif lab == no:
            pred = "incorrect"
        if pred == "correct" and gold == "correct":
            cm["tp"] += 1
        elif pred == "correct" and gold == "incorrect":
            cm["fp"] += 1
        elif pred == "incorrect" and gold == "correct":
            cm["fn"] += 1
        elif pred == "incorrect" and gold == "incorrect":
            cm["tn"] += 1
        recs.append({"text": it["text"], "corruption_type": it.get("corruption_type"), "gold": gold,
                     "raw": raw, "pred": pred, "gold_label": lab and gold})
    n_unparsed = sum(r["pred"] is None for r in recs)
    m = mcc(cm["tp"], cm["tn"], cm["fp"], cm["fn"])
    acc = (cm["tp"] + cm["tn"]) / max(1, len(recs))
    print(f"{args.dataset} {Path(args.patch).stem if args.patch else 'base'}: n={len(recs)} acc={acc:.4f} "
          f"MCC={m:.4f} unparsed={n_unparsed} | tp {cm['tp']} tn {cm['tn']} fp {cm['fp']} fn {cm['fn']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"{args.dataset}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
