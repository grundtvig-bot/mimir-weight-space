"""Mimir on multilingual GSM-symbolic: generate, score, and record the failure shape.

Baseline / edited-checkpoint evaluation. Greedy decoding, Gemma chat template, PrefixLM mask
(the prompt is one bidirectional block), instructions from the multilingual-gsm-symbolic package.

Usage:
  python src/eval_gsm.py --model danish-foundation-models/DFM-Mimir-v1.5 --langs dan eng \
      --device cuda:0 --n 2 --out results/base
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import get_dataset_split_names, load_dataset
from multilingual_gsm_symbolic import load_instruction
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import module_labels  # noqa: E402

DATASET = "danish-foundation-models/multilingual-gsm-symbolic"
DS_REVISION = "v0.5.0"


def load_questions(lang: str, n: int, max_templates: int | None, revision: str, variant: int | None = None) -> list[dict]:
    split = next(s for s in get_dataset_split_names(DATASET, lang, revision=revision) if s.startswith("test_synthetic"))
    ds = load_dataset(DATASET, lang, split=split, revision=revision)
    per_template: dict[str, list[dict]] = {}
    for row in ds:
        per_template.setdefault(row["source_id"], []).append(row)
    out = []
    for sid in sorted(per_template, key=int)[:max_templates]:
        rows = per_template[sid]
        if variant is not None:
            rows = [rows[variant]] if variant < len(rows) else []
        else:
            rows = rows[:n]
        for row in rows:
            out.append({"template": sid, "question": row["question"], "gold": float(row["target"])})
    return out


def parse_number(s: str) -> float | None:
    while re.search(r"\d,\d{3}(?!\d)", s):
        s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = s.replace("\\%", "").replace("%", "").replace("$", "").replace("\\$", "")
    s = re.sub(r"\\(text|mathrm)\{[^}]*\}", "", s)
    frac = re.fullmatch(r"\s*-?\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}\s*", s)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + len("\\boxed{")
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + len("\\boxed{") : j - 1]


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_answer(text: str) -> tuple[float | None, bool]:
    """The last \\boxed{} if present, else the last number in the response. Returns (value, from_box)."""
    if "\\boxed{" in text:
        boxed = extract_boxed(text)
        return (parse_number(boxed) if boxed is not None else None), True
    nums = list(NUMBER.finditer(text))
    if not nums:
        return None, False
    return parse_number(nums[-1].group()), False


def build_prompt(tok, instruction: str, question: str) -> list[int]:
    msgs = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(tok, model, prompts: list[list[int]], device, batch_size=16, max_new_tokens=1024):
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
            token_type_ids=att.clone().to(device),  # PrefixLM: prompt is one bidirectional block
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        for row in gen[:, L:].tolist():
            if tok.eos_token_id in row:
                row = row[: row.index(tok.eos_token_id) + 1]
            outs.append([t for t in row if t != tok.pad_token_id])
        print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    return outs


# a "direct" answer: no chain of thought, the response is essentially just the final answer
COT_MARKERS = ("<<", "=", "først", "first", "step", "trin", "derfor", "therefore", "so ", "så ")


def looks_direct(text: str, n_tokens: int) -> bool:
    body = text.replace("\\boxed{", "").strip()
    return n_tokens < 40 or sum(m in body.lower() for m in COT_MARKERS) == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danish-foundation-models/DFM-Mimir-v1.5")
    ap.add_argument("--langs", nargs="+", default=["dan", "eng"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2, help="instances per template")
    ap.add_argument("--variant", type=int, default=None, help="use only instance #k of every template (dev/test split)")
    ap.add_argument("--patch", default=None, help="JSON edit spec applied to the weights before evaluation")
    ap.add_argument("--max-templates", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--revision", default=DS_REVISION)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]).to(args.device).eval()
    model.requires_grad_(False)
    if args.patch:
        from weights import apply_edits, load_spec, total_params

        spec = load_spec(args.patch)
        changed = apply_edits(model, spec["edits"])
        n_changed = sum(changed.values())
        print(f"[patch] {spec.get('name', args.patch)}: {n_changed:,} weights "
              f"({100 * n_changed / total_params(model):.4f}% of {total_params(model):,})", flush=True)
        for k, v in changed.items():
            print(f"  {k}: {v:,}", flush=True)
    print(f"{args.model}: {module_labels(model)}", flush=True)

    for lang in args.langs:
        qs = load_questions(lang, args.n, args.max_templates, args.revision, args.variant)
        instruction = load_instruction(lang)
        prompts = [build_prompt(tok, instruction, q["question"]) for q in qs]
        print(f"[{lang}] {len(prompts)} questions", flush=True)
        resps = generate(tok, model, prompts, args.device, args.batch_size, args.max_new_tokens)
        recs = []
        for q, r in zip(qs, resps):
            text = tok.decode(r, skip_special_tokens=True)
            pred, boxed = final_answer(text)
            correct = pred is not None and abs(pred - q["gold"]) < 1e-4 * max(1, abs(q["gold"]))
            recs.append(
                {
                    "lang": lang,
                    **q,
                    "response": text,
                    "pred": pred,
                    "correct": bool(correct),
                    "boxed": boxed,
                    "n_resp_tokens": len(r),
                    "finished": bool(r) and r[-1] == tok.eos_token_id,
                    "direct": looks_direct(text, len(r)),
                }
            )
        with open(out_dir / f"{lang}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        acc = float(np.mean([rec["correct"] for rec in recs]))
        direct = float(np.mean([rec["direct"] for rec in recs]))
        trunc = float(np.mean([not rec["finished"] for rec in recs]))
        acc_reasoned = float(np.mean([rec["correct"] for rec in recs if not rec["direct"]])) if any(
            not rec["direct"] for rec in recs
        ) else float("nan")
        print(
            f"[{lang}] n={len(recs)} acc={acc:.3f} direct={direct:.3f} "
            f"acc|reasoned={acc_reasoned:.3f} truncated={trunc:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
