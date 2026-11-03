"""Stream a corpus, tokenize with the Mimir (Gemma 4) tokenizer and count token frequencies.

Usage: python src/count_unigrams.py {pile,dynaword} N_TOKENS
Writes results/unigram_<corpus>.npy (int64 counts over the full vocab).
"""

import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"
CORPORA = {
    "pile": ("monology/pile-uncopyrighted", None),
    "dynaword": ("danish-foundation-models/danish-dynaword", None),
}


def main(corpus: str, n_tokens: int) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    name, cfg = CORPORA[corpus]
    ds = load_dataset(name, cfg, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    counts = np.zeros(len(tok), dtype=np.int64)
    seen, batch = 0, []
    for row in ds:
        batch.append(row["text"][:20_000])  # cap very long docs so no single doc dominates
        if len(batch) < 256:
            continue
        ids = np.concatenate([np.asarray(x, dtype=np.int64) for x in tok(batch, add_special_tokens=False)["input_ids"]])
        counts += np.bincount(ids, minlength=len(counts))[: len(counts)]
        seen += len(ids)
        batch = []
        print(f"{corpus}: {seen:,} tokens, {np.count_nonzero(counts):,} distinct", flush=True)
        if seen >= n_tokens:
            break
    out = Path("results")
    out.mkdir(exist_ok=True)
    np.save(out / f"unigram_{corpus}.npy", counts)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])))
