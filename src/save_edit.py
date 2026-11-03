"""Apply an edit spec to DFM-Mimir-v1.5 and save a standalone checkpoint.

    python src/save_edit.py --patch specs/e1.json --out checkpoints/e1

The saved directory has the original config, tokenizer and chat template, the edited BF16 weights,
and the edit spec itself (`edit_spec.json`) plus a hash of the changed tensor list, so a checkpoint
is always traceable to the spec that produced it.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from weights import apply_edits, load_spec, total_params  # noqa: E402

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_spec(args.patch)
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    before = {
        e["param"]: model.get_parameter(e["param"]).detach().clone() for e in spec["edits"]
    }
    changed = apply_edits(model, spec["edits"])
    n_changed = sum(changed.values())
    print(f"{spec.get('name')}: {n_changed:,} weights = {100 * n_changed / total_params(model):.4f}% of the model")
    for param, n in changed.items():
        w0, w1 = before[param], model.get_parameter(param).detach()
        d = (w1.float() - w0.float())
        print(f"  {param}: {n:,} weights, |delta| max {d.abs().max():.4f} mean {d.abs().mean():.5f}")

    model.to(torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "edit_spec.json").write_text(json.dumps(spec, indent=1))

    h = hashlib.sha256()
    for f in sorted(out.glob("*.safetensors")):
        h.update(f.read_bytes())
    digest = h.hexdigest()[:16]
    (out / "EDIT.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "edit": spec.get("name"),
                "spec": args.patch,
                "n_weights_changed": n_changed,
                "fraction_of_model": n_changed / total_params(model),
                "weights_sha256_16": digest,
            },
            indent=1,
        )
    )
    print(f"saved {out} (weights sha256 {digest})")


if __name__ == "__main__":
    main()
