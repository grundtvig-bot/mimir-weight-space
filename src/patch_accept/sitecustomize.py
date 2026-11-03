"""Inject the acceptability edit into EuroEval's vLLM backend (screening without saving checkpoints).

The edit adds `alpha * <h, u>` to the `no` logit and subtracts it from the `yes` logit, which is exactly
what writing +alpha*u into the `no` row and -alpha*u into the `yes` row of the output head does. Doing it
inside `compute_logits` lets us sweep alpha with the real benchmark harness while the *weights* stay fixed.

Env:
  MIMIR_ACC_ALPHA   strength (0 = off)
  MIMIR_ACC_DIR     torch file with {"u": tensor}
  MIMIR_ACC_ROWS    "no_id,yes_id[,no_id2,yes_id2...]" token ids
"""

import os

import torch

ALPHA = float(os.environ.get("MIMIR_ACC_ALPHA", "0"))
DIRECTION = os.environ.get("MIMIR_ACC_DIR", "results/accept_probe_scala/u_gold.pt")
ROWS = [int(x) for x in os.environ.get("MIMIR_ACC_ROWS", "1904,4443").split(",")]
BACKEND = os.environ.get("MIMIR_ACC_BACKEND", "vllm")


def _patch_vllm() -> None:
    from vllm.model_executor.models.hrm_text import HrmTextForCausalLM

    u = torch.load(DIRECTION, weights_only=False)["u"].float()
    original = HrmTextForCausalLM.compute_logits
    pairs = [(ROWS[i], ROWS[i + 1]) for i in range(0, len(ROWS), 2)]

    def compute_logits(self, hidden_states):
        logits = original(self, hidden_states)
        if logits is None:
            return logits
        proj = hidden_states.float() @ u.to(hidden_states.device)
        for no_id, yes_id in pairs:
            logits[:, no_id] += ALPHA * proj
            logits[:, yes_id] -= ALPHA * proj
        return logits

    HrmTextForCausalLM.compute_logits = compute_logits
    print(f"[mimir-acc] vLLM patched: alpha={ALPHA} rows={pairs} u={DIRECTION}", flush=True)


if ALPHA != 0 and BACKEND == "vllm":
    _patch_vllm()
