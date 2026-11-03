"""Shared helpers: load Mimir, capture the prediction state y after every module application."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "danish-foundation-models/DFM-Mimir-v1.5"


def load(device: str = "cuda:0", dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    return tok, model


def module_labels(model) -> list[str]:
    """Names of every module application in execution order, e.g. L1 L2 L3 H1 L4 L5 L6 H2."""
    cfg, out, n_l, n_h = model.config, [], 0, 0
    for _ in range(cfg.H_cycles):
        for _ in range(cfg.L_cycles):
            n_l += 1
            out.append(f"L{n_l}")
        n_h += 1
        out.append(f"H{n_h}")
    return out


class StateRecorder:
    """Forward hooks on the L- and H-stacks. Each stack ends in a (weightless) RMSNorm, so every
    recorded output lives in the same normalised space as the final prediction state y."""

    def __init__(self, model):
        self.states: list[torch.Tensor] = []
        self.handles = [
            model.model.L_module.register_forward_hook(self._hook),
            model.model.H_module.register_forward_hook(self._hook),
        ]

    def _hook(self, _mod, _inp, out):
        self.states.append(out.detach().float())

    def clear(self):
        self.states = []

    def remove(self):
        for h in self.handles:
            h.remove()
