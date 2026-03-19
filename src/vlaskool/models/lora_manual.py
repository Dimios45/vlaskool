"""Manual LoRA injection — fallback when PEFT is unavailable or incompatible."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Replaces an nn.Linear with W + BA (LoRA)."""

    def __init__(
        self,
        original: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = original.in_features, original.out_features

        # Freeze original weight
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(original.bias.data.clone(), requires_grad=False)
            if original.bias is not None
            else None
        )

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        # x: (..., in_f); lora_A: (r, in_f); lora_B: (out_f, r)
        lora = nn.functional.linear(self.dropout(x), self.lora_A)  # (..., r)
        lora = self.scaling * nn.functional.linear(lora, self.lora_B)  # (..., out_f)
        return base + lora


def inject_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 32,
    target_names: Optional[list[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace target nn.Linear modules with LoRALinear in-place."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if target_names is not None and leaf not in target_names:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else _get_submodule(model, parent_name)
        setattr(parent, attr, LoRALinear(module, r=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    # Freeze non-LoRA parameters
    for param in model.parameters():
        if not isinstance(param, nn.Parameter) or not hasattr(param, "_is_lora"):
            param.requires_grad = False
    # Un-freeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Manual LoRA: replaced {replaced} modules, {n_train:,}/{n_total:,} trainable")
    return model


def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    cur = model
    for p in parts:
        cur = getattr(cur, p)
    return cur
