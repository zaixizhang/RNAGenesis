"""
Structured dropout utilities used inside the Evoformer-style trunk.
"""

import torch
import torch.nn as nn


class DropoutRowwise(nn.Module):
    """Drop entire rows of a 2-D representation during training."""

    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, R, L, D)  or  (B, L, L, D)
        if not self.training or self.p == 0.0:
            return x
        shape = list(x.shape)
        shape[-2] = 1                           # drop whole row
        mask = x.new_empty(shape).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class DropoutColumnwise(nn.Module):
    """Drop entire columns of a 2-D representation during training."""

    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, R, L, D)
        if not self.training or self.p == 0.0:
            return x
        shape = list(x.shape)
        shape[-3] = 1
        mask = x.new_empty(shape).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask
