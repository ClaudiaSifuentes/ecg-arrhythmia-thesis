"""Focal loss for multi-class beat classification.

Lin et al. (2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Down-weights easy, well-classified beats (mostly class N) so gradient signal
concentrates on hard/rare beats (SVEB, VEB) without needing SMOTE to rebalance
the training distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1 - pt).clamp(min=0).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
