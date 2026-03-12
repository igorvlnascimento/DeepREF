from torch import nn
import torch

import torch.nn.functional as F

class FocalLossLabelSmoothing(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        # cross entropy
        ce_loss = -(true_dist * log_probs).sum(dim=1)

        # focal weight
        pt = (true_dist * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss