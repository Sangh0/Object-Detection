from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self, label_smooth: Optional[float]=None, reduction: str='none'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.label_smooth = label_smooth
        
    def forward(self, logits, labels):
        if self.label_smooth:
            labels = torch.where(labels==0, self.smooth, 1 - self.smooth)
        loss = F.mse_loss(logits, labels, reduction=self.reduction)
        return loss


class BCEWithLogitsLoss(nn.Module):

    def __init__(
        self, 
        pos_weight: Optional[float]=None,
        label_smooth: Optional[float]=None, 
        reduction: str='none',
    ):
        self.pos_weight = pos_weight
        self.label_smooth = label_smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        if self.label_smooth:
            labels = torch.where(labels==0, self.smooth, 1 - self.smooth)

        loss = F.binary_cross_entropy_with_logits(
            logits, 
            labels, 
            reduction=self.reduction=, 
            pos_weight=torch.Tensor([self.pos_weight]),
        )
        return loss
