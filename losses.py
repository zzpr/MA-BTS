import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return self.real_label.expand_as(input)
        else:
            return self.fake_label.expand_as(input)

    def __call__(self, input, target_is_real, device=None):
        target_tensor = self.get_target_tensor(input, target_is_real).to(device)
        return self.loss(input, target_tensor)


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super(BCEDiceLoss, self).__init__()

#     def forward(self, input, target):
#         # bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         bce = F.binary_cross_entropy(input, target)

#         return bce + dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # # bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        bce = F.binary_cross_entropy(input, target)

        return bce + dice
    

class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source_features, target_features):
        d = source_features.size(1)
        
        source_mean = torch.mean(source_features, dim=0, keepdim=True)
        target_mean = torch.mean(target_features, dim=0, keepdim=True)
        source_centered = source_features - source_mean
        target_centered = target_features - target_mean
        
        cov_source = torch.mm(source_centered.t(), source_centered) / (source_features.size(0) - 1)
        cov_target = torch.mm(target_centered.t(), target_centered) / (target_features.size(0) - 1)
        
        loss = torch.norm(cov_source - cov_target, p='fro') / (4 * d * d)
        return loss


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, logit1, logit2):
        
        log_prob1 = F.log_softmax(logit1, dim=1)
        prob2 = F.softmax(logit2, dim=1)

        kl_div = F.kl_div(log_prob1, prob2, reduction='batchmean')
        return kl_div
