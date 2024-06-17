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


# # 原版
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
        """
        :param source_features: 形状为 (batch_size, feature_dim) 的张量
        :param target_features: 形状为 (batch_size, feature_dim) 的张量
        """
        # 计算源特征和目标特征的协方差矩阵
        d = source_features.size(1)
        
        # 减去均值
        source_mean = torch.mean(source_features, dim=0, keepdim=True)
        target_mean = torch.mean(target_features, dim=0, keepdim=True)
        source_centered = source_features - source_mean
        target_centered = target_features - target_mean
        
        # 计算协方差
        cov_source = torch.mm(source_centered.t(), source_centered) / (source_features.size(0) - 1)
        cov_target = torch.mm(target_centered.t(), target_centered) / (target_features.size(0) - 1)
        
        # 计算源协方差矩阵和目标协方差矩阵之间的弗罗贝尼乌斯范数
        loss = torch.norm(cov_source - cov_target, p='fro') / (4 * d * d)
        return loss


class OptimalTransportLoss(nn.Module):
    def __init__(self):
        super(OptimalTransportLoss, self).__init__()

    def forward(self, x, y):
        """
        计算最优传输损失
        参数:
        - x: 第一个分布的样本，维度为 (n_samples_x, n_features)
        - y: 第二个分布的样本，维度为 (n_samples_y, n_features)
        返回:
        - loss: 最优传输损失
        """
        # 计算成本矩阵
        M = ot.dist(x.cpu().numpy(), y.cpu().numpy())
        M /= M.max()

        # 均匀分布
        a, b = np.ones((len(x),)) / len(x), np.ones((len(y),)) / len(y)

        # 计算最优传输矩阵
        transport_matrix = ot.emd(a, b, M)

        # 计算损失
        loss = np.sum(transport_matrix * M)

        return torch.tensor(loss, requires_grad=True)


class AffinityKDLoss(nn.Module):
    def __init__(self):
        super(AffinityKDLoss, self).__init__()

    def forward(self, s, t):
        """
        计算亲和性引导的知识蒸馏损失，适用于2D图像
        :param s: 源域分支的2D特征映射或logits列表，每个元素的形状可以是 (batch_size, channels, height, width)
        :param t: 目标域分支的2D特征映射或logits列表，每个元素的形状可以是 (batch_size, channels, height, width)
        """
        kd_losses = 0

        diagonal_matrix = torch.eye(5)
        diagonal_matrix[diagonal_matrix == 0] = 0.01

        for i, s_item in enumerate(s):
            for j, t_item in enumerate(t):
                if s_item.shape != t_item.shape:
                    t_item = F.interpolate(t_item, size=s_item.shape[-2:], mode='bilinear', align_corners=False)                         
                _loss = torch.mean(torch.abs(s_item - t_item).pow(2), (2,3))    # feature
                score = torch.sum(s_item * t_item, (2,3)) / (torch.norm(torch.flatten(s_item,2), dim=(2)) * torch.norm(torch.flatten(t_item,2), dim=(2)) + 1e-40)   # a
                score = (score + 1)/2
                _score = score / torch.unsqueeze(torch.sum(score,1),1)
                kd_loss = _score * _loss
                kd_losses += kd_loss.mean()
        
        return torch.mean(kd_losses) * diagonal_matrix
    

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, logit1, logit2):
        """
        计算两组logit之间的KL散度。

        参数:
        logit1 (torch.Tensor): 第一组logit。
        logit2 (torch.Tensor): 第二组logit。

        返回值:
        torch.Tensor: 两个分布之间的KL散度。
        """

        # 使用log_softmax将logit1转换为对数概率分布，使用softmax将logit2转换为概率分布
        log_prob1 = F.log_softmax(logit1, dim=1)
        prob2 = F.softmax(logit2, dim=1)

        # 计算KL散度，确保使用对数概率和概率
        kl_div = F.kl_div(log_prob1, prob2, reduction='batchmean')
        return kl_div