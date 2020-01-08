# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file license.md
import torch
import cv2
import numpy as np

def caffe_eucl_loss(x, y):
    return torch.sum((x - y)**2) / 2

def AdapWingLoss(pre_hm, gt_hm):
    # pre_hm = pre_hm.to('cpu')
    # gt_hm = gt_hm.to('cpu')
    theta = 0.5
    alpha = 2.1
    w = 14
    e = 1
    A = w * (1 / (1 + torch.pow(theta / e, alpha - gt_hm))) * (alpha - gt_hm) * torch.pow(theta / e,
                                                                                          alpha - gt_hm - 1) * (1 / e)
    C = (theta * A - w * torch.log(1 + torch.pow(theta / e, alpha - gt_hm)))

    batch_size = gt_hm.size()[0]
    hm_num = gt_hm.size()[1]

    mask = torch.zeros_like(gt_hm)
    # print(mask.size())
    # W = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    for i in range(batch_size):
        img_list = []

        img_list.append(np.round(gt_hm[i].cpu().numpy() * 255))
        img_merge = cv2.merge(img_list)
        # print(img_merge.shape)
        img_dilate = cv2.morphologyEx(img_merge, cv2.MORPH_DILATE, kernel)
        img_dilate[img_dilate < 51] = 1  # 0*W+1
        img_dilate[img_dilate >= 51] = 11  # 1*W+1
        img_dilate = np.array(img_dilate, dtype=np.int)

        mask[i] = torch.tensor(img_dilate)

    diff_hm = torch.abs(gt_hm - pre_hm)
    AWingLoss = A * diff_hm - C
    idx = diff_hm < theta
    AWingLoss[idx] = w * torch.log(1 + torch.pow(diff_hm / e, alpha - gt_hm))[idx]

    AWingLoss *= mask
    sum_loss = torch.sum(AWingLoss)
    all_pixel = torch.sum(mask)
    mean_loss = sum_loss  # / all_pixel

    return mean_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(input, target):
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1-input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()
        #return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Motivated by https://arxiv.org/pdf/1809.00076.pdf
    :param beta: to bring the dice and focal losses at similar scale.
    :param gamma: gamma value used in the focal loss.
    :param alpha: alpha value used in the focal loss.
    """
    def __init__(self, beta=1, gamma=0.01, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, input, target):
        dc_loss = dice_loss(input, target)
        fc_loss = self.focal(input, target)

        # used to fine tune beta
        # with torch.no_grad():
        #     print('DICE loss:', dc_loss.cpu().numpy(), 'Focal loss:', fc_loss.cpu().numpy())
        #     log_dc_loss = torch.log(torch.clamp(dc_loss, 1e-7))
        #     log_fc_loss = torch.log(torch.clamp(fc_loss, 1e-7))
        #     print('Log DICE loss:', log_dc_loss.cpu().numpy(), 'Log Focal loss:', log_fc_loss.cpu().numpy())
        #     print('*'*20)

        loss = torch.log(torch.clamp(fc_loss, 1e-7)) - self.beta * torch.log(torch.clamp(dc_loss, 1e-7))

        return loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss: https://arxiv.org/pdf/1707.03237
    """
    def __init__(self, epsilon=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        input = input.view(-1)
        target = target.view(-1)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = nn.Parameter(1. / (target_sum * target_sum).clamp(min=self.epsilon))

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


def loss_l1(x, y):
    return torch.sum((y - x))

