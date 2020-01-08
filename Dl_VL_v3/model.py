# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file license.md
""" Counception Model
A Pytorch implementation of Count-ception designed for intervertebrae disc detection
Inspired by: https://arxiv.org/abs/1703.08710
modifed from github https://github.com/roggirg/count-ception_mbm/blob/master/train.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=nn.LeakyReLU()):
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=3, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=5, pad=1, activation=activation)
        self.conv3 = ConvBlock(in_chan, out_chan_3x3, ksize=9, pad=3, activation=activation)
        self.MP = nn.MaxPool2d(1)

    def forward(self, x):
        # print('ok')
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        # print(conv1_out.size())
        # print(conv2_out.size())
        # print(conv3_out.size())
        output = torch.cat([conv1_out, conv2_out, conv3_out], 1)
        output = self.MP(output)
        return output


class ModelCountception_v2(nn.Module):
    def __init__(self, inplanes=3, outplanes=1, use_logits=False, logits_per_output=12, debug=False):
        super(ModelCountception_v2, self).__init__()
        # params
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.3)
        self.patch_size = 40
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output
        self.debug = debug

        torch.LongTensor()

        self.conv1 = ConvBlock(self.inplanes, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(48, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(80, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(208, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(128, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(120, 32, 96, activation=self.activation)
        self.conv3 = ConvBlock(224, 32, ksize=20, activation=self.activation)
        self.conv4 = ConvBlock(32, 64, ksize=10, activation=self.activation)
        self.conv5 = ConvBlock(64, 32, ksize=9, activation=self.activation)
        if use_logits:
            self.conv6 = nn.ModuleList([ConvBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(outplanes)])
        else:
            self.conv6 = ConvBlock(32, self.outplanes, ksize=20, pad=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _print(self, x):
        if self.debug:
            print(x.size())

    def forward(self, x):
        net = self.conv1(x)  # 32
        self._print(net)

        net = self.simple1(net)

        # self._print(net)
        net = self.simple2(net)

        self._print(net)

        net = self.conv2(net)
        self._print(net)
        net = self.simple3(net)
        self._print(net)
        net = self.simple4(net)
        self._print(net)
        net = self.simple5(net)
        self._print(net)
        net = self.simple6(net)
        self._print(net)
        net = self.conv3(net)
        self._print(net)
        net = self.conv4(net)
        self._print(net)
        net = self.conv5(net)

        self._print(net)

        if self.use_logits:
            net = [c(net) for c in self.conv6]
            [self._print(n) for n in net]
        else:
            net = self.conv6(net)

            self._print(net)
        return net

    def name(self):
        return 'countception_VL_DL'
