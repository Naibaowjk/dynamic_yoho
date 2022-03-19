from glayernorm import GLayerNorm
from convmulti import ConvMulti
import numpy as np
import torch.nn as nn

np.random.seed(42)


class BottleneckMulti(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=7, stride=1, downsample=None, expansion=1):
        super(BottleneckMulti, self).__init__()
        inplanes_ = inplanes * expansion
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(inplanes, inplanes_, kernel_size=1, bias=False)
        self.bn1 = GLayerNorm(inplanes_)
        # self.conv2 = nn.Conv1d(inplanes_, inplanes_, kernel_size=kernel_size, stride=stride,
        #                        padding=pad, bias=False, groups=inplanes_)
        # self.bn2 = GLayerNorm(inplanes_)
        self.conv2 = ConvMulti(inplanes_, inplanes_, stride)
        self.conv3 = nn.Conv1d(inplanes_, planes, kernel_size=1, bias=1)
        self.bn3 = GLayerNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out