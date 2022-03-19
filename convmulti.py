import torch
import numpy as np
import torch.nn as nn
from glayernorm import GLayerNorm
np.random.seed(42)


class ConvMulti(nn.Module):
    def __init__(self, inplanes, planes, stride=4):
        super(ConvMulti, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_7 = nn.Conv1d(inplanes, planes//2, kernel_size=7, stride=stride,
                                padding=3, bias=False, groups=planes//2)
        self.bn_7 = GLayerNorm(planes//2)

        self.conv_5 = nn.Conv1d(inplanes, planes//2, kernel_size=5, stride=stride,
                                padding=2, bias=False, groups=planes//2)
        self.bn_5 = GLayerNorm(planes//2)

    def forward(self, x):
        output_7 = self.bn_7(self.conv_7(x))
        output_5 = self.bn_5(self.conv_5(x))
        return self.relu(torch.cat([output_7, output_5], dim=1))