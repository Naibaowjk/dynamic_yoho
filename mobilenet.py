import numpy as np
import torch.nn as nn
from encoder import Encoder
from glayernorm import GLayerNorm
from bottleneckmulti import BottleneckMulti
np.random.seed(42)

class MobileNetV2(nn.Module):
    def __init__(self, block, layers, n_spk=4, num_emed=128, mode='train'):
        super(MobileNetV2, self).__init__()
        self.mode = mode
        self.inplanes = 32
        self.n_spk = n_spk
        self.num_emed = num_emed
        # self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3, bias=False)
        self.conv1 = Encoder(1, 32)
        self.bn1 = GLayerNorm(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=4, expansion=6)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=4, expansion=6)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=4, expansion=6)
        self.layer5 = self._make_layer(block, 96, layers[4], stride=1, expansion=6)
        self.layer6 = self._make_layer(block, 160, layers[5], stride=4, expansion=6)
        self.layer7 = self._make_layer(block, 320, layers[6], stride=1, expansion=6)
        self.conv8 = nn.Conv1d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # fc layer
        # 
        # input size: the output of abstraction module.
        # output size: 
        self.fc = nn.Linear(1280, num_emed*n_spk)

    def _make_layer(self, block, planes, blocks, stride, expansion):
        downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            GLayerNorm(planes),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv8(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        features =[]
        start = 0
        for i in range(self.n_spk):
            f = x[:, start:start+self.num_emed]
            start += self.num_emed
            features.append(f)
        return features

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


def mobilenet_19(**kwargs):
    """Constructs a MobileNetV2-19 model.
    """
    n_spk = kwargs["n_spk"] if "n_spk" in kwargs else 4
    num_emed = kwargs["num_emed"] if "num_emed" in kwargs else 128
    mode = kwargs["mode"] if "mode" in kwargs else 'train'
    model = MobileNetV2(BottleneckMulti, [1, 2, 3, 4, 3, 3, 1], n_spk, num_emed, mode)
    return model