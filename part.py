import numpy as np
import torch.nn as nn
np.random.seed(42)


# define the distributed ia-net-lite
# backbone is mobilenet v2

class Part_1(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_1, self).__init__()
        self.mode = mode
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_2(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_2, self).__init__()
        self.mode = mode
        self.layer5 = backbone.layer5
        self.layer6 = backbone.layer6

    def forward_once(self, x):
        x = self.layer5(x)
        x = self.layer6(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_3(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_3, self).__init__()
        self.num_emed = backbone.num_emed
        self.n_spk = backbone.n_spk
        self.mode = mode
        self.layer7 = backbone.layer7
        self.conv8 = backbone.conv8
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = backbone.fc

    def forward_once(self, x):
        x = self.layer7(x)
        x = self.conv8(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        features = []
        start = 0
        for i in range(self.n_spk):
            f = x[:, start:start + self.num_emed]
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


def build_part(backbone, mode='train'):
    """Constructs a MobileNetV2-19 model.
    """
    # backbone = mobilenet_19()
    part1 = Part_1(backbone, mode=mode)
    part2 = Part_2(backbone, mode=mode)
    part3 = Part_3(backbone, mode=mode)
    return part1, part2, part3