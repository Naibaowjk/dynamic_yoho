import torch.nn as nn
import numpy as np
np.random.seed(42)

# Last part has FC layer, we should split it
class Part_Conv_3(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_Conv_3, self).__init__()
        self.num_emed = backbone.num_emed
        self.n_spk = backbone.n_spk
        self.mode = mode
        self.layer7 = backbone.layer7
        self.conv8 = backbone.conv8

    def forward_once(self, x):
        x = self.layer7(x)
        x = self.conv8(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_FC_3(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_FC_3, self).__init__()
        self.num_emed = backbone.num_emed
        self.n_spk = backbone.n_spk
        self.mode = mode
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = backbone.fc

    def forward_once(self, x):
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