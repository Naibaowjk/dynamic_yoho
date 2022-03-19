# create the part model & show result
import torch
from mobilenet import mobilenet_19
import os
import numpy as np
import torch.nn as nn
import librosa
import time
from sklearn import preprocessing
from mobilenet_split import conv_layers 
np.random.seed(42)


# split net
class ConvLayers(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(ConvLayers, self).__init__()
        self.mode = mode
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer5 = backbone.layer5
        self.layer6 = backbone.layer6
        self.layer7 = backbone.layer7
        self.conv8 = backbone.conv8


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
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class FCLayers(nn.Module):
    def __init__(self, backbone, mode='train') -> None:
        super(FCLayers, self).__init__()
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


