import torch
import numpy as np
import torch.nn as nn
np.random.seed(42)

# Encoder: Input Size (1,m) , m is the sample point nums, discrete signal.
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_7 = nn.Conv1d(in_channels, out_channels//2, kernel_size=7, stride=4, padding=3, bias=False)
        self.conv_5 = nn.Conv1d(in_channels, out_channels//2, kernel_size=5, stride=4, padding=2, bias=False)

    def forward(self, x):
        output_7 = self.conv_7(x)
        output_5 = self.conv_5(x)
        return torch.cat([output_7, output_5], dim=1)

# Discuss:
#           in_channels = 1, out_channels = 32, input size (h * w) = 1 * m 
#   conv_7: 
#           h_out = (1 - 7 + 2*3)/4 + 1 = 1
#           w_out = (m - 7 + 2*3)/4 + 1 = (m - 1)/4 + 1
#           channels = 16
#   conv_5:
#           h_out = (1 - 5 + 2*2)/4 + 1 = 1
#           w_out = (m - 5 + 2*2)/4 + 1 = (m - 1)/4 + 1
#           channels = 16
#   output:
#           h_out = 1
#           w_out = (m - 1)/2 + 2
#           channels = 16
