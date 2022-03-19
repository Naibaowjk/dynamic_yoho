import torch
import numpy as np
import torch.nn as nn
np.random.seed(42)


# define GLayerNorm, which is used in the Backbone

class GLayerNorm(nn.Module):
    """Global Layer Normalization for TasNet."""

    def __init__(self, channels, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.norm_dim = channels
        self.gamma = nn.Parameter(torch.Tensor(channels))
        self.beta = nn.Parameter(torch.Tensor(channels))
        # self.register_parameter('weight', self.gamma)
        # self.register_parameter('bias', self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, sample):
        """Forward function.
        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # Mean and variance [N, 1, 1]
        mean = torch.mean(sample, (1, 2), keepdim=True)
        var = torch.mean((sample - mean) ** 2, (1, 2), keepdim=True)
        sample = (sample - mean) / torch.sqrt(var + self.eps) * \
                 self.gamma + self.beta
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample