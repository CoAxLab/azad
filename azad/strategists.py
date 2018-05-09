"""ANNs that learn using strategy"""
import torch
import torch.nn as nn


def estimate_expected_value(x, dim):
    """Estimate the expected value (of a Q-value tensor)"""

    return torch.max(x, dim)


def estimate_alp_value(x, dim):
    """Estimate the expected value using the Muyesser method
    
    As described in:
    
    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """

    x = estimate_expected_value(x, dim)
    x = (x + 1) / 2
    return x


def estimate_hot_cold(x, zeta):
    """Estimate the hot/coldness of elements in x

    Hot positions are greater than zeta. Cold are lower.
    """

    if zeta < 0:
        raise ValueError("zeta must be >= 0.")

    cold = torch.zeros_likes(x)
    hot = torch.ones_like(x)
    x = torch.where(x < zeta, x, cold)
    x = torch.where(x >= zeta, x, hot)

    return x


class HotCold(nn.Module):
    """Layers for a Hot-Cold strategy
    
    As described in:
    
    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based 
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """

    def __init__(self, in_channels=2):
        super(HotCold, self).__init__()
        self.fc1 = nn.Linear(in_channels, 15)
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2
