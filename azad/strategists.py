"""ANNs that learn using strategy"""
import torch
import torch.nn as nn


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
