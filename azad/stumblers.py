"""ANNs that learn only by trial-and-error"""
import torch
import torch as th


class DM(nn.Module):
    """Layer model for a Deep Q Network

    Code taken from:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self):
        super(DM, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class LM(object):
    """A learned lookup table for Q values"""

    def forward(self, x):
        th.max()
