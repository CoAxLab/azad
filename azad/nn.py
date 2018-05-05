import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    """A layer model for a Deep Q Network

    Code modified from:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, N_action):
        super(DM, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, N_action)

    def forward(self, x, bias):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1)) + bias


class LQN(th.nn.Module):
    """A layer model for a Q 'Lookup Network'.
    
    NOTE: this assumes states are defined in the input. Use
    DQN if you are working with raw pixels, or similar some such.
    """

    def __init__(self, N_state, N_action):
        self.lin = nn.Linear(N_state, N_action)

    def forward(self, x, bias):
        x = self.lin(x)
        return x + bias
