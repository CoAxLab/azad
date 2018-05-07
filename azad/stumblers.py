import random
import torch
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from azad.util import ReplayMemory


class DQN(tnn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        """Layers for a Deep Q Network

        Code modified from:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        
        Params
        ------
        in_channels: Tensor size 
            Number of channel of input. i.e The number of most recent frames 
            stacked together
        num_actions: int 
            Number of action-value to output, one-to-one correspondence 
            to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class QN(tnn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        """Layers for a Q Network"""

        super(DQN, self).__init__()
        self.lin(in_channels, num_actions)

    def forward(self, x):
        x = self.lin(x)
        return x
