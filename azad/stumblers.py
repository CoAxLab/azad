import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        """Layers for a Deep Q Network

        Based on:
        Minh, V. et al, 2015. Human-level control through deep reinforcement 
        learning. Nature, 518, pp.529–533. Available at: 
        http://dx.doi.org/10.1038/nature14236.
        
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


class TwoQN(nn.Module):
    """Simple 1-layer Q Network"""

    def __init__(self, in_channels, num_actions, num_hidden=200):
        super(TwoQN, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class ThreeQN(nn.Module):
    """Simple 1-layer Q Network"""

    def __init__(self,
                 in_channels,
                 num_actions,
                 num_hidden1=200,
                 num_hidden2=100):
        super(ThreeQN, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))


class QN(object):
    def __init__(self, in_channels=4, num_actions=2):
        """One layer linear Q model.
        
        Note: this functions as a Q-agent in the Sutton and 
        Barto sense, wanted a torch optim compatible implementation.
        ...We waste some electrons to simplify the code...
        """

        super(QN, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_actions)

    def forward(self, x):
        return self.fc1(x)
