import random
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from copy import deepcopy


class ReplayMemory(object):
    """A very generic memory system, with a finite capacity."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        if batch_size >= len(self.memory):
            return self.memory
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_hot1(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self, m, n, num_actions):
        super(DQN_hot1, self).__init__()
        self.fc1 = nn.Linear(m * n, 15)
        self.fc2 = nn.Linear(15, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN_hot2(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self, m, n, num_actions):
        super(DQN_hot2, self).__init__()
        self.fc1 = nn.Linear(m * n, 100)
        self.fc2 = nn.Linear(100, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN_hot3(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self, m, n, num_actions):
        super(DQN_hot3, self).__init__()
        self.fc1 = nn.Linear(m * n, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_hot4(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self, m, n, num_actions):
        super(DQN_hot4, self).__init__()
        self.fc1 = nn.Linear(m * n, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_hot5(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    
    Params
    ------
    m, n: int
        Board size 
    num_actions: int 
        Number of action-value to output, one-to-one correspondence 
        to action in game.    
    """
    def __init__(self, m, n, num_actions):
        super(DQN_hot5, self).__init__()
        self.fc1 = nn.Linear(m * n, 1000)
        self.fc2 = nn.Linear(1000, 2000)
        self.fc3 = nn.Linear(2000, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_xy1(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a (x,y) coordinate board/action representation. 
    """
    def __init__(self):
        super(DQN_xy1, self).__init__()
        self.fc1 = nn.Linear(4, 15)
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN_xy2(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self):
        super(DQN_xy2, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN_xy3(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self):
        super(DQN_xy3, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_xy4(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """
    def __init__(self):
        super(DQN_xy4, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_xy5(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    
    Params
    ------
    m, n: int
        Board size 
    num_actions: int 
        Number of action-value to output, one-to-one 
        correspondence to action in game.    
    """
    def __init__(self):
        super(DQN_xy5, self).__init__()
        self.fc1 = nn.Linear(4, 1000)
        self.fc2 = nn.Linear(1000, 2000)
        self.fc3 = nn.Linear(2000, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_optuna(nn.Module):
    """An A MLP for DQN learning, topology as optimized
    by optuna"""
    def __init__(self):
        super(DQN_optuna, self).__init__()
        # Build network:
        # Params taken from wythoff_exp71 (Makefile).
        in_features = 4  # Initial
        out_features = [10, 11, 13]
        layers = []
        for out_feature in out_features:
            layers.append(nn.Linear(in_features, out_feature))
            layers.append(nn.ReLU())
            in_features = deepcopy(out_feature)

        # Output layer topo is fixed
        layers.append(nn.Linear(in_features, 1))

        # Net-ify
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DQN_conv1(nn.Module):
    def __init__(self, m, n, num_actions):
        """A ConvNet for DQN Learning
        
        Params
        ------
        m,n: int
            Board size 
        num_actions: int 
            Number of action-value to output, one-to-one correspondence 
            to action in game.
        """
        super(DQN_conv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(16 * (n - 2) * (m - 2), 10)
        self.fc2 = nn.Linear(10, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class DQN_conv2(nn.Module):
    def __init__(self, m, n, num_actions):
        """A ConvNet for DQN Learning
        
        Params
        ------
        m,n: int
            Board size 
        num_actions: int 
            Number of action-value to output, one-to-one correspondence 
            to action in game.
        """
        super(DQN_conv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(16 * (n - (2 * 3)) * (m - (2 * 3)), 20)
        self.fc5 = nn.Linear(20, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DQN_conv3(nn.Module):
    def __init__(self, m, n, num_actions):
        """A ConvNet for DQN Learning
        
        Params
        ------
        m,n: int
            Board size 
        num_actions: int 
            Number of action-value to output, one-to-one correspondence 
            to action in game.
        """
        super(DQN_conv3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(64 * (n - (2 * 3)) * (m - (2 * 3)), 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class HotCold2(nn.Module):
    """Layers for a Hot-Cold strategy

    As described in:

    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=15):
        super(HotCold2, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


class HotCold3(nn.Module):
    """Two layers for a Hot-Cold strategy

    Related to the model described in:

    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=100, num_hidden2=25):
        super(HotCold3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)


class DeepTable3(nn.Module):
    """A deep differentialable 'Table' for learning one-hot input and output.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hidden1=200,
                 num_hidden2=100):
        super(DeepTable3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1, bias=False)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2, bias=False)
        self.fc3 = nn.Linear(num_hidden2, out_channels, bias=False)

        self.fc1.weight.data.uniform_(0.0, 0.0)
        self.fc3.weight.data.uniform_(0.0, 0.0)
        # self.fc3.bias.data.uniform_(0, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(self.fc2(x))
        return self.fc3(x)


class Table(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        """A differentialable 'Table' for learning one-hot input and output."""

        super(Table, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias=False)
        self.fc1.weight.data.uniform_(0, 0)

    def forward(self, x):
        return self.fc1(x)


class LinQN1(nn.Module):
    def __init__(self, in_channels=4, num_actions=2):
        """One layer linear Q model.

        Note: this functions as a Q-agent in the Sutton and
        Barto sense, wanted a torch optim compatible implementation.
        ...We waste some electrons to simplify the code...
        """

        super(LinQN1, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_actions)

    def forward(self, x):
        return self.fc1(x)


# class MLP(nn.Module):
#     class DQN_mlp1(nn.Module):
#     """Layers for a Deep Q Network, based on a simple MLP."""
#     def __init__(self, m, n, num_actions, num_hidden1=1000):
#         super(DQN_mlp1, self).__init__()
#         self.m = m
#         self.n = n
#         self.num_hidden1 = num_hidden1

#         self.fc1 = nn.Linear(m * n, num_hidden1)
#         self.fc2 = nn.Linear(num_hidden1, num_actions)

# class ReLu2(nn.Module):
#     """2-layer ReLu Network"""
#     def __init__(self, in_channels, num_actions, num_hidden=200):
#         super(ReLu2, self).__init__()
#         self.fc1 = nn.Linear(in_channels, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, num_actions)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return F.relu(self.fc2(x))

# class ReLu3(nn.Module):
#     """3-layer ReLu Network"""
#     def __init__(self,
#                  in_channels,
#                  num_actions,
#                  num_hidden1=200,
#                  num_hidden2=100):
#         super(ReLu3, self).__init__()
#         self.fc1 = nn.Linear(in_channels, num_hidden1)
#         self.fc2 = nn.Linear(num_hidden1, num_hidden2)
#         self.fc3 = nn.Linear(num_hidden2, num_actions)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
